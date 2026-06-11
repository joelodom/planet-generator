//! Load the embedded P0 flora `.glb` models into engine vegetation data.
//!
//! Each model (a formcast bake — see `flora-revamp/`) is glTF 2.0: +Y-up, metres,
//! base on the ground plane, with semantic submeshes (`trunk`, `canopy`, …) that
//! each carry a baseColorTexture + UVs. The renderer's vegetation pass is textured
//! and instanced, so this module turns every model into:
//!
//! * a [`VegMesh`] — all submeshes concatenated, **recentred on its base and scaled
//!   so its height is exactly 1.0 render unit** (the placer multiplies by the
//!   species' target size), every vertex tagged with its material's texture **layer**;
//! * a shared [`TextureArray`] — every material's baseColorTexture resampled to a
//!   common square and stacked as array layers (+ a box-filter mip chain), uploaded
//!   once as one `texture_2d_array`.
//!
//! glTF samplers default to REPEAT, which is what the tiling trunk/bark UVs want, so
//! the renderer pairs this array with a repeat sampler. Foliage/spine materials are
//! authored `alphaMode=MASK`; the shader alpha-tests, so the texture alpha is what
//! cuts the leaf-card silhouettes. (The models also embed a provenance photo as an
//! *unreferenced* glTF image — we only ever decode images a material points at, so
//! that photo is skipped.)

use crate::mesh::{VegMesh, VegVertex};

/// Edge (texels) of every vegetation texture-array layer. Each material's
/// baseColorTexture is resampled to this square so all layers share one array.
/// 512 keeps the array light (~32 MB incl. mips for ~23 layers); a future settings
/// preset can raise it on big GPUs (the Windows/RTX 5090 target).
pub const VEG_TEX_SIZE: u32 = 512;

/// The decoded, GPU-ready vegetation textures: every archetype material's
/// baseColorTexture, resampled to [`VEG_TEX_SIZE`] and stacked as array layers,
/// each with a precomputed box-filter mip chain. Uploaded once as a
/// `texture_2d_array` (see `gfx::VegGpu`).
pub struct TextureArray {
    /// Edge of mip 0 (`= VEG_TEX_SIZE`).
    pub size: u32,
    /// `layers[layer][mip]` — tightly-packed sRGB RGBA8 for that layer + mip level
    /// (mip 0 is `size × size`, each level halves until 1×1).
    pub layers: Vec<Vec<Vec<u8>>>,
    /// Alpha-weighted average RGB (0..1) of each layer's mip 0. Used only by the
    /// headless gallery/closeup tools, which draw veg geometry through the
    /// untextured terrain shader and so need a representative flat colour per layer.
    #[cfg_attr(not(test), allow(dead_code))]
    pub layer_avg: Vec<[f32; 3]>,
}

impl TextureArray {
    /// Mip levels in the chain (`log2(size) + 1`, down to 1×1).
    pub fn mip_levels(&self) -> u32 {
        32 - (self.size.max(1)).leading_zeros() // floor(log2(size)) + 1
    }
}

/// The whole P0 library: one [`VegMesh`] per input model (in the order given) and
/// the shared [`TextureArray`] their vertices index into.
pub struct Library {
    pub meshes: Vec<VegMesh>,
    pub textures: TextureArray,
}

/// Load `glbs` (each `(label, bytes)`, label used only for error messages) into
/// normalised meshes + a shared texture array. Texture-array layers are assigned
/// globally in load order. Panics on a malformed model — these are build-time
/// embedded assets, so a bad one is a bug to fix, not a runtime condition to
/// degrade around.
pub fn load(glbs: &[(&str, &[u8])]) -> Library {
    let mut meshes = Vec::with_capacity(glbs.len());
    let mut layers: Vec<Vec<Vec<u8>>> = Vec::new();
    let mut layer_avg: Vec<[f32; 3]> = Vec::new();

    for &(label, bytes) in glbs {
        let gltf = gltf::Gltf::from_slice(bytes).unwrap_or_else(|e| panic!("flora model {label}: parse failed: {e}"));
        let blob = gltf.blob.as_deref().unwrap_or_else(|| panic!("flora model {label}: GLB has no binary blob"));

        let mut vertices: Vec<VegVertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        // Within one model, several primitives can share a source image; map the
        // glTF image index → the global texture-array layer so we decode each once.
        let mut image_layer: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();

        for mesh in gltf.document.meshes() {
            for prim in mesh.primitives() {
                let reader = prim.reader(|buffer| if buffer.index() == 0 { Some(blob) } else { None });
                let positions: Vec<[f32; 3]> = reader
                    .read_positions()
                    .unwrap_or_else(|| panic!("flora model {label}: primitive has no POSITION"))
                    .collect();
                if positions.is_empty() {
                    continue;
                }
                // Normals: use the model's if present, else flat-compute from faces below.
                let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|it| it.collect());
                let uvs: Vec<[f32; 2]> = match reader.read_tex_coords(0) {
                    Some(tc) => tc.into_f32().collect(),
                    None => vec![[0.0, 0.0]; positions.len()],
                };
                let prim_indices: Vec<u32> = match reader.read_indices() {
                    Some(ix) => ix.into_u32().collect(),
                    None => (0..positions.len() as u32).collect(),
                };

                // Resolve this primitive's texture → a global array layer.
                let img_index = prim
                    .material()
                    .pbr_metallic_roughness()
                    .base_color_texture()
                    .map(|info| info.texture().source().index())
                    .unwrap_or_else(|| panic!("flora model {label}: a primitive has no baseColorTexture"));
                let layer = *image_layer.entry(img_index).or_insert_with(|| {
                    let (rgba, w, h) = decode_image(&gltf, blob, img_index, label);
                    let resized = resample_rgba(&rgba, w, h, VEG_TEX_SIZE, VEG_TEX_SIZE);
                    layer_avg.push(alpha_weighted_avg(&resized));
                    layers.push(build_mips(resized, VEG_TEX_SIZE));
                    (layers.len() - 1) as u32
                });

                // Compute flat normals only if the model omitted them.
                let computed = normals.is_none().then(|| flat_normals(&positions, &prim_indices));
                let base = vertices.len() as u32;
                for i in 0..positions.len() {
                    let normal = normals.as_ref().map(|n| n[i]).unwrap_or_else(|| computed.as_ref().unwrap()[i]);
                    vertices.push(VegVertex { pos: positions[i], normal, uv: uvs[i], layer });
                }
                indices.extend(prim_indices.iter().map(|&i| base + i));
            }
        }

        normalize_mesh(&mut vertices, label);
        meshes.push(VegMesh { vertices, indices });
    }

    let textures = TextureArray { size: VEG_TEX_SIZE, layers, layer_avg };
    tracing::info!(
        models = meshes.len(),
        layers = textures.layers.len(),
        tex_size = VEG_TEX_SIZE,
        mips = textures.mip_levels(),
        verts = meshes.iter().map(|m| m.vertices.len()).sum::<usize>(),
        "flora model library loaded"
    );
    Library { meshes, textures }
}

/// Recentre a model on its base (min-Y → 0, X/Z centroid → 0) and scale it so its
/// height is exactly 1.0 render unit. Uniform scale leaves normals unit-length, so
/// only positions change. Metres→units falls out for free: dividing by the model's
/// own (metre) height normalises it; the caller multiplies by a unit target height.
fn normalize_mesh(vertices: &mut [VegVertex], label: &str) {
    if vertices.is_empty() {
        panic!("flora model {label}: produced an empty mesh");
    }
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for v in vertices.iter() {
        for a in 0..3 {
            lo[a] = lo[a].min(v.pos[a]);
            hi[a] = hi[a].max(v.pos[a]);
        }
    }
    let height = (hi[1] - lo[1]).max(1e-6);
    let inv = 1.0 / height;
    let cx = (lo[0] + hi[0]) * 0.5;
    let cz = (lo[2] + hi[2]) * 0.5;
    for v in vertices.iter_mut() {
        v.pos[0] = (v.pos[0] - cx) * inv;
        v.pos[1] = (v.pos[1] - lo[1]) * inv;
        v.pos[2] = (v.pos[2] - cz) * inv;
    }
}

/// Per-vertex flat normals, accumulated from incident triangle normals — only used
/// when a model omits NORMAL (the P0 models all carry their own).
fn flat_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    use glam::Vec3;
    let mut accum = vec![Vec3::ZERO; positions.len()];
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (pa, pb, pc) = (Vec3::from(positions[a]), Vec3::from(positions[b]), Vec3::from(positions[c]));
        let n = (pb - pa).cross(pc - pa);
        accum[a] += n;
        accum[b] += n;
        accum[c] += n;
    }
    accum.iter().map(|n| n.normalize_or_zero().into()).collect()
}

/// Decode the PNG that glTF image `img_index` points at, out of the GLB blob.
fn decode_image(gltf: &gltf::Gltf, blob: &[u8], img_index: usize, label: &str) -> (Vec<u8>, u32, u32) {
    let image = gltf.document.images().nth(img_index).unwrap_or_else(|| panic!("flora model {label}: image {img_index} missing"));
    match image.source() {
        gltf::image::Source::View { view, mime_type } => {
            assert!(mime_type.contains("png"), "flora model {label}: image {img_index} is {mime_type}, expected png");
            let start = view.offset();
            let bytes = &blob[start..start + view.length()];
            decode_png(bytes).unwrap_or_else(|| panic!("flora model {label}: image {img_index} PNG decode failed"))
        }
        gltf::image::Source::Uri { .. } => panic!("flora model {label}: image {img_index} is an external URI; embedded GLB expected"),
    }
}

/// Decode a PNG to tightly-packed RGBA8 (RGB sources gain opaque alpha). Returns
/// `None` on any decode error (caller turns it into a labelled panic).
fn decode_png(bytes: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let (w, h) = (info.width, info.height);
    let used = &buf[..info.buffer_size()];
    let rgba = match info.color_type {
        png::ColorType::Rgba => used.to_vec(),
        png::ColorType::Rgb => {
            let mut v = Vec::with_capacity((w * h * 4) as usize);
            for px in used.chunks_exact(3) {
                v.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
            v
        }
        png::ColorType::Grayscale => used.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        png::ColorType::GrayscaleAlpha => used.chunks_exact(2).flat_map(|p| [p[0], p[0], p[0], p[1]]).collect(),
        png::ColorType::Indexed => return None, // formcast emits RGB/RGBA, never paletted
    };
    Some((rgba, w, h))
}

/// Bilinear resample tightly-packed RGBA8 from `sw×sh` to `dw×dh`.
fn resample_rgba(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let mut out = vec![0u8; (dw * dh * 4) as usize];
    let (sx, sy) = (sw as f32 / dw as f32, sh as f32 / dh as f32);
    for y in 0..dh {
        // Sample at pixel centres so the mapping is symmetric.
        let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, (sh - 1) as f32);
        let (y0, y1) = (fy.floor() as u32, (fy.floor() as u32 + 1).min(sh - 1));
        let ty = fy - y0 as f32;
        for x in 0..dw {
            let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, (sw - 1) as f32);
            let (x0, x1) = (fx.floor() as u32, (fx.floor() as u32 + 1).min(sw - 1));
            let tx = fx - x0 as f32;
            let px = |xi: u32, yi: u32| {
                let i = ((yi * sw + xi) * 4) as usize;
                [src[i], src[i + 1], src[i + 2], src[i + 3]]
            };
            let (p00, p10, p01, p11) = (px(x0, y0), px(x1, y0), px(x0, y1), px(x1, y1));
            let o = ((y * dw + x) * 4) as usize;
            for c in 0..4 {
                let top = p00[c] as f32 * (1.0 - tx) + p10[c] as f32 * tx;
                let bot = p01[c] as f32 * (1.0 - tx) + p11[c] as f32 * tx;
                out[o + c] = (top * (1.0 - ty) + bot * ty).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Build a full box-filter mip chain from a `size×size` RGBA8 base, down to 1×1.
/// Alpha is averaged with the colour (so masked foliage thins gently with distance
/// rather than shimmering); the shader's cutoff keeps near detail crisp.
fn build_mips(base: Vec<u8>, size: u32) -> Vec<Vec<u8>> {
    let mut chain = vec![base];
    let mut w = size;
    while w > 1 {
        let prev = chain.last().unwrap();
        let nw = w / 2;
        let mut next = vec![0u8; (nw * nw * 4) as usize];
        for y in 0..nw {
            for x in 0..nw {
                let o = ((y * nw + x) * 4) as usize;
                for c in 0..4 {
                    let mut sum = 0u32;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let i = (((2 * y + dy) * w + (2 * x + dx)) * 4) as usize + c;
                            sum += prev[i] as u32;
                        }
                    }
                    next[o + c] = (sum / 4) as u8;
                }
            }
        }
        chain.push(next);
        w = nw;
    }
    chain
}

/// Alpha-weighted average RGB (0..1) of a tightly-packed RGBA8 layer — so a leaf
/// card's transparent background doesn't wash out its representative colour.
fn alpha_weighted_avg(rgba: &[u8]) -> [f32; 3] {
    let (mut sum, mut wsum) = ([0.0f64; 3], 0.0f64);
    for px in rgba.chunks_exact(4) {
        let a = px[3] as f64;
        for c in 0..3 {
            sum[c] += px[c] as f64 * a;
        }
        wsum += a;
    }
    if wsum > 0.0 {
        [(sum[0] / wsum / 255.0) as f32, (sum[1] / wsum / 255.0) as f32, (sum[2] / wsum / 255.0) as f32]
    } else {
        // Fully transparent layer (shouldn't happen): fall back to a plain average.
        let n = (rgba.len() / 4).max(1) as f64;
        let mut s = [0.0f64; 3];
        for px in rgba.chunks_exact(4) {
            for c in 0..3 {
                s[c] += px[c] as f64;
            }
        }
        [(s[0] / n / 255.0) as f32, (s[1] / n / 255.0) as f32, (s[2] / n / 255.0) as f32]
    }
}
