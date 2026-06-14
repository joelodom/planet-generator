//! CPU-side mesh data and the geometry builders that turn planet samples into
//! triangles: terrain chunks (with crack-hiding skirts), deterministic
//! vegetation instances, and the static base meshes (trees, shrubs, water,
//! fullscreen triangle) the renderer instances and reuses.

use crate::flora;
use crate::lod::ChunkKey;
use crate::planet::{self, Planet, Biome, FACES, PLANET_RADIUS};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use rand::{RngExt, SeedableRng};
use rand::rngs::StdRng;
use std::f32::consts::TAU;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// Detail settings that are baked into chunk geometry, shared (lock-free) with
/// the meshing workers so the graphics menu can retune them at runtime. Changing
/// any of these requires rebuilding chunks (the renderer drops its cache).
pub struct MeshConfig {
    grid: AtomicU32,
    veg_min_level: AtomicU32,
    veg_density: AtomicU32,
}

impl MeshConfig {
    pub fn new(grid: u32, veg_min_level: u32, veg_density: u32) -> Arc<Self> {
        Arc::new(Self {
            grid: AtomicU32::new(grid),
            veg_min_level: AtomicU32::new(veg_min_level),
            veg_density: AtomicU32::new(veg_density),
        })
    }

    /// Reasonable fixed config for tests and the offscreen smoke render.
    #[cfg(test)]
    pub fn standard() -> Arc<Self> {
        Self::new(20, 13, 70)
    }

    pub fn set(&self, grid: u32, veg_min_level: u32, veg_density: u32) {
        self.grid.store(grid, Ordering::Relaxed);
        self.veg_min_level.store(veg_min_level, Ordering::Relaxed);
        self.veg_density.store(veg_density, Ordering::Relaxed);
    }

    fn snapshot(&self) -> (usize, u32, usize) {
        (
            self.grid.load(Ordering::Relaxed) as usize,
            self.veg_min_level.load(Ordering::Relaxed),
            self.veg_density.load(Ordering::Relaxed) as usize,
        )
    }
}

/// Per-vertex terrain attributes. Plain data, uploaded straight to the GPU.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

/// Per-vertex vegetation attributes. Unlike terrain, plants are **textured**: the
/// 3D models (see `crate::models`) carry UVs into a shared `texture_2d_array`, and
/// each vertex names its material's `layer` in that array. 36 B, no padding —
/// uploaded straight to the GPU as the vegetation base-mesh vertex.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VegVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub layer: u32, // index into the shared vegetation texture array
}

/// An indexed vegetation mesh in local space (base at the origin, +Y up, height
/// normalised to 1.0 render unit — the placer scales by the species' target size).
/// One per archetype; the renderer concatenates them into a single base buffer.
pub struct VegMesh {
    pub vertices: Vec<VegVertex>,
    pub indices: Vec<u32>,
}

/// One planted instance: the local→world transform (columns of a `Mat4`) and a
/// colour tint. The renderer draws a species' shared base mesh once per instance,
/// so a chunk stores ~80 bytes per plant instead of a baked copy of its geometry.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VegInstance {
    pub model: [[f32; 4]; 4],
    pub tint: [f32; 4], // rgb tint; a unused
}

/// A chunk's vegetation as instances grouped by archetype mesh: `instances` is one
/// flat buffer, and `draws` is a `(mesh_index, start, count)` run per mesh so the
/// renderer issues one instanced draw of each archetype's base mesh.
#[derive(Default)]
pub struct VegChunk {
    pub instances: Vec<VegInstance>,
    pub draws: Vec<(u32, u32, u32)>,
}

/// Everything a worker produces for one chunk: the terrain mesh and its vegetation
/// as per-species instances (drawn against the shared base meshes the renderer holds
/// — see `crate::flora`). Species geometry is *not* baked in, so a chunk's veg costs
/// only a handful of bytes per plant.
pub struct CpuChunk {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub veg: VegChunk,
}

// Crack-hiding skirts around each chunk edge.
const SKIRT_DEPTH_FACTOR: f32 = 3.0; // skirt depth ≈ this × a terrain quad's width
const SKIRT_MIN_DEPTH: f32 = 2.0; // render units

// Vegetation scatter. The density knob and min LOD level are runtime settings (see
// MeshConfig); the rest is fixed tuning.
const VEG_MIN_GROUND_HEIGHT: f32 = 1.0; // skip water/waterline (render units)
const VEG_MAX_STEEPNESS: f32 = 0.5; // skip cliffs
// Bury a plant's base this fraction of its height to hide the ground seam. Models
// sit base-on-ground (y=0) and `scale` is now the plant's true height in units, so
// this is small: a ~30 m tree sinks ~1.5 m, grass a few cm. (Was 0.15 when `scale`
// was a ~1.0 jitter; lowered for the real-height archetype models.)
const VEG_SINK: f32 = 0.05;
const VEG_TINT_JITTER: f32 = 0.06; // ± per-plant brightness so a stand isn't uniform

// Area-proportional placement. The old model made a flat `density` attempts for
// EVERY chunk regardless of size, so a fine near chunk packed the same count into
// ~1/4-per-LOD-level the ground area of a coarse far one — the near-field "wall of
// trees", and a total plant count that ballooned as you zoomed in. Instead we scale
// attempts by the chunk's actual ground area, so *areal* plant density is
// LOD-independent (a patch looks as dense from orbit as underfoot), and cap it so a
// large coarse chunk can't request tens of thousands of plants.
// (These two were halved 2026-06-08 on in-world feedback that flora ran too dense —
// doubling the reference area and halving the cap cuts attempts ~50% at every LOD.)
pub const VEG_REFERENCE_AREA: f32 = 2_200.0; // ground area (unit²) at which `density` == raw attempts
const VEG_MAX_ATTEMPTS: usize = 192; // per-chunk attempt (hence instance) cap — bounds CPU + Pool-B memory

// Per-plant age/size: a reverse-J distribution (many young/small, few mature) drawn
// as a skewed power law, so a stand isn't all one size — and, since most plants come
// out small, scenes read more open. Multiplies the species' mature size on top of
// the fine ± scale jitter.
const VEG_AGE_MIN: f32 = 0.45; // youngest plant = this × the species' mature size
const VEG_AGE_SKEW: f32 = 1.8; // >1 skews the population toward young/small

// Per-species clustering. Each species (see `flora`) has its own linear-decay
// radius: certain at a seed point, fading to zero that far out. A species' seed
// points sit on a per-species Worley grid (per cube face) spaced SEED_SPACING ×
// its radius, so plant types cluster at wildly different scales — trees blanketing
// regions, ground cover in tight patches — and thin to nothing between seeds.
const SEED_SPACING: f32 = 2.0; // seed spacing, in units of a species' cluster radius

impl CpuChunk {
    /// Build the terrain mesh and vegetation for one quadtree node, at the detail
    /// level given by `cfg` (terrain grid resolution + vegetation rules).
    pub fn build(planet: &Planet, key: ChunkKey, cfg: &MeshConfig) -> CpuChunk {
        let (grid, veg_min_level, veg_density) = cfg.snapshot();
        let face = &FACES[key.face as usize];
        let (u0, v0, size) = key.face_rect();

        let n = grid + 1;
        let mut vertices: Vec<Vertex> = Vec::with_capacity(n * n);
        let mut dirs: Vec<Vec3> = Vec::with_capacity(n * n);

        // Surface grid. The ocean is part of this same mesh: ocean vertices sit at
        // sea level (height clamped to 0), so the sea is always on the exact same
        // grid as the land at every LOD. That's what keeps land above water (no
        // overflow) and avoids any separate-water-surface z-fighting/poke-through.
        // The deeper water is still colored darker (from the true height).
        for r in 0..n {
            for c in 0..n {
                let u = u0 + size * (c as f32 / grid as f32);
                let v = v0 + size * (r as f32 / grid as f32);
                let cube = face.base + face.right * u + face.up * v;
                let dir = planet::cube_to_sphere(cube);
                // Lean terrain sample: meshing needs only height + color, so this
                // skips the slope probes everywhere they can't change the biome.
                let (height, color) = planet.sample_terrain(dir);
                let radius = PLANET_RADIUS + height.max(0.0);
                let pos = dir * radius;
                dirs.push(dir);
                vertices.push(Vertex { pos: pos.into(), normal: dir.into(), color: color.into() });
            }
        }

        // Indices for the grid quads (two triangles each).
        let mut indices: Vec<u32> = Vec::with_capacity(grid * grid * 6);
        let idx = |r: usize, c: usize| (r * n + c) as u32;
        for r in 0..grid {
            for c in 0..grid {
                let a = idx(r, c);
                let b = idx(r, c + 1);
                let d = idx(r + 1, c);
                let e = idx(r + 1, c + 1);
                indices.extend_from_slice(&[a, d, b, b, d, e]);
            }
        }

        // Smooth normals from accumulated triangle normals, forced to point
        // outward from the planet centre (so lighting is correct regardless of
        // per-face winding).
        let mut accum = vec![Vec3::ZERO; n * n];
        for tri in indices.chunks_exact(3) {
            let (ia, ib, ic) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            let pa = Vec3::from(vertices[ia].pos);
            let pb = Vec3::from(vertices[ib].pos);
            let pc = Vec3::from(vertices[ic].pos);
            let nrm = (pb - pa).cross(pc - pa);
            accum[ia] += nrm;
            accum[ib] += nrm;
            accum[ic] += nrm;
        }
        for (i, v) in vertices.iter_mut().enumerate() {
            let mut nrm = accum[i].normalize_or_zero();
            if nrm.dot(dirs[i]) < 0.0 {
                nrm = -nrm;
            }
            if nrm == Vec3::ZERO {
                nrm = dirs[i];
            }
            v.normal = nrm.into();
        }

        // Skirts: a downward apron around all four edges so neighbouring chunks
        // at a coarser LOD can't reveal cracks/gaps to the sky behind them.
        let quad_arc = (size / grid as f32) * PLANET_RADIUS;
        let skirt = (quad_arc * SKIRT_DEPTH_FACTOR).max(SKIRT_MIN_DEPTH);
        let add_skirt = |edge: &[usize], vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>| {
            let start = vertices.len() as u32;
            for &gi in edge {
                let v = vertices[gi];
                let p = Vec3::from(v.pos);
                let down = p.normalize() * skirt;
                vertices.push(Vertex { pos: (p - down).into(), normal: v.normal, color: v.color });
            }
            for w in 0..edge.len() - 1 {
                let g0 = edge[w] as u32;
                let g1 = edge[w + 1] as u32;
                let s0 = start + w as u32;
                let s1 = start + w as u32 + 1;
                // Two windings emitted; terrain draws with culling disabled so
                // the apron is visible from either side.
                indices.extend_from_slice(&[g0, s0, g1, g1, s0, s1]);
            }
        };
        let top: Vec<usize> = (0..n).map(|c| idx(0, c) as usize).collect();
        let bottom: Vec<usize> = (0..n).map(|c| idx(grid, c) as usize).collect();
        let left: Vec<usize> = (0..n).map(|r| idx(r, 0) as usize).collect();
        let right: Vec<usize> = (0..n).map(|r| idx(r, grid) as usize).collect();
        add_skirt(&top, &mut vertices, &mut indices);
        add_skirt(&bottom, &mut vertices, &mut indices);
        add_skirt(&left, &mut vertices, &mut indices);
        add_skirt(&right, &mut vertices, &mut indices);

        let veg = place_vegetation(planet, key, face, u0, v0, size, veg_min_level, veg_density);

        CpuChunk { vertices, indices, veg }
    }
}

/// Area-proportional, capped placement-attempt count for a chunk of face-space edge
/// `size` (cube coords, where a full face spans `[-1, 1]`) at detail `density`.
/// Ground area ≈ `(size · PLANET_RADIUS)²`; attempts scale with it against
/// [`VEG_REFERENCE_AREA`] so areal density is LOD-independent, capped by
/// [`VEG_MAX_ATTEMPTS`].
fn veg_attempts(size: f32, density: usize) -> usize {
    let area = (size * PLANET_RADIUS).powi(2);
    ((density as f32 * area / VEG_REFERENCE_AREA).round() as usize).min(VEG_MAX_ATTEMPTS)
}

/// Deterministically scatter vegetation across a chunk according to biome rules.
/// Seeded by the chunk key so the same ground always grows the same plants.
/// `min_level` gates how far out plants appear; `density` sets the areal placement
/// rate, turned into a capped, area-proportional attempt count (see [`veg_attempts`]).
#[allow(clippy::too_many_arguments)]
fn place_vegetation(
    planet: &Planet,
    key: ChunkKey,
    face: &planet::CubeFace,
    u0: f32,
    v0: f32,
    size: f32,
    min_level: u32,
    density: usize,
) -> VegChunk {
    if key.level < min_level {
        return VegChunk::default();
    }

    let mut rng = StdRng::seed_from_u64(key.hash(planet.seed));
    let mut presence = [0.0f32; flora::SPECIES_PER_BIOME]; // scratch, refilled per attempt
    let mut planted: Vec<(u32, VegInstance)> = Vec::new();
    for _ in 0..veg_attempts(size, density) {
        let u = u0 + size * rng.random::<f32>();
        let v = v0 + size * rng.random::<f32>();
        let cube = face.base + face.right * u + face.up * v;
        let dir = planet::cube_to_sphere(cube);

        // Biome dithered across boundaries so flora intermixes over the same band
        // the colours blend across. Nothing grows in water or on bare cliffs.
        let tj = rng.random::<f32>() * 2.0 - 1.0;
        let mj = rng.random::<f32>() * 2.0 - 1.0;
        let s = planet.sample_blended(dir, tj, mj);
        if s.height < VEG_MIN_GROUND_HEIGHT || s.steepness > VEG_MAX_STEEPNESS {
            continue;
        }
        let lushness = biome_coverage(s.biome);
        if lushness <= 0.0 {
            continue;
        }
        let species_ids = planet.flora.biome_species(s.biome);
        if species_ids.is_empty() {
            continue;
        }

        // Each species' presence here is a linear decay from its nearest seed point
        // at that species' own cluster scale, times its mix weight. The sum drives
        // how likely a plant grows; the per-species presences weight which one does.
        let mut sum = 0.0f32;
        for (k, &id) in species_ids.iter().enumerate() {
            let sp = planet.flora.species(id);
            let w = species_presence(planet.seed, key.face, u, v, id, sp.cluster_radius) * sp.weight;
            presence[k] = w;
            sum += w;
        }
        if sum <= 0.0 || rng.random::<f32>() >= lushness * sum.min(1.0) {
            continue;
        }
        // Weighted pick: which species grows here, proportional to local presence.
        let mut pick = rng.random::<f32>() * sum;
        let mut chosen = species_ids[0];
        for (k, &id) in species_ids.iter().enumerate() {
            if pick < presence[k] {
                chosen = id;
                break;
            }
            pick -= presence[k];
        }
        let species = planet.flora.species(chosen);

        // Plant it: upright on the surface, with a yaw spin and a size jitter — emit
        // an instance (transform + tint) rather than baking the geometry.
        let up = dir;
        let yaw = rng.random_range(0.0..TAU);
        // Reverse-J age: most plants young/small, a few mature (see VEG_AGE_*).
        let age = VEG_AGE_MIN + (1.0 - VEG_AGE_MIN) * rng.random::<f32>().powf(VEG_AGE_SKEW);
        let scale = rng.random_range(species.scale_min..species.scale_max) * age;
        // Sink the base to hide the ground seam, scaled by the plant so a small young
        // one isn't buried whole (the seam to hide scales with the plant's footprint).
        let pos = up * (PLANET_RADIUS + s.height - VEG_SINK * scale);
        let rot = planet::upright_rotation(up, yaw);
        let model = Mat4::from_scale_rotation_translation(Vec3::splat(scale), rot, pos);
        let t = 1.0 + (rng.random::<f32>() - 0.5) * VEG_TINT_JITTER;
        // Key the instance by its archetype MESH index (not species id): species that
        // share a mesh — e.g. the boulder across biomes — then merge into one draw.
        planted.push((species.mesh_index, VegInstance { model: model.to_cols_array_2d(), tint: [t, t, t, 1.0] }));
    }

    // Group instances by mesh into contiguous runs, one instanced draw each.
    planted.sort_by_key(|(id, _)| *id);
    let mut veg = VegChunk::default();
    for (id, inst) in planted {
        let start = veg.instances.len() as u32;
        veg.instances.push(inst);
        match veg.draws.last_mut() {
            Some((sid, _, count)) if *sid == id => *count += 1,
            _ => veg.draws.push((id, start, 1)),
        }
    }
    veg
}

/// Bake a chunk's instances back into one world-space mesh — for the visual tests
/// only (at runtime the renderer draws them instanced). Mirrors `vegetation.wgsl`.
#[cfg(test)]
impl VegChunk {
    pub fn bake(&self, flora: &crate::flora::Flora) -> MeshData {
        // Veg meshes are textured (no per-vertex colour), but this baker feeds the
        // untextured terrain shader, so it flattens each vertex to its texture
        // layer's average colour — enough for the silhouette/clustering visual tools.
        let avg = &flora.textures().layer_avg;
        let mut m = MeshData { vertices: Vec::new(), indices: Vec::new() };
        for &(mesh_id, start, count) in &self.draws {
            let src = flora.mesh(mesh_id);
            for inst in &self.instances[start as usize..(start + count) as usize] {
                let model = Mat4::from_cols_array_2d(&inst.model);
                let nmat = glam::Mat3::from_mat4(model);
                let tint = Vec3::new(inst.tint[0], inst.tint[1], inst.tint[2]);
                let base = m.vertices.len() as u32;
                for v in &src.vertices {
                    let p = model.transform_point3(Vec3::from(v.pos));
                    let n = (nmat * Vec3::from(v.normal)).normalize_or_zero();
                    let c = (Vec3::from(avg[v.layer as usize]) * tint).clamp(Vec3::ZERO, Vec3::ONE);
                    m.vertices.push(Vertex { pos: p.into(), normal: n.into(), color: c.into() });
                }
                m.indices.extend(src.indices.iter().map(|&i| base + i));
            }
        }
        m
    }
}

/// Linear-decay presence of one species at a point: 1 at its nearest seed point,
/// fading to 0 at the species' cluster radius. Seeds sit on a per-species Worley
/// grid (per cube face) spaced SEED_SPACING × that radius, so each species clusters
/// at its own scale — metres for some ground cover, hundreds of km for forest trees.
fn species_presence(seed: u64, face: u8, u: f32, v: f32, species_id: u32, radius_units: f32) -> f32 {
    let radius_uv = radius_units / PLANET_RADIUS;
    if radius_uv <= 0.0 {
        return 0.0;
    }
    let cell = (radius_uv * SEED_SPACING) as f64;
    let cu = (u as f64 / cell).floor() as i64;
    let cv = (v as f64 / cell).floor() as i64;
    let mut best = 0.0f32;
    for di in -1..=1 {
        for dj in -1..=1 {
            let (gi, gj) = (cu + di, cv + dj);
            let h = seed_hash(seed, face, species_id, gi, gj);
            // Seed point jittered inside the cell (two 16-bit fractions from h).
            let jx = (h & 0xFFFF) as f64 / 65535.0;
            let jy = ((h >> 16) & 0xFFFF) as f64 / 65535.0;
            let su = (gi as f64 + jx) * cell;
            let sv = (gj as f64 + jy) * cell;
            let du = u as f64 - su;
            let dv = v as f64 - sv;
            let d = (du * du + dv * dv).sqrt() as f32;
            best = best.max(1.0 - d / radius_uv); // linear decay; negatives clamp to 0 via max
        }
    }
    best
}

/// Stable hash of a per-species seed-grid cell (per planet, per face, per species).
fn seed_hash(seed: u64, face: u8, species: u32, gi: i64, gj: i64) -> u64 {
    let mut h = seed ^ ((face as u64) << 56) ^ (species as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for x in [gi as u64, gj as u64] {
        h ^= x.wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_add(h << 6).wrapping_add(h >> 2);
        h = h.wrapping_mul(0x0100_0000_01B3);
    }
    h
}

/// Peak vegetation coverage for a biome (probability a candidate at a stand core
/// becomes a plant). Lush biomes are dense; harsh ones sparse; barren ones zero.
fn biome_coverage(biome: Biome) -> f32 {
    match biome {
        Biome::TropicalForest => 0.95,
        Biome::TemperateForest => 0.85,
        Biome::Grassland => 0.80,
        Biome::BorealForest => 0.70,
        Biome::Tundra => 0.40,
        Biome::Mountain => 0.30,
        Biome::Beach => 0.30,
        Biome::Desert => 0.25,
        Biome::Snow => 0.15,     // a few snow-capped/dark forms scattered on white
        Biome::PolarIce => 0.06, // near-barren — the rare ice block or stranded boulder
        Biome::Ocean => 0.0,     // sea floor — barren
    }
}

// ---------------------------------------------------------------------------
// Mesh container
// ---------------------------------------------------------------------------

/// A simple indexed mesh of [`Vertex`]. Now only the headless visual tools use it,
/// to bake instanced vegetation back to world space for the terrain shader
/// (`VegChunk::bake`); runtime terrain builds `Vec<Vertex>` directly and runtime
/// vegetation uses [`VegMesh`].
#[cfg(test)]
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presence_is_bounded_and_clusters_at_its_scale() {
        // Walk a fine line (≈50 m steps) sampling a broad species and a tiny one.
        // Presence stays in [0, 1]; the tiny species flips between patches far faster
        // than the broad one — i.e. it clusters at a much smaller scale.
        let big_r = 30_000.0_f32; // ~300 km
        let small_r = 5.0_f32; // ~50 m
        let step = 5.0 / PLANET_RADIUS; // ≈50 m in UV
        let v = 0.1f32;
        let n = 4000;
        let (mut max_small, mut min_small) = (0.0f32, 1.0f32);
        let (mut big_var, mut small_var) = (0.0f64, 0.0f64);
        let mut prev_big = species_presence(7, 2, 0.0, v, 1, big_r);
        let mut prev_small = species_presence(7, 2, 0.0, v, 2, small_r);
        for i in 1..n {
            let u = i as f32 * step;
            let pb = species_presence(7, 2, u, v, 1, big_r);
            let ps = species_presence(7, 2, u, v, 2, small_r);
            assert!((0.0..=1.0).contains(&pb) && (0.0..=1.0).contains(&ps), "presence out of range");
            max_small = max_small.max(ps);
            min_small = min_small.min(ps);
            big_var += (pb - prev_big).abs() as f64;
            small_var += (ps - prev_small).abs() as f64;
            prev_big = pb;
            prev_small = ps;
        }
        // A real linear-decay field: near 1 at seed points, ~0 in the gaps between.
        assert!(max_small > 0.8 && min_small < 0.2, "presence should span near-1 to near-0 (max {max_small}, min {min_small})");
        // Across the same 50 m steps the tiny species varies far faster than the broad
        // one (clusters at metres, not hundreds of km).
        assert!(small_var > big_var * 5.0, "small-radius species should vary far faster ({small_var:.1} vs {big_var:.1})");
    }

    #[test]
    fn veg_attempts_scale_with_area_and_cap() {
        // Edge length (cube coords) of a chunk at a given LOD level: a face spans [-1, 1].
        let edge = |level: u32| 2.0 / (1u32 << level) as f32;

        // Doubling the LOD edge quadruples ground area, so ~4× the attempts — i.e.
        // areal density is LOD-independent (the whole point of the redesign).
        let fine = veg_attempts(edge(16), 100);
        let coarse = veg_attempts(edge(15), 100);
        assert!(fine > 0, "a fine chunk should still get some attempts");
        assert!((coarse as f32 / fine as f32 - 4.0).abs() < 0.5, "≈4× area ⇒ ≈4× attempts ({fine} → {coarse})");

        // A large coarse chunk is bounded by the cap, never tens of thousands.
        assert_eq!(veg_attempts(edge(10), 750), VEG_MAX_ATTEMPTS, "coarse chunk must hit the cap");

        // Zero density ⇒ no vegetation.
        assert_eq!(veg_attempts(edge(16), 0), 0);
    }
}
