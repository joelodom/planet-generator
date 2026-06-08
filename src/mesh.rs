//! CPU-side mesh data and the geometry builders that turn planet samples into
//! triangles: terrain chunks (with crack-hiding skirts), deterministic
//! vegetation instances, and the static base meshes (trees, shrubs, water,
//! fullscreen triangle) the renderer instances and reuses.

use crate::lod::ChunkKey;
use crate::planet::{self, Planet, Biome, FACES, PLANET_RADIUS};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use rand::{RngExt, SeedableRng};
use rand::rngs::StdRng;

/// Per-vertex terrain/veg attributes. Plain data, uploaded straight to the GPU.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

/// One instanced placement (a tree or shrub): a full model matrix plus a color
/// tint. 80 bytes; cheap to stream a few thousand of per visible region.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct InstanceRaw {
    pub model: [[f32; 4]; 4],
    pub color: [f32; 4],
}

impl InstanceRaw {
    fn new(model: Mat4, color: Vec3) -> Self {
        Self { model: model.to_cols_array_2d(), color: [color.x, color.y, color.z, 1.0] }
    }
}

/// Everything a worker produces for one chunk: the terrain mesh and the
/// vegetation that grows on it, split by base mesh so each draws in one call.
pub struct CpuChunk {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub trees: Vec<InstanceRaw>,
    pub shrubs: Vec<InstanceRaw>,
}

/// Terrain tessellation per chunk side (quads). Total grid is (GRID+1)^2 verts.
pub const GRID: usize = 20;


/// Below this quadtree level, chunks are too coarse / too far to bother placing
/// individual plants on.
// Vegetation only appears on the finest chunks (≈ sub-km) — at coarser levels an
// Earth-sized chunk spans tens of km and individual plants would be invisible.
pub const VEG_MIN_LEVEL: u32 = 13;

// Crack-hiding skirts around each chunk edge.
const SKIRT_DEPTH_FACTOR: f32 = 3.0; // skirt depth ≈ this × a terrain quad's width
const SKIRT_MIN_DEPTH: f32 = 2.0; // render units

// Vegetation scatter.
const VEG_ATTEMPTS_PER_CHUNK: usize = 70;
const VEG_MIN_GROUND_HEIGHT: f32 = 1.0; // skip water/waterline (render units)
const VEG_MAX_STEEPNESS: f32 = 0.5; // skip cliffs
const TREE_SCALE_MIN: f32 = 0.8; // ~8–26 m trees (render units)
const TREE_SCALE_MAX: f32 = 2.6;
const SHRUB_SCALE_MIN: f32 = 0.25; // ~2.5–8 m shrubs
const SHRUB_SCALE_MAX: f32 = 0.8;
const TREE_TINT_JITTER: f32 = 0.08; // ± per-plant color variation
const SHRUB_TINT_JITTER: f32 = 0.10;
const SHRUB_TINT_BRIGHTEN: f32 = 1.1; // shrubs a touch lighter than the biome tint

impl CpuChunk {
    /// Build the terrain mesh and vegetation for one quadtree node.
    pub fn build(planet: &Planet, key: ChunkKey) -> CpuChunk {
        let face = &FACES[key.face as usize];
        let (u0, v0, size) = key.face_rect();

        let n = GRID + 1;
        let mut vertices: Vec<Vertex> = Vec::with_capacity(n * n);
        let mut dirs: Vec<Vec3> = Vec::with_capacity(n * n);

        // Surface grid. The ocean is part of this same mesh: ocean vertices sit at
        // sea level (height clamped to 0), so the sea is always on the exact same
        // grid as the land at every LOD. That's what keeps land above water (no
        // overflow) and avoids any separate-water-surface z-fighting/poke-through.
        // The deeper water is still colored darker (from the true height).
        for r in 0..n {
            for c in 0..n {
                let u = u0 + size * (c as f32 / GRID as f32);
                let v = v0 + size * (r as f32 / GRID as f32);
                let cube = face.base + face.right * u + face.up * v;
                let dir = planet::cube_to_sphere(cube);
                let s = planet.sample(dir);
                let radius = PLANET_RADIUS + s.height.max(0.0);
                let pos = dir * radius;
                dirs.push(dir);
                vertices.push(Vertex { pos: pos.into(), normal: dir.into(), color: s.color.into() });
            }
        }

        // Indices for the grid quads (two triangles each).
        let mut indices: Vec<u32> = Vec::with_capacity(GRID * GRID * 6);
        let idx = |r: usize, c: usize| (r * n + c) as u32;
        for r in 0..GRID {
            for c in 0..GRID {
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
        let quad_arc = (size / GRID as f32) * PLANET_RADIUS;
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
        let bottom: Vec<usize> = (0..n).map(|c| idx(GRID, c) as usize).collect();
        let left: Vec<usize> = (0..n).map(|r| idx(r, 0) as usize).collect();
        let right: Vec<usize> = (0..n).map(|r| idx(r, GRID) as usize).collect();
        add_skirt(&top, &mut vertices, &mut indices);
        add_skirt(&bottom, &mut vertices, &mut indices);
        add_skirt(&left, &mut vertices, &mut indices);
        add_skirt(&right, &mut vertices, &mut indices);

        let (trees, shrubs) = place_vegetation(planet, key, face, u0, v0, size);

        CpuChunk { vertices, indices, trees, shrubs }
    }
}

/// Deterministically scatter vegetation across a chunk according to biome rules.
/// Seeded by the chunk key so the same ground always grows the same plants.
fn place_vegetation(
    planet: &Planet,
    key: ChunkKey,
    face: &planet::CubeFace,
    u0: f32,
    v0: f32,
    size: f32,
) -> (Vec<InstanceRaw>, Vec<InstanceRaw>) {
    let mut trees = Vec::new();
    let mut shrubs = Vec::new();
    if key.level < VEG_MIN_LEVEL {
        return (trees, shrubs);
    }

    let mut rng = StdRng::seed_from_u64(key.hash(planet.seed));
    // More candidate slots at deeper levels (smaller chunks) keeps on-screen
    // density roughly constant as you descend.
    let attempts = VEG_ATTEMPTS_PER_CHUNK;

    for _ in 0..attempts {
        let u = u0 + size * rng.random::<f32>();
        let v = v0 + size * rng.random::<f32>();
        let cube = face.base + face.right * u + face.up * v;
        let dir = planet::cube_to_sphere(cube);
        let s = planet.sample(dir);

        // Nothing grows in water, on ice/snow, on bare rock, or on cliffs.
        if s.height < VEG_MIN_GROUND_HEIGHT || s.steepness > VEG_MAX_STEEPNESS {
            continue;
        }
        let (tree_p, shrub_p, tint) = match s.biome {
            Biome::TropicalForest => (0.85, 0.5, Vec3::new(0.10, 0.45, 0.16)),
            Biome::TemperateForest => (0.72, 0.4, Vec3::new(0.18, 0.45, 0.20)),
            Biome::BorealForest => (0.6, 0.35, Vec3::new(0.12, 0.30, 0.18)),
            Biome::Grassland => (0.10, 0.6, Vec3::new(0.40, 0.52, 0.24)),
            Biome::Tundra => (0.0, 0.30, Vec3::new(0.36, 0.40, 0.30)),
            Biome::Desert => (0.0, 0.07, Vec3::new(0.40, 0.50, 0.25)),
            Biome::Beach => (0.0, 0.05, Vec3::new(0.35, 0.5, 0.25)),
            _ => (0.0, 0.0, Vec3::ZERO),
        };

        let up = dir;
        let ground = PLANET_RADIUS + s.height;
        let roll = rng.random::<f32>();
        if roll < tree_p {
            let scale = rng.random_range(TREE_SCALE_MIN..TREE_SCALE_MAX);
            let yaw = rng.random_range(0.0..std::f32::consts::TAU);
            let pos = up * ground;
            let model = Mat4::from_scale_rotation_translation(
                Vec3::splat(scale),
                planet::upright_rotation(up, yaw),
                pos,
            );
            let var = (rng.random::<f32>() - 0.5) * TREE_TINT_JITTER;
            trees.push(InstanceRaw::new(model, (tint + Vec3::splat(var)).clamp(Vec3::ZERO, Vec3::ONE)));
        } else if roll < tree_p + shrub_p {
            let scale = rng.random_range(SHRUB_SCALE_MIN..SHRUB_SCALE_MAX);
            let yaw = rng.random_range(0.0..std::f32::consts::TAU);
            let pos = up * ground;
            let model = Mat4::from_scale_rotation_translation(
                Vec3::splat(scale),
                planet::upright_rotation(up, yaw),
                pos,
            );
            let var = (rng.random::<f32>() - 0.5) * SHRUB_TINT_JITTER;
            shrubs.push(InstanceRaw::new(model, (tint * SHRUB_TINT_BRIGHTEN + Vec3::splat(var)).clamp(Vec3::ZERO, Vec3::ONE)));
        }
    }

    (trees, shrubs)
}

// ---------------------------------------------------------------------------
// Static base meshes
// ---------------------------------------------------------------------------

/// A simple indexed mesh of [`Vertex`].
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

/// A tree: a tapered trunk plus a conical canopy. Trunk verts are brown; canopy
/// verts are white so the per-instance color tint sets their green. Unit-ish
/// height (~3.2), scaled by the instance.
pub fn tree_mesh() -> MeshData {
    let mut m = MeshData { vertices: Vec::new(), indices: Vec::new() };
    let brown = Vec3::new(0.32, 0.22, 0.13);
    // Trunk: hexagonal prism.
    cylinder(&mut m, 0.18, 0.13, 1.3, 7, brown, 0.0);
    // Canopy: stacked cones, colored white (tinted per-instance).
    cone(&mut m, 0.95, 1.4, 9, Vec3::ONE, 1.1);
    cone(&mut m, 0.70, 1.2, 9, Vec3::ONE, 2.0);
    cone(&mut m, 0.45, 1.0, 9, Vec3::ONE, 2.8);
    m
}

/// A shrub: a low hemisphere, colored white (tinted per-instance).
pub fn shrub_mesh() -> MeshData {
    let mut m = MeshData { vertices: Vec::new(), indices: Vec::new() };
    hemisphere(&mut m, 0.6, 8, 4, Vec3::ONE, 0.1);
    m
}

fn cylinder(m: &mut MeshData, r_bottom: f32, r_top: f32, height: f32, sides: usize, color: Vec3, y0: f32) {
    use std::f32::consts::TAU;
    let start = m.vertices.len() as u32;
    for s in 0..=sides {
        let a = TAU * s as f32 / sides as f32;
        let (c, sn) = (a.cos(), a.sin());
        let nrm = Vec3::new(c, 0.3, sn).normalize();
        m.vertices.push(Vertex { pos: [c * r_bottom, y0, sn * r_bottom], normal: nrm.into(), color: color.into() });
        m.vertices.push(Vertex { pos: [c * r_top, y0 + height, sn * r_top], normal: nrm.into(), color: color.into() });
    }
    for s in 0..sides {
        let b = start + (s * 2) as u32;
        m.indices.extend_from_slice(&[b, b + 1, b + 2, b + 2, b + 1, b + 3]);
    }
}

fn cone(m: &mut MeshData, radius: f32, height: f32, sides: usize, color: Vec3, y0: f32) {
    use std::f32::consts::TAU;
    let start = m.vertices.len() as u32;
    let apex = Vec3::new(0.0, y0 + height, 0.0);
    m.vertices.push(Vertex { pos: apex.into(), normal: [0.0, 1.0, 0.0], color: color.into() });
    for s in 0..=sides {
        let a = TAU * s as f32 / sides as f32;
        let (c, sn) = (a.cos(), a.sin());
        let nrm = Vec3::new(c, 0.5, sn).normalize();
        m.vertices.push(Vertex { pos: [c * radius, y0, sn * radius], normal: nrm.into(), color: color.into() });
    }
    for s in 0..sides {
        let b = start + 1 + s as u32;
        m.indices.extend_from_slice(&[start, b + 1, b]);
    }
}

fn hemisphere(m: &mut MeshData, radius: f32, sectors: usize, rings: usize, color: Vec3, y0: f32) {
    use std::f32::consts::PI;
    let start = m.vertices.len() as u32;
    for i in 0..=rings {
        let phi = (PI * 0.5) * i as f32 / rings as f32; // 0..pi/2
        for j in 0..=sectors {
            let theta = 2.0 * PI * j as f32 / sectors as f32;
            let dir = Vec3::new(phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin());
            m.vertices.push(Vertex { pos: (dir * radius + Vec3::new(0.0, y0, 0.0)).into(), normal: dir.into(), color: color.into() });
        }
    }
    let stride = (sectors + 1) as u32;
    for i in 0..rings as u32 {
        for j in 0..sectors as u32 {
            let a = start + i * stride + j;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            m.indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
}
