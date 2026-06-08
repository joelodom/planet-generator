//! CPU-side mesh data and the geometry builders that turn planet samples into
//! triangles: terrain chunks (with crack-hiding skirts), deterministic
//! vegetation instances, and the static base meshes (trees, shrubs, water,
//! fullscreen triangle) the renderer instances and reuses.

use crate::lod::ChunkKey;
use crate::planet::{self, Planet, Biome, FACES, METERS_PER_UNIT, PLANET_RADIUS};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};
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

/// Per-vertex terrain/veg attributes. Plain data, uploaded straight to the GPU.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

/// Everything a worker produces for one chunk: the terrain mesh and a single
/// baked vegetation mesh (every plant on the chunk, in world space, ready to draw
/// in one call). Procedural per-planet species are baked in here rather than
/// instanced, so a chunk can carry unlimited plant variety at no extra draw cost.
pub struct CpuChunk {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub veg: MeshData,
}

// Crack-hiding skirts around each chunk edge.
const SKIRT_DEPTH_FACTOR: f32 = 3.0; // skirt depth ≈ this × a terrain quad's width
const SKIRT_MIN_DEPTH: f32 = 2.0; // render units

// Vegetation scatter. Density (attempts per chunk) and the min LOD level are
// runtime settings (see MeshConfig); the rest is fixed tuning.
const VEG_MIN_GROUND_HEIGHT: f32 = 1.0; // skip water/waterline (render units)
const VEG_MAX_STEEPNESS: f32 = 0.5; // skip cliffs
const VEG_SINK: f32 = 0.15; // bury each plant's base this deep to hide the seam
const VEG_TINT_JITTER: f32 = 0.06; // ± per-plant brightness so a stand isn't uniform

// Same-species "stands". A Worley cell grid laid over each cube face, independent
// of the LOD chunk grid so stands cross chunk seams seamlessly. Every cell owns a
// jittered seed point and (per biome) one species; nearby plants adopt it, so one
// kind of plant clusters together the way real stands do.
const CLUSTER_CELL_METERS: f32 = 320.0; // ~ stand diameter
const CLUSTER_CELL_UV: f32 = CLUSTER_CELL_METERS / METERS_PER_UNIT / PLANET_RADIUS;
// Density falls off from each stand's seed: a dense core thinning to gaps, plus a
// sparse floor of stragglers between stands. Distances are normalised to the cell.
const CLUSTER_CORE: f32 = 0.18; // within this radius of a seed: full density
const CLUSTER_EDGE: f32 = 0.62; // past this: bare ground
const CLUSTER_FLOOR: f32 = 0.06; // baseline density between stands (loose mixing)
const CLUSTER_MIX: f32 = 0.5; // odds, near a border, of taking the neighbour's species

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

/// Deterministically scatter vegetation across a chunk according to biome rules.
/// Seeded by the chunk key so the same ground always grows the same plants.
/// `min_level` gates how far out plants appear; `density` is attempts per chunk.
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
) -> MeshData {
    let mut veg = MeshData { vertices: Vec::new(), indices: Vec::new() };
    if key.level < min_level {
        return veg;
    }

    let mut rng = StdRng::seed_from_u64(key.hash(planet.seed));
    for _ in 0..density {
        let u = u0 + size * rng.random::<f32>();
        let v = v0 + size * rng.random::<f32>();
        let cube = face.base + face.right * u + face.up * v;
        let dir = planet::cube_to_sphere(cube);
        let s = planet.sample(dir);

        // Nothing grows in water, on bare cliffs, or in a biome with no flora.
        if s.height < VEG_MIN_GROUND_HEIGHT || s.steepness > VEG_MAX_STEEPNESS {
            continue;
        }
        let coverage = biome_coverage(s.biome);
        if coverage <= 0.0 || !planet.flora.has_vegetation(s.biome) {
            continue;
        }

        // Which stand are we in? Its local density decides whether a plant grows
        // here (dense cores, thinning to gaps); its species hash decides which.
        let stand = cluster_lookup(planet.seed, key.face, u, v, &mut rng);
        if rng.random::<f32>() > coverage * stand.density {
            continue;
        }
        let Some(species_id) = planet.flora.pick(s.biome, stand.species_hash) else { continue };
        let species = planet.flora.species(species_id);

        // Plant it: upright on the surface, with a yaw spin and a size jitter.
        let up = dir;
        let yaw = rng.random_range(0.0..TAU);
        let scale = rng.random_range(species.scale_min..species.scale_max);
        let pos = up * (PLANET_RADIUS + s.height - VEG_SINK);
        let rot = planet::upright_rotation(up, yaw);
        let model = Mat4::from_scale_rotation_translation(Vec3::splat(scale), rot, pos);
        let nmat = Mat3::from_quat(rot);
        let tint = Vec3::splat(1.0 + (rng.random::<f32>() - 0.5) * VEG_TINT_JITTER);
        bake_plant(&mut veg, &species.mesh, model, nmat, tint);
    }

    veg
}

/// Append one plant's local-space mesh into a chunk's vegetation mesh, baked to
/// world space (positions via `model`, normals via the rotation `nmat`), tinted.
fn bake_plant(dst: &mut MeshData, src: &MeshData, model: Mat4, nmat: Mat3, tint: Vec3) {
    let base = dst.vertices.len() as u32;
    for v in &src.vertices {
        let p = model.transform_point3(Vec3::from(v.pos));
        let n = (nmat * Vec3::from(v.normal)).normalize_or_zero();
        let c = (Vec3::from(v.color) * tint).clamp(Vec3::ZERO, Vec3::ONE);
        dst.vertices.push(Vertex { pos: p.into(), normal: n.into(), color: c.into() });
    }
    dst.indices.extend(src.indices.iter().map(|&i| base + i));
}

/// The stand covering a point: how dense vegetation is here, and a stable hash
/// selecting its species.
struct Stand {
    density: f32,
    species_hash: u64,
}

/// Worley-cell lookup over the stand grid. Finds the nearest seed point (and the
/// runner-up, for soft borders), returning the local stand density and a species
/// hash that's constant across a stand's core so one species clusters together.
fn cluster_lookup(seed: u64, face: u8, u: f32, v: f32, rng: &mut StdRng) -> Stand {
    let cu = (u / CLUSTER_CELL_UV).floor() as i64;
    let cv = (v / CLUSTER_CELL_UV).floor() as i64;
    let (mut d1, mut d2) = (f32::INFINITY, f32::INFINITY);
    let (mut h1, mut h2) = (0u64, 0u64);
    for di in -1..=1 {
        for dj in -1..=1 {
            let (gi, gj) = (cu + di, cv + dj);
            let h = cell_hash(seed, face, gi, gj);
            // Jittered seed point inside the cell (two 16-bit fractions from h).
            let jx = (h & 0xFFFF) as f32 / 65535.0;
            let jy = ((h >> 16) & 0xFFFF) as f32 / 65535.0;
            let su = (gi as f32 + jx) * CLUSTER_CELL_UV;
            let sv = (gj as f32 + jy) * CLUSTER_CELL_UV;
            let d = (((u - su).powi(2) + (v - sv).powi(2)).sqrt()) / CLUSTER_CELL_UV;
            if d < d1 {
                d2 = d1;
                h2 = h1;
                d1 = d;
                h1 = h;
            } else if d < d2 {
                d2 = d;
                h2 = h;
            }
        }
    }
    let density = CLUSTER_FLOOR.max(1.0 - planet::smoothstep(CLUSTER_CORE, CLUSTER_EDGE, d1));
    // Near a stand border, sometimes adopt the neighbour's species so stands
    // interleave instead of meeting on hard Voronoi lines.
    let species_hash = if d2 < d1 * (1.0 + CLUSTER_MIX) && rng.random::<f32>() < CLUSTER_MIX {
        h2
    } else {
        h1
    };
    Stand { density, species_hash }
}

/// Stable hash of a stand-grid cell (per planet, per face).
fn cell_hash(seed: u64, face: u8, gi: i64, gj: i64) -> u64 {
    let mut h = seed ^ (face as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
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
        _ => 0.0,
    }
}

// ---------------------------------------------------------------------------
// Mesh container
// ---------------------------------------------------------------------------

/// A simple indexed mesh of [`Vertex`]. Used for terrain chunks and as the
/// container the flora module grows plant species into (see `crate::flora`).
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}
