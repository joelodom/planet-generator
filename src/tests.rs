//! Headless sanity checks for the procedural pipeline — everything that doesn't
//! need a GPU. These guard the invariants the renderer relies on: determinism,
//! finite geometry, sensible value ranges, and a varied world.

use crate::lod::{self, ChunkKey};
use crate::mesh::CpuChunk;
use crate::planet::{Biome, Planet, PLANET_RADIUS, HEIGHT_SCALE};
use glam::Vec3;

fn dirs(n: usize) -> Vec<Vec3> {
    // A deterministic spread of directions over the sphere (golden spiral).
    let mut v = Vec::with_capacity(n);
    let ga = std::f32::consts::PI * (3.0 - 5f32.sqrt());
    for i in 0..n {
        let y = 1.0 - (i as f32 / (n as f32 - 1.0)) * 2.0;
        let r = (1.0 - y * y).max(0.0).sqrt();
        let t = ga * i as f32;
        v.push(Vec3::new(t.cos() * r, y, t.sin() * r));
    }
    v
}

#[test]
fn height_is_finite_and_bounded() {
    let p = Planet::new(42);
    for d in dirs(4000) {
        let h = p.height(d);
        assert!(h.is_finite(), "height not finite");
        // Comfortably within the design envelope around sea level.
        assert!(h > -HEIGHT_SCALE && h < HEIGHT_SCALE * 1.5, "height {h} out of range");
    }
}

#[test]
fn generation_is_deterministic() {
    let a = Planet::new(12345);
    let b = Planet::new(12345);
    for d in dirs(1000) {
        assert_eq!(a.height(d).to_bits(), b.height(d).to_bits(), "same seed must match");
    }
    assert_eq!(a.sun_dir, b.sun_dir);
}

#[test]
fn different_seeds_differ() {
    let a = Planet::new(1);
    let b = Planet::new(2);
    let mut diff = 0;
    for d in dirs(500) {
        if (a.height(d) - b.height(d)).abs() > 1.0 {
            diff += 1;
        }
    }
    assert!(diff > 100, "different seeds should produce different terrain (got {diff})");
}

#[test]
fn world_has_land_and_sea_and_variety() {
    let p = Planet::new(777);
    let mut land = 0;
    let mut sea = 0;
    let mut biomes = std::collections::HashSet::new();
    for d in dirs(6000) {
        let s = p.sample(d);
        if s.height >= 0.0 { land += 1 } else { sea += 1 }
        biomes.insert(s.biome as u8);
    }
    assert!(land > 200, "expected some land (got {land})");
    assert!(sea > 200, "expected some ocean (got {sea})");
    assert!(biomes.len() >= 5, "expected biome variety (got {})", biomes.len());
    // Poles should be cold.
    let pole = p.sample(Vec3::Y);
    assert!(matches!(pole.biome, Biome::Snow | Biome::PolarIce | Biome::Ocean | Biome::Tundra), "pole biome was {:?}", pole.biome);
}

#[test]
fn chunk_mesh_is_well_formed() {
    let p = Planet::new(99);
    // A deep chunk where vegetation should appear.
    let key = ChunkKey { face: 2, level: 7, i: 40, j: 40 };
    let c = CpuChunk::build(&p, key);
    assert!(!c.vertices.is_empty() && !c.indices.is_empty());
    assert_eq!(c.indices.len() % 3, 0, "indices must form triangles");
    let vmax = c.vertices.len() as u32;
    for &idx in &c.indices {
        assert!(idx < vmax, "index {idx} out of bounds {vmax}");
    }
    for v in &c.vertices {
        for comp in v.pos.iter().chain(v.normal.iter()).chain(v.color.iter()) {
            assert!(comp.is_finite(), "non-finite vertex component");
        }
        let r = Vec3::from(v.pos).length();
        // Every vertex sits within the planet shell (skirts dip below).
        assert!(r > PLANET_RADIUS * 0.7 && r < PLANET_RADIUS * 1.3, "radius {r} implausible");
        let n = Vec3::from(v.normal).length();
        assert!((n - 1.0).abs() < 0.01, "normal not unit length: {n}");
    }
}

#[test]
fn chunk_build_is_deterministic() {
    let p = Planet::new(5);
    let key = ChunkKey { face: 0, level: 6, i: 10, j: 20 };
    let a = CpuChunk::build(&p, key);
    let b = CpuChunk::build(&p, key);
    assert_eq!(a.vertices.len(), b.vertices.len());
    assert_eq!(a.trees.len(), b.trees.len());
    assert_eq!(a.shrubs.len(), b.shrubs.len());
    if let (Some(x), Some(y)) = (a.vertices.first(), b.vertices.first()) {
        assert_eq!(x.pos, y.pos);
    }
}

#[test]
fn lod_selection_covers_visible_world() {
    let p = Planet::new(2024);
    // From orbit, with all roots "ready", we should draw something and not panic.
    let cam = Vec3::new(0.0, 0.0, 1.0) * (PLANET_RADIUS + 2000.0);
    let roots: std::collections::HashSet<_> = ChunkKey::roots().into_iter().collect();
    let sel = lod::select(&p, cam, &|k| roots.contains(&k));
    assert!(!sel.draw.is_empty(), "should draw visible roots from orbit");
    // Closer in, it should want finer chunks than it currently has.
    let near = Vec3::new(0.0, 0.0, 1.0) * (PLANET_RADIUS + 30.0);
    let sel2 = lod::select(&p, near, &|k| roots.contains(&k));
    assert!(!sel2.want.is_empty(), "near the surface it should request detail");
}
