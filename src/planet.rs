//! Procedural planet generation: the seeded "source of truth" for a world.
//!
//! Everything about a planet derives deterministically from a single `u64` seed:
//! the terrain height field, biome classification, vegetation, the direction of
//! the sun, and the tint of its atmosphere. The same seed + same coordinate
//! always yields the same result, on any thread, so chunk generation can be
//! parallelised freely.
//!
//! This module deliberately knows nothing about the GPU. It produces plain CPU
//! data (`glam` vectors, colors, mesh buffers) that the renderer consumes. That
//! separation is what makes the roadmap (animals, NPCs, weather, ...) tractable:
//! new systems query `Planet` for ground truth without touching rendering.

use glam::{Quat, Vec3};
use noise::{Fbm, MultiFractal, NoiseFn, Perlin, RidgedMulti};
use std::f32::consts::PI;

/// One render unit in metres. The render world is kept ~10x smaller than real
/// metres so `f32` precision stays comfortable (≈0.75 m at the surface) while the
/// planet is still Earth-sized: `PLANET_RADIUS * METERS_PER_UNIT` = 6,371,000 m.
pub const METERS_PER_UNIT: f32 = 10.0;

/// Base radius of the planet in render units. At 10 m/unit this is Earth's radius
/// (6,371 km). Large enough that flying/curvature feel planetary, small enough to
/// avoid the f32 precision wall (no camera-relative rendering needed).
pub const PLANET_RADIUS: f32 = 637_100.0;

/// Vertical scale applied to the normalised height field, in render units. This
/// carries a deliberate ~2x vertical exaggeration (as Google Earth and most
/// terrain renderers do): at true Earth proportions an 8 km peak is only 0.13%
/// of the radius and reads as flat, so peaks here reach ~+17 km. Mountains are
/// then clearly visible and snow-capped while the planet still feels huge.
pub const HEIGHT_SCALE: f32 = 1300.0;

/// Sea level sits exactly at `PLANET_RADIUS`; anything below is underwater.
pub const SEA_LEVEL: f32 = PLANET_RADIUS;

/// Terrain height (render units) below which a vertex is ocean floor.
pub const SHORE: f32 = 0.0;

/// Base permanent-snow elevation (render units), raised toward the equator by
/// temperature in `classify`, so high peaks wear snow caps.
pub const SNOW_BASE: f32 = 650.0;

/// The six faces of the cube that we inflate into a sphere. Each face is a unit
/// square parameterised by (u, v) in [-1, 1], embedded in 3D by an origin axis
/// plus two tangent axes.
pub struct CubeFace {
    pub base: Vec3,
    pub right: Vec3,
    pub up: Vec3,
}

pub const FACES: [CubeFace; 6] = [
    // +X
    CubeFace { base: Vec3::new(1.0, 0.0, 0.0), right: Vec3::new(0.0, 0.0, -1.0), up: Vec3::new(0.0, 1.0, 0.0) },
    // -X
    CubeFace { base: Vec3::new(-1.0, 0.0, 0.0), right: Vec3::new(0.0, 0.0, 1.0), up: Vec3::new(0.0, 1.0, 0.0) },
    // +Y
    CubeFace { base: Vec3::new(0.0, 1.0, 0.0), right: Vec3::new(1.0, 0.0, 0.0), up: Vec3::new(0.0, 0.0, -1.0) },
    // -Y
    CubeFace { base: Vec3::new(0.0, -1.0, 0.0), right: Vec3::new(1.0, 0.0, 0.0), up: Vec3::new(0.0, 0.0, 1.0) },
    // +Z
    CubeFace { base: Vec3::new(0.0, 0.0, 1.0), right: Vec3::new(1.0, 0.0, 0.0), up: Vec3::new(0.0, 1.0, 0.0) },
    // -Z
    CubeFace { base: Vec3::new(0.0, 0.0, -1.0), right: Vec3::new(-1.0, 0.0, 0.0), up: Vec3::new(0.0, 1.0, 0.0) },
];

/// Map a point on the cube surface (components in [-1, 1]) onto the unit sphere.
/// This "spherified cube" formula spreads vertices far more evenly than a plain
/// `normalize()`, which keeps chunk triangles from bunching up at face corners.
pub fn cube_to_sphere(p: Vec3) -> Vec3 {
    let x2 = p.x * p.x;
    let y2 = p.y * p.y;
    let z2 = p.z * p.z;
    Vec3::new(
        p.x * (1.0 - y2 * 0.5 - z2 * 0.5 + y2 * z2 / 3.0).max(0.0).sqrt(),
        p.y * (1.0 - z2 * 0.5 - x2 * 0.5 + z2 * x2 / 3.0).max(0.0).sqrt(),
        p.z * (1.0 - x2 * 0.5 - y2 * 0.5 + x2 * y2 / 3.0).max(0.0).sqrt(),
    )
    .normalize()
}

/// The biomes recognised in phase one. Ocean is a terrain classification (the
/// sea floor); the animated water surface is rendered separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Biome {
    Ocean,
    Beach,
    PolarIce,
    Tundra,
    BorealForest,
    Grassland,
    TemperateForest,
    Desert,
    TropicalForest,
    Mountain,
    Snow,
}

/// What a single surface sample resolves to: where it is, how it's lit, what
/// grows there. Returned by [`Planet::sample`].
#[derive(Clone, Copy)]
pub struct Surface {
    pub height: f32,
    pub biome: Biome,
    pub color: Vec3,
    pub steepness: f32,
}

/// A fully-resolved procedural planet. Cheap to clone-share via `Arc`; all noise
/// sources are immutable after construction.
pub struct Planet {
    pub seed: u64,
    /// Unit vector toward the sun. Varies per seed so lighting differs per world.
    pub sun_dir: Vec3,
    /// Atmosphere / horizon tint, also used for distance fog. Per seed.
    pub atmosphere: Vec3,

    continents: Fbm<Perlin>,
    mountains: RidgedMulti<Perlin>,
    detail: Fbm<Perlin>,
    warp: Fbm<Perlin>,
    moisture: Fbm<Perlin>,
    temp_var: Fbm<Perlin>,
}

/// SplitMix64 — turns a seed into a stream of well-mixed sub-seeds so each noise
/// octave gets an independent basis.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Planet {
    pub fn new(seed: u64) -> Self {
        let mut s = seed;
        let sub = |s: &mut u64| splitmix64(s) as u32;

        let continents = Fbm::<Perlin>::new(sub(&mut s))
            .set_octaves(5)
            .set_frequency(0.9)
            .set_persistence(0.5)
            .set_lacunarity(2.1);
        // Higher-frequency ridged noise makes steep, local mountain ranges (rather
        // than continent-wide gentle swells), so peaks read as real rocky/snowy
        // mountains with dramatic relief.
        let mountains = RidgedMulti::<Perlin>::new(sub(&mut s))
            .set_octaves(6)
            .set_frequency(4.2)
            .set_lacunarity(2.4);
        let detail = Fbm::<Perlin>::new(sub(&mut s))
            .set_octaves(5)
            .set_frequency(13.0)
            .set_persistence(0.5);
        let warp = Fbm::<Perlin>::new(sub(&mut s))
            .set_octaves(3)
            .set_frequency(1.3);
        let moisture = Fbm::<Perlin>::new(sub(&mut s))
            .set_octaves(4)
            .set_frequency(1.1);
        let temp_var = Fbm::<Perlin>::new(sub(&mut s))
            .set_octaves(3)
            .set_frequency(0.8);

        // Sun direction: a deterministic but varied point on the sphere.
        let a = (splitmix64(&mut s) as f64 / u64::MAX as f64) as f32 * std::f32::consts::TAU;
        let b = ((splitmix64(&mut s) as f64 / u64::MAX as f64) as f32 - 0.5) * 1.4;
        let sun_dir = Vec3::new(a.cos() * b.cos(), b.sin(), a.sin() * b.cos()).normalize();

        // Atmosphere tint: mostly blue-ish, but nudged per seed (alien skies).
        let h = splitmix64(&mut s) as f64 / u64::MAX as f64;
        let atmosphere = hsv_to_rgb(0.55 + (h as f32 - 0.5) * 0.18, 0.55, 1.0);

        Self { seed, sun_dir, atmosphere, continents, mountains, detail, warp, moisture, temp_var }
    }

    /// Terrain height in world units at a point on the sphere (positive = above
    /// sea level, negative = below). `dir` need not be normalised.
    pub fn height(&self, dir: Vec3) -> f32 {
        let d = dir.normalize();

        // Domain warp displaces the sample point for more organic coastlines.
        let wp = [d.x as f64 * 1.3, d.y as f64 * 1.3, d.z as f64 * 1.3];
        let warp = self.warp.get(wp) as f32;
        let w = (d + Vec3::splat(warp) * 0.06).normalize();
        let p = [w.x as f64, w.y as f64, w.z as f64];

        // Continents: a low-frequency mask. Biased so ~40% of the surface is land.
        let c = self.continents.get(p) as f32; // ~[-1, 1]
        let continent = c - 0.12;

        // Land mask ramps in past the coastline so mountains only rise inland.
        let land = smoothstep(-0.02, 0.30, continent);

        // Ridged mountains, only meaningful on land. Sharper power + higher weight
        // gives prominent ranges with steep flanks.
        let m = self.mountains.get(p) as f32;
        let mountains = (m * 0.5 + 0.5).powf(1.7) * land;

        // Fine detail everywhere (surface roughness as you zoom in).
        let detail = self.detail.get(p) as f32 * 0.08;

        let h_unit = continent * 0.5 + mountains * 1.05 + detail;
        h_unit * HEIGHT_SCALE
    }

    /// Resolve everything about a surface point: height, biome, color, slope.
    pub fn sample(&self, dir: Vec3) -> Surface {
        let d = dir.normalize();
        let height = self.height(d);
        let steepness = self.steepness(d, height);

        let p = [d.x as f64, d.y as f64, d.z as f64];
        let moisture = (self.moisture.get(p) as f32 * 0.5 + 0.5).clamp(0.0, 1.0);

        // Temperature: hot at equator, cold at poles, colder with altitude, plus
        // a little noise so biome bands aren't perfect latitude rings.
        let lat = d.y.clamp(-1.0, 1.0).asin().abs() / (PI * 0.5); // 0 equator .. 1 pole
        let tvar = self.temp_var.get(p) as f32 * 0.10;
        // Altitude cooling: high peaks run much colder (so they hold snow).
        let temp = (1.0 - lat - (height.max(0.0) / 3000.0) + tvar).clamp(0.0, 1.0);

        let biome = classify(height, temp, moisture, steepness);
        let color = biome_color(biome, height, temp, moisture, self.detail.get([p[0] * 40.0, p[1] * 40.0, p[2] * 40.0]) as f32);

        Surface { height, biome, color, steepness }
    }

    /// Approximate surface slope at a point: 0 = flat, 1 = ~vertical cliff.
    /// Computed by sampling height a small step away along two tangents.
    pub fn steepness(&self, dir: Vec3, h0: f32) -> f32 {
        let d = dir.normalize();
        let (t, b) = tangent_basis(d);
        let eps = 0.0015;
        let ha = self.height((d + t * eps).normalize());
        let hb = self.height((d + b * eps).normalize());
        let arc = eps * PLANET_RADIUS;
        let grad = (((ha - h0) / arc).powi(2) + ((hb - h0) / arc).powi(2)).sqrt();
        (grad * 0.9).min(1.0)
    }

    /// Radius of the walkable surface at a direction (terrain, but never below
    /// sea level — you hover over water, not under it). Used for the camera.
    pub fn surface_radius(&self, dir: Vec3) -> f32 {
        PLANET_RADIUS + self.height(dir).max(0.0)
    }
}

/// Build an orthonormal tangent/bitangent basis for a point on the sphere.
pub fn tangent_basis(n: Vec3) -> (Vec3, Vec3) {
    let reference = if n.y.abs() < 0.99 { Vec3::Y } else { Vec3::X };
    let t = reference.cross(n).normalize();
    let b = n.cross(t);
    (t, b)
}

/// Orientation that maps local +Y onto the surface normal, with a yaw spin.
/// Used to plant vegetation upright relative to the planet's local "up".
pub fn upright_rotation(up: Vec3, yaw: f32) -> Quat {
    let align = Quat::from_rotation_arc(Vec3::Y, up.normalize());
    align * Quat::from_rotation_y(yaw)
}

fn classify(height: f32, temp: f32, moisture: f32, steep: f32) -> Biome {
    if height < SHORE {
        return Biome::Ocean;
    }
    if height < 8.0 && temp > 0.25 {
        return Biome::Beach;
    }
    // Snow: cold poles at any altitude, or cold-enough high ground. Snow line
    // rises toward the warm equator, so peaks wear caps and ranges go white.
    let snow_line = SNOW_BASE + temp * 1000.0;
    if temp < 0.08 || height > snow_line {
        return Biome::Snow;
    }
    // Steep high rock reads as bare mountain regardless of biome band.
    if steep > 0.5 && height > 520.0 {
        return Biome::Mountain;
    }
    if temp < 0.18 {
        return Biome::PolarIce;
    }
    if temp < 0.34 {
        return if moisture > 0.45 { Biome::BorealForest } else { Biome::Tundra };
    }
    if temp < 0.66 {
        return if moisture > 0.5 { Biome::TemperateForest } else { Biome::Grassland };
    }
    // Warm
    if moisture < 0.35 {
        Biome::Desert
    } else {
        Biome::TropicalForest
    }
}

fn biome_color(biome: Biome, height: f32, _temp: f32, moisture: f32, n: f32) -> Vec3 {
    let jitter = n * 0.05;
    let base = match biome {
        Biome::Ocean => {
            // Depth-shaded sea floor (darker the deeper it is).
            let depth = (-height / (HEIGHT_SCALE * 0.6)).clamp(0.0, 1.0);
            Vec3::new(0.20, 0.30, 0.34).lerp(Vec3::new(0.04, 0.07, 0.13), depth)
        }
        Biome::Beach => Vec3::new(0.80, 0.74, 0.55),
        Biome::PolarIce => Vec3::new(0.82, 0.88, 0.93),
        Biome::Snow => Vec3::new(0.93, 0.95, 0.98),
        Biome::Tundra => Vec3::new(0.55, 0.52, 0.44),
        Biome::BorealForest => Vec3::new(0.13, 0.27, 0.18),
        Biome::Grassland => Vec3::new(0.50, 0.58, 0.27),
        Biome::TemperateForest => Vec3::new(0.20, 0.40, 0.20),
        Biome::Desert => Vec3::new(0.78, 0.62, 0.38),
        Biome::TropicalForest => Vec3::new(0.13, 0.42, 0.18),
        Biome::Mountain => {
            let g = 0.34 + moisture * 0.05;
            Vec3::new(g, g * 0.98, g * 0.95)
        }
    };
    (base + Vec3::splat(jitter)).clamp(Vec3::ZERO, Vec3::ONE)
}

/// Smooth Hermite interpolation, matching the GLSL `smoothstep`.
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Vec3 {
    let h = (h.fract() + 1.0).fract() * 6.0;
    let i = h.floor() as i32;
    let f = h - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i % 6 {
        0 => Vec3::new(v, t, p),
        1 => Vec3::new(q, v, p),
        2 => Vec3::new(p, v, t),
        3 => Vec3::new(p, q, v),
        4 => Vec3::new(t, p, v),
        _ => Vec3::new(v, p, q),
    }
}
