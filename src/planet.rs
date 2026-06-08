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

use crate::flora::Flora;
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

// --- Noise configuration (octaves / frequency / persistence / lacunarity) ---
// Frequency ≈ cycles across the planet, so wavelength ≈ radius / frequency.
const CONTINENT_OCTAVES: usize = 5;
const CONTINENT_FREQ: f64 = 0.9; // ~7000 km landmasses
const CONTINENT_PERSISTENCE: f64 = 0.5;
const CONTINENT_LACUNARITY: f64 = 2.1;
const MOUNTAIN_OCTAVES: usize = 7;
const MOUNTAIN_FREQ: f64 = 34.0; // ~190 km ranges, with fractal detail below
const MOUNTAIN_LACUNARITY: f64 = 2.2;
const DETAIL_OCTAVES: usize = 5;
const DETAIL_FREQ: f64 = 13.0;
const DETAIL_PERSISTENCE: f64 = 0.5;
const WARP_OCTAVES: usize = 3;
const WARP_FREQ: f64 = 1.3;
const MOISTURE_OCTAVES: usize = 4;
const MOISTURE_FREQ: f64 = 1.1;
const TEMP_OCTAVES: usize = 3;
const TEMP_FREQ: f64 = 0.8;

// --- Per-seed sun & atmosphere ---
const SUN_TILT_RANGE: f32 = 1.4; // how far off the equator the sun can sit (rad)
const ATMOSPHERE_HUE: f32 = 0.55; // base blue; nudged per seed for alien skies
const ATMOSPHERE_HUE_RANGE: f32 = 0.18;
const ATMOSPHERE_SATURATION: f32 = 0.55;
const ATMOSPHERE_VALUE: f32 = 1.0;

// --- Height field shaping ---
const WARP_SAMPLE_FREQ: f64 = 1.3; // coordinate scale fed to the domain-warp noise
const WARP_STRENGTH: f32 = 0.06; // how far the warp displaces the sample direction
const CONTINENT_SEA_BIAS: f32 = 0.12; // subtracted from continent noise (~40% land)
const LAND_MASK_LO: f32 = -0.02; // continent value where land starts ...
const LAND_MASK_HI: f32 = 0.30; // ... and where it's fully inland
const MOUNTAIN_POWER: f32 = 1.4; // sharpens ridges
const DETAIL_AMPLITUDE: f32 = 0.08;
const CONTINENT_WEIGHT: f32 = 0.5;
const MOUNTAIN_WEIGHT: f32 = 1.25;

// --- Temperature / moisture for biomes ---
const TEMP_NOISE_AMPLITUDE: f32 = 0.10; // breaks up perfect latitude bands
const ALTITUDE_COOLING_SCALE: f32 = 3000.0; // render units per ~1.0 of cooling
const COLOR_DETAIL_FREQ: f64 = 40.0; // high-freq noise for per-vertex color jitter

// --- Slope estimation ---
const STEEPNESS_EPS: f32 = 0.0015; // angular step used to sample the gradient (rad)
const STEEPNESS_GAIN: f32 = 0.9; // maps gradient → 0..1 steepness

// --- Biome thresholds (height in render units; temp/moisture in 0..1) ---
const BEACH_MAX_HEIGHT: f32 = 8.0;
const BEACH_MIN_TEMP: f32 = 0.25;
const SNOW_TEMP_RANGE: f32 = 1000.0; // how much the snow line rises toward the equator
const POLAR_SNOW_TEMP: f32 = 0.08; // below this it's snow at any altitude
const MOUNTAIN_MIN_STEEP: f32 = 0.5; // steep + high ⇒ bare rock
const MOUNTAIN_MIN_HEIGHT: f32 = 520.0;
const POLAR_TEMP: f32 = 0.18;
const COLD_TEMP: f32 = 0.34;
const TEMPERATE_TEMP: f32 = 0.66;
const BOREAL_MOISTURE: f32 = 0.45;
const FOREST_MOISTURE: f32 = 0.5;
const DESERT_MOISTURE: f32 = 0.35;

// --- Coloring ---
const COLOR_JITTER: f32 = 0.05; // per-vertex color variation
const OCEAN_DEPTH_FRACTION: f32 = 0.6; // of HEIGHT_SCALE → fully-dark sea floor
const MOUNTAIN_ROCK_GREY: f32 = 0.34; // base grey of bare mountain rock
const MOUNTAIN_ROCK_MOISTURE: f32 = 0.05; // wetter rock is a touch lighter

/// Pole-proximity threshold (|y|) for choosing a tangent reference axis.
const POLE_AXIS_THRESHOLD: f32 = 0.99;

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

/// Number of [`Biome`] variants. `biome as usize` indexes a per-biome table
/// (e.g. the flora library's species lists).
pub const BIOME_COUNT: usize = 11;

/// All biomes in `as usize` order. Used to drive per-biome generation loops.
pub const BIOMES: [Biome; BIOME_COUNT] = [
    Biome::Ocean,
    Biome::Beach,
    Biome::PolarIce,
    Biome::Tundra,
    Biome::BorealForest,
    Biome::Grassland,
    Biome::TemperateForest,
    Biome::Desert,
    Biome::TropicalForest,
    Biome::Mountain,
    Biome::Snow,
];

/// What a single surface sample resolves to for queries: where it is, how steep
/// it is, and what grows there. Vertex *color* is a rendering concern computed by
/// [`Planet::sample_terrain`], not part of this query result. Returned by
/// [`Planet::sample`].
#[derive(Clone, Copy)]
pub struct Surface {
    pub height: f32,
    pub biome: Biome,
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
    /// The procedurally generated plant species for this world, grouped by biome.
    /// Built once at construction; shared read-only with the meshing workers.
    pub flora: Flora,

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
            .set_octaves(CONTINENT_OCTAVES)
            .set_frequency(CONTINENT_FREQ)
            .set_persistence(CONTINENT_PERSISTENCE)
            .set_lacunarity(CONTINENT_LACUNARITY);
        // Ridged noise for mountains. The base frequency sets the size of a
        // mountain range (~radius / freq), so a too-low frequency just makes one
        // continent-wide swell that reads as flat — see MOUNTAIN_FREQ.
        let mountains = RidgedMulti::<Perlin>::new(sub(&mut s))
            .set_octaves(MOUNTAIN_OCTAVES)
            .set_frequency(MOUNTAIN_FREQ)
            .set_lacunarity(MOUNTAIN_LACUNARITY);
        let detail = Fbm::<Perlin>::new(sub(&mut s))
            .set_octaves(DETAIL_OCTAVES)
            .set_frequency(DETAIL_FREQ)
            .set_persistence(DETAIL_PERSISTENCE);
        let warp = Fbm::<Perlin>::new(sub(&mut s))
            .set_octaves(WARP_OCTAVES)
            .set_frequency(WARP_FREQ);
        let moisture = Fbm::<Perlin>::new(sub(&mut s))
            .set_octaves(MOISTURE_OCTAVES)
            .set_frequency(MOISTURE_FREQ);
        let temp_var = Fbm::<Perlin>::new(sub(&mut s))
            .set_octaves(TEMP_OCTAVES)
            .set_frequency(TEMP_FREQ);

        // Sun direction: a deterministic but varied point on the sphere.
        let a = (splitmix64(&mut s) as f64 / u64::MAX as f64) as f32 * std::f32::consts::TAU;
        let b = ((splitmix64(&mut s) as f64 / u64::MAX as f64) as f32 - 0.5) * SUN_TILT_RANGE;
        let sun_dir = Vec3::new(a.cos() * b.cos(), b.sin(), a.sin() * b.cos()).normalize();

        // Atmosphere tint: mostly blue-ish, but nudged per seed (alien skies).
        let h = splitmix64(&mut s) as f64 / u64::MAX as f64;
        let atmosphere = hsv_to_rgb(
            ATMOSPHERE_HUE + (h as f32 - 0.5) * ATMOSPHERE_HUE_RANGE,
            ATMOSPHERE_SATURATION,
            ATMOSPHERE_VALUE,
        );

        // Procedural plant library for this world (pure function of the seed).
        let flora = Flora::generate(seed);

        Self { seed, sun_dir, atmosphere, flora, continents, mountains, detail, warp, moisture, temp_var }
    }

    /// Terrain height in world units at a point on the sphere (positive = above
    /// sea level, negative = below). `dir` need not be normalised.
    pub fn height(&self, dir: Vec3) -> f32 {
        let d = dir.normalize();

        // Domain warp displaces the sample point for more organic coastlines.
        let wp = [d.x as f64 * WARP_SAMPLE_FREQ, d.y as f64 * WARP_SAMPLE_FREQ, d.z as f64 * WARP_SAMPLE_FREQ];
        let warp = self.warp.get(wp) as f32;
        let w = (d + Vec3::splat(warp) * WARP_STRENGTH).normalize();
        let p = [w.x as f64, w.y as f64, w.z as f64];

        // Continents: a low-frequency mask, biased so part of the surface is land.
        let c = self.continents.get(p) as f32; // ~[-1, 1]
        let continent = c - CONTINENT_SEA_BIAS;

        // Land mask ramps in past the coastline so mountains only rise inland.
        let land = smoothstep(LAND_MASK_LO, LAND_MASK_HI, continent);

        // Ridged mountains, only meaningful on land. Sharper power + higher weight
        // gives prominent ranges with steep flanks.
        let m = self.mountains.get(p) as f32;
        let mountains = (m * 0.5 + 0.5).powf(MOUNTAIN_POWER) * land;

        // Fine detail everywhere (surface roughness as you zoom in).
        let detail = self.detail.get(p) as f32 * DETAIL_AMPLITUDE;

        let h_unit = continent * CONTINENT_WEIGHT + mountains * MOUNTAIN_WEIGHT + detail;
        h_unit * HEIGHT_SCALE
    }

    /// Resolve a surface point for queries: its height, biome, and slope. (Vertex
    /// *colour* is a rendering output — use [`Self::sample_terrain`] for that, so
    /// this gameplay/HUD path doesn't compute a colour it would ignore.)
    pub fn sample(&self, dir: Vec3) -> Surface {
        let d = dir.normalize();
        let height = self.height(d);
        let steepness = self.steepness(d, height);
        let (temp, moisture) = self.climate(d, height);
        Surface { height, biome: classify(height, temp, moisture, steepness), steepness }
    }

    /// Like [`Self::sample`] but dithers the biome across boundaries by nudging the
    /// climate within ±[`BIOME_BLEND`] (`*_jitter` in -1..1). The mesher feeds it
    /// per-plant random jitter so flora from neighbouring biomes intermixes across
    /// the same band the colours blend over, instead of snapping at a hard edge.
    /// Deep water is unaffected — vegetation is gated by height, not biome.
    pub fn sample_blended(&self, dir: Vec3, temp_jitter: f32, moisture_jitter: f32) -> Surface {
        let d = dir.normalize();
        let height = self.height(d);
        let steepness = self.steepness(d, height);
        let (temp, moisture) = self.climate(d, height);
        let temp = (temp + temp_jitter * BIOME_BLEND).clamp(0.0, 1.0);
        let moisture = (moisture + moisture_jitter * BIOME_BLEND).clamp(0.0, 1.0);
        Surface { height, biome: classify(height, temp, moisture, steepness), steepness }
    }

    /// Lean terrain sample for chunk meshing: the height and smoothly biome-blended
    /// vertex colour the mesh needs. Slope only shifts the colour on high ground
    /// (the bare-rock blend, gated by `MOUNTAIN_MIN_HEIGHT`), so the two extra
    /// `height()` probes [`Self::steepness`] costs are skipped at or below that line
    /// — the bulk of every chunk's vertices (proven slope-independent there by the
    /// `color_is_slope_independent_below_mountains` test).
    pub fn sample_terrain(&self, dir: Vec3) -> (f32, Vec3) {
        let d = dir.normalize();
        let height = self.height(d);
        let steepness = if height > MOUNTAIN_MIN_HEIGHT { self.steepness(d, height) } else { 0.0 };
        let (temp, moisture) = self.climate(d, height);
        let p = [d.x as f64, d.y as f64, d.z as f64];
        let cnoise = self.detail.get([p[0] * COLOR_DETAIL_FREQ, p[1] * COLOR_DETAIL_FREQ, p[2] * COLOR_DETAIL_FREQ]) as f32;
        (height, blended_color(height, temp, moisture, steepness, cnoise))
    }

    /// Temperature and moisture (each 0..1) at a point — the climate fields the
    /// biome and its colour derive from.
    fn climate(&self, d: Vec3, height: f32) -> (f32, f32) {
        let p = [d.x as f64, d.y as f64, d.z as f64];
        let moisture = (self.moisture.get(p) as f32 * 0.5 + 0.5).clamp(0.0, 1.0);

        // Temperature: hot at equator, cold at poles, colder with altitude, plus
        // a little noise so biome bands aren't perfect latitude rings.
        let lat = d.y.clamp(-1.0, 1.0).asin().abs() / (PI * 0.5); // 0 equator .. 1 pole
        let tvar = self.temp_var.get(p) as f32 * TEMP_NOISE_AMPLITUDE;
        // Altitude cooling: high peaks run much colder (so they hold snow).
        let temp = (1.0 - lat - (height.max(0.0) / ALTITUDE_COOLING_SCALE) + tvar).clamp(0.0, 1.0);

        (temp, moisture)
    }

    /// Approximate surface slope at a point: 0 = flat, 1 = ~vertical cliff.
    /// Computed by sampling height a small step away along two tangents.
    pub fn steepness(&self, dir: Vec3, h0: f32) -> f32 {
        let d = dir.normalize();
        let (t, b) = tangent_basis(d);
        let eps = STEEPNESS_EPS;
        let ha = self.height((d + t * eps).normalize());
        let hb = self.height((d + b * eps).normalize());
        let arc = eps * PLANET_RADIUS;
        let grad = (((ha - h0) / arc).powi(2) + ((hb - h0) / arc).powi(2)).sqrt();
        (grad * STEEPNESS_GAIN).min(1.0)
    }

    /// Radius of the walkable surface at a direction (terrain, but never below
    /// sea level — you hover over water, not under it). Used for the camera.
    pub fn surface_radius(&self, dir: Vec3) -> f32 {
        PLANET_RADIUS + self.height(dir).max(0.0)
    }
}

/// Build an orthonormal tangent/bitangent basis for a point on the sphere.
pub fn tangent_basis(n: Vec3) -> (Vec3, Vec3) {
    let reference = if n.y.abs() < POLE_AXIS_THRESHOLD { Vec3::Y } else { Vec3::X };
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
    if height < BEACH_MAX_HEIGHT && temp > BEACH_MIN_TEMP {
        return Biome::Beach;
    }
    // Snow: cold poles at any altitude, or cold-enough high ground. Snow line
    // rises toward the warm equator, so peaks wear caps and ranges go white.
    let snow_line = SNOW_BASE + temp * SNOW_TEMP_RANGE;
    if temp < POLAR_SNOW_TEMP || height > snow_line {
        return Biome::Snow;
    }
    // Steep high rock reads as bare mountain regardless of biome band.
    if steep > MOUNTAIN_MIN_STEEP && height > MOUNTAIN_MIN_HEIGHT {
        return Biome::Mountain;
    }
    if temp < POLAR_TEMP {
        return Biome::PolarIce;
    }
    if temp < COLD_TEMP {
        return if moisture > BOREAL_MOISTURE { Biome::BorealForest } else { Biome::Tundra };
    }
    if temp < TEMPERATE_TEMP {
        return if moisture > FOREST_MOISTURE { Biome::TemperateForest } else { Biome::Grassland };
    }
    // Warm
    if moisture < DESERT_MOISTURE {
        Biome::Desert
    } else {
        Biome::TropicalForest
    }
}

// --- Smooth biome colouring ----------------------------------------------------
// Biomes cross-fade at their junctions instead of snapping, over a band BIOME_BLEND
// wide in climate-variable (0..1) space. Temperature is dominated by latitude
// (~1.0 across a pole-to-equator quarter ≈ 10,000 km), so a temperature-driven edge
// smears over roughly `BIOME_BLEND * 20,000` km (a few hundred km here); the noisier
// moisture edges are shorter. Deep water is the exception — the coastline stays
// crisp (handled first in `blended_color`).
const BIOME_BLEND: f32 = 0.05; // half-width of the climate cross-fade (0..1)

// Base biome colours (named per the no-magic-numbers rule).
const COL_OCEAN_SHALLOW: Vec3 = Vec3::new(0.20, 0.30, 0.34);
const COL_OCEAN_DEEP: Vec3 = Vec3::new(0.04, 0.07, 0.13);
const COL_BEACH: Vec3 = Vec3::new(0.80, 0.74, 0.55);
const COL_POLAR_ICE: Vec3 = Vec3::new(0.82, 0.88, 0.93);
const COL_SNOW: Vec3 = Vec3::new(0.93, 0.95, 0.98);
const COL_TUNDRA: Vec3 = Vec3::new(0.55, 0.52, 0.44);
const COL_BOREAL: Vec3 = Vec3::new(0.13, 0.27, 0.18);
const COL_GRASS: Vec3 = Vec3::new(0.50, 0.58, 0.27);
const COL_TEMPERATE: Vec3 = Vec3::new(0.20, 0.40, 0.20);
const COL_DESERT: Vec3 = Vec3::new(0.78, 0.62, 0.38);
const COL_TROPICAL: Vec3 = Vec3::new(0.13, 0.42, 0.18);

// Override blend widths for the height/slope-driven layers.
const SHORE_BLEND: f32 = 4.0; // beach↔inland fade (render units of height) — coasts stay crisp
const MOUNTAIN_BLEND_H: f32 = 200.0; // bare rock fades in this far above MOUNTAIN_MIN_HEIGHT
const MOUNTAIN_BLEND_S: f32 = 0.12; // ... and over this slope band (0..1)
const SNOW_BLEND_H: f32 = 250.0; // snow-line softness (render units)

/// Smoothly biome-blended terrain colour for a point's climate, height and slope.
/// Neighbouring biomes cross-fade (no hard borders); the ocean is kept crisp at the
/// shoreline. `cnoise` is the fine per-vertex colour jitter.
fn blended_color(height: f32, temp: f32, moisture: f32, steepness: f32, cnoise: f32) -> Vec3 {
    // Deep water — the exception: a crisp coastline and depth shading, no smear.
    if height < SHORE {
        let depth = (-height / (HEIGHT_SCALE * OCEAN_DEPTH_FRACTION)).clamp(0.0, 1.0);
        return COL_OCEAN_SHALLOW.lerp(COL_OCEAN_DEEP, depth);
    }

    // Each temperature band is a smooth dry↔wet blend across its moisture split.
    let cold = COL_TUNDRA.lerp(COL_BOREAL, smoothstep(BOREAL_MOISTURE - BIOME_BLEND, BOREAL_MOISTURE + BIOME_BLEND, moisture));
    let temperate = COL_GRASS.lerp(COL_TEMPERATE, smoothstep(FOREST_MOISTURE - BIOME_BLEND, FOREST_MOISTURE + BIOME_BLEND, moisture));
    let warm = COL_DESERT.lerp(COL_TROPICAL, smoothstep(DESERT_MOISTURE - BIOME_BLEND, DESERT_MOISTURE + BIOME_BLEND, moisture));

    // Cross-fade across the temperature bands as it cools: warm→temperate→cold→polar.
    let mut col = warm;
    col = col.lerp(temperate, smoothstep(TEMPERATE_TEMP + BIOME_BLEND, TEMPERATE_TEMP - BIOME_BLEND, temp));
    col = col.lerp(cold, smoothstep(COLD_TEMP + BIOME_BLEND, COLD_TEMP - BIOME_BLEND, temp));
    col = col.lerp(COL_POLAR_ICE, smoothstep(POLAR_TEMP + BIOME_BLEND, POLAR_TEMP - BIOME_BLEND, temp));

    // Beach on low, warm ground near the shore — a narrow height fade keeps it crisp.
    let beach = smoothstep(BEACH_MAX_HEIGHT, BEACH_MAX_HEIGHT - SHORE_BLEND, height)
        * smoothstep(BEACH_MIN_TEMP - BIOME_BLEND, BEACH_MIN_TEMP + BIOME_BLEND, temp);
    col = col.lerp(COL_BEACH, beach);

    // Bare mountain rock: steep AND high. The height gate starts exactly at
    // MOUNTAIN_MIN_HEIGHT, so below it the colour is slope-independent — which is
    // what lets `sample_terrain` skip the slope probe there.
    let g = MOUNTAIN_ROCK_GREY + moisture * MOUNTAIN_ROCK_MOISTURE;
    let rock = Vec3::new(g, g * 0.98, g * 0.95);
    let mtn = smoothstep(MOUNTAIN_MIN_HEIGHT, MOUNTAIN_MIN_HEIGHT + MOUNTAIN_BLEND_H, height)
        * smoothstep(MOUNTAIN_MIN_STEEP - MOUNTAIN_BLEND_S, MOUNTAIN_MIN_STEEP + MOUNTAIN_BLEND_S, steepness);
    col = col.lerp(rock, mtn);

    // Snow on top: cold poles at any altitude, or anything above the (warmth-raised)
    // snow line — so high peaks wear caps over the rock.
    let snow_line = SNOW_BASE + temp * SNOW_TEMP_RANGE;
    let snow = smoothstep(snow_line - SNOW_BLEND_H, snow_line + SNOW_BLEND_H, height)
        .max(smoothstep(POLAR_SNOW_TEMP + BIOME_BLEND, POLAR_SNOW_TEMP - BIOME_BLEND, temp));
    col = col.lerp(COL_SNOW, snow);

    (col + Vec3::splat(cnoise * COLOR_JITTER)).clamp(Vec3::ZERO, Vec3::ONE)
}

/// Smooth Hermite interpolation, matching the GLSL `smoothstep`.
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub(crate) fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Vec3 {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic spread of unit directions over the sphere (golden spiral).
    fn dirs(n: usize) -> Vec<Vec3> {
        let ga = PI * (3.0 - 5f32.sqrt());
        (0..n)
            .map(|i| {
                let y = 1.0 - (i as f32 / (n as f32 - 1.0)) * 2.0;
                let r = (1.0 - y * y).max(0.0).sqrt();
                let t = ga * i as f32;
                Vec3::new(t.cos() * r, y, t.sin() * r)
            })
            .collect()
    }

    #[test]
    fn color_is_slope_independent_below_mountains() {
        // Below MOUNTAIN_MIN_HEIGHT the bare-rock blend is gated off, so colour must
        // not depend on slope — which is exactly why `sample_terrain` may skip the
        // slope probe there (the H1 optimisation stays valid).
        let p = Planet::new(2024);
        for d in dirs(4000) {
            let dn = d.normalize();
            let h = p.height(dn);
            if h > MOUNTAIN_MIN_HEIGHT {
                continue;
            }
            let (temp, moisture) = p.climate(dn, h);
            let flat = blended_color(h, temp, moisture, 0.0, 0.0);
            let steep = blended_color(h, temp, moisture, p.steepness(dn, h), 0.0);
            assert_eq!(flat.to_array(), steep.to_array(), "slope changed colour below the mountain line at h={h}");
        }
    }

    #[test]
    #[ignore = "micro-benchmark; run: cargo test --release planet::tests::sampling_cost -- --ignored --nocapture"]
    fn sampling_cost() {
        use std::time::Instant;
        let p = Planet::new(2024);
        let ds = dirs(20_000);
        let mut acc = 0.0f32; // keep the calls from being optimised away

        // Full query path (slope always computed) vs the lean meshing path that
        // skips the slope probe below the mountain line and blends the colour.
        let t0 = Instant::now();
        for _ in 0..5 {
            for &d in &ds {
                acc += p.sample(d).height;
            }
        }
        let full = t0.elapsed();
        let t1 = Instant::now();
        for _ in 0..5 {
            for &d in &ds {
                acc += p.sample_terrain(d).0;
            }
        }
        let lean = t1.elapsed();
        eprintln!("sample(full slope)={full:?}  sample_terrain(skip+colour)={lean:?}  (acc {acc})");
    }
}
