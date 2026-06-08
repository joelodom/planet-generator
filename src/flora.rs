//! Procedural, per-planet vegetation.
//!
//! At world construction we generate a *library* of plant **species** — up to
//! [`SPECIES_PER_BIOME`] distinct, procedurally-built plants for every vegetated
//! biome. No two species share geometry: each is grown from its own seeded RNG
//! into a low-poly mesh with its own form (conifer, broadleaf, palm, shrub,
//! cactus, flower, grass tuft, dead snag…), size, foliage colour, and quirks
//! (flowering, autumn colour, bare/twiggy). Harsh biomes get small, hardy plants;
//! lush ones get huge trees. A per-planet "personality" rotates the foliage
//! palette and, rarely, lets a world grow plants in colours never seen on Earth.
//!
//! The library lives on [`crate::planet::Planet`] and is shared read-only with the
//! meshing workers, which *bake* chosen species into each chunk's vegetation mesh
//! (see `mesh::place_vegetation`). The renderer never sees this module — plants are
//! ordinary world-space triangles by the time they reach the GPU.
//!
//! All geometry here is **local space**: base at the origin, growing up +Y, at the
//! plant's true size in render units (1 unit = 10 m). Placement applies only an
//! upright rotation onto the planet's surface and a small per-plant scale jitter.

use crate::mesh::{MeshData, Vertex};
use crate::planet::{hsv_to_rgb, Biome, BIOMES, BIOME_COUNT, METERS_PER_UNIT};
use glam::Vec3;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::f32::consts::TAU;

/// Procedurally-generated species per vegetated biome. The user-facing "X types
/// per biome". Barren biomes (ocean, ice, snow) get none; sparse ones fewer.
pub const SPECIES_PER_BIOME: usize = 24;

// --- Per-planet flora "personality" --------------------------------------------
const FLORA_SEED_SALT: u64 = 0xF10A_5EED_C0FF_EE01; // decorrelate flora from terrain
const FLORA_HUE_DRIFT: f32 = 0.06; // ± global foliage hue rotation for this world
const EXOTIC_BIAS_POW: f32 = 3.5; // >1 pushes most worlds Earthlike (rarely alien)
const EXOTIC_MAX: f32 = 0.45; // cap on the fraction of alien-coloured species

// --- Per-plant placement scale jitter (multiplies the baked size) --------------
const SCALE_JITTER_LO: f32 = 0.82;
const SCALE_JITTER_HI: f32 = 1.20;

// --- Foliage / bloom / bark colour ---------------------------------------------
const GREEN_HUE_MIN: f32 = 0.26; // hue in 0..1; ~0.33 = pure green
const GREEN_HUE_MAX: f32 = 0.42;
const GREEN_SAT_MIN: f32 = 0.40;
const GREEN_SAT_MAX: f32 = 0.85;
const GREEN_VAL_MIN: f32 = 0.20; // deep, shadowed green …
const GREEN_VAL_MAX: f32 = 0.58; // … to bright new growth
const CONIFER_BLUE_SHIFT: f32 = 0.05; // needles skew a touch blue-green
const FALL_HUE_MAX: f32 = 0.13; // autumn: red(0) → orange → yellow(0.13)
const FALL_SAT_MIN: f32 = 0.65;
const FALL_VAL_MIN: f32 = 0.45;
const FOLIAGE_VERT_JITTER: f32 = 0.05; // ± per-vertex colour noise within foliage
const BARK_VAL_MIN: f32 = 0.10; // near-black bark …
const BARK_VAL_MAX: f32 = 0.34; // … to pale grey/tan
const BARK_SAT_MAX: f32 = 0.45;
const STEM_COLOR: Vec3 = Vec3::new(0.22, 0.38, 0.14); // flower/grass stalk green

/// Saturated flower-bloom palette (hue, sat, val) in 0..1 — white, gold, pink,
/// red, violet, blue, orange. Picked per flowering species.
const BLOOM_HSV: [(f32, f32, f32); 8] = [
    (0.00, 0.00, 1.00), // white
    (0.13, 0.85, 1.00), // gold
    (0.92, 0.55, 1.00), // pink
    (0.99, 0.85, 0.95), // red
    (0.78, 0.70, 0.95), // violet
    (0.60, 0.65, 0.95), // blue
    (0.07, 0.85, 1.00), // orange
    (0.83, 0.80, 0.95), // magenta
];

// --- Mesh resolution (low: a chunk may bake hundreds of these) -----------------
const TRUNK_SIDES: usize = 5;
const CONE_SIDES: usize = 7;
const BLOB_RINGS: usize = 3;
const BLOB_SECTORS: usize = 5;
const BLOB_LUMP: f32 = 0.22; // organic radius wobble on canopy/shrub blobs

/// One procedurally-generated plant: a finished local-space mesh plus the
/// per-plant scale-jitter range applied when it's planted.
pub struct Species {
    pub mesh: MeshData,
    pub scale_min: f32,
    pub scale_max: f32,
    /// Linear-decay clustering radius in render units: this species is certain at a
    /// seed point and fades to zero this far out. Varies by form — trees blanket
    /// regions, ground cover clumps tightly (see [`form_cluster_km`]).
    pub cluster_radius: f32,
}

/// The whole world's plant library, grouped by biome.
pub struct Flora {
    species: Vec<Species>,
    by_biome: [Vec<u32>; BIOME_COUNT],
}

impl Flora {
    /// Grow the full species library for a world — a pure function of its seed,
    /// so every machine and every meshing thread agrees on the plants.
    pub fn generate(seed: u64) -> Flora {
        let mut species: Vec<Species> = Vec::new();
        let mut by_biome: [Vec<u32>; BIOME_COUNT] = std::array::from_fn(|_| Vec::new());

        // This planet's flora personality: a small global hue rotation, plus a
        // (usually tiny) chance per species of fully-alien foliage colour.
        let mut prng = StdRng::seed_from_u64(seed ^ FLORA_SEED_SALT);
        let hue_shift = (prng.random::<f32>() - 0.5) * 2.0 * FLORA_HUE_DRIFT;
        let exotic = prng.random::<f32>().powf(EXOTIC_BIAS_POW) * EXOTIC_MAX;

        for biome in BIOMES {
            let Some(profile) = biome_profile(biome) else { continue };
            for i in 0..profile.count {
                let mut rng = StdRng::seed_from_u64(mix(seed, biome as u64, i as u64));
                let sp = build_species(&mut rng, &profile, hue_shift, exotic);
                by_biome[biome as usize].push(species.len() as u32);
                species.push(sp);
            }
        }

        tracing::info!(
            species = species.len(),
            exotic = exotic,
            hue_shift = hue_shift,
            "flora library generated"
        );
        Flora { species, by_biome }
    }

    /// The species with the given global id.
    pub fn species(&self, id: u32) -> &Species {
        &self.species[id as usize]
    }

    /// The species ids that grow in `biome` (empty if barren). The mesher evaluates
    /// each one's clustering at a point to decide which (if any) grows there.
    pub fn biome_species(&self, biome: Biome) -> &[u32] {
        &self.by_biome[biome as usize]
    }
}

// ---------------------------------------------------------------------------
// Biome → what grows there
// ---------------------------------------------------------------------------

/// Structural archetype of a plant. The renderer is form-agnostic; this only
/// drives the local mesh builder.
#[derive(Clone, Copy, PartialEq)]
enum Form {
    Conifer,   // stacked-cone evergreen (spruce/pine/fir)
    Broadleaf, // trunk + branches + rounded leafy crown (oak/maple)
    Palm,      // tall bare trunk crowned with radiating fronds
    Bush,      // big multi-lobe shrub, often flowering
    Shrub,     // small ground shrub
    Cactus,    // ribbed column with arms
    Flower,    // thin stalk + bright bloom head
    Grass,     // tuft of splayed blades
    Snag,      // bare, twiggy, dead-looking wood
}

/// Per-biome recipe: which forms appear (weighted), how big they run, and the
/// odds of flowering / autumn colour / bare-twiggy variants.
struct Profile {
    forms: &'static [(Form, f32)],
    /// Multiplies every form's intrinsic height — the harsh/lush size dial.
    size_scale: f32,
    flower_chance: f32,
    fall_chance: f32,
    bare_chance: f32,
    count: usize,
}

/// Intrinsic height range (render units, pre-`size_scale`) for each form.
fn form_height(form: Form) -> (f32, f32) {
    match form {
        Form::Conifer => (1.8, 4.2),
        Form::Broadleaf => (1.6, 3.8),
        Form::Palm => (2.2, 4.6),
        Form::Bush => (0.45, 1.1),
        Form::Shrub => (0.28, 0.7),
        Form::Cactus => (0.5, 1.8),
        Form::Flower => (0.06, 0.18),
        Form::Grass => (0.12, 0.34),
        Form::Snag => (1.3, 3.2),
    }
}

/// Render units per kilometre (1 unit = `METERS_PER_UNIT` m).
const UNITS_PER_KM: f32 = 1000.0 / METERS_PER_UNIT;

/// Per-form clustering scale as a (min, max) kilometre range for a species'
/// linear-decay radius (drawn log-uniform per species). Trees blanket regions —
/// tens to hundreds of km; ground cover clumps tightly — down to tens of metres.
fn form_cluster_km(form: Form) -> (f32, f32) {
    match form {
        Form::Conifer | Form::Broadleaf | Form::Palm | Form::Snag => (30.0, 400.0),
        Form::Cactus => (3.0, 80.0),
        Form::Bush => (2.0, 50.0),
        Form::Shrub => (0.2, 20.0),
        Form::Grass => (0.05, 6.0),
        Form::Flower => (0.02, 3.0),
    }
}

fn biome_profile(biome: Biome) -> Option<Profile> {
    use Form::*;
    let p = match biome {
        Biome::TropicalForest => Profile {
            forms: &[(Broadleaf, 0.45), (Palm, 0.25), (Bush, 0.18), (Flower, 0.12)],
            size_scale: 1.6,
            flower_chance: 0.30,
            fall_chance: 0.0,
            bare_chance: 0.0,
            count: SPECIES_PER_BIOME,
        },
        Biome::TemperateForest => Profile {
            forms: &[(Broadleaf, 0.45), (Conifer, 0.25), (Bush, 0.15), (Flower, 0.10), (Snag, 0.05)],
            size_scale: 1.1,
            flower_chance: 0.18,
            fall_chance: 0.35, // maples & oaks in autumn
            bare_chance: 0.04,
            count: SPECIES_PER_BIOME,
        },
        Biome::BorealForest => Profile {
            forms: &[(Conifer, 0.6), (Snag, 0.15), (Shrub, 0.18), (Flower, 0.07)],
            size_scale: 0.95,
            flower_chance: 0.06,
            fall_chance: 0.06,
            bare_chance: 0.05,
            count: SPECIES_PER_BIOME,
        },
        Biome::Grassland => Profile {
            forms: &[(Grass, 0.42), (Flower, 0.26), (Bush, 0.16), (Broadleaf, 0.16)],
            size_scale: 1.0,
            flower_chance: 0.50,
            fall_chance: 0.12,
            bare_chance: 0.03,
            count: SPECIES_PER_BIOME,
        },
        Biome::Tundra => Profile {
            forms: &[(Grass, 0.35), (Flower, 0.22), (Shrub, 0.30), (Snag, 0.13)],
            size_scale: 0.5,
            flower_chance: 0.25,
            fall_chance: 0.10,
            bare_chance: 0.18,
            count: SPECIES_PER_BIOME,
        },
        Biome::Desert => Profile {
            forms: &[(Cactus, 0.45), (Snag, 0.20), (Shrub, 0.25), (Flower, 0.10)],
            size_scale: 0.8,
            flower_chance: 0.20,
            fall_chance: 0.0,
            bare_chance: 0.22,
            count: SPECIES_PER_BIOME,
        },
        Biome::Beach => Profile {
            forms: &[(Palm, 0.30), (Grass, 0.40), (Shrub, 0.25), (Flower, 0.05)],
            size_scale: 0.95,
            flower_chance: 0.10,
            fall_chance: 0.0,
            bare_chance: 0.02,
            count: SPECIES_PER_BIOME,
        },
        Biome::Mountain => Profile {
            forms: &[(Conifer, 0.45), (Snag, 0.25), (Shrub, 0.25), (Flower, 0.05)],
            size_scale: 0.7,
            flower_chance: 0.08,
            fall_chance: 0.10,
            bare_chance: 0.18,
            count: SPECIES_PER_BIOME / 2,
        },
        // No vegetation on open ocean, polar ice, or permanent snow.
        Biome::Ocean | Biome::PolarIce | Biome::Snow => return None,
    };
    Some(p)
}

// ---------------------------------------------------------------------------
// Growing one species
// ---------------------------------------------------------------------------

fn build_species(rng: &mut StdRng, profile: &Profile, hue_shift: f32, exotic: f32) -> Species {
    let form = pick_weighted(rng, profile.forms);
    let (h0, h1) = form_height(form);
    let height = rng.random_range(h0..h1) * profile.size_scale;

    let bark = bark_color(rng);
    let flowering = rng.random::<f32>() < profile.flower_chance;
    let fall = rng.random::<f32>() < profile.fall_chance;
    let bare = rng.random::<f32>() < profile.bare_chance;
    let conifer_tint = form == Form::Conifer;
    let foliage = if fall {
        fall_foliage(rng)
    } else {
        green_foliage(rng, hue_shift, exotic, conifer_tint)
    };
    let bloom = bloom_color(rng);

    let mut m = MeshData { vertices: Vec::new(), indices: Vec::new() };
    match form {
        Form::Conifer => build_conifer(&mut m, rng, height, bark, foliage),
        Form::Broadleaf => build_broadleaf(&mut m, rng, height, bark, foliage, flowering, bloom, bare),
        Form::Palm => build_palm(&mut m, rng, height, bark, foliage),
        Form::Bush => build_shrub(&mut m, rng, height, foliage, flowering, bloom, bare, bark, true),
        Form::Shrub => build_shrub(&mut m, rng, height, foliage, flowering, bloom, bare, bark, false),
        Form::Cactus => build_cactus(&mut m, rng, height, foliage, flowering, bloom),
        Form::Flower => build_flower(&mut m, rng, height, bloom),
        Form::Grass => build_grass(&mut m, rng, height, foliage),
        Form::Snag => build_snag(&mut m, rng, height, bark),
    }

    // Clustering scale for this species: log-uniform within its form's km range, so
    // each plant type spreads (or clumps) at its own distance scale.
    let (kmin, kmax) = form_cluster_km(form);
    let cluster_radius = kmin * (kmax / kmin).powf(rng.random::<f32>()) * UNITS_PER_KM;

    Species { mesh: m, scale_min: SCALE_JITTER_LO, scale_max: SCALE_JITTER_HI, cluster_radius }
}

fn pick_weighted(rng: &mut StdRng, items: &[(Form, f32)]) -> Form {
    let total: f32 = items.iter().map(|x| x.1).sum();
    let mut r = rng.random::<f32>() * total;
    for &(form, w) in items {
        if r < w {
            return form;
        }
        r -= w;
    }
    items[items.len() - 1].0
}

// --- Form builders -------------------------------------------------------------

fn build_conifer(m: &mut MeshData, rng: &mut StdRng, height: f32, bark: Vec3, foliage: Vec3) {
    const TRUNK_R: f32 = 0.035; // trunk radius as a fraction of height
    // A bit of bare trunk shows beneath the skirt of branches.
    segment(m, Vec3::ZERO, Vec3::Y * height * 0.45, height * TRUNK_R, height * TRUNK_R * 0.6, TRUNK_SIDES, bark, bark);

    let tiers = rng.random_range(4..8);
    let base_r = height * rng.random_range(0.22..0.30);
    for i in 0..tiers {
        let f = i as f32 / (tiers - 1) as f32;
        let y = height * (0.12 + 0.80 * f);
        let r = base_r * (1.0 - 0.85 * f);
        let ch = height * (0.20 + 0.12 * (1.0 - f));
        // Lower tiers a touch darker (self-shadowing), upper a touch brighter.
        let col = (foliage * (0.82 + 0.22 * f)).clamp(Vec3::ZERO, Vec3::ONE);
        cone(m, Vec3::Y * y, ch, r.max(height * 0.02), CONE_SIDES, col);
    }
}

#[allow(clippy::too_many_arguments)]
fn build_broadleaf(
    m: &mut MeshData,
    rng: &mut StdRng,
    height: f32,
    bark: Vec3,
    foliage: Vec3,
    flowering: bool,
    bloom: Vec3,
    bare: bool,
) {
    let trunk_h = height * rng.random_range(0.40..0.60);
    let r0 = height * rng.random_range(0.04..0.07);
    segment(m, Vec3::ZERO, Vec3::Y * trunk_h, r0, r0 * 0.6, TRUNK_SIDES, bark, bark * 1.06);

    // A few main limbs reaching up and out from the upper trunk.
    let branches = rng.random_range(3..6);
    for _ in 0..branches {
        let az = rng.random_range(0.0..TAU);
        let el: f32 = rng.random_range(0.5..1.1);
        let dir = Vec3::new(az.cos() * el.cos(), el.sin(), az.sin() * el.cos()).normalize();
        let p0 = Vec3::Y * (trunk_h * rng.random_range(0.6..1.0));
        let len = height * rng.random_range(0.25..0.45);
        segment(m, p0, p0 + dir * len, r0 * 0.5, r0 * 0.2, 4, bark, bark);
    }

    if bare {
        return; // bare winter/dead frame — branches only
    }

    // Rounded crown: a clump of overlapping leafy blobs above the trunk.
    let blobs = rng.random_range(3..7);
    let crown_r = height * rng.random_range(0.32..0.50);
    let crown_c = Vec3::Y * (height * rng.random_range(0.75..0.95));
    for _ in 0..blobs {
        let off = Vec3::new(sym(rng), sym(rng) * 0.6, sym(rng)) * crown_r;
        let rr = crown_r * rng.random_range(0.6..1.0);
        let radii = Vec3::new(rr, rr * rng.random_range(0.7..1.0), rr);
        ellipsoid(m, crown_c + off, radii, foliage, rng);
    }
    if flowering {
        let dots = rng.random_range(8..16);
        for _ in 0..dots {
            let off = Vec3::new(sym(rng), sym(rng) * 0.5, sym(rng)) * crown_r;
            let rr = crown_r * 0.13;
            ellipsoid(m, crown_c + off + Vec3::Y * crown_r * 0.2, Vec3::splat(rr), bloom, rng);
        }
    }
}

fn build_palm(m: &mut MeshData, rng: &mut StdRng, height: f32, bark: Vec3, foliage: Vec3) {
    let trunk_h = height * 0.85;
    let lean = sym(rng) * 0.12 * height;
    // Gently-curved trunk from a few stacked segments.
    let segs = 3;
    let mut prev = Vec3::ZERO;
    let mut r = height * 0.045;
    for k in 1..=segs {
        let t = k as f32 / segs as f32;
        let p = Vec3::new(lean * t * t, trunk_h * t, 0.0);
        segment(m, prev, p, r, r * 0.82, TRUNK_SIDES, bark, bark);
        prev = p;
        r *= 0.82;
    }
    let crown = prev;
    let fronds = rng.random_range(7..12);
    let droop = height * 0.5;
    for i in 0..fronds {
        let az = TAU * i as f32 / fronds as f32 + sym(rng) * 0.2;
        let out = Vec3::new(az.cos(), 0.0, az.sin());
        let len = height * rng.random_range(0.5..0.8);
        frond(m, crown, out, len, height * 0.18, droop, height * 0.045, foliage, foliage * 0.8);
    }
}

#[allow(clippy::too_many_arguments)]
fn build_shrub(
    m: &mut MeshData,
    rng: &mut StdRng,
    height: f32,
    foliage: Vec3,
    flowering: bool,
    bloom: Vec3,
    bare: bool,
    bark: Vec3,
    big: bool,
) {
    let base_r = height * if big { 0.6 } else { 0.5 };
    if bare {
        // Twiggy and brown: a fan of bare branches, no leaves.
        let twigs = rng.random_range(3..7);
        for _ in 0..twigs {
            let az = rng.random_range(0.0..TAU);
            let el: f32 = rng.random_range(0.7..1.3);
            let dir = Vec3::new(az.cos() * el.cos(), el.sin(), az.sin() * el.cos()).normalize();
            segment(m, Vec3::ZERO, dir * height * rng.random_range(0.6..1.0), height * 0.05, height * 0.012, 4, bark, bark);
        }
        return;
    }
    let lobes = if big { rng.random_range(3..6) } else { rng.random_range(1..4) };
    for _ in 0..lobes {
        let off = Vec3::new(sym(rng), 0.0, sym(rng)) * base_r;
        let rr = base_r * rng.random_range(0.5..0.9);
        let c = Vec3::new(off.x, off.y.abs() + rr * 0.5, off.z);
        ellipsoid(m, c, Vec3::new(rr, rr * 0.8, rr), foliage, rng);
    }
    if flowering {
        let dots = rng.random_range(6..14);
        for _ in 0..dots {
            let off = Vec3::new(sym(rng), rng.random::<f32>(), sym(rng)) * base_r;
            ellipsoid(m, off + Vec3::Y * base_r * 0.4, Vec3::splat(base_r * 0.13), bloom, rng);
        }
    }
}

fn build_cactus(m: &mut MeshData, rng: &mut StdRng, height: f32, col: Vec3, flowering: bool, bloom: Vec3) {
    let r = height * 0.13;
    segment(m, Vec3::ZERO, Vec3::Y * height, r, r * 0.9, CONE_SIDES, col, col);
    let arms = rng.random_range(0..3);
    for _ in 0..arms {
        let az = rng.random_range(0.0..TAU);
        let out = Vec3::new(az.cos(), 0.0, az.sin());
        let y0 = height * rng.random_range(0.3..0.6);
        let elbow = Vec3::Y * y0 + out * (height * 0.2);
        let tip = elbow + Vec3::Y * height * rng.random_range(0.25..0.45);
        segment(m, Vec3::Y * y0, elbow, r * 0.6, r * 0.55, 6, col, col);
        segment(m, elbow, tip, r * 0.55, r * 0.5, 6, col, col);
        if flowering {
            ellipsoid(m, tip, Vec3::splat(r * 0.5), bloom, rng);
        }
    }
    if flowering {
        ellipsoid(m, Vec3::Y * height, Vec3::splat(r * 0.6), bloom, rng);
    }
}

fn build_flower(m: &mut MeshData, rng: &mut StdRng, height: f32, bloom: Vec3) {
    segment(m, Vec3::ZERO, Vec3::Y * height, height * 0.05, height * 0.035, 4, STEM_COLOR, STEM_COLOR);
    let head = Vec3::Y * height;
    // A bright centre ringed by petals.
    ellipsoid(m, head, Vec3::splat(height * 0.20), bloom * 0.8, rng);
    let petals = rng.random_range(5..9);
    for i in 0..petals {
        let az = TAU * i as f32 / petals as f32;
        let out = Vec3::new(az.cos(), 0.0, az.sin());
        let pc = head + out * (height * 0.28);
        ellipsoid(m, pc, Vec3::new(height * 0.22, height * 0.09, height * 0.22), bloom, rng);
    }
}

fn build_grass(m: &mut MeshData, rng: &mut StdRng, height: f32, foliage: Vec3) {
    let tip = (foliage * 0.8 + Vec3::new(0.18, 0.16, 0.04)).clamp(Vec3::ZERO, Vec3::ONE);
    let blades = rng.random_range(5..10);
    for _ in 0..blades {
        let az = rng.random_range(0.0..TAU);
        let out = Vec3::new(az.cos(), 0.0, az.sin());
        let lean = rng.random_range(0.12..0.40);
        frond(m, Vec3::ZERO, out, height * lean, height, height * 0.45, height * 0.05, foliage, tip);
    }
}

fn build_snag(m: &mut MeshData, rng: &mut StdRng, height: f32, bark: Vec3) {
    // Greyed, weathered wood.
    let wood = (bark * 0.7 + Vec3::splat(0.12)).clamp(Vec3::ZERO, Vec3::ONE);
    let trunk_h = height * rng.random_range(0.7..1.0);
    segment(m, Vec3::ZERO, Vec3::Y * trunk_h, height * 0.05, height * 0.02, TRUNK_SIDES, wood, wood * 1.1);
    let branches = rng.random_range(2..6);
    for _ in 0..branches {
        let az = rng.random_range(0.0..TAU);
        let el: f32 = rng.random_range(0.6..1.2);
        let dir = Vec3::new(az.cos() * el.cos(), el.sin(), az.sin() * el.cos()).normalize();
        let p0 = Vec3::Y * (trunk_h * rng.random_range(0.4..0.9));
        let len = height * rng.random_range(0.2..0.4);
        segment(m, p0, p0 + dir * len, height * 0.025, height * 0.008, 4, wood, wood);
    }
}

// --- Colour ---------------------------------------------------------------------

fn green_foliage(rng: &mut StdRng, hue_shift: f32, exotic: f32, conifer: bool) -> Vec3 {
    let mut hue = rng.random_range(GREEN_HUE_MIN..GREEN_HUE_MAX) + hue_shift;
    if conifer {
        hue += CONIFER_BLUE_SHIFT;
    }
    // Rarely, a world grows plants in colours never seen on Earth.
    if rng.random::<f32>() < exotic {
        hue = rng.random::<f32>();
    }
    let sat = rng.random_range(GREEN_SAT_MIN..GREEN_SAT_MAX);
    let val = rng.random_range(GREEN_VAL_MIN..GREEN_VAL_MAX);
    hsv_to_rgb(hue, sat, val)
}

fn fall_foliage(rng: &mut StdRng) -> Vec3 {
    let hue = rng.random::<f32>() * FALL_HUE_MAX;
    let sat = rng.random_range(FALL_SAT_MIN..1.0);
    let val = rng.random_range(FALL_VAL_MIN..0.95);
    hsv_to_rgb(hue, sat, val)
}

fn bloom_color(rng: &mut StdRng) -> Vec3 {
    let (h, s, v) = BLOOM_HSV[rng.random_range(0..BLOOM_HSV.len())];
    hsv_to_rgb(h, s, v)
}

fn bark_color(rng: &mut StdRng) -> Vec3 {
    // Brown→grey: low-saturation warm hue, value from near-black to pale.
    let hue = rng.random_range(0.05..0.10); // warm brown band
    let sat = rng.random::<f32>() * BARK_SAT_MAX;
    let val = rng.random_range(BARK_VAL_MIN..BARK_VAL_MAX);
    hsv_to_rgb(hue, sat, val)
}

// ---------------------------------------------------------------------------
// Mesh primitives (local space, +Y up). Colours baked per vertex.
// ---------------------------------------------------------------------------

fn vert(p: Vec3, n: Vec3, c: Vec3) -> Vertex {
    Vertex { pos: p.into(), normal: n.into(), color: c.into() }
}

fn sym(rng: &mut StdRng) -> f32 {
    rng.random::<f32>() * 2.0 - 1.0
}

/// Two unit vectors spanning the plane perpendicular to `n`.
fn ortho_basis(n: Vec3) -> (Vec3, Vec3) {
    let reference = if n.y.abs() < 0.99 { Vec3::Y } else { Vec3::X };
    let u = reference.cross(n).normalize();
    (u, n.cross(u))
}

/// A tapered tube from `a` (radius `ra`) to `b` (radius `rb`) — trunks, limbs,
/// twigs, cactus columns. Colour gradients base→tip. Normals point radially out.
#[allow(clippy::too_many_arguments)]
fn segment(m: &mut MeshData, a: Vec3, b: Vec3, ra: f32, rb: f32, sides: usize, ca: Vec3, cb: Vec3) {
    let axis = b - a;
    let len = axis.length();
    if len < 1e-6 {
        return;
    }
    let dir = axis / len;
    let (u, v) = ortho_basis(dir);
    let start = m.vertices.len() as u32;
    for s in 0..=sides {
        let ang = TAU * s as f32 / sides as f32;
        let radial = u * ang.cos() + v * ang.sin();
        m.vertices.push(vert(a + radial * ra, radial, ca));
        m.vertices.push(vert(b + radial * rb, radial, cb));
    }
    for s in 0..sides as u32 {
        let base = start + s * 2;
        m.indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 1, base + 3]);
    }
}

/// An upright cone (apex on +Y) — conifer tiers. Side normals are slope-aware.
fn cone(m: &mut MeshData, base: Vec3, height: f32, radius: f32, sides: usize, color: Vec3) {
    let start = m.vertices.len() as u32;
    let slope = (radius / height).max(0.0);
    m.vertices.push(vert(base + Vec3::Y * height, Vec3::Y, color));
    for s in 0..=sides {
        let ang = TAU * s as f32 / sides as f32;
        let radial = Vec3::new(ang.cos(), 0.0, ang.sin());
        let n = (radial + Vec3::Y * slope).normalize();
        m.vertices.push(vert(base + radial * radius, n, color));
    }
    for s in 0..sides as u32 {
        let rim = start + 1 + s;
        m.indices.extend_from_slice(&[start, rim + 1, rim]);
    }
}

/// A lumpy ellipsoid blob — leafy canopy clumps, shrub lobes, flower heads.
/// Position and colour are jittered per vertex for an organic, non-CG look.
fn ellipsoid(m: &mut MeshData, center: Vec3, radii: Vec3, color: Vec3, rng: &mut StdRng) {
    let start = m.vertices.len() as u32;
    for i in 0..=BLOB_RINGS {
        let phi = std::f32::consts::PI * i as f32 / BLOB_RINGS as f32; // 0..π
        for j in 0..=BLOB_SECTORS {
            let theta = TAU * j as f32 / BLOB_SECTORS as f32;
            let dir = Vec3::new(phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin());
            let lump = 1.0 + sym(rng) * BLOB_LUMP;
            let p = center + dir * radii * lump;
            let n = (dir / radii).normalize_or_zero();
            let n = if n == Vec3::ZERO { dir } else { n };
            let cv = (color + Vec3::splat(sym(rng) * FOLIAGE_VERT_JITTER)).clamp(Vec3::ZERO, Vec3::ONE);
            m.vertices.push(vert(p, n, cv));
        }
    }
    let stride = (BLOB_SECTORS + 1) as u32;
    for i in 0..BLOB_RINGS as u32 {
        for j in 0..BLOB_SECTORS as u32 {
            let a = start + i * stride + j;
            let (b, c, d) = (a + 1, a + stride, a + stride + 1);
            m.indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
}

/// A flat, drooping blade/frond strip from `base` heading along horizontal `out`:
/// rises (`rise`), then droops (`droop`), tapering to a tip. Grass and palm leaves.
/// Drawn one-sided but the pipeline disables culling, so it shows from both faces.
#[allow(clippy::too_many_arguments)]
fn frond(m: &mut MeshData, base: Vec3, out: Vec3, length: f32, rise: f32, droop: f32, width: f32, col_base: Vec3, col_tip: Vec3) {
    const SEGS: usize = 4;
    let side = {
        let s = out.cross(Vec3::Y);
        if s.length_squared() < 1e-6 { Vec3::X } else { s.normalize() }
    };
    // Roughly-upward normal, tilted back against the direction of travel.
    let normal = (Vec3::Y * 2.0 - out).normalize();
    let start = m.vertices.len() as u32;
    for k in 0..=SEGS {
        let t = k as f32 / SEGS as f32;
        let y = rise * t - droop * t * t;
        let center = base + out * (length * t) + Vec3::Y * y;
        let w = width * (1.0 - 0.85 * t);
        let col = col_base.lerp(col_tip, t);
        m.vertices.push(vert(center + side * w, normal, col));
        m.vertices.push(vert(center - side * w, normal, col));
    }
    for k in 0..SEGS as u32 {
        let b = start + k * 2;
        m.indices.extend_from_slice(&[b, b + 1, b + 2, b + 2, b + 1, b + 3]);
    }
}

/// Mix a few integers into a well-distributed seed (sub-seeds per species).
fn mix(a: u64, b: u64, c: u64) -> u64 {
    let mut h = a ^ 0x9E37_79B9_7F4A_7C15;
    for v in [b, c] {
        h ^= v.wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_add(h << 6).wrapping_add(h >> 2);
        h = h.wrapping_mul(0x0100_0000_01B3);
    }
    h
}
