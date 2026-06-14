//! Per-planet vegetation: the species library and biome assignment.
//!
//! Vegetation is built from a library of **photoreal 3D archetype models** (see
//! [`crate::models`], loaded from embedded `.glb`) — boulders and rock outcrops,
//! deadwood, hardwoods and conifers, palms, cacti, shrubs, grasses and forbs —
//! replacing the old procedurally-grown plants. Each biome maps to a weighted set of those archetypes
//! at biome-specific sizes (a shared mesh can be dwarfed where a biome needs it
//! smaller). The library is a pure function of the model assets plus the world seed
//! (which only sets per-species clustering), so every machine and meshing thread
//! agrees on the plants.
//!
//! The library lives on [`crate::planet::Planet`], shared read-only with the meshing
//! workers; `mesh::place_vegetation` instances the archetypes deterministically.
//! The renderer draws the (textured) archetype meshes instanced — see
//! `crate::models`, `shaders/vegetation.wgsl`, and `gfx::VegGpu`.

use crate::mesh::VegMesh;
use crate::models::{self, TextureArray};
use crate::planet::{Biome, BIOMES, BIOME_COUNT, METERS_PER_UNIT};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Max archetype slots any one biome lists — sizes the per-attempt presence scratch
/// array in `mesh::place_vegetation`. The table peaks at 11 (temperate/boreal forest)
/// now that the full P0+P1+P2 set has landed; 12 leaves a little headroom.
pub const SPECIES_PER_BIOME: usize = 12;

/// Render units per kilometre (1 unit = `METERS_PER_UNIT` m).
const UNITS_PER_KM: f32 = 1000.0 / METERS_PER_UNIT;

// ---------------------------------------------------------------------------
// The archetype model library — order defines mesh + texture-array indices
// ---------------------------------------------------------------------------

/// One archetype: its embedded model, its on-planet **height range** in render
/// units (1 unit = 10 m; carries the world's deliberate ~2× vertical exaggeration —
/// see the scale notes in CLAUDE.md), and its clustering-radius range in km.
struct Archetype {
    id: &'static str,
    glb: &'static [u8],
    height: (f32, f32),     // (min, max) render units — becomes the placer's scale jitter
    cluster_km: (f32, f32), // linear-decay clustering radius range (drawn log-uniform per world)
}

macro_rules! archetype {
    ($id:literal, $h0:expr, $h1:expr, $k0:expr, $k1:expr) => {
        Archetype {
            id: $id,
            glb: include_bytes!(concat!("../assets/models/", $id, ".glb")),
            height: ($h0, $h1),
            cluster_km: ($k0, $k1),
        }
    };
}

/// The archetypes (9 P0 + 22 P1 + 20 P2 = 51). **Index = mesh index = texture-load
/// order**; the biome tables and [`models::load`] both key off this order. P0
/// heights/clustering per `flora-revamp/FLORA_P0_INTEGRATION_PLAN.md` §3; the P1/P2
/// rows follow the same scheme — rock/deadwood sit low and scatter broadly, trees are
/// tall and tightly clustered, shrubs/forbs/cushions are small.
const ARCHETYPES: [Archetype; 51] = [
    // --- P0 ---
    archetype!("granite-boulder", 0.10, 0.45, 2.0, 8.0),
    archetype!("broadleaf-hardwood", 1.8, 3.6, 30.0, 400.0),
    archetype!("spreading-oak", 1.5, 3.0, 30.0, 400.0),
    archetype!("spruce-spire-conifer", 2.0, 4.4, 30.0, 400.0),
    archetype!("tropical-emergent-tree", 3.4, 6.0, 30.0, 400.0),
    archetype!("feather-frond-palm", 2.0, 4.2, 10.0, 200.0),
    archetype!("bunchgrass-tussock", 0.10, 0.26, 0.05, 6.0),
    archetype!("savanna-acacia", 1.4, 2.6, 8.0, 120.0),
    archetype!("columnar-cactus", 0.5, 1.6, 3.0, 80.0),
    // --- P1 (first 15) ---
    archetype!("rock-outcrop", 0.12, 0.55, 2.0, 10.0),
    archetype!("fallen-log", 0.12, 0.30, 8.0, 120.0),
    archetype!("white-birch", 1.6, 3.4, 30.0, 400.0),
    archetype!("understory-shrub", 0.30, 0.80, 8.0, 150.0),
    archetype!("dense-fir", 2.0, 4.2, 30.0, 400.0),
    archetype!("long-needle-pine", 2.2, 4.6, 30.0, 400.0),
    archetype!("tree-fern", 0.8, 1.8, 10.0, 200.0),
    archetype!("broadleaf-understory", 0.4, 1.0, 8.0, 150.0),
    archetype!("hanging-liana", 1.0, 2.6, 10.0, 200.0),
    archetype!("sagebrush", 0.20, 0.50, 1.0, 20.0),
    archetype!("meadow-wildflower", 0.08, 0.20, 0.5, 10.0),
    archetype!("sandstone-hoodoo", 0.5, 2.0, 3.0, 60.0),
    archetype!("barrel-cactus", 0.10, 0.35, 1.0, 30.0),
    archetype!("creosote-shrub", 0.20, 0.60, 1.0, 25.0),
    archetype!("dwarf-shrub", 0.06, 0.18, 0.5, 8.0),
    // --- P1 (final 7) — cold/coastal/alpine fills + a wetland sedge ---
    archetype!("cotton-grass", 0.10, 0.24, 0.3, 6.0),       // tundra/bog sedge tuft, drifts
    archetype!("lichen-boulder", 0.12, 0.50, 2.0, 10.0),    // alpine/moraine boulder
    archetype!("leaning-palm", 1.8, 4.0, 10.0, 200.0),      // coastal palm
    archetype!("dune-grass", 0.12, 0.30, 0.3, 6.0),         // beach bunchgrass tussock
    archetype!("talus-rock", 0.12, 0.55, 1.5, 10.0),        // angular scree boulder
    archetype!("krummholz-conifer", 0.9, 2.2, 20.0, 300.0), // stunted treeline conifer, mats
    archetype!("snow-laden-conifer", 2.0, 4.2, 30.0, 400.0), // boreal/subalpine conifer
    // --- P2 (20) — sparse fills, understory, deadwood, and the snow/ice biomes ---
    archetype!("standing-dead-snag", 1.2, 2.8, 8.0, 150.0),  // standing deadwood, forest/boreal
    archetype!("fern", 0.20, 0.50, 1.0, 20.0),               // damp shaded forest floor
    archetype!("woodland-wildflower", 0.08, 0.22, 0.5, 12.0), // forest clearings/edges
    archetype!("juniper-scrub", 0.18, 0.45, 2.0, 40.0),      // low sprawling boreal conifer
    archetype!("moss-rock", 0.12, 0.50, 2.0, 10.0),          // mossy boulder, damp boreal floor
    archetype!("buttress-root-tree", 3.0, 5.5, 30.0, 400.0), // tropical buttress-root giant
    archetype!("bromeliad", 0.15, 0.40, 1.0, 30.0),          // tropical rosette epiphyte
    archetype!("grassland-rock", 0.12, 0.45, 2.0, 12.0),     // low sun-bleached plains rock
    archetype!("agave-rosette", 0.25, 0.70, 2.0, 40.0),      // desert sword-leaf rosette
    archetype!("dead-bush", 0.15, 0.40, 1.0, 25.0),          // leafless desert bush
    archetype!("cushion-plant", 0.04, 0.12, 0.3, 8.0),       // tundra ground-hugging dome
    archetype!("driftwood-log", 0.12, 0.35, 5.0, 80.0),      // bleached beach driftwood
    archetype!("coastal-scrub", 0.20, 0.55, 2.0, 40.0),      // salt-pruned back-beach shrub
    archetype!("sea-stack", 0.20, 0.80, 3.0, 30.0),          // wet surf-rounded shore rock
    archetype!("rock-spire", 0.8, 3.0, 3.0, 50.0),           // jagged alpine pinnacle
    archetype!("alpine-cushion", 0.04, 0.12, 0.3, 8.0),      // tiny cushion among rocks
    archetype!("snow-boulder", 0.12, 0.50, 2.0, 12.0),       // snow-capped rounded boulder
    archetype!("snow-snag", 1.0, 2.4, 8.0, 150.0),           // dark dead snag in snow
    archetype!("ice-hummock", 0.15, 0.60, 2.0, 20.0),        // buckled fractured-ice block
    archetype!("glacial-erratic", 0.12, 0.55, 3.0, 40.0),    // dark boulder stranded on ice
];

// Archetype indices (into ARCHETYPES) used by the biome tables.
const BOULDER: usize = 0;
const HARDWOOD: usize = 1;
const OAK: usize = 2;
const SPRUCE: usize = 3;
const EMERGENT: usize = 4;
const PALM: usize = 5;
const BUNCHGRASS: usize = 6;
const ACACIA: usize = 7;
const CACTUS: usize = 8;
// P1
const ROCK_OUTCROP: usize = 9;
const FALLEN_LOG: usize = 10;
const BIRCH: usize = 11;
const UNDERSTORY: usize = 12;
const FIR: usize = 13;
const PINE: usize = 14;
const TREE_FERN: usize = 15;
const BROADLEAF_UNDER: usize = 16;
const LIANA: usize = 17;
const SAGEBRUSH: usize = 18;
const WILDFLOWER: usize = 19;
const HOODOO: usize = 20;
const BARREL: usize = 21;
const CREOSOTE: usize = 22;
const DWARF_SHRUB: usize = 23;
const COTTON_GRASS: usize = 24;
const LICHEN_BOULDER: usize = 25;
const LEANING_PALM: usize = 26;
const DUNE_GRASS: usize = 27;
const TALUS_ROCK: usize = 28;
const KRUMMHOLZ: usize = 29;
const SNOW_CONIFER: usize = 30;
// P2
const DEAD_SNAG: usize = 31;
const FERN: usize = 32;
const WOOD_WILDFLOWER: usize = 33;
const JUNIPER: usize = 34;
const MOSS_ROCK: usize = 35;
const BUTTRESS_TREE: usize = 36;
const BROMELIAD: usize = 37;
const GRASS_ROCK: usize = 38;
const AGAVE: usize = 39;
const DEAD_BUSH: usize = 40;
const CUSHION: usize = 41;
const DRIFTWOOD: usize = 42;
const COASTAL_SCRUB: usize = 43;
const SEA_STACK: usize = 44;
const ROCK_SPIRE: usize = 45;
const ALPINE_CUSHION: usize = 46;
const SNOW_BOULDER: usize = 47;
const SNOW_SNAG: usize = 48;
const ICE_HUMMOCK: usize = 49;
const GLACIAL_ERRATIC: usize = 50;

/// One entry in a biome's planting table: which archetype, its relative weight in
/// the local mix, and a multiplier on the archetype's base height range (so a
/// shared mesh can be dwarfed — e.g. mountain spruce, tundra grass).
struct Plant {
    arch: usize,
    weight: f32,
    scale: f32,
}
const fn plant(arch: usize, weight: f32, scale: f32) -> Plant {
    Plant { arch, weight, scale }
}

// Per-biome planting tables. Weights are relative *within* a biome — they set the
// mix, not the absolute count; `mesh::biome_coverage` gates how dense each biome is
// (lush forest vs. sparse snowfield). The granite boulder + rock-outcrop are the
// cross-biome ground objects; most biomes also carry a tinted boulder variant
// (moss/lichen/grassland/sea/snow/glacial). The full P0+P1+P2 set is wired here,
// including the once-barren Snow and Polar Ice biomes (kept sparse via biome_coverage).
const TEMPERATE_FOREST: &[Plant] = &[plant(HARDWOOD, 0.19, 1.0), plant(OAK, 0.14, 1.0), plant(BIRCH, 0.12, 1.0), plant(UNDERSTORY, 0.11, 1.0), plant(FERN, 0.10, 1.0), plant(SPRUCE, 0.07, 1.0), plant(WOOD_WILDFLOWER, 0.06, 1.0), plant(FALLEN_LOG, 0.05, 1.0), plant(DEAD_SNAG, 0.05, 1.0), plant(ROCK_OUTCROP, 0.055, 1.0), plant(BOULDER, 0.055, 1.0)];
const BOREAL_FOREST: &[Plant] = &[plant(SPRUCE, 0.22, 1.0), plant(FIR, 0.17, 1.0), plant(SNOW_CONIFER, 0.13, 1.0), plant(PINE, 0.11, 1.0), plant(JUNIPER, 0.08, 1.0), plant(BIRCH, 0.06, 0.9), plant(MOSS_ROCK, 0.05, 1.0), plant(FALLEN_LOG, 0.05, 1.0), plant(DEAD_SNAG, 0.05, 1.0), plant(ROCK_OUTCROP, 0.04, 1.0), plant(BOULDER, 0.04, 1.0)];
const TROPICAL_FOREST: &[Plant] = &[plant(EMERGENT, 0.20, 1.0), plant(PALM, 0.17, 1.0), plant(BUTTRESS_TREE, 0.14, 1.0), plant(TREE_FERN, 0.13, 1.0), plant(BROADLEAF_UNDER, 0.12, 1.0), plant(BROMELIAD, 0.08, 1.0), plant(LIANA, 0.07, 1.0), plant(FALLEN_LOG, 0.05, 1.0), plant(BOULDER, 0.04, 1.0)];
const GRASSLAND: &[Plant] = &[plant(BUNCHGRASS, 0.50, 1.0), plant(WILDFLOWER, 0.15, 1.0), plant(SAGEBRUSH, 0.11, 1.0), plant(ACACIA, 0.08, 1.0), plant(GRASS_ROCK, 0.06, 1.0), plant(ROCK_OUTCROP, 0.05, 1.0), plant(BOULDER, 0.05, 1.0)];
const DESERT: &[Plant] = &[plant(CACTUS, 0.24, 1.0), plant(BARREL, 0.15, 1.0), plant(AGAVE, 0.13, 1.0), plant(CREOSOTE, 0.13, 1.0), plant(HOODOO, 0.10, 1.0), plant(DEAD_BUSH, 0.09, 1.0), plant(SAGEBRUSH, 0.08, 0.9), plant(BOULDER, 0.08, 1.0)];
const BEACH: &[Plant] = &[plant(DUNE_GRASS, 0.32, 1.0), plant(PALM, 0.16, 1.0), plant(LEANING_PALM, 0.14, 1.0), plant(COASTAL_SCRUB, 0.12, 1.0), plant(DRIFTWOOD, 0.08, 1.0), plant(SEA_STACK, 0.08, 1.0), plant(ROCK_OUTCROP, 0.05, 1.0), plant(BOULDER, 0.05, 1.0)];
const TUNDRA: &[Plant] = &[plant(COTTON_GRASS, 0.30, 1.0), plant(DWARF_SHRUB, 0.22, 1.0), plant(CUSHION, 0.16, 1.0), plant(LICHEN_BOULDER, 0.13, 1.0), plant(ROCK_OUTCROP, 0.10, 1.0), plant(BOULDER, 0.09, 1.0)];
const MOUNTAIN: &[Plant] = &[plant(TALUS_ROCK, 0.22, 1.0), plant(KRUMMHOLZ, 0.16, 1.0), plant(ROCK_OUTCROP, 0.16, 1.0), plant(ROCK_SPIRE, 0.14, 1.0), plant(BOULDER, 0.14, 1.0), plant(SNOW_CONIFER, 0.10, 0.7), plant(ALPINE_CUSHION, 0.08, 1.0)];
// Snow & Polar Ice were barren until the P2 set; kept sparse by biome_coverage so they
// read as a few capped/dark forms scattered across white, not a packed field.
const SNOW: &[Plant] = &[plant(SNOW_CONIFER, 0.30, 0.9), plant(SNOW_BOULDER, 0.24, 1.0), plant(SNOW_SNAG, 0.18, 1.0), plant(GLACIAL_ERRATIC, 0.16, 1.0), plant(BOULDER, 0.12, 1.0)];
const POLAR_ICE: &[Plant] = &[plant(ICE_HUMMOCK, 0.45, 1.0), plant(GLACIAL_ERRATIC, 0.33, 1.0), plant(SNOW_BOULDER, 0.22, 1.0)];

/// Per-biome planting table (`None` for barren ocean/ice/snow — the placer skips them).
fn biome_plants(biome: Biome) -> Option<&'static [Plant]> {
    match biome {
        Biome::TemperateForest => Some(TEMPERATE_FOREST),
        Biome::BorealForest => Some(BOREAL_FOREST),
        Biome::TropicalForest => Some(TROPICAL_FOREST),
        Biome::Grassland => Some(GRASSLAND),
        Biome::Desert => Some(DESERT),
        Biome::Beach => Some(BEACH),
        Biome::Tundra => Some(TUNDRA),
        Biome::Mountain => Some(MOUNTAIN),
        Biome::Snow => Some(SNOW),
        Biome::PolarIce => Some(POLAR_ICE),
        Biome::Ocean => None, // sea floor — no above-surface flora
    }
}

// ---------------------------------------------------------------------------
// Species library
// ---------------------------------------------------------------------------

/// A plantable species: an archetype mesh plus the per-instance size range, mix
/// weight, and clustering radius the placer uses. Several species can share one
/// `mesh_index` (e.g. the boulder grows in every biome at different sizes/weights);
/// the renderer draws by `mesh_index`, so shared meshes upload once.
pub struct Species {
    /// Index into [`Flora::mesh`] / the renderer's combined base-mesh buffer.
    pub mesh_index: u32,
    pub scale_min: f32,
    pub scale_max: f32,
    /// Relative likelihood this species wins a placement among the biome's mix.
    pub weight: f32,
    /// Linear-decay clustering radius in render units: certain at a seed point,
    /// fading to zero this far out (drawn log-uniform within the archetype's range).
    pub cluster_radius: f32,
}

/// The whole world's vegetation: the shared archetype meshes + texture array, the
/// per-biome species built from the planting tables, and the biome→species index.
pub struct Flora {
    meshes: Vec<VegMesh>,
    textures: TextureArray,
    species: Vec<Species>,
    by_biome: [Vec<u32>; BIOME_COUNT],
}

impl Flora {
    /// Build the species library for a world. The archetype meshes/textures come
    /// from the embedded models (identical everywhere); the seed only sets each
    /// species' clustering radius, so the library is deterministic per seed.
    pub fn generate(seed: u64) -> Flora {
        let inputs: Vec<(&str, &[u8])> = ARCHETYPES.iter().map(|a| (a.id, a.glb)).collect();
        let lib = models::load(&inputs);

        let mut species: Vec<Species> = Vec::new();
        let mut by_biome: [Vec<u32>; BIOME_COUNT] = std::array::from_fn(|_| Vec::new());
        for biome in BIOMES {
            let Some(plants) = biome_plants(biome) else { continue };
            debug_assert!(plants.len() <= SPECIES_PER_BIOME, "biome {biome:?} lists more plants than SPECIES_PER_BIOME");
            for (i, pl) in plants.iter().enumerate() {
                let arch = &ARCHETYPES[pl.arch];
                // Clustering radius: log-uniform within the archetype's km range, sub-
                // seeded per (world, biome, slot) — same splitmix64 style as before.
                let mut rng = StdRng::seed_from_u64(mix(seed, biome as u64, i as u64));
                let (kmin, kmax) = arch.cluster_km;
                let cluster_radius = kmin * (kmax / kmin).powf(rng.random::<f32>()) * UNITS_PER_KM;
                by_biome[biome as usize].push(species.len() as u32);
                species.push(Species {
                    mesh_index: pl.arch as u32,
                    scale_min: arch.height.0 * pl.scale,
                    scale_max: arch.height.1 * pl.scale,
                    weight: pl.weight,
                    cluster_radius,
                });
            }
        }

        tracing::info!(meshes = lib.meshes.len(), species = species.len(), "flora library generated");
        Flora { meshes: lib.meshes, textures: lib.textures, species, by_biome }
    }

    /// The species with the given global id.
    pub fn species(&self, id: u32) -> &Species {
        &self.species[id as usize]
    }

    /// How many species exist (ids `0..species_count`). Test/diagnostic use.
    #[cfg(test)]
    pub fn species_count(&self) -> usize {
        self.species.len()
    }

    /// The species ids that grow in `biome` (empty if barren).
    pub fn biome_species(&self, biome: Biome) -> &[u32] {
        &self.by_biome[biome as usize]
    }

    /// The unique archetype base meshes (indexed by [`Species::mesh_index`]). The
    /// renderer concatenates these into one base buffer.
    pub fn meshes(&self) -> &[VegMesh] {
        &self.meshes
    }

    /// One archetype base mesh by index (used by the headless gallery/closeup tools).
    #[cfg(test)]
    pub fn mesh(&self, index: u32) -> &VegMesh {
        &self.meshes[index as usize]
    }

    /// How many unique archetype meshes exist. Test/diagnostic use.
    #[cfg(test)]
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    /// The shared vegetation texture array the meshes index into.
    pub fn textures(&self) -> &TextureArray {
        &self.textures
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every archetype loads into a well-formed, bounded, texture-tagged mesh.
    #[test]
    fn archetype_meshes_load_well_formed() {
        // Cap on any one archetype's geometry (the P0 models run ~10–22k verts).
        const MAX_VERTS: usize = 60_000;
        let flora = Flora::generate(0x1234_5678);
        assert_eq!(flora.mesh_count(), ARCHETYPES.len(), "every archetype should load");
        let layer_count = flora.textures().layers.len();
        assert!(layer_count > 0, "no texture layers loaded");

        for i in 0..flora.mesh_count() as u32 {
            let mesh = flora.mesh(i);
            let nv = mesh.vertices.len();
            assert!(nv > 0, "archetype {i} produced an empty mesh");
            assert!(nv < MAX_VERTS, "archetype {i} ballooned to {nv} verts (> {MAX_VERTS})");
            assert_eq!(mesh.indices.len() % 3, 0, "archetype {i}: index count not whole triangles");
            for &idx in &mesh.indices {
                assert!((idx as usize) < nv, "archetype {i}: index {idx} out of range ({nv} verts)");
            }
            // Normalised to base at y=0, height 1.0, X/Z centred; layers in range.
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for v in &mesh.vertices {
                assert!(v.pos.iter().all(|c| c.is_finite()), "archetype {i}: non-finite position");
                assert!((v.layer as usize) < layer_count, "archetype {i}: layer {} out of range", v.layer);
                lo = lo.min(v.pos[1]);
                hi = hi.max(v.pos[1]);
            }
            assert!(lo.abs() < 1e-3, "archetype {i}: base not at y=0 (min y {lo})");
            assert!((hi - 1.0).abs() < 1e-3, "archetype {i}: height not normalised to 1.0 (max y {hi})");
        }
    }

    /// Every vegetated biome lists at least one species, and each species references
    /// a real mesh; barren biomes list none.
    #[test]
    fn biomes_map_to_valid_species() {
        let flora = Flora::generate(7);
        for biome in BIOMES {
            let ids = flora.biome_species(biome);
            match biome {
                Biome::Ocean => assert!(ids.is_empty(), "{biome:?} should be barren"),
                _ => {
                    assert!(!ids.is_empty(), "{biome:?} should have species");
                    for &id in ids {
                        let sp = flora.species(id);
                        assert!((sp.mesh_index as usize) < flora.mesh_count(), "species {id} bad mesh index");
                        assert!(sp.scale_max >= sp.scale_min && sp.scale_min > 0.0, "species {id} bad scale");
                    }
                }
            }
        }
    }

    /// Same seed → identical library (the meshing workers rely on this). Meshes come
    /// from static assets; only the per-species clustering radius is seed-driven.
    #[test]
    fn flora_generation_is_deterministic() {
        let a = Flora::generate(99);
        let b = Flora::generate(99);
        assert_eq!(a.species_count(), b.species_count());
        assert_eq!(a.mesh_count(), b.mesh_count());
        for biome in BIOMES {
            assert_eq!(a.biome_species(biome), b.biome_species(biome), "{biome:?} species differ");
        }
        for id in 0..a.species_count() as u32 {
            assert_eq!(a.species(id).cluster_radius, b.species(id).cluster_radius, "species {id} cluster differs");
            assert_eq!(a.species(id).mesh_index, b.species(id).mesh_index, "species {id} mesh differs");
        }
    }
}
