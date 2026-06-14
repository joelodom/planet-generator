//! Runtime graphics settings, adjustable from the ESC overlay's GRAPHICS tab, so
//! the same build can run lean on a basic laptop GPU and crank all the way up on a
//! 5090. There is no longer a single opaque "detail" master: every parameter the
//! renderer actually consumes is exposed on its own row, in real units, so nothing
//! is hidden behind a percentage.
//!
//! The exposed parameters:
//!   - **Geometry LOD** (`split_factor`) — how eagerly the quadtree subdivides
//!     (a chunk splits while the camera is within this many of its own edge
//!     lengths). Higher = finer terrain persists farther out. *Live.*
//!   - **Terrain mesh** (`mesh_res`) — quads per chunk edge. *Rebuild.*
//!   - **Plant distance** (`veg_min_level`) — the coarsest quadtree level that gets
//!     vegetation; lower level = bigger/farther chunks get plants = longer plant
//!     draw distance. Shown as the approximate real-world reach. *Rebuild.*
//!   - **Plant density** (`veg_density`) — areal vegetation density, shown as
//!     plants/km². *Rebuild.*
//!   - **Memory budget** (`mem_budget_mb`) — resident-geometry target; the renderer
//!     evicts cached chunks to stay under it. *Live.*
//!
//! "Live" parameters are read fresh each frame; "Rebuild" parameters are baked into
//! chunk geometry, so they take effect on a re-mesh when the menu closes (see
//! [`Graphics::rebuild_signature`]).

use crate::mesh::VEG_REFERENCE_AREA;
use crate::planet::{METERS_PER_UNIT, PLANET_RADIUS};
use crate::units::{self, Units};

// --- Geometry LOD (split factor): dimensionless. A chunk subdivides while the
// camera is within SPLIT × its world-space edge length. Higher = more chunks,
// finer geometry at distance. Min is the laptop floor; max pushes a 5090. ---
const SPLIT_MIN: f32 = 1.8;
const SPLIT_MAX: f32 = 10.0;
const SPLIT_STEP: f32 = 0.5;

// --- Terrain tessellation: quads per chunk edge (the mesh is GRID×GRID quads). ---
const GRID_MIN: u32 = 24;
const GRID_MAX: u32 = 96;
const GRID_STEP: u32 = 2;

// --- Vegetation reach: the minimum quadtree level a chunk must reach before it is
// populated with plants. LOWER level = larger/farther chunks get vegetation = plants
// drawn farther from the camera. So detail rises as this value falls. ---
const VEG_LEVEL_NEAR: u32 = 15; // Low: plants only on the smallest, nearest chunks
const VEG_LEVEL_FAR: u32 = 11; // Ultra: plants out toward the horizon

// --- Vegetation areal density: raw placement attempts per VEG_REFERENCE_AREA patch
// (the unit the mesher consumes). Surfaced to the user as plants/km². ---
const DENSITY_MIN: u32 = 100;
const DENSITY_MAX: u32 = 1000;
const DENSITY_STEP: u32 = 25;

// --- Resident-geometry memory budget, in MB. Like the other knobs the slider spans
// [Low, Ultra]: 1 GB suits a basic laptop GPU, 24 GB gives a 5090 room to keep a lot
// of fine geometry resident. ---
const MEM_MIN_MB: u32 = 1_024; // == the Low preset, so every slider reads empty at Low
const MEM_MAX_MB: u32 = 24_576; // 24 GB — Ultra on a 32 GB 5090
const MEM_STEP_MB: u32 = 256;

const SQ_M_PER_SQ_KM: f32 = 1_000_000.0;
/// A cube face spans [-1, 1] in face coordinates, so its full edge is 2 units wide;
/// level L quarters the area, i.e. halves the edge L times.
const FACE_EDGE_SPAN: f32 = 2.0;

/// (name, split factor, mesh quads/side, veg min level, veg density, memory MB) —
/// five tiers from a basic laptop (Low) to a 5090 pushed to its limits (Ultra). The
/// detail parameters are spaced evenly between Low and Ultra; the memory budgets are
/// the deliberate per-tier targets (roughly geometric, not linear).
const PRESETS: [(&str, f32, u32, u32, u32, u32); 5] = [
    ("Low", 1.8, 24, 15, 100, 1_024),       // basic/integrated laptop GPU
    ("Medium", 3.9, 42, 14, 325, 3_072),    // decent laptop / entry desktop
    ("High", 6.0, 60, 13, 550, 8_192),      // solid gaming GPU / 16 GB+ unified memory
    ("Very High", 8.0, 78, 12, 775, 16_384), // high-end desktop
    ("Ultra", 10.0, 96, 11, 1_000, 24_576), // 5090 / 32 GB — pushed to the limit
];
const DEFAULT_PRESET: usize = 2; // High
const CUSTOM: &str = "Custom";

/// ESC-overlay tabs.
pub const TAB_HELP: usize = 0;
pub const TAB_GRAPHICS: usize = 1;
pub const TAB_COUNT: usize = 2;

/// GRAPHICS-tab rows, in display order. Row 0 is the preset selector; the rest are
/// one exposed parameter each.
pub const ROW_PRESET: usize = 0;
pub const ROW_COUNT: usize = 6;

/// One rendered menu row: a label, a value string, and (for sliders) a 0..1 fill.
pub struct Row {
    pub label: &'static str,
    pub value: String,
    /// `Some(fill)` → draw a slider bar; `None` → a `< choice >` selector.
    pub frac: Option<f32>,
}

#[derive(Clone, Copy, PartialEq)]
pub struct Graphics {
    pub split_factor: f32,  // chunk subdivides while camera is within this × its edge (live LOD)
    pub mesh_res: u32,      // terrain quads per chunk edge (rebuild)
    pub veg_min_level: u32, // coarsest quadtree level that gets vegetation (rebuild)
    pub veg_density: u32,   // areal vegetation attempts, raw mesher units (rebuild)
    pub mem_budget_mb: u32, // resident-geometry memory target (live)
    pub preset: &'static str,
}

impl Default for Graphics {
    fn default() -> Self {
        let mut g = Graphics {
            split_factor: SPLIT_MIN,
            mesh_res: GRID_MIN,
            veg_min_level: VEG_LEVEL_NEAR,
            veg_density: DENSITY_MIN,
            mem_budget_mb: MEM_MIN_MB,
            preset: CUSTOM,
        };
        g.apply_preset(DEFAULT_PRESET);
        g
    }
}

impl Graphics {
    /// Build the named detail preset, or `None` if the name isn't a known tier.
    /// Matching is insensitive to case and to spaces/hyphens, so `--video-preset`
    /// accepts `"Very High"`, `very-high`, or `veryhigh` alike.
    pub fn from_preset(name: &str) -> Option<Self> {
        let idx = PRESETS.iter().position(|p| preset_key(p.0) == preset_key(name))?;
        let mut g = Graphics::default();
        g.apply_preset(idx);
        Some(g)
    }

    /// The preset tier names, lowest → highest — for `--video-preset` help/errors.
    pub fn preset_names() -> Vec<&'static str> {
        PRESETS.iter().map(|p| p.0).collect()
    }

    /// Resident-geometry budget in bytes — the renderer evicts cached chunks to keep
    /// real GPU memory (terrain + vegetation) under this.
    pub fn mem_budget_bytes(&self) -> usize {
        (self.mem_budget_mb as usize) << 20
    }

    /// The values baked into chunk geometry; when this changes the world rebuilds.
    /// `split_factor` and the memory budget are excluded — they apply live.
    pub fn rebuild_signature(&self) -> (u32, u32, u32) {
        (self.mesh_res, self.veg_min_level, self.veg_density)
    }

    /// Approximate camera distance (render units) out to which plants are drawn.
    /// Vegetation lands on chunks at level >= `veg_min_level`; the coarsest of those
    /// (level `veg_min_level`) is the rendered LOD out to ~`split_factor` × its
    /// *parent's* edge — beyond that only coarser, plant-less chunks remain. So that
    /// product is the true reach. Couples both LOD knobs on purpose: it's the real
    /// distance, not a single hidden number.
    fn plant_view_units(&self) -> f32 {
        let parent_level = self.veg_min_level.saturating_sub(1);
        let parent_edge = FACE_EDGE_SPAN / (1u32 << parent_level) as f32 * PLANET_RADIUS;
        self.split_factor * parent_edge
    }

    /// Areal vegetation density as plants per km² (the mesher stores raw attempts per
    /// [`VEG_REFERENCE_AREA`] patch; this converts that to a real areal density).
    fn plants_per_km2(&self) -> u32 {
        let patch_sq_m = VEG_REFERENCE_AREA * METERS_PER_UNIT * METERS_PER_UNIT;
        (self.veg_density as f32 * SQ_M_PER_SQ_KM / patch_sq_m).round() as u32
    }

    fn apply_preset(&mut self, idx: usize) {
        let p = PRESETS[idx];
        self.split_factor = p.1;
        self.mesh_res = p.2;
        self.veg_min_level = p.3;
        self.veg_density = p.4;
        self.mem_budget_mb = p.5;
        self.preset = p.0;
    }

    /// Adjust the GRAPHICS row at `index` by one step in `dir` (-1 / +1).
    pub fn adjust(&mut self, index: usize, dir: i32) {
        if index == ROW_PRESET {
            self.cycle_preset(dir);
            return;
        }
        match index {
            1 => self.split_factor = round1((self.split_factor + dir as f32 * SPLIT_STEP).clamp(SPLIT_MIN, SPLIT_MAX)),
            2 => self.mesh_res = step_u32(self.mesh_res, dir, GRID_STEP, GRID_MIN, GRID_MAX),
            // More detail = LOWER min level (plants reach farther), so invert `dir`.
            3 => self.veg_min_level = (self.veg_min_level as i32 - dir).clamp(VEG_LEVEL_FAR as i32, VEG_LEVEL_NEAR as i32) as u32,
            4 => self.veg_density = step_u32(self.veg_density, dir, DENSITY_STEP, DENSITY_MIN, DENSITY_MAX),
            5 => self.mem_budget_mb = step_u32(self.mem_budget_mb, dir, MEM_STEP_MB, MEM_MIN_MB, MEM_MAX_MB),
            _ => {}
        }
        self.preset = CUSTOM;
    }

    fn cycle_preset(&mut self, dir: i32) {
        let cur = PRESETS.iter().position(|p| p.0 == self.preset).map(|i| i as i32).unwrap_or(-1);
        let n = PRESETS.len() as i32;
        let next = if cur < 0 {
            if dir >= 0 { 0 } else { n - 1 }
        } else {
            (cur + dir).rem_euclid(n)
        };
        self.apply_preset(next as usize);
    }

    pub fn rows(&self, sys: Units) -> Vec<Row> {
        vec![
            Row { label: "Preset", value: self.preset.to_string(), frac: None },
            Row {
                label: "Geometry LOD",
                value: format!("{:.1}x", self.split_factor),
                frac: Some(frac(self.split_factor, SPLIT_MIN, SPLIT_MAX)),
            },
            Row {
                label: "Terrain mesh",
                value: format!("{}/side", self.mesh_res),
                frac: Some(frac_u32(self.mesh_res, GRID_MIN, GRID_MAX)),
            },
            Row {
                label: "Plant distance",
                value: units::distance(self.plant_view_units(), sys),
                // Lower min level = farther plants = more detail, so the fill inverts.
                frac: Some(frac_u32_inv(self.veg_min_level, VEG_LEVEL_FAR, VEG_LEVEL_NEAR)),
            },
            Row {
                label: "Plant density",
                value: format!("{}/km2", self.plants_per_km2()),
                frac: Some(frac_u32(self.veg_density, DENSITY_MIN, DENSITY_MAX)),
            },
            Row {
                label: "Memory budget",
                value: format_mem(self.mem_budget_mb),
                frac: Some(frac_u32(self.mem_budget_mb, MEM_MIN_MB, MEM_MAX_MB)),
            },
        ]
    }
}

fn format_mem(mb: u32) -> String {
    if mb >= 1024 {
        format!("{:.1} GB", mb as f32 / 1024.0)
    } else {
        format!("{} MB", mb)
    }
}

/// 0..1 position of `v` within `[lo, hi]`, for a slider fill.
fn frac(v: f32, lo: f32, hi: f32) -> f32 {
    ((v - lo) / (hi - lo)).clamp(0.0, 1.0)
}

fn frac_u32(v: u32, lo: u32, hi: u32) -> f32 {
    (v.clamp(lo, hi) - lo) as f32 / (hi - lo) as f32
}

/// Like [`frac_u32`] but inverted: `lo` reads full, `hi` reads empty. Used where a
/// smaller stored value means more detail (the vegetation min level).
fn frac_u32_inv(v: u32, lo: u32, hi: u32) -> f32 {
    (hi - v.clamp(lo, hi)) as f32 / (hi - lo) as f32
}

fn round1(v: f32) -> f32 {
    (v * 10.0).round() / 10.0
}

/// Normalize a preset name for tolerant matching: drop non-alphanumerics (spaces,
/// hyphens) and lowercase, so `"Very High"` == `very-high` == `veryhigh`.
fn preset_key(name: &str) -> String {
    name.chars().filter(|c| c.is_alphanumeric()).flat_map(|c| c.to_lowercase()).collect()
}

fn step_u32(v: u32, dir: i32, step: u32, lo: u32, hi: u32) -> u32 {
    (v as i64 + dir as i64 * step as i64).clamp(lo as i64, hi as i64) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cycle the preset selector until it lands on `name`.
    fn preset(name: &str) -> Graphics {
        let mut g = Graphics::default();
        while g.preset != name {
            g.adjust(ROW_PRESET, 1);
        }
        g
    }

    const ORDER: [&str; 5] = ["Low", "Medium", "High", "Very High", "Ultra"];

    #[test]
    fn default_is_high() {
        let g = Graphics::default();
        assert_eq!(g.preset, "High");
        assert_eq!(g.rows(Units::Metric).len(), ROW_COUNT);
    }

    #[test]
    fn low_is_the_floor() {
        let g = preset("Low");
        assert_eq!(g.split_factor, SPLIT_MIN);
        assert_eq!(g.mesh_res, GRID_MIN);
        assert_eq!(g.veg_min_level, VEG_LEVEL_NEAR);
        assert_eq!(g.veg_density, DENSITY_MIN);
        assert_eq!(g.mem_budget_mb, MEM_MIN_MB); // Low == every slider's floor
    }

    #[test]
    fn ultra_pushes_the_limits() {
        let g = preset("Ultra");
        assert_eq!(g.split_factor, SPLIT_MAX);
        assert_eq!(g.mesh_res, GRID_MAX);
        assert_eq!(g.veg_min_level, VEG_LEVEL_FAR);
        assert_eq!(g.veg_density, DENSITY_MAX);
        assert_eq!(g.mem_budget_mb, MEM_MAX_MB);
    }

    #[test]
    fn presets_increase_monotonically() {
        let tiers: Vec<Graphics> = ORDER.iter().map(|n| preset(n)).collect();
        for w in tiers.windows(2) {
            let (lo, hi) = (&w[0], &w[1]);
            assert!(hi.split_factor > lo.split_factor, "split must climb");
            assert!(hi.mesh_res > lo.mesh_res, "mesh must climb");
            assert!(hi.veg_min_level < lo.veg_min_level, "veg level must fall (reach farther)");
            assert!(hi.veg_density > lo.veg_density, "density must climb");
            assert!(hi.mem_budget_mb > lo.mem_budget_mb, "memory must climb");
            // Plants reach farther and the slider fills more at higher tiers.
            assert!(hi.plant_view_units() > lo.plant_view_units());
        }
    }

    #[test]
    fn adjusting_a_row_drops_to_custom_and_clamps() {
        let mut g = preset("Ultra");
        g.adjust(1, 1); // try to push Geometry LOD past Ultra's max
        assert_eq!(g.preset, CUSTOM);
        assert_eq!(g.split_factor, SPLIT_MAX); // clamped, not exceeded
    }

    #[test]
    fn veg_row_inverts_direction() {
        let mut g = preset("High");
        let before = g.veg_min_level;
        g.adjust(3, 1); // +1 = more detail = farther plants = LOWER level
        assert_eq!(g.veg_min_level, before - 1);
        assert!(g.plant_view_units() > preset("High").plant_view_units());
    }

    #[test]
    fn mem_budget_bytes_tracks_the_setting() {
        // High's 8192 MB budget converts to bytes (MB << 20).
        assert_eq!(preset("High").mem_budget_bytes(), 8192usize << 20);
    }

    #[test]
    fn plant_density_reads_in_plants_per_km2() {
        // Raw attempts convert to a plausible areal density, and Ultra > Low.
        assert!(preset("Ultra").plants_per_km2() > preset("Low").plants_per_km2());
        let row = &preset("High").rows(Units::Metric)[4];
        assert!(row.value.ends_with("/km2"));
    }

    #[test]
    fn plant_distance_row_uses_real_units() {
        let metric = &preset("Ultra").rows(Units::Metric)[3];
        assert!(metric.value.ends_with("km") || metric.value.ends_with('m'));
        let us = &preset("Ultra").rows(Units::Us)[3];
        assert!(us.value.ends_with("mi") || us.value.ends_with("ft"));
    }

    #[test]
    fn memory_formats_with_units() {
        assert_eq!(format_mem(512), "512 MB");
        assert_eq!(format_mem(8192), "8.0 GB");
    }
}
