//! Runtime graphics settings, adjustable from the ESC overlay so the same build
//! can run lean on a laptop and crank way up on a big GPU.
//!
//! Five knobs cover the things that matter most for visible detail:
//!   - **terrain detail** — how eagerly the LOD subdivides (applies live)
//!   - **mesh resolution** — triangles per chunk (needs a rebuild)
//!   - **tree distance**   — how far out vegetation appears (needs a rebuild)
//!   - **vegetation**      — plant density (needs a rebuild)
//!   - **memory budget**   — how much geometry stays resident (applies live)
//!
//! "Applies live" settings take effect next frame; the rest are baked into chunk
//! meshes, so changing them rebuilds the visible world (done when the menu closes).

// Slider ranges and step sizes.
const DETAIL_MIN: f32 = 1.0;
const DETAIL_MAX: f32 = 4.5;
const DETAIL_STEP: f32 = 0.2;
const GRID_MIN: u32 = 12;
const GRID_MAX: u32 = 64;
const GRID_STEP: u32 = 4;
const TREE_LEVEL_NEAR: u32 = 16; // higher level = trees only on smaller/closer chunks
const TREE_LEVEL_FAR: u32 = 10;
const DENSITY_MAX: u32 = 400;
const DENSITY_STEP: u32 = 20;
const BUDGET_MIN: usize = 800;
const BUDGET_MAX: usize = 14_000;
const BUDGET_STEP: usize = 600;

/// (name, terrain_detail, mesh_res, veg_min_level, veg_density, chunk_budget)
const PRESETS: [(&str, f32, u32, u32, u32, usize); 4] = [
    ("Low", 1.4, 16, 14, 40, 1_200),
    ("Medium", 2.2, 28, 12, 110, 3_000),
    ("High", 3.0, 40, 11, 220, 6_000),
    ("Ultra", 4.2, 56, 10, 380, 12_000),
];
const DEFAULT_PRESET: usize = 1; // Medium — a clear step up from the old fixed values
const CUSTOM: &str = "Custom";

/// Menu rows, in display order. Row 0 is the preset selector.
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
    pub terrain_detail: f32, // LOD split factor (live)
    pub mesh_res: u32,       // terrain grid per chunk side (rebuild)
    pub veg_min_level: u32,  // lowest LOD level that grows plants (rebuild)
    pub veg_density: u32,    // vegetation attempts per chunk (rebuild)
    pub chunk_budget: usize, // resident chunk cap (live)
    pub preset: &'static str,
}

impl Default for Graphics {
    fn default() -> Self {
        let mut g = Graphics {
            terrain_detail: 1.0,
            mesh_res: 16,
            veg_min_level: 13,
            veg_density: 70,
            chunk_budget: 1_800,
            preset: CUSTOM,
        };
        g.apply_preset(DEFAULT_PRESET);
        g
    }
}

impl Graphics {
    /// The fields baked into chunk meshes; when this changes the world rebuilds.
    pub fn rebuild_signature(&self) -> (u32, u32, u32) {
        (self.mesh_res, self.veg_min_level, self.veg_density)
    }

    fn apply_preset(&mut self, idx: usize) {
        let p = PRESETS[idx];
        self.terrain_detail = p.1;
        self.mesh_res = p.2;
        self.veg_min_level = p.3;
        self.veg_density = p.4;
        self.chunk_budget = p.5;
        self.preset = p.0;
    }

    /// Adjust the setting at `index` by one step in `dir` (-1 / +1).
    pub fn adjust(&mut self, index: usize, dir: i32) {
        if index == ROW_PRESET {
            self.cycle_preset(dir);
            return;
        }
        match index {
            1 => self.terrain_detail = round1((self.terrain_detail + dir as f32 * DETAIL_STEP).clamp(DETAIL_MIN, DETAIL_MAX)),
            2 => self.mesh_res = step_u32(self.mesh_res, dir, GRID_STEP, GRID_MIN, GRID_MAX),
            // Higher "distance" = lower min level, so +1 lowers the level.
            3 => self.veg_min_level = ((self.veg_min_level as i32 - dir).clamp(TREE_LEVEL_FAR as i32, TREE_LEVEL_NEAR as i32)) as u32,
            4 => self.veg_density = step_u32(self.veg_density, dir, DENSITY_STEP, 0, DENSITY_MAX),
            5 => self.chunk_budget = step_usize(self.chunk_budget, dir, BUDGET_STEP, BUDGET_MIN, BUDGET_MAX),
            _ => {}
        }
        self.preset = CUSTOM;
    }

    fn cycle_preset(&mut self, dir: i32) {
        // Current index among named presets, or -1 if Custom.
        let cur = PRESETS.iter().position(|p| p.0 == self.preset).map(|i| i as i32).unwrap_or(-1);
        let n = PRESETS.len() as i32;
        // From Custom, +1 → first preset, -1 → last.
        let next = if cur < 0 {
            if dir >= 0 { 0 } else { n - 1 }
        } else {
            (cur + dir).rem_euclid(n)
        };
        self.apply_preset(next as usize);
    }

    pub fn rows(&self) -> Vec<Row> {
        vec![
            Row { label: "Preset", value: self.preset.to_string(), frac: None },
            Row {
                label: "Terrain detail",
                value: format!("{:.1}", self.terrain_detail),
                frac: Some((self.terrain_detail - DETAIL_MIN) / (DETAIL_MAX - DETAIL_MIN)),
            },
            Row {
                label: "Mesh resolution",
                value: format!("{}", self.mesh_res),
                frac: Some((self.mesh_res - GRID_MIN) as f32 / (GRID_MAX - GRID_MIN) as f32),
            },
            Row {
                label: "Tree distance",
                value: tree_distance_word(self.veg_min_level).to_string(),
                frac: Some((TREE_LEVEL_NEAR - self.veg_min_level) as f32 / (TREE_LEVEL_NEAR - TREE_LEVEL_FAR) as f32),
            },
            Row {
                label: "Vegetation",
                value: format!("{}", self.veg_density),
                frac: Some(self.veg_density as f32 / DENSITY_MAX as f32),
            },
            Row {
                label: "Memory budget",
                value: format!("{}", self.chunk_budget),
                frac: Some((self.chunk_budget - BUDGET_MIN) as f32 / (BUDGET_MAX - BUDGET_MIN) as f32),
            },
        ]
    }
}

fn tree_distance_word(level: u32) -> &'static str {
    match level {
        l if l >= 15 => "close",
        14 => "near",
        13 => "medium",
        12 => "far",
        11 => "very far",
        _ => "maximum",
    }
}

fn round1(v: f32) -> f32 {
    (v * 10.0).round() / 10.0
}

fn step_u32(v: u32, dir: i32, step: u32, lo: u32, hi: u32) -> u32 {
    (v as i32 + dir * step as i32).clamp(lo as i32, hi as i32) as u32
}

fn step_usize(v: usize, dir: i32, step: usize, lo: usize, hi: usize) -> usize {
    (v as i64 + dir as i64 * step as i64).clamp(lo as i64, hi as i64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_medium() {
        let g = Graphics::default();
        assert_eq!(g.preset, "Medium");
        assert_eq!(g.rows().len(), ROW_COUNT);
    }

    #[test]
    fn adjusting_a_slider_makes_it_custom_and_clamps() {
        let mut g = Graphics::default();
        g.adjust(1, 1); // bump terrain detail
        assert_eq!(g.preset, "Custom");
        for _ in 0..100 {
            g.adjust(1, 1);
        }
        assert!(g.terrain_detail <= DETAIL_MAX);
        for _ in 0..100 {
            g.adjust(1, -1);
        }
        assert!(g.terrain_detail >= DETAIL_MIN);
    }

    #[test]
    fn tree_distance_increases_as_level_drops() {
        let mut g = Graphics::default();
        let before = g.veg_min_level;
        g.adjust(3, 1); // "more distance"
        assert!(g.veg_min_level < before);
    }

    #[test]
    fn preset_cycles_and_changes_rebuild_signature() {
        let mut g = Graphics::default();
        let sig = g.rebuild_signature();
        g.adjust(ROW_PRESET, 1); // Medium → High
        assert_eq!(g.preset, "High");
        assert_ne!(g.rebuild_signature(), sig);
    }
}
