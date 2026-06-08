//! Runtime graphics settings, adjustable from the ESC overlay's GRAPHICS tab, so
//! the same build can run lean on a cheap laptop GPU and crank way up on a 5090.
//!
//! Two knobs:
//!   - **Detail** — a single master that drives LOD subdivision, terrain mesh
//!     resolution, and vegetation (distance + density) together. Future "detail
//!     object" types should derive from it too, rather than adding sliders.
//!   - **Memory budget** — a real memory target (MB/GB) for resident geometry;
//!     the resident-chunk cap is derived from it and the current mesh resolution,
//!     so the same budget holds fewer chunks at higher detail.
//!
//! LOD and the memory budget apply live (read fresh each frame); mesh resolution
//! and vegetation are baked into chunk geometry, so they take effect on a rebuild
//! when the menu closes.

// Master-detail end points. `detail` is 0..1; these are its min/max effects.
// Maxed out is intentionally punishing (a laptop should struggle on Ultra while a
// high-end GPU looks great).
const SPLIT_MIN: f32 = 1.3; // LOD split factor (higher = finer, more chunks)
const SPLIT_MAX: f32 = 6.5;
const GRID_MIN: u32 = 16; // terrain quads per chunk side
const GRID_MAX: u32 = 88;
const VEG_LEVEL_NEAR: u32 = 15; // veg only on small/near chunks (low detail) ...
const VEG_LEVEL_FAR: u32 = 11; // ... out to bigger/farther chunks (high detail)
const DENSITY_MIN: u32 = 30; // vegetation attempts per chunk
// Capped so a fully-subdivided dense forest's *drawn* chunks fit the memory budget:
// plants baked into each chunk mesh are the dominant cost, and a covering that
// overshoots the budget forces eviction of chunks we need this frame, thrashing the
// LOD (the treetop flashing). Vegetation instancing (BACKLOG.md, High) is what lets
// this go back up without the memory blow-up.
const DENSITY_MAX: u32 = 250;

const DETAIL_STEP: f32 = 0.05;

// Memory budget for resident geometry, in MB. The low end suits a 2–4 GB laptop
// GPU; the high end gives a 5090 room to keep a lot of fine geometry resident.
const MEM_MIN_MB: u32 = 256;
const MEM_MAX_MB: u32 = 8_192; // 8 GB
const MEM_STEP_MB: u32 = 256;

/// (name, detail 0..1, memory budget MB) — tiers from a cheap laptop GPU to a 5090.
const PRESETS: [(&str, f32, u32); 4] = [
    ("Low", 0.10, 512),     // weak/integrated laptop GPU
    ("Medium", 0.35, 1_536), // decent laptop / entry desktop
    ("High", 0.65, 4_096),   // solid gaming GPU
    ("Ultra", 1.00, MEM_MAX_MB), // 5090 — maxes out everything
];
const DEFAULT_PRESET: usize = 2; // High
const CUSTOM: &str = "Custom";

/// ESC-overlay tabs.
pub const TAB_HELP: usize = 0;
pub const TAB_GRAPHICS: usize = 1;
pub const TAB_COUNT: usize = 2;

/// GRAPHICS-tab rows, in display order. Row 0 is the preset selector.
pub const ROW_PRESET: usize = 0;
pub const ROW_COUNT: usize = 3;

/// One rendered menu row: a label, a value string, and (for sliders) a 0..1 fill.
pub struct Row {
    pub label: &'static str,
    pub value: String,
    /// `Some(fill)` → draw a slider bar; `None` → a `< choice >` selector.
    pub frac: Option<f32>,
}

#[derive(Clone, Copy, PartialEq)]
pub struct Graphics {
    pub detail: f32,        // master detail, 0..1
    pub mem_budget_mb: u32, // resident-geometry memory target
    pub preset: &'static str,
}

impl Default for Graphics {
    fn default() -> Self {
        let mut g = Graphics { detail: 0.0, mem_budget_mb: MEM_MIN_MB, preset: CUSTOM };
        g.apply_preset(DEFAULT_PRESET);
        g
    }
}

impl Graphics {
    // --- derived detail values (the single `detail` knob fans out to these) ---
    pub fn split_factor(&self) -> f32 {
        lerp(SPLIT_MIN, SPLIT_MAX, self.detail)
    }
    pub fn mesh_res(&self) -> u32 {
        lerp_u32(GRID_MIN, GRID_MAX, self.detail)
    }
    pub fn veg_min_level(&self) -> u32 {
        // Higher detail lowers the level (vegetation appears farther out).
        lerp_u32(VEG_LEVEL_NEAR, VEG_LEVEL_FAR, self.detail)
    }
    pub fn veg_density(&self) -> u32 {
        lerp_u32(DENSITY_MIN, DENSITY_MAX, self.detail)
    }

    /// Resident-geometry budget in bytes — the renderer evicts cached chunks to keep
    /// real GPU memory (terrain + vegetation) under this. Finer detail makes each
    /// chunk bigger, so fewer fit — but that now falls out of the *actual* sizes the
    /// renderer tracks, not an estimate.
    pub fn mem_budget_bytes(&self) -> usize {
        (self.mem_budget_mb as usize) << 20
    }

    /// The values baked into chunk geometry; when this changes the world rebuilds.
    pub fn rebuild_signature(&self) -> (u32, u32, u32) {
        (self.mesh_res(), self.veg_min_level(), self.veg_density())
    }

    fn apply_preset(&mut self, idx: usize) {
        let p = PRESETS[idx];
        self.detail = p.1;
        self.mem_budget_mb = p.2;
        self.preset = p.0;
    }

    /// Adjust the GRAPHICS row at `index` by one step in `dir` (-1 / +1).
    pub fn adjust(&mut self, index: usize, dir: i32) {
        if index == ROW_PRESET {
            self.cycle_preset(dir);
            return;
        }
        match index {
            1 => self.detail = round2((self.detail + dir as f32 * DETAIL_STEP).clamp(0.0, 1.0)),
            2 => self.mem_budget_mb = step_u32(self.mem_budget_mb, dir, MEM_STEP_MB, MEM_MIN_MB, MEM_MAX_MB),
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

    pub fn rows(&self) -> Vec<Row> {
        vec![
            Row { label: "Preset", value: self.preset.to_string(), frac: None },
            Row {
                label: "Detail",
                value: format!("{}%", (self.detail * 100.0).round() as i32),
                frac: Some(self.detail),
            },
            Row {
                label: "Memory budget",
                value: format_mem(self.mem_budget_mb),
                frac: Some((self.mem_budget_mb - MEM_MIN_MB) as f32 / (MEM_MAX_MB - MEM_MIN_MB) as f32),
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

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

fn lerp_u32(a: u32, b: u32, t: f32) -> u32 {
    (a as f32 + (b as f32 - a as f32) * t.clamp(0.0, 1.0)).round() as u32
}

fn round2(v: f32) -> f32 {
    (v * 100.0).round() / 100.0
}

fn step_u32(v: u32, dir: i32, step: u32, lo: u32, hi: u32) -> u32 {
    (v as i64 + dir as i64 * step as i64).clamp(lo as i64, hi as i64) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_high() {
        let g = Graphics::default();
        assert_eq!(g.preset, "High");
        assert_eq!(g.rows().len(), ROW_COUNT);
    }

    #[test]
    fn ultra_maxes_everything() {
        let mut g = Graphics::default();
        while g.preset != "Ultra" {
            g.adjust(ROW_PRESET, 1);
        }
        assert_eq!(g.detail, 1.0);
        assert_eq!(g.mem_budget_mb, MEM_MAX_MB);
        assert_eq!(g.mesh_res(), GRID_MAX);
        assert_eq!(g.veg_density(), DENSITY_MAX);
        assert_eq!(g.veg_min_level(), VEG_LEVEL_FAR);
    }

    #[test]
    fn mem_budget_bytes_tracks_the_setting() {
        let g = Graphics { detail: 0.5, mem_budget_mb: 2048, preset: "Custom" };
        assert_eq!(g.mem_budget_bytes(), 2048usize << 20);
    }

    #[test]
    fn detail_fans_out_monotonically() {
        let lo = Graphics { detail: 0.0, mem_budget_mb: 2048, preset: "Custom" };
        let hi = Graphics { detail: 1.0, mem_budget_mb: 2048, preset: "Custom" };
        assert!(hi.split_factor() > lo.split_factor());
        assert!(hi.mesh_res() > lo.mesh_res());
        assert!(hi.veg_density() > lo.veg_density());
        assert!(hi.veg_min_level() < lo.veg_min_level());
    }

    #[test]
    fn memory_formats_with_units() {
        assert_eq!(format_mem(512), "512 MB");
        assert_eq!(format_mem(8192), "8.0 GB");
    }
}
