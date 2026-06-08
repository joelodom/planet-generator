//! The ESC help overlay: turns lines of text into screen-space colored quads
//! using the embedded bitmap font. Pure CPU geometry (no GPU types here) so it's
//! unit-testable; the renderer just uploads the instances and draws them.

use crate::font8x8::FONT8X8;
use crate::settings::Graphics;
use bytemuck::{Pod, Zeroable};

/// Width (chars) of a settings slider bar, e.g. `[####------]`.
const BAR_WIDTH: usize = 10;

/// One solid-color rectangle in normalized device coordinates. `rect` is
/// `(x, y, w, h)` where `(x, y)` is the top-left and `h` is negative (NDC y is
/// up, our layout is top-down). Matches the `overlay.wgsl` instance inputs.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct OverlayInstance {
    pub rect: [f32; 4],
    pub color: [f32; 4],
}

impl OverlayInstance {
    /// Build from a pixel-space rect (top-left origin, y down) + screen size.
    fn px(x: f32, y: f32, w: f32, h: f32, color: [f32; 4], sw: f32, sh: f32) -> Self {
        Self {
            rect: [x / sw * 2.0 - 1.0, 1.0 - y / sh * 2.0, w / sw * 2.0, -(h / sh * 2.0)],
            color,
        }
    }
}

/// Title + build stamp (matches the log) — shown at the top of the overlay.
fn header_lines() -> Vec<String> {
    let built = env!("BUILD_DATE").replace('T', " ").replace('Z', " UTC");
    vec![
        "PLANET EXPLORER".to_string(),
        format!("version {}   build {}", env!("CARGO_PKG_VERSION"), env!("GIT_HASH")),
        format!("built {built}"),
    ]
}

/// The key-bindings block.
fn control_lines() -> Vec<String> {
    let quit = if cfg!(target_os = "macos") { "Cmd-Q" } else { "Ctrl-Q" };
    vec![
        "CONTROLS  (Google Earth style, keyboard)".to_string(),
        "Arrow keys    pan  /  move + adjust settings".to_string(),
        "W / S  (+ -)  zoom in / out".to_string(),
        "A / D         rotate     Q / E   tilt".to_string(),
        "Shift         move faster (hold)".to_string(),
        "T  tour    R  teleport    P  location    G  wireframe".to_string(),
        format!("ESC  close menu              {quit}  quit"),
    ]
}

/// Plain help text (no live settings) — used by the offscreen smoke test.
#[cfg(test)]
pub fn help_lines() -> Vec<String> {
    let mut v = header_lines();
    v.push(String::new());
    v.extend(control_lines());
    v
}

fn bar(frac: f32) -> String {
    let fill = (frac.clamp(0.0, 1.0) * BAR_WIDTH as f32).round() as usize;
    let mut s = String::with_capacity(BAR_WIDTH + 2);
    s.push('[');
    for i in 0..BAR_WIDTH {
        s.push(if i < fill { '#' } else { '-' });
    }
    s.push(']');
    s
}

/// Build the full ESC menu — graphics settings (with sliders) above the controls
/// — and the row index that should be highlighted (the selected setting).
pub fn menu(graphics: &Graphics, selected: usize) -> (Vec<String>, usize) {
    let mut lines = header_lines();
    lines.push(String::new());
    lines.push("GRAPHICS   up/down: select    left/right: adjust".to_string());
    let settings_start = lines.len();
    for (i, row) in graphics.rows().iter().enumerate() {
        let marker = if i == selected { '>' } else { ' ' };
        let body = match row.frac {
            Some(f) => format!("{}  {:>6}", bar(f), row.value),
            None => format!("  < {} >", row.value),
        };
        lines.push(format!("{} {:<15} {}", marker, row.label, body));
    }
    let highlight = settings_start + selected;
    lines.push(String::new());
    lines.extend(control_lines());
    (lines, highlight)
}

/// Result of laying out the help overlay: solid-color quads (backdrop, panel,
/// text) plus the rectangle where the planet image should be drawn.
pub struct OverlayGeometry {
    pub quads: Vec<OverlayInstance>,
    /// NDC rect for the planet image (textured separately by the renderer).
    pub image: OverlayInstance,
}

// Overlay layout tuning. The embedded font is 8×8 px per glyph.
const FONT_PX: f32 = 8.0; // glyph cell width
const LINE_PX: f32 = 10.0; // glyph height + 2 px inter-line gap
const GAP_CELLS: f32 = 3.0; // gap between text block and image, in glyph cells
const PAD_CELLS: f32 = 7.0; // panel padding (× scale)
const SCALE_MIN: f32 = 2.0; // smallest/largest integer pixel scale tried
const SCALE_MAX: f32 = 5.0;
const FIT_FRACTION: f32 = 0.9; // the group must fit within this fraction of the screen
const IMG_HEIGHT_FRACTION: f32 = 0.94; // planet disc a touch shorter than the text block
const IMG_MAX_ROWS: f32 = 11.0; // cap the disc so a tall settings panel isn't dwarfed
const BORDER_PX: f32 = 2.0; // panel border thickness (unscaled)

/// Lay out `lines` centered on a `screen_w` x `screen_h` surface, with the planet
/// image to the right of the text inside one bordered panel. `highlight` is the
/// index of a row to draw as the selected setting (use `usize::MAX` for none).
pub fn layout(lines: &[String], highlight: usize, screen_w: u32, screen_h: u32) -> OverlayGeometry {
    let sw = screen_w.max(1) as f32;
    let sh = screen_h.max(1) as f32;

    let cols = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0).max(1) as f32;
    let rows = lines.len().max(1) as f32;

    // The group is [ text block | gap | square planet image (~ text height) ].
    // Pick the largest pixel scale (2..=5) that fits the whole group with margins.
    // Capped at 5 (one notch smaller than before) to leave room for the image.
    let group_cells_w = cols * FONT_PX + GAP_CELLS * FONT_PX + rows * LINE_PX; // image ≈ block_h square
    let mut scale = SCALE_MAX;
    while scale > SCALE_MIN {
        if group_cells_w * scale <= sw * FIT_FRACTION && rows * LINE_PX * scale <= sh * FIT_FRACTION {
            break;
        }
        scale -= 1.0;
    }

    let char_adv = FONT_PX * scale;
    let line_h = LINE_PX * scale;
    let block_w = cols * char_adv;
    let block_h = rows * line_h;
    let img = (block_h * IMG_HEIGHT_FRACTION).min(IMG_MAX_ROWS * line_h); // capped planet disc
    let gap = GAP_CELLS * char_adv;
    let group_w = block_w + gap + img;

    let ox = ((sw - group_w) * 0.5).round(); // text origin
    let oy = ((sh - block_h) * 0.5).round();
    let img_x = ox + block_w + gap;
    let img_y = oy + (block_h - img) * 0.5;
    let pad = PAD_CELLS * scale;

    let mut quads = Vec::new();
    let mut push = |x, y, w, h, c| quads.push(OverlayInstance::px(x, y, w, h, c, sw, sh));

    // Dim the whole scene, then a bordered panel behind the group.
    push(0.0, 0.0, sw, sh, [0.0, 0.0, 0.0, 0.55]);
    let (px0, py0, pw, ph) = (ox - pad, oy - pad, group_w + 2.0 * pad, block_h + 2.0 * pad);
    push(px0 - BORDER_PX, py0 - BORDER_PX, pw + 2.0 * BORDER_PX, ph + 2.0 * BORDER_PX, [0.30, 0.52, 0.74, 0.95]); // border
    push(px0, py0, pw, ph, [0.05, 0.07, 0.12, 0.94]); // panel

    let title = [0.55, 0.80, 1.0, 1.0];
    let body = [0.92, 0.95, 1.0, 1.0];
    let accent = [1.0, 0.86, 0.35, 1.0]; // selected settings row

    // Highlight bar behind the selected row.
    if highlight < lines.len() {
        let hy = oy + highlight as f32 * line_h;
        push(ox - pad * 0.5, hy - line_h * 0.1, block_w + pad, line_h, [0.20, 0.34, 0.52, 0.85]);
    }

    for (r, line) in lines.iter().enumerate() {
        let color = if r == highlight { accent } else if r == 0 { title } else { body };
        let base_y = oy + r as f32 * line_h;
        for (k, ch) in line.chars().enumerate() {
            let code = ch as usize;
            if code >= 128 {
                continue; // non-ASCII not in the font; skip
            }
            let glyph = &FONT8X8[code];
            let cx = ox + k as f32 * char_adv;
            for (row, bits) in glyph.iter().enumerate() {
                if *bits == 0 {
                    continue;
                }
                for col in 0..8u32 {
                    if bits & (1 << col) != 0 {
                        let x = cx + col as f32 * scale;
                        let y = base_y + row as f32 * scale;
                        push(x, y, scale, scale, color);
                    }
                }
            }
        }
    }

    let image = OverlayInstance::px(img_x, img_y, img, img, [1.0, 1.0, 1.0, 1.0], sw, sh);
    OverlayGeometry { quads, image }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_nonempty_and_in_ndc() {
        let lines = help_lines();
        let geo = layout(&lines, usize::MAX, 1280, 800);
        // Backdrop + border + panel + many glyph pixels.
        assert!(geo.quads.len() > 1000, "expected substantial text geometry, got {}", geo.quads.len());
        for q in geo.quads.iter().chain(std::iter::once(&geo.image)) {
            for c in q.rect {
                assert!(c.is_finite());
            }
            // Top-left corner stays within the NDC square (allow the full range).
            assert!(q.rect[0] >= -1.001 && q.rect[0] <= 1.001, "x {} out of NDC", q.rect[0]);
            assert!(q.rect[1] >= -1.001 && q.rect[1] <= 1.001, "y {} out of NDC", q.rect[1]);
        }
    }

    #[test]
    fn scales_down_for_tiny_windows() {
        // A tiny surface must still produce geometry without panicking.
        let lines = help_lines();
        let geo = layout(&lines, usize::MAX, 320, 240);
        assert!(!geo.quads.is_empty());
    }
}
