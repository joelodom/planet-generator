//! The ESC help overlay: turns lines of text into screen-space colored quads
//! using the embedded bitmap font. Pure CPU geometry (no GPU types here) so it's
//! unit-testable; the renderer just uploads the instances and draws them.

use crate::font8x8::FONT8X8;
use crate::settings::{self, Graphics};
use crate::units::Units;
use bytemuck::{Pod, Zeroable};

/// Width (chars) of a settings slider bar, e.g. `[####------]`.
const BAR_WIDTH: usize = 10;
/// Width (chars) of the HELP tab's key column (left), before the description.
const KEY_COL: usize = 13;

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
    // BUILD_DATE is already local and display-ready (see build.rs), e.g.
    // "2026-06-08 07:16:59 EDT".
    vec![
        "PLANET EXPLORER".to_string(),
        format!("version {}   build {}", env!("CARGO_PKG_VERSION"), env!("GIT_HASH")),
        format!("built {}", env!("BUILD_DATE")),
    ]
}

/// The HELP tab body: two aligned columns — key(s) on the left, what they do on
/// the right.
fn help_body() -> Vec<String> {
    let quit = if cfg!(target_os = "macos") { "Cmd-Q" } else { "Ctrl-Q" };
    let rows: [(&str, &str); 13] = [
        ("Arrow keys", "Pan across the surface"),
        ("W / S", "Zoom in"),
        ("+ / -", "Zoom out"),
        ("A / D", "Rotate (spin) the view"),
        ("Q / E", "Tilt (top-down to horizon)"),
        ("Shift", "Move faster (hold)"),
        ("T", "Guided tour (autopilot)"),
        ("R", "Teleport to a random spot"),
        ("P", "Print location to the log"),
        ("G", "Toggle wireframe"),
        ("Tab", "Switch HELP / GRAPHICS tab"),
        ("Esc", "Close this menu"),
        (quit, "Quit (or close the window)"),
    ];
    rows.iter().map(|(k, d)| format!("{:<width$}{}", k, d, width = KEY_COL)).collect()
}

/// The GRAPHICS tab body: a hint line then the setting rows. Returns the body and
/// the index *within the body* of the selected row.
fn graphics_body(graphics: &Graphics, selected: usize, sys: Units) -> (Vec<String>, usize) {
    let mut v = vec!["up/down: select     left/right: adjust".to_string()];
    for (i, row) in graphics.rows(sys).iter().enumerate() {
        let marker = if i == selected { '>' } else { ' ' };
        let body = match row.frac {
            Some(f) => format!("{}  {:>6}", bar(f), row.value),
            None => format!("  < {} >", row.value),
        };
        v.push(format!("{} {:<15} {}", marker, row.label, body));
    }
    (v, 1 + selected) // +1: the hint line precedes the rows
}

fn tab_bar(tab: usize) -> String {
    // The active tab is bracketed.
    let (h, g) = if tab == settings::TAB_GRAPHICS {
        ("  HELP  ", "[ GRAPHICS ]")
    } else {
        ("[ HELP ]", "  GRAPHICS  ")
    };
    format!("{h}    {g}        Tab: switch")
}

/// Plain help text (no live settings) — used by the offscreen smoke test.
#[cfg(test)]
pub fn help_lines() -> Vec<String> {
    let mut v = header_lines();
    v.push(String::new());
    v.extend(help_body());
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

/// Build the ESC menu for the active `tab` (HELP or GRAPHICS) and the row index
/// to highlight (the selected setting on the GRAPHICS tab; none on HELP).
///
/// Both tabs are padded to a common width and height so the panel size and font
/// scale never change when you switch tabs.
pub fn menu(graphics: &Graphics, tab: usize, selected: usize, sys: Units) -> (Vec<String>, usize) {
    let help = help_body();
    let (gfx, gfx_hl) = graphics_body(graphics, selected, sys);
    let body_rows = help.len().max(gfx.len());

    let mut lines = header_lines();
    lines.push(String::new());
    lines.push(tab_bar(tab));
    lines.push(String::new());
    let body_start = lines.len();

    let highlight = if tab == settings::TAB_HELP {
        lines.extend(help.iter().cloned());
        usize::MAX
    } else {
        lines.extend(gfx.iter().cloned());
        body_start + gfx_hl
    };
    // Pad the body so the footer and panel height match across tabs.
    while lines.len() < body_start + body_rows {
        lines.push(String::new());
    }

    lines.push(String::new());
    lines.push("Tab: switch tab      Esc: close".to_string());

    // Pad every line to a common width — the max over BOTH tabs' content — so the
    // panel width and font scale are identical regardless of which tab shows.
    let width = lines
        .iter()
        .chain(help.iter())
        .chain(gfx.iter())
        .map(|l| l.chars().count())
        .max()
        .unwrap_or(0);
    for l in &mut lines {
        let n = l.chars().count();
        if n < width {
            l.push_str(&" ".repeat(width - n));
        }
    }

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
