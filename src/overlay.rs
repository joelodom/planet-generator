//! The ESC help overlay: turns lines of text into screen-space colored quads
//! using the embedded bitmap font. Pure CPU geometry (no GPU types here) so it's
//! unit-testable; the renderer just uploads the instances and draws them.

use crate::font8x8::FONT8X8;
use bytemuck::{Pod, Zeroable};

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

/// The help text. Composed here so the build version/date (stamped via `build.rs`)
/// show on the overlay exactly as in the logs.
pub fn help_lines() -> Vec<String> {
    let built = env!("BUILD_DATE").replace('T', " ").replace('Z', " UTC");
    // Quit shortcut differs by platform (Cmd on macOS, Ctrl elsewhere).
    let quit = if cfg!(target_os = "macos") { "Cmd-Q" } else { "Ctrl-Q" };
    vec![
        "PLANET EXPLORER".to_string(),
        format!("version {}    build {}", env!("CARGO_PKG_VERSION"), env!("GIT_HASH")),
        format!("built {built}"),
        String::new(),
        "CONTROLS".to_string(),
        "Left-drag     orbit the planet".to_string(),
        "Scroll        zoom  orbit <-> surface".to_string(),
        "WASD          move across the surface".to_string(),
        "Space / C     ascend / descend".to_string(),
        "Shift         sprint (move faster)".to_string(),
        "Right-drag/F  look around  (F toggles free-look)".to_string(),
        "+ / -         adjust movement speed".to_string(),
        "R             teleport to a random spot".to_string(),
        "P             print location to the log".to_string(),
        "G             toggle wireframe".to_string(),
        String::new(),
        "ESC           toggle this help".to_string(),
        format!("{quit:<13} quit  (or close the window)"),
    ]
}

const PIXEL: f32 = 1.0; // font cell columns are drawn 1:1 then scaled

/// Lay out `lines` centered on a `screen_w` x `screen_h` surface, returning the
/// quads for a dim backdrop, a bordered panel, and the text.
pub fn layout(lines: &[String], screen_w: u32, screen_h: u32) -> Vec<OverlayInstance> {
    let sw = screen_w.max(1) as f32;
    let sh = screen_h.max(1) as f32;

    let cols = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0).max(1) as f32;
    let rows = lines.len().max(1) as f32;

    // Pick the largest integer pixel scale (2..=6) that fits with margins, so the
    // overlay stays crisp and readable on both small and HiDPI surfaces.
    let mut scale = 6.0f32;
    while scale > 2.0 {
        let bw = cols * 8.0 * scale;
        let bh = rows * 10.0 * scale;
        if bw <= sw * 0.88 && bh <= sh * 0.88 {
            break;
        }
        scale -= 1.0;
    }

    let char_adv = 8.0 * scale;
    let line_h = 10.0 * scale; // 8px glyph + 2px gap
    let block_w = cols * char_adv;
    let block_h = rows * line_h;
    let ox = ((sw - block_w) * 0.5).round();
    let oy = ((sh - block_h) * 0.5).round();
    let pad = 7.0 * scale;

    let mut out = Vec::new();
    let mut push = |x, y, w, h, c| out.push(OverlayInstance::px(x, y, w, h, c, sw, sh));

    // Dim the whole scene, then a bordered panel behind the text.
    push(0.0, 0.0, sw, sh, [0.0, 0.0, 0.0, 0.55]);
    let (px0, py0, pw, ph) = (ox - pad, oy - pad, block_w + 2.0 * pad, block_h + 2.0 * pad);
    push(px0 - 2.0, py0 - 2.0, pw + 4.0, ph + 4.0, [0.30, 0.52, 0.74, 0.95]); // border
    push(px0, py0, pw, ph, [0.05, 0.07, 0.12, 0.94]); // panel

    let title = [0.55, 0.80, 1.0, 1.0];
    let body = [0.92, 0.95, 1.0, 1.0];

    for (r, line) in lines.iter().enumerate() {
        let color = if r == 0 { title } else { body };
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
                        let x = cx + col as f32 * scale * PIXEL;
                        let y = base_y + row as f32 * scale;
                        push(x, y, scale, scale, color);
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_nonempty_and_in_ndc() {
        let lines = help_lines();
        let inst = layout(&lines, 1280, 800);
        // Backdrop + border + panel + many glyph pixels.
        assert!(inst.len() > 1000, "expected substantial text geometry, got {}", inst.len());
        for q in &inst {
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
        let inst = layout(&lines, 320, 240);
        assert!(!inst.is_empty());
    }
}
