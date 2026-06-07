# planet-explorer

A real-time 3D procedural planet explorer in Rust, built directly on **wgpu**
(Metal on macOS) — no game engine. Generate a planet from a seed and fly
seamlessly from orbit down to the grass.

```bash
cargo run                 # random seed
cargo run -- --seed 12345 # reproducible planet
```

The seed is printed on launch and shown in the window title. The same seed
always produces the same planet.

## Build a macOS app bundle

```bash
./package_macos.sh
```

Produces `dist/Planet Explorer.app` (ad-hoc signed, double-clickable) and copies
it to `/Users/Shared/Planet Explorer.app` so a different macOS account can grab
it. To launch with a specific seed:

```bash
"dist/Planet Explorer.app/Contents/MacOS/planet-explorer" --seed 12345
```

If macOS Gatekeeper ever complains, right-click the app → **Open**, or run
`xattr -dr com.apple.quarantine "Planet Explorer.app"`.

## Controls

**Keyboard only.** Twin-stick style: **WASD moves, the arrow keys look around.**

| Input | Action |
|-------|--------|
| WASD | Move / strafe |
| Arrow keys | Look around (turn / pitch) |
| Space / C | Ascend / descend (zoom orbit ↔ surface; drives LOD) |
| Shift | Sprint (move faster) |
| + / − | Adjust movement speed |
| R | Teleport to a random spot on the surface |
| P | Print current location & seed to stdout |
| G | Toggle wireframe |
| Esc | Toggle the help overlay (key bindings + build version) |
| Cmd-Q / Ctrl-Q | Quit (or close the window) |

## How it works

The camera is a single continuum — there are no hard "orbit" vs "surface" modes.
It stores an *anchor direction* (the lat/long it's above) and an *altitude*;
position is always `anchor · (surface_radius + altitude)`. Zooming just shrinks
the altitude, and the controls reinterpret the same state as you descend, so the
transition from space to ground is perfectly continuous and you can never end up
underground.

The planet is a **cube-sphere**: six faces, each a quadtree that subdivides as
the camera nears and merges as it recedes. Chunks are meshed on a background
thread pool and cached by coordinate; downward "skirts" hide the cracks between
LOD levels. Terrain height comes from layered fractal noise (continents + ridged
mountains + detail, with domain warping). Biomes are assigned from latitude,
altitude, and moisture, then vertex-colored. Vegetation (instanced trees and
shrubs) is scattered deterministically per chunk by biome rules.

Rendering is a single depth-tested pass: a seeded starfield + atmospheric rim
glow, then lit/fogged terrain, instanced vegetation, and an animated
transparent ocean. The sun direction and atmosphere tint are derived from the
seed, so every world is lit differently.

### Module layout

| Module | Responsibility |
|--------|----------------|
| `planet` | Seeded source of truth: height field, biomes, sun, atmosphere |
| `mesh` | Planet samples → triangles, skirts, vegetation instances, base meshes |
| `lod` | Cube-sphere quadtree selection + background meshing pool |
| `camera` | The seamless orbit↔surface control continuum |
| `gfx` | wgpu renderer (sky / terrain / vegetation / water) |

Each system queries `Planet` for ground truth without touching the others —
the seam along which the roadmap (animals, NPCs, manipulable objects, weather)
will grow.

## Tech

Rust (stable) · wgpu 29 · winit 0.30 · glam · noise-rs · rodio (looping
soundtrack) · png · bytemuck · pollster

The planet artwork and soundtrack are embedded in the binary (`assets/`), so the
app is self-contained. The soundtrack loops while the app is open, and the planet
image appears on the ESC help overlay. The macOS app icon is built from it too.

## Tests

```bash
cargo test
```

Headless checks cover generation determinism, value ranges, biome variety, mesh
well-formedness, and LOD selection — plus an **offscreen GPU smoke test** that
compiles the real shaders, builds every pipeline, renders a frame, and reads it
back to confirm geometry actually draws (no window required).
