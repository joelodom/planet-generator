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

The planet is **Earth-sized** (6,371 km radius), with Earth-like elevations and
real-world units in the HUD — metric by default, or `--units us` for imperial.

**Keyboard only.** Google Earth–style navigation.

| Input | Action |
|-------|--------|
| Arrow keys | Pan across the surface |
| W / S (or + / −) | Zoom in / out (drives LOD) |
| A / D | Rotate (spin) the view |
| Q / E | Tilt (top-down ↔ horizon) |
| Shift | Move faster (hold) |
| R | Teleport to a random spot on the surface |
| P | Print current location & seed (real units) |
| G | Toggle wireframe |
| Esc | Open the overlay (**HELP** / **GRAPHICS** tabs) |
| Cmd-Q / Ctrl-Q | Quit (or close the window) |

### Graphics settings (Esc → GRAPHICS tab)

Press **Esc** for an on-screen overlay with two tabs (**Tab** switches): HELP
(these key bindings) and GRAPHICS. On the GRAPHICS tab, ↑/↓ select a row and
←/→ adjust:

| Setting | What it does |
|---------|--------------|
| Preset | Low / Medium / High / Ultra |
| Detail | One master slider — drives LOD subdivision, terrain mesh resolution, and vegetation (and future detail objects) together |
| Memory budget | How much geometry stays resident as you look around |

LOD detail and the memory budget apply immediately; mesh/vegetation re-mesh the
world when you close the menu. **Ultra is intentionally punishing** — tune it
down on a laptop, crank it up on a big GPU.

## How it works

The camera works like Google Earth: it **orbits a focus point on the surface**,
parameterized by `focus` (the lat/long you're looking at), `distance` (zoom),
`heading` (spin), and `tilt` (top-down ↔ horizon). Pan/zoom speeds scale with
distance, so it feels the same from orbit down to ground level. The planet is
Earth-sized (6,371 km); the world is rendered in 10 m "units" to stay within f32
precision without a camera-relative renderer, and near/far planes track the
visible horizon each frame so the globe never clips and the ground never z-fights.

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
