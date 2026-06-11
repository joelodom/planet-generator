# Backlog

A running, user-managed list of ideas and future work for Planet Explorer,
grouped by priority. (The point-in-time engineering assessment lives in
`ARCHITECTURE_REVIEW.md`; the standards new work is held to live in
`ARCHITECTURE_GUIDELINES.md`.)

## Critical

_(none)_

## High

- [ ] **Day/night terminator (whole-planet lighting).** The planet has no real
  night side: terrain/veg shading applies a flat ambient (~0.42) everywhere, so the
  hemisphere facing away from the sun is still ~42 %-lit — a uniformly-lit globe
  with no terminator, which reads wrong. Drive ambient from the sun geometry
  (`dot(surface_up, sun_dir)`) so it decays through a soft `smoothstep` terminator
  to a low twilight floor (`NIGHT_FLOOR`) — dark but still navigable (starfield +
  a faint cool atmosphere/moonlight fill), warm-tinted in the band. Shader-only
  (`terrain.wgsl`, `vegetation.wgsl`; `sky.wgsl` already brightens the sunlit rim),
  a few ALU ops/fragment, **no new memory**, stays deterministic from the per-world
  sun direction. Keep the tour/free-fly readable on the night side. Effort: **S–M**.
  _(Highest visual impact for the cost.)_

- [ ] **Flora look overhaul — shading + colour + per-instance variation.** Three
  ~memory-free changes that together move vegetation from "CG" toward believable:
  - **Tier-0 shading** (`vegetation.wgsl` + a per-vertex material/AO byte on the
    base meshes): leaf back-light **translucency** (the biggest "plastic → leaf"
    cue), **half-Lambert wrap** to soften the terminator, a weak leaf **sheen**, and
    baked **AO + foliage-normal break-up** so a canopy stops shading like a balloon.
  - **Climate-driven colour:** tie foliage hue/value/saturation to the sample's
    temperature & moisture (the planet already computes both) instead of
    `green_foliage`'s pure-RNG pick, so stands read cohesive and place-appropriate.
  - **Per-instance variation:** small lean + gentle non-uniform scale + wider tint
    in the *already-emitted* model matrix / tint, to kill the "clone army."
  Pool A +~0.2 MB, Pool B ×1.0; gen-time AO stays off the main thread. Effort: **M**.
  _(Recursive branches + leaf cards and the area-proportional density redesign have
  already shipped — this is the shading/colour layer on top.)_

## Medium

- [ ] **World generation uses a non-reproducible RNG (`rand::StdRng`).**
  _[Correctness — latent, no bug today]_ Generation **seeds** are derived with the
  hand-rolled splitmix64 (`mix`, `ChunkKey::hash`) — stable — but the RNG *stream*
  is `rand::rngs::StdRng` (`flora.rs`, `mesh.rs::place_vegetation`). **Within a
  single process and a pinned `rand` version it is fully deterministic** (and the
  in-process determinism tests pass), so nothing is broken now. The risk is
  *across versions*: `StdRng` is an unspecified alias the `rand` crate may change
  between releases, and its own docs say not to rely on it for reproducibility. A
  future `rand` upgrade could silently re-roll **every** world — every saved seed
  renders a different planet; gallery baselines shift. `tests.rs::generation_is_deterministic`
  compares two planets in the *same* process, so it cannot catch this. **Fix:** drive
  generation from a documented-stable PRNG — pin `rand_chacha::ChaCha*Rng` explicitly,
  or generate the stream from the existing splitmix64/PCG and drop `rand` from the
  generation path (keep `rand` only for the deliberately non-deterministic uses:
  random seed, teleport, tour drift). Add a **golden-value** test (assert a known
  `height(seed, dir).to_bits()`) so any future RNG/algorithm change trips CI.
  Effort: **S–M**. _(Aligns the code with `ARCHITECTURE_GUIDELINES.md` §1, which
  already names "splitmix64 sub-seeding" as the determinism exemplar.)_

- [ ] **Tame `split_factor` (draw-call count).** _(The memory pressure here is gone —
  instancing made the working set tiny; this is now just about draw calls.)_ At High,
  `split_factor` ≈ 4.7 draws ~3,800 chunks/frame — fine on memory, but a lot of draw
  calls (one `draw_indexed` per terrain chunk + one per species per veg chunk).
  Lowering `settings.rs` `SPLIT_MAX` draws fewer, larger chunks (cuts draw calls,
  mostly free visually). The density/poly levers are no longer needed for memory —
  touch them only for looks/CPU. Effort: **S** (tuning).
  _**Prime suspect for the Windows / RTX 5090 report (2026-06-08): "GPU 100 % util but
  ~50 % power, low FPS."** That signature is submission / draw-call-bound, not
  compute-bound — thousands of tiny per-chunk draws keep the GPU front-end and the
  (heavier-on-DX12) driver busy without saturating the ALUs. Fewer/larger chunks here
  plus the buffer pooling/batching below directly target it._
  _(Update 2026-06-08: the terrain arena below shipped, collapsing the per-chunk
  buffer-*bind* overhead; this item is now about the residual per-chunk `draw_indexed`
  COUNT, which lower `SPLIT_MAX` still reduces.)_

- [x] **✅ Terrain suballocation — DONE (2026-06-08).** Resident terrain now lives in a
  shared `MeshArena` (fixed-size slots in a few large blocks, since every chunk is the
  same size for a grid): per-chunk buffer create/free is gone, and the draw loop binds
  each block ONCE and draws every chunk in it by base_vertex/first_index — terrain
  buffer binds dropped from ~2 per drawn chunk to ~one per block (~20 at High vs
  thousands). Veg also stopped rebinding a base buffer per species (all species' base
  meshes are concatenated into one buffer, selected by base_vertex/first_index). This
  targets the Windows/5090 draw-call bottleneck. _(LRU eviction shipped earlier with
  the byte budget.)_

- [ ] **Remaining draw-call headroom (veg instances + indirect).** Two follow-ups now
  that terrain is pooled: (a) veg *instance* buffers are still per-chunk (one bind per
  veg chunk) — route them through an arena too so the instance buffer binds once per
  block; (b) collapse the residual per-chunk `draw_indexed` calls with
  multi-draw-indirect — **DX12/Vulkan only (not Metal in wgpu)**, so gate it behind a
  feature check and keep the current path for macOS. Effort: **M** (veg arena) → **L**
  (indirect). Measure on the 5090 first — the terrain arena may already be enough.

- [ ] **Wind sway.** Animate vegetation in `vegetation.wgsl`'s vertex stage —
  amplitude scaling up the plant (base fixed, tips move), from a wind vector, time
  (`camera_pos.w`), and a per-instance phase. **Precision caveat (ARCH §8):** do
  *not* phase off absolute world position (~637 k units loses `sin` phase) — use a
  per-instance phase scalar or local/fractional coords (as the water animation does).
  Render-time only; the world stays deterministic. Pool B ×1.0–1.05. Effort: **S**.

## Low

- [ ] **Seasons — different parts of the planet in different seasons at once.**
  Drive a per-sample seasonal state from latitude + hemisphere (and an axial-tilt /
  time-of-year phase), so at any instant the two hemispheres sit in opposite seasons
  and the equator stays roughly aseasonal — the way a real planet looks from space.
  Season then modulates the *appearance* layers that already exist: foliage
  hue/value (deciduous greens → autumn golds/reds → bare/sparse winter), the snow
  line (`classify()`'s `SNOW_BASE`/`SNOW_TEMP_RANGE` push equatorward in local
  winter), and grassland/tundra colour. Keep it **deterministic** — derive the
  year-phase from the seed (and, if it should animate, from the same time source the
  water/wind use), never from wall-clock; biome *classification* should stay stable
  (seasons recolour and adjust the snow line, they don't re-roll the world). Prefer a
  shader-side tint/snow-line shift (no new memory) over regenerating chunks. New
  tuning (tilt angle, season→colour ramps, snow-line swing) must be named consts per
  the no-magic-numbers rule. Effort: **M–L**. _(Stacks on the day/night terminator and
  the flora colour work — same shading layer, same determinism constraints.)_

- [ ] **Flora photoreal leap — leaf-card textures + impostors + MSAA.** The only flora
  items that add (fixed, preset-gated) VRAM: alpha-tested **leaf-card** clusters + a
  small bark/leaf atlas (Pool C +5–50 MB) for lacy canopy silhouettes; **vegetation
  LOD impostors** (far plants → one billboard) which *reduce* far-field Pool B and GPU
  and let a low `veg_min_level` coexist with sane near-field density; and **MSAA / FXAA**
  to fix thin-trunk/leaf-edge aliasing for *all* geometry (MSAA: render-target VRAM ×
  `sample_count`, ~120 MB @ 1440p 4×). Gate every VRAM knob behind `settings.rs`
  presets (the 5090 cranks; the laptop preset stays lean). Effort: **L**.

- [ ] **Evaluation: render the flora gallery through the veg pipeline.** The offscreen
  `gallery` (`gfx.rs`) currently renders flora via the **terrain** shader (baked to
  world space), so `vegetation.wgsl` shading work won't show. Extend it to draw via
  the vegetation pipeline, and add a back-lit view (translucency) and a wide-stand
  view (density). Prerequisite for evaluating the flora-look items above headlessly.
  Effort: **S**.

- [ ] **Tour camera — keep motion generally forward.** During the guided tour, the
  camera's movement should always read as the viewer moving *forward*. A little
  randomized sideways slipping is fine, but it should never pull backward or drift
  directly sideways — generally forward, with some sideways slip.
  _Today the cruise phase picks a fully random great-circle drift direction
  (`tour.rs` → `begin_cruise` / `drift_axis`), so it can head backward or straight
  sideways relative to where the camera is looking._

## Done (recent)

- [x] **Vegetation instancing (2026-06-08).** Each species' base mesh is uploaded once
  and drawn with per-instance transform + tint (`shaders/vegetation.wgsl`,
  `mesh::VegChunk`) — ~95× less veg memory, which fixed the high-detail treetop
  flashing and let density/budget caps go back up. The old per-vertex baking survives
  only as the test helper `VegChunk::bake`.
- [x] **Area-proportional vegetation density (2026-06-08).** Density is now
  plants-per-unit-area with a per-chunk cap (LOD-independent, budgetable), tuned to
  ecological per-biome targets — replacing the old fixed attempts-per-chunk.
