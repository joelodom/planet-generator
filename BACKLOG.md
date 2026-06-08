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

- [ ] **GPU buffer pooling / suballocation.** `upload_chunk` creates 2–4 fresh
  `wgpu::Buffer`s per chunk and frees them on eviction — constant allocate/free churn
  as the camera moves (the review's H2; the log showed thousands of uploads/2 s near
  the budget cap). Pool freed buffers by size class and reuse, or suballocate chunk
  meshes from a few large growable buffers, to cut allocation overhead and
  fragmentation. The bigger structural win for the 5090 is collapsing the per-chunk
  draws via batching / indirect / multi-draw (pairs with the `split_factor` item).
  Effort: **M** (a size-classed free list) → **L** (indirect draw).
  _(LRU eviction — formerly bundled here — shipped 2026-06-08 with the byte budget.)_

- [ ] **Wind sway.** Animate vegetation in `vegetation.wgsl`'s vertex stage —
  amplitude scaling up the plant (base fixed, tips move), from a wind vector, time
  (`camera_pos.w`), and a per-instance phase. **Precision caveat (ARCH §8):** do
  *not* phase off absolute world position (~637 k units loses `sin` phase) — use a
  per-instance phase scalar or local/fractional coords (as the water animation does).
  Render-time only; the world stays deterministic. Pool B ×1.0–1.05. Effort: **S**.

## Low

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
