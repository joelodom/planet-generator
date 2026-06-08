# Backlog

A running, user-managed list of ideas and future work for Planet Explorer,
grouped by priority. (The point-in-time engineering assessment lives in
`ARCHITECTURE_REVIEW.md`; the standards new work is held to live in
`ARCHITECTURE_GUIDELINES.md`.)

## Critical

- [ ] **🔴 UNRESOLVED: treetop vegetation flashing.** At low altitude / treetop in
  dense forest at the **High** preset, the scene rapidly draws-and-erases ("flashes")
  — confirmed in the perf log by `draw` oscillating ~11 ↔ ~1191 with `mem_mb` pinned
  at the 4 GB budget. **Diagnosis:** the drawn LOD covering needs more memory than the
  budget, so eviction is forced to drop chunks needed *this frame* → the LOD collapses
  to root chunks → re-streams → repeats. **Tried, did NOT fix it:** byte-accurate
  eviction, LRU eviction, request-throttling at the budget, and capping veg
  `DENSITY_MAX` 750→250. Since the density cut didn't help, either the covering *still*
  exceeds the budget, and/or a second cause is in play: resident is pinned at the 4 GB
  budget by the LRU cache, and 4 GB of GPU geometry may exceed the MacBook's available
  memory → driver paging (also reads as a flash), independent of veg density.
  **Next steps, in order:** (1) **drop the Memory slider live (e.g. to 1 GB)** — if the
  flash stops, it's GPU/cache pressure; fix by capping the cache well below the slider
  value or lowering the default budget; (2) **vegetation instancing** (High item below)
  to shrink the covering so it genuinely fits; (3) cap LOD subdivision by a memory
  target so `select` never asks for more than fits. **Do not ship High as default
  until fixed.**

## High

_(From the 2026-06-08 memory-efficiency analysis. `#1` byte-accurate budgeting and
`#2` memory logging are already done; these are the structural follow-ups. Use the
new `mem_mb` HUD/log readout to measure before/after.)_

- [ ] **Vegetation instancing.** _(the biggest memory win; preserves visual quality)_
  Today every plant is baked as unique world-space geometry into its chunk
  (`mesh::place_vegetation` → `bake_plant`), so identical plants are duplicated
  thousands of times. One dense chunk holds ~150k veg verts (~5+ MB) and this
  dominates resident memory.
  **Goal:** upload each species' base mesh **once** and *instance* it per placement
  with a per-instance transform + tint, instead of baking full geometry per chunk.
  **Sketch:** keep `flora::Species.mesh` as the base local-space mesh in a
  per-species GPU mesh table on the `Renderer`; have `place_vegetation` emit a
  compact per-chunk instance list `(species_id, model/{pos,scale,yaw}, tint)`
  (~tens of bytes each) rather than baked vertices; group a chunk's instances by
  species and issue one instanced `draw_indexed(.., instances)` per species, with an
  instance-step vertex buffer carrying transform+tint, via a small vegetation
  pipeline/shader (apply per-instance model + normal matrix + tint, then light/fog
  like terrain). Memory goes from O(plants × plant_verts) → O(plants) +
  O(species × base_verts), roughly a 10–50× cut on veg.
  **Watch:** keep determinism (instances seeded per chunk key — unchanged), the
  LOD-independence, and the biome-blend behaviour. This **reverses** the earlier
  deliberate "no instancing, unlimited variety in one draw" decision (see the
  flora-vegetation design notes) — it trades a few more draw calls for a large
  memory cut. Effort: **L**.

- [ ] **Shrink the working set: tame `split_factor` + veg cost.** _(quicker; partial
  quality trade)_ Even with byte-accurate budgeting, the *drawn* set is the memory
  floor (eviction can't drop on-screen chunks), and at High that's ~3,800 chunks per
  frame (`split_factor` ≈ 4.7), each carrying dense veg — so the floor can still be
  multiple GB. Levers (all named consts, tune against the `mem_mb` readout):
  - `settings.rs` `SPLIT_MIN`/`SPLIT_MAX` — High's `split_factor` subdivides very
    eagerly; lowering the top end draws fewer, larger chunks (cuts draw calls **and**
    memory) and is mostly free visually.
  - `settings.rs` `DENSITY_MIN`/`DENSITY_MAX` — vegetation attempts per chunk (up to
    750); lower to cut plant counts.
  - `flora.rs` `BLOB_RINGS`/`BLOB_SECTORS`, `TRUNK_SIDES`, `CONE_SIDES`, frond `SEGS`
    — drop plant poly counts (halving blob rings/sectors ~halves veg verts).
  - Consider gating vegetation to only the finest 1–2 LOD levels (raise the effective
    `veg_min_level`) so fewer chunks carry plants at all.
  **Trade-off:** density/poly cuts reduce lushness; the `split_factor` cut is the
  safe one. Effort: **S–M** (mostly tuning + measuring).

## Medium

- [ ] **GPU buffer pooling / suballocation.** `upload_chunk` creates 2–4 fresh
  `wgpu::Buffer`s per chunk and frees them on eviction — constant allocate/free churn
  as the camera moves (the review's H2; the log showed thousands of uploads/2 s near
  the budget cap). Pool freed buffers by size class and reuse, or suballocate chunk
  meshes from a few large growable buffers, to cut allocation overhead and
  fragmentation. Effort: **M** (a size-classed free list).
  _(LRU eviction — formerly bundled here — shipped 2026-06-08 with the byte budget:
  `Renderer::evict` now drops least-recently-used first, which also fixed a
  zoom-never-refines regression.)_

## Low

- [ ] **Tour camera — keep motion generally forward.** During the guided tour, the
  camera's movement should always read as the viewer moving *forward*. A little
  randomized sideways slipping is fine, but it should never pull backward or drift
  directly sideways — generally forward, with some sideways slip.
  _Today the cruise phase picks a fully random great-circle drift direction
  (`tour.rs` → `begin_cruise` / `drift_axis`), so it can head backward or straight
  sideways relative to where the camera is looking._
