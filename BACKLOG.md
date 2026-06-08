# Backlog

A running, user-managed list of ideas and future work for Planet Explorer,
grouped by priority. (The point-in-time engineering assessment lives in
`ARCHITECTURE_REVIEW.md`; the standards new work is held to live in
`ARCHITECTURE_GUIDELINES.md`.)

## Critical

_(none — the treetop vegetation flashing was **resolved 2026-06-08 by vegetation
instancing**; see the ✅ item under High.)_

## High

_(From the 2026-06-08 memory-efficiency analysis. `#1` byte-accurate budgeting and
`#2` memory logging are already done; these are the structural follow-ups. Use the
new `mem_mb` HUD/log readout to measure before/after.)_

- [x] **✅ Vegetation instancing — DONE (2026-06-08).** Each species' base mesh is
  uploaded to the GPU once (`Renderer.species_meshes`); `mesh::place_vegetation` emits
  per-plant `VegInstance`s (model matrix + tint, ~80 bytes each) grouped by species
  into `CpuChunk.veg: VegChunk`, drawn with one instanced `draw_indexed` per species
  via `shaders/vegetation.wgsl` (lit/fogged like terrain). **Cut veg memory ~95×** (a
  dense forest chunk: ~54 KB of instances vs ~5 MB baked) — which **fixed the treetop
  flashing** (the baked covering used to exceed the budget), and let `DENSITY_MAX` go
  back to 750 and the memory caps go up (High 8 GB). The old per-vertex baking survives
  only as the test helper `VegChunk::bake`.

- [ ] **Tame `split_factor` (draw-call count).** _(the memory pressure here is gone —
  instancing made the working set tiny; this is now just about draw calls)_ At High,
  `split_factor` ≈ 4.7 draws ~3,800 chunks/frame — fine on memory, but a lot of draw
  calls (terrain + one per species per veg chunk). Lowering `settings.rs` `SPLIT_MAX`
  draws fewer, larger chunks (cuts draw calls, mostly free visually). The density/poly
  levers (`DENSITY_MAX`, `flora.rs` `BLOB_RINGS`/`BLOB_SECTORS`/…) are no longer needed
  for memory — touch them only for looks/CPU. Effort: **S** (tuning).

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
