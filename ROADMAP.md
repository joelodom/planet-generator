# Planet Explorer — Engineering Roadmap

**Written:** 2026-07-19 • **Reviewed tree:** commit `b5ff9c9` (clean working tree,
`main`) • **Reviewer:** full-source expert pass (every Rust module + all five WGSL
shaders + packaging/scripts), with measurements taken on the dev Mac.

This is the **forward-looking engineering plan**: what to improve, why it matters,
and concretely how to build each item. It complements the other three documents
rather than replacing them:

| Document | Role |
|---|---|
| [`ARCHITECTURE_GUIDELINES.md`](ARCHITECTURE_GUIDELINES.md) | The standing standard (priority hierarchy: correctness → security → performance → maintainability → portability → polish). Every item below is designed to score on it. |
| [`ARCHITECTURE_REVIEW.md`](ARCHITECTURE_REVIEW.md) | Point-in-time assessment of 2026-06-08. Now ~6 weeks / ~2,400 LOC stale; §1.2 below is the up-to-date ledger of its findings. |
| [`BACKLOG.md`](BACKLOG.md) | The user-managed wishlist. Items from it that this roadmap absorbs are cross-referenced as *(BACKLOG: …)* — the backlog entry stays the user-facing tracker; this document adds the engineering why/what/how. |

**How to read an item.** Each has a stable ID (`A1`, `G4`, …), tags for the
guideline dimension it serves, an effort grade (**S** < ½ day, **M** 1–2 days,
**L** ~1 week, **XL** multi-week), and three parts: **Why** (grounded in the
current code, with evidence), **What** (the design), **How** (implementation
steps, specific functions/constants/APIs). Items are *not* in execution order —
the **Sequencing** section at the end gives the recommended phases and the
dependency graph.
Deliberately excluded: pure nitpicks (the tree is clippy-clean and has zero
`TODO`/`unsafe`; there is no lint-level debt worth listing).

---

## 1. Current state — evidence snapshot

Everything below was verified on this tree, today (not carried forward from the
old review):

- **Size:** 7,969 lines of Rust across 17 modules + 363 lines of WGSL across 5
  shaders. Largest files: `gfx.rs` 1,943 (≈940 of it `#[cfg(test)]` harnesses),
  `main.rs` 1,048, `planet.rs` 615, `mesh.rs` 534, `tour.rs` 517.
- **Health:** `cargo clippy --all-targets` — **0 warnings**. `cargo test
  --release` — **35 passed, 0 failed, 2 ignored** (the ignored two are a
  micro-benchmark and a slow visual scan, by design). The suite includes real
  GPU smoke tests (offscreen pipeline validation + framebuffer readback) that
  skip cleanly without an adapter.
- **Measured costs** (release, Apple Silicon dev Mac):
  - `Planet::sample` (full, always-slope): **~3.85 µs/sample**;
    `Planet::sample_terrain` (lean meshing path): **~2.42 µs/sample**
    (via the in-repo `planet::tests::sampling_cost` bench). A single `height()`
    is ~1.2 µs — it evaluates ~20 noise octaves in f64.
  - `Flora::generate` (51 embedded GLBs → parse + PNG decode + resample to 512²
    + CPU mip chains): **~0.7 s**, single-threaded, on the main thread before
    the window opens.
- **Assets:** `assets/` is **280 MB**, of which `assets/models/` is **242 MB**
  of GLBs — all embedded via `include_bytes!`, so the release binary is ~¼ GB
  and every link step re-swallows it. `flora-revamp/` adds **421 MB** of
  intermediate images/bakes to the git history.
- **No CI** (`.github/` absent), no `rustfmt.toml`, no `rust-toolchain.toml`,
  no `cargo-deny`/`audit` config.
- **The shared log file does not currently exist on disk** — so there is no
  fresh in-field perf data to cite. (The log is append-forever by design and was
  evidently cleaned up manually; see D3.)
- **Windows status:** packaging exists and works (`package_windows.ps1` + `.cmd`
  wrapper, static CRT, embedded icon); the RTX 5090 box has run the app at least
  once and produced the "GPU 100 % util / ~50 % power / low FPS" report recorded
  in `BACKLOG.md` — the draw-submission bottleneck that the terrain `MeshArena`
  was built to attack.

### 1.1 What is genuinely strong (don't churn these)

The architecture calls in `ARCHITECTURE_REVIEW.md` §2 all still hold, and the
six weeks since added several more things worth protecting: the terrain
**`MeshArena`** (fixed-slot suballocation, block-bind + `base_vertex` draws) is
exactly the right shape for the draw-call problem; **byte-budgeted LRU
eviction** (`GpuChunk::last_used`, roots pinned) replaced the naive
keep-set-only evictor; the meshing workers are now **`catch_unwind`-supervised**;
`logging::init` **degrades to stderr** instead of panicking; the `want` sort is
**decorate-sorted** (`sort_by_cached_key` on distance bits); the quadtree walk
hoists its invariants into a `Walk` context struct; and the headless `--video`
recorder + `Renderer::new_offscreen` give the project a real offscreen render
path that tests and tools already share. The flora stack (embedded GLB archetype
library → per-biome weighted tables → Worley-clustered deterministic placement →
one instanced draw per species run) is a clean, extensible pipeline.

### 1.2 Ledger — where the 2026-06-08 review findings stand

| Finding | Status today |
|---|---|
| H1 steepness triple-sampling | ✅ Fixed (in-review): `sample_terrain` skips the slope probe below `MOUNTAIN_MIN_HEIGHT`. |
| H2 per-chunk buffers + unbatched draws | ◑ Half-fixed: terrain lives in the shared `MeshArena`; **veg instance buffers are still allocated per chunk** (`upload_chunk` → `create_buffer_init`) and draws are still one `draw_indexed` per chunk (terrain) + per species-run (veg). → A6, J2. |
| M1 veg baked per chunk | ✅ Fixed: per-species instancing (~95× memory win). |
| M2 eviction thrash at ceiling | ✅ Fixed by the LRU + byte budget (recent uploads survive; roots pinned). |
| M3 unsupervised workers | ✅ Fixed: `catch_unwind` in `spawn_worker`. |
| M4 `logging::init` panic | ✅ Fixed: degrades to stderr-only with a warning. |
| M5 oversized files | ❌ Open, worse: `gfx.rs` 1,943 / `main.rs` 1,048. → E1. |
| M6 water ripple feeds ~637k coords into `sin()` | ❌ Open: `terrain.wgsl` still phases off `in.world`. → C1. |
| L1 clippy lints | ✅ Fixed (clean run). |
| L2 water detected by radius heuristic | ❌ Open (`WATER_DETECT_EPS` = 1.5 units). → G5 fixes it properly. |
| L3 duplicated headless harnesses | ❌ Open (×3: `smoke`, `gallery::render_and_save`, `headless_renderer_draws_a_frame`). → E1. |
| L4 world-writable shared log | ➖ Open by design (documented trade-off; workbench-only). |
| L5 per-frame scratch/normalize | ✅ Fixed (`Walk` struct; remaining per-frame `Vec`s are negligible next to A1). |
| L6 `build.rs` shells `date -u` | ❌ Open — and it matters slightly more now: on the real Windows box `date` isn't an executable, so `BUILD_DATE` is empty in the overlay/log. → J3. |
| L7 `altitude()` is a distance proxy | ❌ Open. → C3. |
| L8 flora startup cost | ~ Superseded: procedural species are gone; the replacement (model decode) costs ~0.7 s. → F1/F2. |
| BACKLOG "non-reproducible RNG" | ❌ Open — `StdRng` still drives `place_vegetation` and cluster radii. → D1 (promoted: it's the biggest correctness landmine in the tree). |
| BACKLOG day/night terminator | ❌ Open (flat `SUN_AMBIENT = 0.32` everywhere). → G1. |
| BACKLOG wind sway / flora shading / seasons / impostors | ❌ Open. → G3, B5, H5-adjacent. |

---

## 2. Index of roadmap items

| ID | Title | Effort | Primary win |
|---|---|---|---|
| **A — Main-thread & streaming performance** | | | |
| A1 | Cache `surface_radius` per chunk key in the LOD walk | S | ~5–7 ms/frame of main-thread noise gone at High |
| A2 | Bucket the draw list by arena block once per frame | S | Removes O(blocks × draws) hash probes |
| A3 | Per-frame upload budget (bytes), spillover queue | S–M | Kills teleport/zoom hitch spikes |
| A4 | Priority + staleness for the meshing queue | M | Faster perceived streaming, no wasted builds |
| A5 | Frustum culling in `lod::select` | M | 2–4× fewer draws at ground level |
| A6 | Vegetation instance arena (+ optional multi-draw-indirect) | M–L | Finishes H2; the 5090 headroom item |
| A7 | Chunk-build throughput: octave LOD + lazy veg slope + f32 noise epoch | M–L | 2–4× faster chunk builds at distance |
| **B — GPU & rendering pipeline** | | | |
| B1 | Consistent winding + back-face culling for terrain | S–M | Free raster/fragment savings |
| B2 | Reversed-Z depth (infinite far) | S–M | Bulletproof depth precision; simpler `near_far` |
| B3 | MSAA + alpha-to-coverage for vegetation (preset-gated) | M | The single biggest image-quality lever |
| B4 | Vertex/instance compression (20-byte terrain vertex, u16 indices, packed instances) | M | ~45 % geometry memory + bandwidth |
| B5 | Vegetation mesh LOD → impostors | L–XL | Makes "plants to the horizon" affordable |
| **C — Precision & scale** | | | |
| C1 | Fix water ripple phase (camera-relative) | S | Removes f32 phase-precision artifact (M6) |
| C2 | Camera-relative rendering; raise `MAX_LEVEL` past 16 | XL | Sub-metre ground detail; the walk-mode enabler |
| C3 | True altitude-above-ground for HUD/fog/log | S | Honest numbers in a units-meticulous app |
| **D — Correctness & robustness debt** | | | |
| D1 | Pin the generation RNG + golden-value tests (seed stability epoch) | M | Saved seeds survive dependency upgrades |
| D2 | GPU device-loss recovery (Windows TDR) | M | No wedged black window on the 5090 box |
| D3 | Log rotation (size-capped) | S | The append-forever file can't grow unbounded |
| **E — Structure & process** | | | |
| E1 | Split `gfx.rs`/`main.rs`; one shared headless GPU harness | M | Restores the <500-line norm; kills L3 |
| E2 | Single `build_globals()` shared by live + video paths | S | Removes a real drift hazard |
| E3 | Refresh `ARCHITECTURE_REVIEW.md` (or fold into this doc's ledger) | S | Docs match reality |
| E4 | CI: fmt + clippy + tests on macOS/Windows/Linux + cargo-deny | M | Catches cross-platform rot before the 5090 does |
| E5 | `rust-toolchain.toml` + `rustfmt.toml` | S | Two machines, one toolchain |
| E6 | `justfile` encoding build→package→verify workflows | S | "Always package after build" becomes one command |
| **F — Assets, startup & distribution** | | | |
| F1 | Build-time asset pack (preprocessed models/textures) | L | Binary ~260 MB → ~40 MB; startup decode ~0.7 s → ~0 |
| F2 | Parallel/overlapped startup | S | Window appears immediately even without F1 |
| F3 | Repo hygiene for `flora-revamp/` (421 MB) | S | Clone/pack size sanity |
| **G — Visual richness** | | | |
| G1 | Day/night terminator | S–M | Highest visual impact per line of code |
| G2 | Sun disc + seeded moon in the sky pass | S–M | The sky gets a light source |
| G3 | Flora look: climate tint, per-instance variation, wind sway | M | "CG clone army" → believable stands |
| G4 | LOD-stable vegetation placement (no re-roll on zoom) | M–L | Fixes forests reshuffling as you approach |
| G5 | Water v2: explicit water vertex channel, depth color, foam, Gerstner ripple | M–L | Closes L2 properly; coastlines come alive |
| G6 | Atmosphere v2: single-scatter limb + sunset band | M–L | Horizon depth; pairs with G1 |
| G7 | Near-field cascaded shadow maps | XL | Grounded terrain/vegetation at low altitude |
| **H — Features** | | | |
| H1 | Surface walk mode (first-person) | L | The "down to the grass" promise, delivered |
| H2 | Bookmarks: save/load/share locations | S–M | Cheap, high utility |
| H3 | "Wonders" finder: seed-derived points of interest | M | Unique-to-this-app exploration feature |
| H4 | Screenshot key + `--screenshot` CLI | S–M | Shareable stills; visual-regression fodder |
| H5 | Optional planet rotation (day cycle) | M | Motion for the whole sky/lighting system |
| H6 | Planet archetypes (seed-derived world classes) | M | Massive replay variety for ~zero content cost |
| H7 | Ambient audio layer + volume controls | M | The soundscape half of immersion |
| H8 | Gamepad support via `gilrs` | S–M | Couch/demo ergonomics |
| **I — Tooling & developer experience** | | | |
| I1 | Shader hot-reload (debug builds) | S–M | Iteration speed for every G item |
| I2 | On-screen debug HUD (F3) with LOD/chunk introspection | M | See what the streamer is doing, live |
| I3 | Glyph-atlas text rendering (when HUD text goes per-frame) | S–M | 64× fewer overlay instances |
| I4 | Headless benchmark mode (`--bench`) with JSON report | S–M | Before/after numbers for every perf item |
| I5 | Golden-image visual regression harness | M | Shader changes stop being "eyeball only" |
| **J — Windows / RTX 5090 readiness** | | | |
| J1 | Backend/present-mode overrides + on-site validation checklist | S | Turns the next 5090 session into data |
| J2 | Draw-call follow-ups gated on 5090 measurements | — | Pointer item (A6 + `SPLIT_MAX` retune) |
| J3 | Windows polish: `BUILD_DATE` without `date`, log path visibility | S | Small correctness on the second platform |

---
## A — Main-thread & streaming performance

The per-frame path (`App::frame` → `lod::select` → `Renderer::render`) is the
sacred hot path (guidelines §4). The items here are ordered by measured or
strongly-evidenced cost, not speculation.

### A1. Cache `surface_radius` per chunk key in the LOD walk — **S**
*[Performance • the largest known main-thread cost]*

**Why.** `select_node` (`lod.rs`) computes, for **every visited node, every
frame**: `let center = center_dir * w.planet.surface_radius(center_dir)` —
and `surface_radius` is a full `height()` evaluation (~1.2 µs of f64 fBm/ridged
noise, measured via the `sampling_cost` bench). At the High preset ~3,800 chunks
draw, so the walk visits ~5,000+ nodes → **roughly 5–7 ms of noise sampling per
frame on the main thread**, growing linearly with `split_factor` (Ultra will be
worse — exactly where the 5090 wants headroom). The `want`-sort's cached keys
(`main.rs`) already fixed the *sort* half of this; the *walk* half remains. This
is invisible on a vsync-bound Mac at Medium but is a hard CPU floor under every
frame.

**What.** Memoize the per-key surface radius. `Planet` is immutable per run and
`ChunkKey` is a stable, hashable identity, so the cache is a pure, deterministic
function table — no invalidation needed for the planet's lifetime.

**How.**
- Add a `radius_cache: HashMap<ChunkKey, f32>` (or `Vec`-backed per-face map)
  owned by `App` (not `Planet` — keeps `Planet` immutable/`Arc`-shareable) and
  pass it into `select` alongside the `ready` closure, or wrap both in a small
  `SelectCtx` struct.
- On miss: compute once, insert. Entries are 24 B; even 200k visited keys is
  ~5 MB. If unbounded growth offends, clear it whenever `apply_rebuild` runs
  (it already clears the chunk cache) — but it's not geometry-config-dependent
  (only seed-dependent), so even that is optional.
- `center_dir()` (two multiplies + normalize) is cheap enough to leave as-is;
  cache only the height.
- **Verify:** perf-log `avg_ms` at High/Ultra before/after (use I4); the walk
  should drop out of the profile entirely. Determinism untouched (pure cache).

### A2. Bucket the draw list by arena block once per frame — **S**
*[Performance]*

**Why.** `record_scene` (`gfx.rs`) draws terrain as: *for each arena block → for
each key in `draw` → `self.chunks.get(key)`, skip unless it lives in this
block*. That is O(blocks × draws) HashMap probes — with ~20–40 resident blocks
(48 MiB each) and ~3,800 draws, that's **~100k+ hash lookups per frame** just to
route draws, plus it defeats any front-to-back ordering.

**What.** One O(draws) pass that groups draw calls by block, then one loop per
block. While there, sort each block's chunks front-to-back (camera distance is
already computable from the key) to get early-Z rejection for free on the
heavy fragment path.

**How.**
- In `render()`/`record_scene`, build `Vec<Vec<(TerrainSlot, ChunkKey)>>`
  indexed by block (reuse a scratch buffer across frames per guidelines §4 — a
  `Vec` kept on `Renderer`, `clear()`ed each frame).
- Single pass over `draw`: one `chunks.get` per key (needed anyway for the LRU
  stamp — fold `stamp_drawn` into this same pass so the map is probed **once**
  per key per frame instead of twice).
- Optional: within each bucket, `sort_unstable_by_key` on the camera-distance
  bits computed from A1's cached radii (cheap; measure whether early-Z pays
  before keeping the sort).
- **Verify:** CPU frame time at Ultra; renderdoc/Metal capture confirms
  identical draw sets.

### A3. Per-frame upload budget with a spillover queue — **S–M**
*[Performance • hitch elimination]*

**Why.** `MAX_REQUESTS_PER_FRAME = 64` caps *requests*, but `Streamer::poll()`
returns **everything finished**, and `App::frame` uploads it all in one frame.
After a teleport (`R`) or a fast zoom, 50+ chunks complete nearly simultaneously;
at Ultra a chunk is ~(97² + 4·97) verts × 36 B + ~56k indices × 4 B ≈ **560 KB**,
so a burst frame can push 20–30 MB through `queue.write_buffer` + allocate that
many veg instance buffers — a classic hitch spike. The `FRAME_HITCH_MS` warn in
the log exists precisely because this happens.

**What.** Budget uploads per frame (bytes, not count — chunk size varies 25× from
Low to Ultra grid), and carry the remainder in a FIFO the next frames drain.

**How.**
- Add `const UPLOAD_BUDGET_BYTES_PER_FRAME: usize = 8 << 20;` (name per the
  no-magic-numbers rule; tune with I4).
- In `frame()`: drain `streamer.poll()` into a `pending_uploads: VecDeque<(ChunkKey,
  CpuChunk)>` on `App`, then pop-and-upload until the byte budget is spent.
  `CpuChunk` already knows its sizes (`vertices.len()`, `indices.len()`,
  `veg.instances.len()`).
- Guard staleness: entries whose key is no longer wanted *and* not in the draw
  set can be dropped instead of uploaded (cheap check against the current
  selection) — otherwise a fast pan uploads geometry that evicts next frame.
- Order the queue nearest-first (same cached-distance key as the `want` sort) so
  the visible hole fills before the periphery.
- **Verify:** teleport at Ultra while watching `target=perf` hitch warns —
  before: multi-hundred-ms spikes; after: none above `FRAME_HITCH_MS`.

### A4. Priority + staleness for the meshing queue — **M**
*[Performance • perceived streaming latency]*

**Why.** The worker queue (`lod.rs::Queue`) is a plain FIFO `VecDeque`. Requests
are pushed nearest-first *within one frame*, but pushes from previous frames sit
ahead of newer, now-more-urgent work; and once queued, a job is **never
cancelled** — pan away and the pool still meshes chunks you'll never see
(generation only bumps on `clear()`, i.e. settings changes). On a slow CPU at
Ultra (23 ms/chunk build — see A7) the queue backlog is the visible "world
sharpens slowly" lag.

**What.** (a) Pop-time priority: workers take the *nearest currently-wanted*
job. (b) Staleness: each frame, the main thread replaces the queue's want-set;
jobs not re-wanted for N frames are dropped.

**How.**
- Replace `jobs: VecDeque<(u32, ChunkKey)>` with `jobs: Vec<(u32, ChunkKey)>` +
  a shared `cam_cell: AtomicU64` (quantized camera direction+altitude packed by
  the main thread each frame). Workers, holding the lock anyway, pop the min by
  a cheap distance proxy (`key.center_dir()` dot the unpacked camera dir at the
  key's level — no `height()` needed for ordering). The queue is ≤ a few hundred
  entries (64/frame cap), so an O(n) scan under the lock is fine; no heap needed.
- Simplest robust staleness: `Streamer::retain(&HashSet<ChunkKey>)` called once
  per frame with the current `want ∪ draw-children` set; drops queued-but-unwanted
  jobs (in-flight ones finish and get dropped by A3's staleness check).
- Keep the generation guard exactly as is — it's correct and battle-tested.
- **Verify:** time-to-sharp after teleport (I4 can measure "frames until
  `pending == 0`"), and worker idle % (no starvation regressions).

### A5. Frustum culling in `lod::select` — **M**
*[Performance • biggest draw-count lever]*

**Why.** The walk culls only against the **horizon cone** (`select_node`'s
`ang > w.horizon + node_ang + HORIZON_CULL_MARGIN`). Everything on the near
hemisphere — including terrain squarely *behind the camera* — is selected,
drawn, and (with A1 fixed) still costs draw submission + vertex work every
frame. With a 60° vertical FOV, the visible wedge at ground level is a small
fraction of the near hemisphere: **2–4× draw-count reduction** is the normal
yield of adding frustum culling to a planet quadtree.

**What.** Test each node's bounding sphere against the view-frustum planes
during the walk; cull *drawing* aggressively but keep *streaming* (`want`)
slightly wider so a quick 180° spin doesn't stare at holes.

**How.**
- Extract the six planes from `view_proj` once per frame (Gribb-Hartmann; add a
  small `Frustum` type in `camera.rs` — it owns the matrices already). Pass it
  in `Walk`.
- Node bound: center = `center_dir * (PLANET_RADIUS + h_cache)` (A1's cached
  radius), radius = `node.world_size() * BOUND_RADIUS_FACTOR +
  HEIGHT_SCALE * MAX_TERRAIN_FACTOR` — a named-const slack covering terrain
  relief and the skirt; verify with an assertion pass in debug that every drawn
  vertex is inside its node bound.
- Apply plane test *after* the horizon test (horizon kills the far side
  cheaply). If a node is fully outside → return (as horizon cull does). Chunks
  that fail frustum but pass horizon within `STREAM_MARGIN_RADIANS` of the
  frustum boundary still go to `want` (rotation warm-up); the margin is the
  tuning knob between memory and pop-in.
- Wireframe/debug consideration: add a debug toggle (I2) to freeze the frustum
  so culling can be inspected from a detached viewpoint — invaluable when tuning.
- **Verify:** `draw` count in the perf log at ground level (expect ~⅓ of
  today's), zero visible pop when spinning with boost at ground level.

### A6. Vegetation instance arena (+ optional multi-draw-indirect) — **M–L**
*[Performance • the remaining half of H2]* *(BACKLOG: "Remaining draw-call headroom")*

**Why.** Terrain now suballocates, but each chunk's vegetation still gets its
**own `create_buffer_init`'d instance buffer** at upload and its own
`set_vertex_buffer(1, …)` at draw — per-chunk buffer churn (the exact pattern H2
flagged) plus one bind per vegetated chunk per frame. On DX12 (heavier
per-submission driver cost — the 5090 report's signature) this is the next
bottleneck after terrain.

**What.** A second fixed-slot arena for instance data. Unlike terrain, veg
instance counts vary per chunk, so use a size-class arena: slots of
`VEG_MAX_ATTEMPTS × size_of::<VegInstance>()` (= 192 × 80 B ≈ 15 KB — the placer
already hard-caps instances per chunk, so **a single slot size fits every chunk
by construction**). One buffer bind per block; each chunk's runs become
`draw_indexed(…, base_instance_range)` into its slot offset… with the caveat
that `first_instance` requires no special feature in wgpu (it's part of
`draw_indexed`'s instance range) — the instance buffer offset is expressed via
the range, so runs stay exactly as they are, just relative to `slot_start`.

**How.**
- Mirror `MeshArena`: `VegArena { slot_instances: u32 = VEG_MAX_ATTEMPTS, … }`,
  blocks sized to the same 48 MiB target (≈3,200 slots/block). `upload_chunk`
  writes instances via `queue.write_buffer` into the slot; `GpuVeg` becomes
  `{ slot: VegSlot, draws: Vec<(u32, u32, u32)> }` with `start` now
  slot-relative.
- Draw loop: bind the veg block's instance buffer once (vertex slot 1), then for
  each chunk in that block, for each run: instance range =
  `slot_base + start .. slot_base + start + count`.
- **Indirect (second stage, DX12/Vulkan only):** once instances live in shared
  buffers, per-run draws can collapse into `multi_draw_indexed_indirect` with
  a CPU-built (later GPU-built) command buffer per frame. Gate on
  `Features::MULTI_DRAW_INDIRECT` (absent on Metal in wgpu — keep the loop path
  for macOS, per the portability guideline). Only do this if 5090 measurements
  (J1) still show submission-bound after the arena; the arena alone may suffice.
- **Verify:** buffer-creation count per frame → 0 in steady state; 5090 FPS at
  Ultra (the real acceptance test); Metal path unchanged visually (I5).

### A7. Chunk-build throughput: octave LOD, lazy veg slope, f32-noise epoch — **M–L**
*[Performance • worker path]*

**Why.** A chunk build is `(grid+1)² × sample_terrain` ≈ 9,409 × 2.42 µs ≈
**23 ms at Ultra grid 96** — one chunk per core per ~23 ms bounds how fast the
world sharpens (A4 reorders the queue; this shrinks it). Three independent
levers, in increasing invasiveness:

1. **Distant chunks compute octaves they cannot express.** `height()` always
   evaluates 7-octave ridged mountains + 5-octave detail; a level-6 chunk's
   quads are ~150 km wide — its top octaves change the surface by metres,
   invisible at that scale. Clamping octave count by quad wavelength
   (`octaves_for(level)`) cuts coarse-chunk cost 2–3×. Cross-LOD height deltas
   from dropped octaves are bounded by the octave amplitude sum (≈
   `DETAIL_AMPLITUDE · persistenceᵏ · HEIGHT_SCALE` — single-digit units),
   far below the skirt depth (3× quad width) that already hides LOD seams —
   but this **changes generated bytes per LOD**, so land it *with* D1's golden
   tests updated and a one-time visual pass.
2. **Vegetation placement pays for slope it usually discards.**
   `sample_blended` (called per placement attempt) always runs `steepness` —
   two extra `height()` probes — before the cheap gates (`lushness`, species
   presence, the coverage dice) that reject most attempts. Reorder: sample
   height + climate first, run the cheap probability gates, and only compute
   steepness for the survivors right before planting (`VEG_MAX_STEEPNESS` check
   moves late). Saves ~⅔ of veg-placement noise cost with **byte-identical
   output** for all planted instances *only if* the RNG draw order is preserved
   — draw all random numbers up front per attempt (they're already a fixed
   sequence: u, v, tj, mj, yaw, age, scale, tint) so gate reordering can't
   shift the stream. Write the equivalence test first.
3. **The noise stack is f64 scalar.** `noise-rs` evaluates in f64; a planet
   renderer needs f32 quality at 4-wide SIMD speeds. A hand-rolled
   f32 fBm/ridged (or a vetted pure-Rust SIMD noise crate — verify determinism
   across x86/ARM: strict IEEE f32 ops only, no FMA-dependent paths) is a
   2–4× worker speedup — but it **re-rolls every world**, so it must ride the
   same "seed epoch" train as D1 (one announced breakage, golden tests
   re-baselined, `SEED_EPOCH` stamped in the log/overlay).

**How (sequencing within the item):** 2 first (pure win), then 1 (bounded
visual delta), then 3 only bundled with D1's epoch cut. **Verify:** chunk
build ms in a worker-side `trace!` + I4's time-to-sharp metric.

---

## B — GPU & rendering pipeline

### B1. Consistent winding + back-face culling for terrain — **S–M**
*[Performance]*

**Why.** Every pipeline sets `cull_mode: None` with the comment "terrain
faces/skirts have mixed winding". The mixed winding comes from the six
`FACES` bases having different handedness (`right × up` points inward on some
faces), not from anything essential — and it forfeits back-face culling on the
heaviest pass. Hills' far slopes and every below-horizon triangle currently
rasterize and only die at depth test.

**What.** Make the grid triangles wind CCW-from-outside on all six faces, then
enable `Face::Back` culling for the terrain fill pipeline. Skirts stay visible
by emitting both windings *for skirt quads only* (they're 4·grid quads vs.
grid² — a ~7 % index increase in exchange for culling the other 93 %).

**How.**
- In `CpuChunk::build`, compute the face's handedness once
  (`(right × up) · base < 0`) and swap two indices per triangle when emitting
  (`[a, d, b] → [a, b, d]`). The outward-normal accumulation logic is already
  winding-agnostic (it flips by `dot(dirs)`), so nothing else changes.
- Duplicate skirt quads with both windings (or leave skirts in a second, uncull
  draw — but that's a second pipeline bind; double-emitting is simpler).
- Keep `cull_mode: None` for the wireframe pipeline (`G` toggle) and vegetation
  (leaf cards are intentionally double-sided).
- **Verify:** the gallery/smoke PNGs (I5 goldens) byte-match except where
  culling legitimately changes nothing visible; GPU frame time at grazing
  mountain views drops measurably on the 5090.

### B2. Reversed-Z depth with infinite far — **S–M**
*[Correctness margin + simplification]*

**Why.** Depth is `Depth32Float`, `CompareFunction::Less`, cleared to 1.0 — the
classic configuration that wastes float depth precision exactly where a planet
renderer is poorest. Today's per-frame `near_far` horizon derivation is a good
mitigation, but the measured far:near ratio still hits ~10⁵:1 near the ground
(near = `distance × 0.25` ≥ 0.1, far = horizon + mountain slack ≈ 50k units at
eye height 15 m). Reversed-Z on a float buffer makes depth precision effectively
uniform in log-distance, eliminating the whole class of z-artifacts — and it
frees the far plane entirely, deleting tuning constants rather than adding them.

**What.** Standard reversed-Z: clear depth to **0.0**, `Opaque` compares
**`Greater`**, projection becomes `Mat4::perspective_infinite_reverse_rh(fov_y,
aspect, near)`.

**How.**
- `gfx::make_pipeline`: `PassKind::Opaque → (true, Greater)`; sky/overlay stay
  `Always`/no-write. Depth clear in `record_scene` and the smoke tests → 0.0.
- `camera::view_proj`: swap in the infinite-reverse projection; keep `near_far`'s
  **near** logic; the far value survives only if something still wants it (fog
  doesn't — it's density-based; the horizon math stays for A5's stream margin).
- The sky shader reconstructs rays via `inv_view_proj` at NDC z = 1 — with
  reversed-Z, the far plane is at **z = 0**; update `sky.wgsl`'s ray
  reconstruction (`vec4(ndc, 1.0, 1.0)` → `vec4(ndc, 1e-7, 1.0)`-style or
  reconstruct from the near plane and negate) and the fullscreen triangle's
  emitted depth (`out.clip = vec4(p, 0.0, 1.0)` so `Always` still passes with
  depth-write off — cosmetic but keep it consistent).
- Audit both smoke tests (they build their own projections) — this is exactly
  the kind of cross-cutting change the offscreen harness (E1) should make
  one-line.
- **Verify:** I5 goldens; a deliberate stress scene (tilted view at `MIN_DIST`
  over mountains at max zoom-out) shows no z-shimmer before/after on both
  backends.

### B3. MSAA + alpha-to-coverage for vegetation — **M**
*[Polish with a performance budget • preset-gated]* *(BACKLOG: "MSAA / FXAA" under the flora photoreal item)*

**Why.** Everything renders 1×-sampled; thin trunks, leaf-card edges, and the
terrain's high-frequency vertex color all shimmer in motion. Worse, vegetation
cutouts use a hard `discard` at `ALPHA_CUTOFF` — the sharpest-aliasing way to
draw foliage. MSAA 4× + alpha-to-coverage is the standard fix and the single
biggest image-quality jump available to this renderer; the 5090 will not notice
the cost, and even Apple Silicon handles 4× at 1440p comfortably. This is also
the prerequisite that makes the backlog's leaf-card work look good.

**What.** A `Multisample` graphics setting (Off / 4×; preset-gated: Off at Low,
4× at High+). Scene renders into a transient MSAA color + depth target and
resolves into the swapchain/offscreen texture. The vegetation pipeline enables
`alpha_to_coverage` when sampled, replacing the hard discard with
coverage-proportional alpha (keep the `discard` path for 1×).

**How.**
- `Renderer`: create `msaa_color`/`msaa_depth` textures at `sample_count`,
  render pass uses them with `resolve_target: Some(&swapchain_view)`. All
  pipelines take `MultisampleState { count, alpha_to_coverage_enabled }` —
  pipelines must be **rebuilt when the setting changes** (mirror how
  `apply_rebuild` handles geometry settings; sample count is a "live-ish"
  setting that rebuilds pipelines, not chunks).
- `vegetation.wgsl`: when A2C is on, output `tex.a` (optionally sharpened:
  `(tex.a - ALPHA_CUTOFF) / max(fwidth(tex.a), 1e-4) + 0.5`) instead of
  discarding, so coverage dithers the edge.
- Offscreen/video path gets the same option (`--video` at 4× is nearly free
  since it's not real-time).
- Memory: 1440p 4× ≈ 2560×1440×(8 B color + 4 B depth)×4 ≈ **170 MB** — name it
  in the settings row so the budget is honest (the settings UI already displays
  real units; add the MB figure to the row value).
- **Verify:** I5 goldens at both sample counts; eyeball pass on the Mac; fps
  delta recorded via I4.

### B4. Vertex & instance compression — **M**
*[Performance (memory/bandwidth)]*

**Why.** `Vertex` is 36 B (three `f32x3`s) and `VegInstance` is 80 B (a full
`Mat4` + tint). At High the drawn set alone is ~900 MB of terrain geometry;
upload bandwidth (A3) and the memory budget both scale with it. Normals and
colors don't need 12 B each; a veg instance is fully described by
position + yaw + scale + tint.

**What.**
- Terrain vertex → **20 B**: `pos: [f32; 3]` (12 B — full precision until C2),
  `normal: unorm/snorm 10-10-10-2 or 2×i16 octahedral` (4 B), `color:
  Unorm8x4` (4 B). wgpu vertex formats decode these for free
  (`VertexFormat::Snorm16x2`, `Unorm8x4`); WGSL sees the same `vec3<f32>`s.
- Terrain indices → **u16**: max grid 96 ⇒ 97² + 4·97 = 9,797 verts < 65,536.
  Halves index memory; `IndexFormat::Uint16`. (Static assert `GRID_MAX` keeps
  this true — a compile-time guard, not a runtime branch.)
- `VegInstance` → **32 B**: `pos: [f32; 3]`, `yaw+scale: [f16; 2]` or
  `[u16; 2]` quantized, `tint: Unorm8x4`, reconstructing the model matrix in
  the vertex shader from `upright_rotation(normalize(pos), yaw) * scale` — the
  shader already has geocentric up = `normalize(world)`; the rotation
  reconstruction is ~10 ALU ops amortized over 10–22k-vert meshes.
- Net: terrain ≈ **45 % smaller**, veg instances **60 % smaller**, uploads
  proportionally cheaper; the memory-budget slider effectively grows for free.

**How.** Change `mesh::Vertex`/`VegInstance` + the two `vertex_attr_array!`s +
tiny WGSL input adjustments; the octahedral encode is ~15 lines in `mesh.rs`
(with a unit test round-tripping normals to <1° error). Land after B1 (winding
work touches the same builder) and before C2 (which re-bases `pos`).
**Verify:** I5 goldens (color quantization is sub-perceptual at 8 bits; normals
at 1° error are invisible in diffuse lighting), `resident_bytes` at identical
scenes drops ~40 %.

### B5. Vegetation mesh LOD → impostors — **L–XL**
*[Performance • the far-field plant budget]* *(BACKLOG: "vegetation LOD impostors")*

**Why.** Archetype meshes run **10–22k vertices** and are drawn at that full
cost from arm's length to the horizon — the `Plant distance` slider is currently
a *count* knob, not a *cost-per-plant* knob. At Ultra (veg to level 11 ≈ tens of
km) the vertex load is dominated by trees that occupy 4 pixels. Every serious
foliage renderer converges on the same ladder: full mesh → reduced mesh →
billboard impostor.

**What.** Three rungs, keyed by instance distance (computable on CPU at draw
grouping time since chunks already know their distance):
1. **LOD1 mesh** (~25 % triangles): generated offline per archetype during F1's
   asset bake (quadric decimation; the bake pipeline already exists in
   `flora-revamp/` — add a decimation pass), stored alongside LOD0 in the same
   concatenated buffer with its own `MeshRange`.
2. **Impostor**: per archetype, an 8-view billboard atlas rendered at bake time
   from the real model (reuse the offscreen harness), drawn as camera-facing
   quads (2 tris) with normal-from-atlas for lighting; one extra texture array.
3. Per-chunk selection: a chunk's veg draws pick LOD by `chunk_distance /
   plant_view_units` thresholds (named consts `VEG_LOD1_FRACTION`,
   `VEG_IMPOSTOR_FRACTION`); no per-instance branching needed at first —
   per-chunk granularity is visually fine because chunk size scales with
   distance by construction.

**How.** Stage 1+3 first (mesh LOD is pure win, no new art path); impostors
after F1 exists to host baked atlases. Cross-fade at thresholds via a dither
pattern in the fragment shader (`discard` on a screen-space hash below blend
factor) to hide the swap. **Verify:** vertex-count counter in I2's HUD; Ultra
far-field GPU time on the 5090; I5 goldens at each rung.

---

## C — Precision & scale

### C1. Fix the water ripple phase (camera-relative) — **S**
*[Correctness of the f32 scale model • closes review M6]*

**Why.** `terrain.wgsl` still animates the ocean glint with
`sin(p.x * 0.7 + t * 1.4)` where `p = in.world` reaches ~637,000. f32 sine
argument reduction at that magnitude quantizes phase to steps larger than the
wavelength — the guideline's own §8 names this exact pattern as the thing to
avoid, and the backlog's wind-sway item repeats the warning. The glint likely
bands/aliases subtly today and will get worse the moment wave frequency rises
(G5).

**What.** Phase off a small-magnitude coordinate. The camera-relative vector is
already computed in the fragment (`v` uses `g.camera_pos.xyz - in.world`);
ripples matter only within a few km of the camera (fog + specular falloff),
where `rel = in.world - g.camera_pos.xyz` has comfortable f32 precision.

**How.** In the `is_water` branch: `let rel = in.world - g.camera_pos.xyz;`
then feed `rel.x/rel.z` into the existing sin/cos mix (constants unchanged —
the ripple field translates with the camera, which is imperceptible for
sub-metre glint noise; if the sliding ever shows at high zoom, switch to
`fract(world / RIPPLE_TILE_UNITS) * RIPPLE_TILE_UNITS` per-axis instead, which
stays world-anchored at small magnitude). One shader edit; verify against I5's
water golden and by eye on a sun-glint coastline.

### C2. Camera-relative rendering; raise `MAX_LEVEL` past 16 — **XL**
*[The next scale frontier • prerequisite for true ground detail]*

**Why.** The 10 m-unit f32 model is the project's cleverest trade, and it is
also a hard ceiling: `MAX_LEVEL = 16` (~8 m quads, ~0.75 m positional
quantization at the surface) is as fine as absolute f32 coordinates allow —
CLAUDE.md already declares finer detail "a deliberate future step, not a
constant bump." Walk mode (H1), close-up flora, and sub-metre terrain all press
against this ceiling. The standard planetary-renderer answer is camera-relative
(a.k.a. floating-origin) rendering: keep world math in units, but express
GPU-visible positions relative to a nearby origin so f32 only ever sees small
numbers.

**What.** Per-chunk local coordinates + camera-relative transform:
- CPU: each chunk stores vertices **relative to its center** (`f64` center
  computed at build; verts become small f32s — this also makes them compressible
  to f16/snorm later, compounding B4).
- Per frame: for each drawn chunk, compute `offset = chunk_center - eye` in
  **f64 on CPU** (the only f64 in the app; two Vec3s per chunk per frame), cast
  to f32, and supply it per draw. The view matrix loses its translation
  (`look_to_rh` from origin); the vertex shader does
  `clip = view_proj_rot * vec4(v.pos + chunk_offset, 1)`.
- With absolute magnitudes gone from the vertex path, `MAX_LEVEL` can rise to
  18–20 (~2 m → ~0.5 m quads) wherever the split heuristic asks for it.

**How (staged, each stage shippable):**
1. **Plumb per-draw data.** The `MeshArena` draw loop needs a per-chunk
   `vec4 offset`. Options: (a) a dynamic-offset uniform buffer indexed per draw;
   (b) a second instance-rate vertex buffer (one `vec4` per chunk, `step_mode:
   Instance`, `draw_indexed(…, k..k+1)` selects it) — (b) is simpler in wgpu and
   free on all backends; (c) push constants/immediates (`immediate_size` is
   already in the pipeline layouts, currently 0) — smallest change but check
   size limits per backend. Recommend (b).
2. **Re-base the mesher.** `CpuChunk::build` subtracts `center = center_dir *
   surface_radius(center_dir)` (f64 accumulate, f32 store); `ChunkKey` gains a
   `center_f64()`; skirts/normals unchanged (they're differential).
3. **Re-base consumers.** Vegetation instances become chunk-relative too
   (their `pos` is currently absolute world — make it relative to the same
   center so one offset serves both pipelines); `sky.wgsl`'s ray reconstruction
   and the water `length(in.world)` sea-level test switch to
   `camera_pos`-anchored forms (pass `eye_radius` in `Globals` and use
   `length(rel + eye)`-free formulations — the water flag moves to a vertex
   attribute in G5 anyway, decoupling it from magnitude).
4. **Raise the ceiling.** Bump `MAX_LEVEL` behind a preset-gated cap
   (`LOD_MAX_LEVEL` in `settings.rs`, Ultra-only at first), retune
   `SKIRT_MIN_DEPTH` (2 units = 20 m is huge at level 20) to scale with quad
   size, and re-check `STEEPNESS_EPS` (fixed 9.6 km probe stays deliberately
   LOD-independent — no change, but document it).
- **Order:** after B2/B4 (both touch the same structs; do the churn once),
  before H1 (walk mode wants the fidelity).
- **Verify:** goldens must be pixel-identical at current levels (the transform
  is mathematically the same); a new stress test teleports to `MIN_DIST` and
  asserts vertex jitter under camera micro-motion stays sub-pixel (render two
  1-frame-apart images at a fixed camera, diff — jitter is the artifact this
  work eliminates).

### C3. True altitude-above-ground for HUD/fog/log — **S**
*[Correctness/UX • closes review L7]*

**Why.** `Camera::altitude()` returns `self.distance` — the focus distance, not
height above terrain. At `MAX_TILT` (~74.5°) the HUD/`P` printout/perf log
overstate altitude by ~4×; fog density (`fog_density()` keys off the same
value) thins when tilting even though the eye hasn't climbed. For an app that
displays real units everywhere, the flagship number is wrong whenever the view
is scenic.

**What/How.** Compute eye AGL: `eye.length() - planet.surface_radius(eye_dir)`
— one `height()` per frame (A1's cache makes repeated queries near-free, but
even uncached it's 1.2 µs). Keep `distance` for `near_far` (that derivation is
about the focus geometry, correctly) and give fog its own input decision:
AGL-based fog reads better (ground haze while tilted low) — retune
`FOG_FADE_DISTANCE` once, by eye. Update the HUD title, `print_location`, perf
log, and the video heartbeat, all of which funnel through
`units::distance(camera.altitude(), …)` already — the call sites don't change.

---

## D — Correctness & robustness debt

### D1. Pin the generation RNG + golden-value tests (the "seed epoch") — **M**
*[Correctness • the tree's biggest latent landmine]* *(BACKLOG: "World generation uses a non-reproducible RNG" — promoted)*

**Why.** The backlog's analysis is exactly right and worth acting on **before**
seeds start being saved/shared (bookmarks H2 and the video pipeline both raise
the stakes): `StdRng` is an explicitly-unstable alias — a routine `cargo update`
of `rand` may silently re-roll **every world**: flora cluster radii
(`flora.rs::generate`), every plant placement (`mesh::place_vegetation`), and
the video soundtrack order (`audio::shuffled_soundtrack`). The in-process
determinism tests cannot catch it, by construction. Terrain itself is safe
(`noise-rs` + splitmix64), but "the forests moved" is still "my planet
changed."

**What.** (a) Replace `StdRng` in all *generation* paths with an explicitly
pinned stream. (b) Add golden-value tests that fail loudly if any dependency or
refactor shifts generated bytes. (c) Establish the **seed-epoch discipline**:
any future intentional break (A7's f32 noise, G4's placement redesign) bumps a
`SEED_EPOCH: u32` logged at startup and shown on the overlay, with goldens
re-baselined in the same commit.

**How.**
- Mechanical swap: `rand_chacha::ChaCha8Rng::seed_from_u64(…)` (pinned minor
  version; ChaCha8 is ~as fast as StdRng's chacha12 for this volume), or —
  fully dependency-free — a 20-line xoshiro/PCG in `planet.rs` next to
  `splitmix64` (the repo already hand-rolls hashes; this matches house style
  and drops `rand` from the deterministic path entirely). Keep `rand` for
  teleport/tour/live-audio shuffle, which are *deliberately* nondeterministic.
- **This swap itself re-rolls placements once** — that is the epoch-0 cut.
  Announce it in the commit message; nothing user-visible depends on stability
  yet, which is why now is the moment.
- Goldens (new `tests/goldens.rs` or in `tests.rs`): assert exact
  `to_bits()` values for (1) `Planet::height` at ~8 fixed dirs on a fixed seed,
  (2) the first `VegInstance` (model matrix bits + tint) of a fixed chunk,
  (3) `Flora::generate(7)`'s per-species `cluster_radius` bits, (4) the
  `shuffled_soundtrack(7)` order. ~40 lines, permanent tripwire.
- **Verify:** flip `rand` versions locally and watch the goldens catch it.

### D2. GPU device-loss recovery — **M**
*[Robustness • Windows-critical]*

**Why.** `render()` handles *surface* loss (reconfigure) but not **device**
loss. On Windows/DX12, TDRs (driver timeout-and-reset) are a fact of life —
a two-second driver stall (alt-tab under load, another app's GPU hang, an
overclock burp on an enthusiast 5090 box) invalidates the device, and today the
app would wedge into a black window with a flooding error stream: exactly the
"worse than a crash" failure mode the guidelines call out for workers. wgpu
surfaces this via the device-lost callback and uncaptured-error callback.

**What.** Detect device loss, log it (`ERROR`, with reason), and rebuild the
renderer in place: the design already makes this cheap — `Planet`/`Flora` are
untouched CPU state, the streamer keeps producing `CpuChunk`s, and
`App::resumed` shows the full recipe (recreate `Renderer`, re-upload roots,
world re-streams within a second or two).

**How.**
- Register `device.set_device_lost_callback` (sets an `AtomicBool` on `App` via
  a clone — the callback context is 'static) and `device.on_uncaptured_error`
  (route to `tracing::error!` instead of the default panic-ish stderr spew —
  this alone improves field diagnostics).
- In `frame()`: if the flag is set, drop `self.renderer`, rebuild via the same
  code `resumed` uses (extract `fn build_renderer(&mut self, window)` during
  E1's split), re-upload roots, `streamer.clear()` to flush stale generations,
  log `INFO "renderer rebuilt after device loss"`. Rate-limit: two losses
  within N seconds → give up with a clear terminal/log message (a genuinely
  dead driver shouldn't spin).
- **Verify:** on Windows, force a TDR (registry `TdrDelay` test or
  `dxcap -forcetdr`) and watch the app survive; on macOS this path is
  effectively dead code but compiles/cfg-cleanly (no cfg needed — wgpu API is
  uniform).

### D3. Size-capped log rotation — **S**
*[Robustness of the diagnostic path]*

**Why.** The log is one file, **append-forever across restarts** — the design
is right for a workbench artifact, but it grows without bound (perf lines alone
are ~1 KB/2 s ≈ 1.7 MB/hour/session), and the fact that the current file has
evidently been hand-deleted shows the pressure is real. An unbounded file in a
shared world-writable dir is also the least graceful thing to fill a disk with.

**What/How.** At `logging::init`, before opening: if the file exceeds
`LOG_ROTATE_BYTES` (e.g. 50 MB), rename it to `planet-explorer.log.1`
(clobbering any previous `.1`) and start fresh — two generations, ~100 MB
worst case, zero runtime cost (no size checks while running; a session that
alone exceeds the cap is fine — rotation happens at next start). Keep the
world-rw chmod on both files. ~15 lines; test with a pre-seeded oversized temp
file + `$PLANET_LOG`.

---

## E — Structure & process

### E1. Split `gfx.rs`/`main.rs`; one shared headless harness — **M**
*[Maintainability • closes review M5 + L3]*

**Why.** `gfx.rs` is 1,943 lines (≈940 of them the `smoke`/`gallery` test
modules — each of which builds its *own* device/pipeline stack, ~150 duplicated
lines × 3 with `render_and_save` and `headless_renderer_draws_a_frame`);
`main.rs` is 1,048 (the `--video` runner is ~330 of it). The guidelines set a
<500-line norm and every upcoming renderer item (B1–B5, C2, G-series) pays the
navigation tax repeatedly. This is the enabling refactor for the whole B/G
program — do it early, not as cleanup.

**What/How.**
- `src/gfx.rs` → `src/gfx/mod.rs` (renderer proper, ~1,000 lines today,
  shrinking after extraction), `src/gfx/arena.rs` (`MeshArena` + slot tests),
  `src/gfx/veg.rs` (`VegGpu` + texture upload), `src/gfx/harness.rs`
  (`#[cfg(test)] pub(crate) struct TestGpu` — device+queue+pipelines+offscreen
  target+readback-to-png in one constructor), `src/gfx/smoke.rs` /
  `src/gfx/gallery.rs` rewritten on top of `TestGpu` (net deletion of ~300
  lines).
- `main.rs`: move `run_video`/`capture_frame`/`VideoOptions`/`parse_video`/
  `fade_to_black` into `video.rs` (it already owns the encoder; the runner
  belongs with it — `main.rs` keeps only the dispatch), and extract
  `App::frame`'s phases into named methods (`advance_sim`, `stream_chunks`,
  `submit_frame`, `sample_perf`) so the frame reads as a table of contents.
- Zero behavior change; land as one mechanical PR verified by the full suite +
  goldens. (ct's `move-symbol`/`move-lines` make this near-mechanical.)

### E2. Single `build_globals()` for live + video paths — **S**
*[Maintainability • drift hazard]*

**Why.** The 15-line `Globals` assembly (view_proj, inverse, camera_pos+time,
sun+ambient, params, atmosphere+star seed) exists **twice**, character-for-
character, in `App::frame` and `capture_frame`. Every uniform added by G1/G2/
G5/G6 must be added in both or the video mode silently diverges from the live
app — the exact class of bug that wastes a remote-debugging session.

**What/How.** `fn build_globals(camera: &Camera, planet: &Planet, seed: u64,
time: f32) -> Globals` in `gfx.rs` (it owns the struct) or a `Globals::build`
constructor; both call sites shrink to one line. Do it before any G item adds a
uniform. The smoke tests' hand-built `Globals` stay hand-built (they exercise
specific values) — but route the two *scene* tests through it too where the
values aren't the point.

### E3. Refresh `ARCHITECTURE_REVIEW.md` — **S**

**Why/What.** The review is dated 2026-06-08, pre-dating the flora model
revamp, the video recorder, the arena, and the LRU evictor; half its findings
are fixed (§1.2's ledger). Either stamp a "post-review status" block on it
(the H1 entry shows the house pattern) referencing this roadmap's ledger, or
schedule a fresh full review after Phase 1 lands (see Sequencing). Cheap,
prevents the
docs from training future contributors (and future Claude sessions) on stale
facts.

### E4. CI: fmt + clippy + tests, three OSes, cargo-deny — **M**
*[Process • the Windows target's early-warning system]*

**Why.** There is no CI. The suite is *designed* for it — GPU tests skip
cleanly without an adapter — and the project has a second platform it can't
compile-check locally (macOS host can't `cargo check` MSVC/DX12-specific
fallout of a dependency bump). The wgpu 29 / winit 0.30 / rodio 0.22 stack
moves fast; today the first machine to notice a Windows breakage is the 5090
box mid-demo.

**What/How.**
- If the repo has (or gets) a GitHub remote: `.github/workflows/ci.yml` —
  matrix `{macos-latest, windows-latest, ubuntu-latest}` × stable; steps:
  `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
  `cargo test` (debug is fine in CI; the release suite runs locally), plus a
  `cargo-deny` job (advisories, licenses, duplicate-version hygiene) and
  `cargo build --release` on Windows only (link check with the 242 MB embed —
  budget cache accordingly; after F1 this gets cheap).
- If the repo stays remote-less: the same commands in a `just ci` recipe (E6) +
  a `.git/hooks/pre-push` sample checked in as `hooks/pre-push` — the value is
  the *ritual*, the runner is secondary.
- Caveat named openly: CI cannot run the GPU smoke tests (no adapter on hosted
  runners) — the local `cargo test --release` before packaging remains the
  GPU gate; I5's goldens run wherever an adapter exists.

### E5. `rust-toolchain.toml` + `rustfmt.toml` — **S**

**Why.** Edition 2024 requires rustc ≥ 1.85; the Mac and the Windows box are
two independently-updated toolchains, and "works here, ICEs there" is a real
failure mode for a fast-moving edition. Formatting is currently hand-maintained
(consistently ~110–120 columns) with no enforcement — one wrong-IDE session
churns diffs forever.

**What/How.** `rust-toolchain.toml` pinning `channel = "stable"` with a
recorded minimum (or a specific version, bumped deliberately);
`rustfmt.toml` with `max_width = 120` (matching current style so the initial
`cargo fmt` is a no-op-sized diff — verify before committing it) and
`newline_style = "Unix"` (protects the cross-platform diff hygiene). Wire both
into E4.

### E6. `justfile` encoding the workflows — **S**

**Why.** The project has load-bearing rituals that live only in docs and habit:
*every build must end with `./package_macos.sh`* (CLAUDE.md, bolded), video
renders have a five-flag incantation, the bench/golden flows (I4/I5) will add
more. Rituals in prose regress; rituals in a runner don't.

**What/How.** A `justfile` (cross-platform, single static binary) with:
`just ship` (clippy → test → package_macos.sh), `just ship-win` (the PS1),
`just video [seed]`, `just bench` (I4), `just goldens` (I5 re-baseline),
`just ci` (E4's local mirror). Document in README; CLAUDE.md's rule becomes
"use `just ship`".

---

## F — Assets, startup & distribution

### F1. Build-time asset pack (preprocessed models/textures) — **L**
*[Performance (startup) + distribution + build ergonomics — the biggest
quality-of-life item in the roadmap]*

**Why.** Three compounding costs, one root cause — the app embeds **raw source
assets** and does all processing at runtime, every launch:
1. **Binary size:** 242 MB of GLBs (plus ~34 MB of mp3s) ride `include_bytes!`
   into a ~¼ GB executable; every link re-processes it, every copy to
   `/Users/Shared` moves it.
2. **Startup:** `Flora::generate` spends ~0.7 s (measured) parsing GLBs,
   PNG-decoding **full-resolution** textures, bilinear-resampling them down to
   512², and box-filtering mip chains — on the main thread, before the window
   exists. The punchline: *most of those decoded bytes are thrown away* (the
   512² cap discards the source resolution).
3. **Inflexibility:** because resampling happens at runtime from oversized
   sources, offering a 1024² texture tier for the 5090 (the `VEG_TEX_SIZE` doc
   comment already anticipates it) would make startup *slower still*.

**What.** A deterministic bake step that converts `assets/models/*.glb` into a
single compact pack containing exactly what the engine uploads: normalized,
(optionally quantized) vertex/index buffers per archetype + the texture array
layers **pre-resampled with pre-built mips** (stored zstd-compressed;
optionally BC7 later for VRAM — a separate decision, keep the pack format
versioned). The pack is embedded via `include_bytes!` — the single-binary
property (a hard project requirement) is preserved; only the *shape* of the
embedded data changes. Expected: binary ~260 MB → **~30–50 MB**; flora startup
~0.7 s → **milliseconds** (one zstd decode + memcpy).

**How.**
- **Where the bake runs:** a small `tools/bake-assets` bin crate (workspace
  member) run manually via `just bake-assets`, committing the pack
  (`assets/flora.pack`) — *not* `build.rs* (a build.rs bake would re-parse
  242 MB on clean builds and drag the GLBs into every checkout forever;
  a committed pack lets the raw GLBs eventually leave the repo — F3).
- **Format:** dead-simple versioned binary — header (magic, version,
  `VEG_TEX_SIZE`, counts) + per-archetype mesh blobs + per-layer mip blobs,
  all length-prefixed, zstd-compressed as one stream (`zstd` crate, pure-Rust
  `ruzstd` for decode if dependency weight matters). No serde needed; the
  loader is ~100 lines mirroring `models::load`'s output types.
- `models::load` becomes the *bake-side* code path (moved into the tool);
  the engine gains `models::load_pack(bytes) -> Library`. Keep a
  `--features bake-from-glb` dev path only if GLB iteration continues;
  otherwise the tool is the iteration path.
- **Determinism:** the pack is a checked-in artifact — bit-stable by
  definition; the bake tool prints a content hash the pack embeds and the app
  logs at startup (ties a log to exact assets, like `GIT_HASH` does for code).
- **Verify:** `flora::tests` run against the pack (same assertions);
  I5 goldens unchanged; startup timestamp delta in the log
  (`flora library loaded` line already logs counts — add elapsed ms).

### F2. Parallel/overlapped startup — **S**
*[Performance (perceived) • independent of F1, worth doing first]*

**Why.** Launch order is strictly serial on the main thread: `Planet::new`
(flora ~0.7 s) → *then* the event loop starts → `resumed` builds the renderer,
uploads six root chunks, then the window becomes live. Finder-launched users
stare at nothing for over a second. Even after F1 shrinks the flora cost, the
root-chunk builds (6 × grid² samples) and renderer init remain serial.

**What/How.**
- Spawn `Flora::generate`/pack-load on a thread **before** creating the event
  loop (`std::thread::spawn` returning `JoinHandle<Flora>`), join it inside
  `resumed` right where the renderer needs meshes — window appears immediately,
  flora joins ~free.
- Decode the 51 models across threads inside `Flora::generate`
  (`std::thread::scope`, chunk the archetype list by core count — each model is
  independent; the layer *ordering* must stay load-order-deterministic, so
  assign layer indices before parallel decode and write into pre-sized slots).
  This alone cuts the 0.7 s by ~the core count if F1 is deferred.
- Build the six root chunks on the streamer pool instead of inline (they're
  the sanctioned inline exception today — keep one inline *coarsest* fallback
  frame if a truly blank first frame ever shows; in practice the six builds at
  default grid are ~10 ms each and can stay inline if measurements say they're
  invisible).
- **Verify:** time from process start to first `RedrawRequested` logged at
  INFO; target < 300 ms on the Mac.

### F3. Repo hygiene for `flora-revamp/` — **S**

**Why.** `flora-revamp/` carries **421 MB** of reference plates and
intermediate bakes in the working tree (and history), on top of the 242 MB of
shipped models — a 700 MB clone for an 8 kLOC app. This is tooling exhaust, not
source: the plates regenerate from `generate_images.py` + prompts
(`FLORA_MODEL_TARGETS.md` documents exactly this), and the intermediate models
are superseded by `assets/models/`.

**What/How.** Decide the retention policy explicitly: (a) move
`flora-revamp/images` + `flora-revamp/models` to an archive branch or a
sidecar location outside the repo, keeping the docs + scripts (the
reproducibility recipe) in-tree; or (b) adopt `git-lfs` for `*.png`/`*.glb`
under `flora-revamp/` if history matters. Pair with F1 so the *shipped* GLBs
can eventually follow the same route (pack committed, sources archived).
History rewriting is optional and separate — even just stopping the growth
going forward is most of the value.

---

## G — Visual richness

Ordered so each item compounds the previous; G1–G3 are the "one good weekend
each" tier, G4–G7 are the investments.

### G1. Day/night terminator — **S–M**
*[Polish • highest impact per line]* *(BACKLOG: High — absorbed; engineering detail added)*

**Why.** The planet has no night side: `SUN_AMBIENT = 0.32` applies uniformly,
so the hemisphere facing away from the sun renders at 32 % brightness — from
orbit the globe reads as a uniformly-lit ball, the single most "renderer-ish"
tell in the whole image. The fix is shader-only, a few ALU per fragment, no new
memory, and it multiplies the value of G2/G6/H5.

**What.** Ambient becomes a function of sun elevation at the *surface point*:
`day = smoothstep(-TERMINATOR_SOFT, TERMINATOR_SOFT, dot(up, sun))` where
`up = normalize(in.world)`; light = `mix(NIGHT_FLOOR, 1.0, day) * (amb + diff·…)`
with `diff` also gated by `day` (no direct sun below the horizon), plus a warm
tint pulse in the band (`mix` toward an `SUNSET_TINT` when `|dot| <
TERMINATOR_SOFT`) and a faint cool "moonlight" floor so the night side stays
navigable (`NIGHT_FLOOR ≈ 0.05–0.08`, tinted toward `atmosphere.rgb`).

**How.**
- Same ~10 lines in `terrain.wgsl` and `vegetation.wgsl` (E2 note: no new
  uniforms needed — `sun_dir` and world position are already in both). Fog
  color should also darken with `day` (fog currently glows atmosphere-bright at
  night — `fog_color * mix(NIGHT_FOG_FLOOR, 1.0, day)`).
- `sky.wgsl`: the rim glow already scales with `sun_face`; add the same
  `day`-darkening to the rim so the dark limb doesn't halo, and consider
  raising star brightness slightly when the *camera-under-point* is on the
  night side (`Globals` has camera_pos; compute in-shader).
- Named consts per house rule: `TERMINATOR_SOFT` (≈0.12 — a few hundred km
  band), `NIGHT_FLOOR`, `SUNSET_TINT`, `NIGHT_FOG_FLOOR`.
- Tour interaction: the cruise can land on the night side. That's arguably
  atmospheric; if it bothers in practice, bias `pick_destination` to accept
  night-side candidates with lower probability (uses `rand` — the
  intentionally-nondeterministic domain, no epoch concerns).
- **Verify:** I5 goldens for a terminator-crossing viewpoint + the finale
  pullback in a `--video` render (the space view is where this lands hardest).

### G2. Sun disc + seeded moon — **S–M**
*[Polish]*

**Why.** The sky has stars and a rim glow but **no visible sun** — lighting
comes from nowhere — and nothing else in the sky varies per world beyond tint.
A sun disc is ~15 shader lines; a moon gives every seed a second identity
object and sets up eclipse/phase play for free.

**What/How.**
- `sky.wgsl`: `sun_amount = dot(dir, normalize(sun_dir))`; disc =
  `smoothstep(cos(SUN_ANGULAR_RADIUS + SUN_LIMB_SOFT), cos(SUN_ANGULAR_RADIUS),
  sun_amount)` in a warm white scaled above 1.0 (it should bloom-clip), plus a
  broad forward-scatter halo (`pow(max(sun_amount,0), SUN_HALO_POWER) *
  SUN_HALO_STRENGTH * atmosphere.rgb`). Occlusion falls out free: the sky pass
  draws first and terrain covers it — the sun sets *behind the planet*
  geometrically.
- Moon: derive `moon_dir` (slow great-circle orbit if H5 lands; static
  otherwise) and `moon_scale` from the seed in `Planet::new`; pass via a spare
  `Globals` slot (`params.w` frees up after C3, or grow `Globals` — it's one
  vec4). Render as a disc with `hash`-based crater mottling and a
  phase term `max(dot(moon_normal_approx, sun_dir), MOON_EARTHSHINE)` — a
  lit-sphere approximation inside the disc (classic single-quad moon shading,
  ~20 lines).
- **Verify:** goldens; check the disc size against the fog/atmosphere at the
  horizon (sun near the limb should redden via G6 when that lands — fine
  standalone first).

### G3. Flora look: climate tint, per-instance variation, wind sway — **M**
*[Polish]* *(BACKLOG: High "Flora look overhaul" + Medium "Wind sway" — absorbed)*

**Why.** The backlog's diagnosis stands: stands read as a clone army (same hue,
same uprightness) and hold statue-still. All three fixes ride the
**already-emitted** per-instance data — no memory growth.

**What/How.**
- **Climate-driven tint (CPU-side, zero shader cost):** `place_vegetation`
  already samples `sample_blended` — but climate (temp/moisture) is internal to
  it. Expose it (add `temp`/`moisture` to `Surface`, or a
  `sample_blended_full` — `Surface` is `Copy`, 2 more f32s is fine) and derive
  the tint from a per-archetype ramp: hue shift toward yellow-brown as
  moisture ↓, value ↓ as temp ↓ (matching how `blended_color` already blends
  ground colors — reuse those constants' spirit: `VEG_TINT_DRY`,
  `VEG_TINT_COLD` ramps). The existing random `VEG_TINT_JITTER` stays on top.
  Stands then read cohesive *and* place-appropriate: olive grassland acacias,
  deep-green rainforest, sage-grey dry shrubs.
- **Per-instance lean + non-uniform scale:** replace
  `Vec3::splat(scale)` with `vec3(scale·wx, scale·h, scale·wz)` (wx/wz jitter
  ±`VEG_WIDTH_JITTER` ≈ 8 %) and compose a small random lean
  (`Quat::from_axis_angle(random_tangent, VEG_LEAN_MAX · r²)`) into
  `upright_rotation` — trees lean a couple of degrees, rocks tumble more
  (per-archetype `lean_max` in the `Archetype` table). Both are placement-side;
  instances stay 80 B (until B4 packs them — coordinate: packed form must carry
  the lean, so land G3 before or with B4's instance packing).
- **Wind sway (shader):** in `vegetation.wgsl`'s vertex stage, displace by
  `wind_dir · sin(t · WIND_FREQ + phase) · WIND_AMPLITUDE · pos.y²` (bend grows
  quadratically up the plant; base pinned). Per-instance `phase` must **not**
  come from world position (the §8 precision rule) — derive it from the
  instance's tint bits or add it to the packed instance (B4 has spare bits in
  the quantized yaw/scale lane). `wind_dir` = seed-derived constant in
  `Globals` (one new vec4 shared with G2's moon — pack thoughtfully via E2's
  single builder). Amplitude scaled down for rocks via a per-archetype
  `sway: f32` multiplier baked into… simplest: a `sway` scalar in the spare
  instance tint alpha (`tint[3]` is currently unused — it's *already
  uploaded*). Zero new memory, per the backlog's Pool-B ×1.0 promise.
- **Verify:** the flora gallery test (extend it to render through the *veg*
  pipeline — the backlog's own "Evaluation" item; fold it in here as the
  acceptance harness) + goldens with time frozen.

### G4. LOD-stable vegetation placement — **M–L**
*[Correctness-of-experience • new finding, not in any prior list]*

**Why.** Placement is seeded by `key.hash(planet.seed)` — **which includes the
chunk's LOD level**. When a chunk splits (or merges), its four children draw
vegetation from *different RNG streams* than the parent: **every plant in view
re-rolls as you zoom** through each level from `veg_min_level` (13 at High)
down to 16 — three full reshuffles of every visible stand on approach, plus a
per-frame shimmer band at the split boundary as chunks flicker between levels.
Areal density hides the *count* change, but positions/species/sizes all jump.
This is the largest remaining "the world isn't solid" artifact and it gates
walk mode (H1): plants must not teleport as you walk toward them.

**What.** Decouple placement from the *rendering* quadtree: plants exist on a
**fixed virtual placement grid** (level `VEG_PLACEMENT_LEVEL`, e.g. 15 — cells
≈ 16 m… choose ~the finest level that gets veg), and a chunk at any level
renders the union of placements from the fixed cells it covers. Same
determinism story (`seed_hash(face, cell_i, cell_j)` — the species-presence
Worley grid already works exactly this way and is the model to follow), but now
LOD-invariant: zooming refines geometry under plants that stay put.

**How.**
- `place_vegetation` iterates the placement cells overlapping `[u0, u0+size) ×
  [v0, v0+size)` instead of drawing `veg_attempts` random points from the chunk
  RNG. Per cell: a small fixed attempt budget (`VEG_ATTEMPTS_PER_CELL`, derived
  once from density so *areal* density is preserved), each attempt fully seeded
  by `(seed, face, cell, attempt_index)` — position, jitters, species pick,
  age, yaw all from that stream. The per-chunk **cap** becomes a per-cell cap
  (same bound, better distributed).
- Coarse chunks (level ≪ placement level) would cover thousands of cells —
  but they only *need* the sparse survivors: keep the existing
  area-proportional thinning by having coarse chunks sample a deterministic
  subset of cells (stride by `4^(placement_level - level) / cap` with a
  per-cell hash threshold — "keep this cell's plants iff hash(cell) <
  keep_fraction"), which preserves *which* plants exist near the camera as LOD
  rises: the fine level draws a superset of what the coarse level drew.
  Design the subset rule so it's **monotonic** (coarse ⊂ fine) — that's the
  property that kills popping.
- Density/`veg_min_level` settings keep their meaning (they gate which chunks
  draw any veg and the per-cell budget); `VEG_REFERENCE_AREA` folds into the
  per-cell derivation.
- This **re-rolls placements once** → same seed-epoch train as D1/A7 (bundle
  the breaks).
- **Verify:** a new test: build a parent chunk and its four children; assert
  the union of child instances ⊇ parent instances (bit-exact transforms);
  eyeball a zoom-in in the live app — stands must hold still.

### G5. Water v2: explicit vertex channel, depth color, foam, moving ripple — **M–L**
*[Polish • closes review L2 correctly]*

**Why.** Water is currently inferred in the shader from
`length(in.world) < radius + WATER_DETECT_EPS` (1.5 units = 15 m) — sub-15 m
coastal land gets the ocean's specular sheen (L2), and the shader has no idea of
*depth*, so shallows/deeps differ only via baked vertex color. The mesher knows
exactly which vertices are ocean (it clamps them to sea level in
`CpuChunk::build`) and what the true depth is — it just doesn't tell the GPU.

**What.** Give the terrain vertex a water/depth channel and spend it:
crisp water/land classification (no epsilon), depth-graded color and
specular, shoreline foam where depth ≈ 0, and (with C1's phase fix) a slightly
richer two-band ripple. This deliberately stays **within the single-mesh ocean
design** (no separate transparent water surface — that's a much bigger
architectural step; note it as a possible G5.2 with explicit costs:
translucent pass ordering, refraction, underwater camera rules).

**How.**
- B4's compressed vertex has a free lane: pack `water_depth` (0 = land,
  else clamped depth in units / `WATER_DEPTH_MAX`) into the norm/color
  encoding (e.g. color alpha byte). Until B4 lands, add a 4-byte channel to the
  36-B vertex — coordinate the two so the format only changes once.
- `terrain.wgsl`: `is_water = depth > 0`; ocean color =
  `mix(COL_SHALLOW, COL_DEEP, smoothstep(...depth...))` moves from CPU
  (`blended_color`'s ocean branch) into the shader — freeing the vertex color
  to carry the *sea-floor* tint for shallows to show through; foam =
  `smoothstep(FOAM_DEPTH, 0.0, depth) · (noise via the existing hash pattern +
  ripple phase)` — animated shoreline whitecaps, the single highest-payoff
  water feature from altitude; specular/fresnel gate on `is_water` as today.
- **Verify:** goldens for a coastline scene (day + low sun); the L2 artifact
  (glinting beaches) gone by construction.

### G6. Atmosphere v2: single-scatter limb + sunset band — **M–L**
*[Polish]*

**Why.** The rim glow is a single `exp` falloff with a sun-facing multiplier —
solid for phase one, but it can't produce the two effects that sell "planet
from space": wavelength-dependent limb color (blue day limb → orange terminator
limb) and the horizontal sunset band at the terminator seen from low altitude.
Full precomputed scattering (Bruneton/Hillaire LUTs) is heavy machinery; a
closed-form single-scatter approximation gets 80 % of the look for ~40 shader
lines and no textures.

**What/How.**
- `sky.wgsl`: replace the scalar rim with a 4–8 step raymarch along the view
  ray through a thin shell (`ATMOSPHERE_HEIGHT` ≈ 8,000 units): accumulate
  Rayleigh-ish scattering with a per-channel extinction weighted by
  `atmosphere.rgb` (keeping the per-seed alien-sky property — the seed tint
  becomes the scattering coefficient ratio rather than a flat color), phase
  term `(1 + cos²θ)`; sun transmittance approximated by the sun's elevation at
  the sample (`smoothstep` — no LUT). 4 steps × fullscreen is cheap; the pass
  already runs per-pixel math of similar weight for stars.
- Terrain fog joins in: fog color from the same function evaluated at the
  fragment's direction (or cheaper: lerp fog tint by `day` + sun-facing —
  G1 already half-does this). Height falloff on fog density
  (`exp(-altitude / FOG_SCALE_HEIGHT)`) makes valleys pool haze — two lines in
  `apply_fog` using AGL (C3).
- Explicitly **defer**: multiple scattering, LUT precompute, aerial-perspective
  texture — revisit only if G6 output disappoints on the 5090 at 4K.
- **Verify:** golden set spanning noon/terminator/night limb views + a `--video`
  finale render (the pullback is the money shot for this item).

### G7. Near-field cascaded shadow maps — **XL**
*[Polish • the most expensive visual item — schedule deliberately]*

**Why.** Nothing casts shadows; low-sun scenes (which G1/G2 make common and
beautiful) are flat where they should be dramatic — trees float on bright
ground, mountains don't shade valleys. On a planet, whole-world shadowing is a
research topic, but the payoff case is narrow and tractable: **within a few km
of the camera, at low altitude** — exactly where the eye lives during cruise
and walk mode. Beyond that, fog and scale hide the absence.

**What.** 1–2 sun-space cascades covering ~0.5 km / ~4 km around the camera
focus, depth-only pass rendering the already-selected terrain chunks + veg
instances (they're all in view structures already), sampled with 2×2 PCF in
`terrain.wgsl`/`vegetation.wgsl`; fade shadow strength to zero at the outer
cascade edge and above `SHADOW_MAX_ALTITUDE` (from orbit, terrain self-shading
via `diff` is already adequate).

**How (sketch — this item deserves its own design doc when scheduled):**
depth-only pipeline variants (no fragment, bias via `DepthBiasState` — finally
a use for it), cascade matrices built from sun dir × camera focus tangent frame
(stable per world — sun never moves unless H5 lands, which helps caching),
`Depth32Float` 2048² per cascade (32 MB), preset-gated (High+). Vegetation
alpha-cutout needs a fragment shader in the shadow pass for leaf cards (or
accept blob-shadows from LOD1 meshes — cheaper, often fine). Peter-panning vs.
acne tuned per the standard playbook. Depends on: B2 (reversed-Z conventions),
E1 (pipeline factory), I1 (iteration), I2 (cascade visualization).
**Verify:** goldens + fps budget: target ≤ 1.5 ms on the Mac at High.

---

## H — Features

### H1. Surface walk mode — **L**
*[Feature • the promise in the README's first sentence]*

**Why.** "Fly seamlessly from orbit down to the grass" currently ends hovering
at `MIN_DIST` = 15 m with a focus-orbit camera. Standing on the surface —
eye at 1.7 m, looking *up* at the trees the flora program spent six weeks on —
is the payoff moment the whole stack is pointed at, and it's the feature that
turns a five-minute demo into a thirty-minute one.

**What.** A camera mode toggle (auto-engage below ~30 m + a key, e.g. `F`):
first-person eye glued to `surface_radius(eye_dir) + EYE_HEIGHT_UNITS`
(0.17 units = 1.7 m), tangent-plane WASD movement at walking/running speed
(1.4 / 6 m/s × boost), yaw/pitch look (arrow keys to match the keyboard-only
philosophy; optional mouse-look behind a setting), slope-aware speed (no
climbing `VEG_MAX_STEEPNESS`+ walls), jump optional-later. Exits back to orbit
mode by zooming out.

**How.**
- `Camera` grows a `Mode` enum (`Orbit`, `Walk { eye_dir: Vec3, yaw, pitch }`)
  — the orbit math stays untouched; walk mode bypasses focus/distance/tilt and
  produces its own `view()` basis (up = `eye_dir`, forward from yaw/pitch in
  the tangent frame — `tangent_basis` + the pole-safe `frame()` already exist).
  `view_proj`/`near_far` work unchanged off the returned eye (near wants a
  floor of ~0.02 in walk mode — reversed-Z B2 makes that free).
- Ground query per frame: `surface_radius(eye_dir)` — with A1's cache plus one
  live sample; smooth with the tour's asymmetric rise/fall time constants
  (`GROUND_RISE_TAU`/`GROUND_FALL_TAU` — the terrain-follow code in `tour.rs`
  is 80 % of walk-mode's vertical logic already; extract and share it).
- Streaming: walk speed is slow — the streamer coasts. The LOD floor is the
  visible limit: at `MAX_LEVEL` 16, ground quads are ~8 m — acceptable v1
  (Google-Earth-ground-level look), transformative after C2 (which this item
  should *not* wait for; ship v1, let C2 upgrade it).
- Interactions: tour and walk are mutually exclusive (walk cancels tour like
  any manual control); HUD altitude shows AGL (C3 is a prerequisite for the
  numbers to make sense); fog wants the AGL basis too (same C3).
- **Verify:** the tour-smoothness test pattern applied to a scripted walk
  (no NaN, never underground, speed within bounds); manual feel pass.

### H2. Bookmarks: save/load/share locations — **S–M**
*[Feature • cheapest high-utility item in the list]*

**Why.** `P` prints a location; nothing can *return* to one. Every "look what I
found" moment currently dies with the session — in a seeded, deterministic
world where a 60-byte tuple reproduces any view exactly. This is the
determinism guarantee turned into a user feature.

**What.** `B` saves the current view (seed, focus lat/lon, distance, heading,
tilt, mode) with a timestamp; a `LOCATIONS` overlay tab lists saved spots
(number keys teleport); `--at "lat,lon[,alt]"` CLI flag starts there; a
share-string format (`pe://seed=…&lat=…&lon=…` or just the CLI args) printed
alongside `P`.

**How.**
- Storage: a TOML/JSON lines file next to the log
  (`$PLANET_LOG_DIR/bookmarks.toml` — reuses `logging.rs`'s path logic and the
  cross-account sharing property; versioned header per D1's epoch discipline so
  a future seed-epoch bump can annotate stale bookmarks rather than lying).
  Parse defensively (guidelines §3: warn + skip bad entries, never panic).
- Overlay: `settings.rs`'s tab system is already generic
  (`TAB_COUNT`, per-tab bodies in `overlay.rs`) — add `TAB_LOCATIONS` with the
  same row/selection pattern; teleport reuses `camera.teleport` + `set_view`.
- Cross-seed bookmarks: a bookmark stores its seed; selecting one from a
  different world offers "restart with seed N" (print instructions v1 — full
  in-app world reload is a bigger step: `Planet` rebuild + streamer clear —
  actually cheap via the existing `apply_rebuild` pattern + `Planet::new`;
  v1.5).
- **Verify:** round-trip test (save → parse → same view bits); a bookmark file
  from the GUI account loads on the dev account (shared-dir property).

### H3. "Wonders" finder: seed-derived points of interest — **M**
*[Feature • unique differentiator]*

**Why.** Teleport is uniform-random; the tour picks *biomes*, not *drama*. Yet
the most compelling seconds of any session are extremes — the tallest peak, a
fjord-cut coast, an equatorial glacier, an island chain. Those are pure
functions of the seed, findable by sampling — the app just never looks for
them. A "wonders" list gives every world an itinerary and gives the tour/video
a highlight reel.

**What.** Per seed, compute (once, off-thread, cached): highest peak, lowest
ocean trench (for the map/HUD flavor), largest island and largest lake-like
inland sea (coarse flood-fill), steepest relief within a window (max
`steepness` sample), most biome-diverse region (entropy over a sampling disc),
"polar oasis" style rarities (warm pocket at high latitude). Expose: a
`WONDERS` overlay tab (teleport like H2), `W` to cycle, tour option to route
via wonders, and INFO log lines (nice in videos).

**How.**
- A `wonders.rs` worker spawned at startup (post-F2 it overlaps the first
  minutes of flight): pass 1 — golden-spiral sample ~200k dirs (at 3.85 µs ≈
  0.8 s of one core, spread across the pool via chunked `std::thread::scope`)
  recording height/biome/steepness extrema; pass 2 — refine each candidate by
  local hill-climb (sample a shrinking neighborhood, ~thousand more samples);
  island/sea detection on a coarse lat-lon occupancy grid (flood fill,
  ~512×256 cells). All deterministic → cache to
  `$PLANET_LOG_DIR/wonders-<seed>.toml`, versioned with the seed epoch (D1).
- Results arrive asynchronously: the tab shows "surveying…" until the channel
  delivers (degrade, don't block — guideline §2).
- **Verify:** determinism test (two runs, same list); a "peak is actually a
  local max" property test on a few seeds.

### H4. Screenshot key + `--screenshot` CLI — **S–M**
*[Feature + tooling feeder]*

**Why.** The only way to capture the app today is OS-level screen grab (GUI
account) — while the codebase already contains everything needed
(`render_to_rgba`, PNG encode in the test harnesses). A first-class capture
path feeds I5 (goldens), the README, and the user's share impulse.

**What/How.**
- Headless: `--screenshot out.png [--at …] [--seed …] [--size WxH]` — trivially
  composed from `Renderer::new_offscreen` + the `capture_frame` settle loop
  (the `--video` machinery minus ffmpeg; ~50 lines in the E1-refactored
  `video.rs`).
- Live: `F12`/`PrintScreen` — the swapchain texture can't be read back
  portably, so render one extra frame into a `COPY_SRC` offscreen texture:
  factor `record_scene` (already target-agnostic) so the live renderer can own
  a lazily-created capture target of window size; write PNG to
  `/Users/Shared/planet-explorer/screenshots/` (macOS) / pictures-dir
  (Windows) with a seed+latlon filename — self-documenting captures.
  Log at INFO with the path.
- **Verify:** the capture of a fixed seed/camera byte-matches
  `render_to_rgba`'s output (same path, so trivially true — the test guards
  the live-path plumbing).

### H5. Optional planet rotation (day cycle) — **M**
*[Feature]*

**Why.** `sun_dir` is eternally frozen per seed; after G1/G2 exist, a slowly
turning planet makes the terminator/sunset machinery *play* rather than pose.
Off by default (the current fixed look is a valid aesthetic and the tour is
tuned for it), a settings row turns it on.

**What/How.** Don't move the sun — **spin the planet's lighting frame**:
`sun_dir(t) = Quat::from_axis_angle(spin_axis, DAY_RATE · t) · sun_dir₀` with
`spin_axis` = seed-derived (reuse the pole = +Y for simplicity v1),
`DAY_RATE` a settings row in real units (minutes per day; default ~20 min).
Computed on CPU per frame into `Globals` — zero shader change beyond what
G1/G2 already read. Determinism: `t` is sim time (the same `time`/`sim_time`
the water uses), so `--video` renders are reproducible; the live app's `t`
starts at launch (fine — it's presentation, not generation; document that
distinction in the module header). Biome/climate stay sun-independent
(seasonal coupling is the backlog's separate "Seasons" item — explicitly out of
scope here, but H5's axis/day-phase machinery is the substrate it will want).

### H6. Planet archetypes (seed-derived world classes) — **M**
*[Feature • replay value at near-zero content cost]*

**Why.** Every seed today is the same *kind* of world — Earth-tuned constants
(`CONTINENT_SEA_BIAS`, `MOUNTAIN_WEIGHT`, temp/moisture thresholds) with a new
noise phase and sky tint. The generation stack is *already parameterized* by
those constants; deriving a handful of them from the seed yields ocean worlds,
desert planets, ice ages, archipelagos, and pangea supercontinents — an order
of magnitude more variety from ~a hundred lines. This is the highest
variety-per-effort item in the roadmap.

**What.** In `Planet::new`, draw a **world profile** from the seed (via the
existing splitmix stream — order matters for compatibility; this **re-rolls
nothing** if drawn *after* the current draws): sea bias ∈ [−0.05, 0.30] (ocean
→ pangea), mountain weight ∈ [0.8, 1.8], temp offset ∈ [−0.25, +0.15] (ice age
→ hothouse), moisture offset, continent frequency ∈ [0.6, 1.6]
(supercontinent → archipelago). Classify the drawn profile into a named
archetype (`"Ocean world"`, `"Desert world"`, …) shown at startup, in the
HUD title, and in the log.

**How.**
- The constants involved become fields on `Planet` (defaulting to today's
  values), read by `height`/`climate`/`classify` — a mechanical threading;
  `classify`'s biome thresholds stay global (the *inputs* shift instead —
  cleaner and keeps the biome tests meaningful).
- Guard degeneracy: clamp so every world keeps ≥ ~10 % land and ≥ 2 biomes
  (the tour and the flora system assume *some* land; `pick_destination`
  already falls back gracefully, and the `world_has_land_and_sea_and_variety`
  test becomes a multi-seed property test with archetype-aware bounds).
- Ship behind `--worlds varied|earthlike` (default `earthlike` until the tour
  video and flora tables get a pass on extreme worlds; flip the default when
  it's tuned). **Seed compatibility:** default-off means existing seeds render
  identically until the user opts in — no epoch needed if the profile draw is
  appended to the splitmix stream *and* gated; document this in D1's epoch
  notes.
- **Verify:** archetype distribution test over 1,000 seeds (each class
  occurs; no degenerate all-ocean under clamps); tour completes on 10
  extreme seeds (the existing tour test, parameterized).

### H7. Ambient audio layer + volume controls — **M**
*[Feature]*

**Why.** The soundtrack is the only sound; the world itself is mute — no wind
at altitude, no surf on a beach cruise, no forest bed. Rodio already runs a
mixer; the biome under the camera is already computed every perf-sample. Also
missing: any in-app volume control (`AUDIO_VOLUME` is a compile-time constant).

**What/How.**
- Assets: 4–6 loopable CC0 ambience beds (wind-high, wind-ground, surf,
  forest-birds, desert, snow) — a few MB of OGG each (rodio/symphonia decodes
  ogg; smaller than mp3 for loops). Ride F1's pack rather than more
  `include_bytes!` lines.
- `audio.rs`: a second `Player` per active bed; each frame (or each
  title-update tick — 0.4 s is plenty) compute target gains from
  (biome-under-focus, AGL): surf gain from `Beach`/coast proximity, wind from
  altitude bands, forest from the forest biomes; ease gains toward targets
  (`GAIN_TAU` ≈ 2 s) for click-free crossfades. Biome is already sampled for
  the HUD — zero added sampling.
- Controls: `[`/`]` master volume in steps + `M` mute, persisted in the same
  file H2 uses (a general lightweight prefs file: volume, units, last window
  size — fold `--units` persistence in while there); a `SOUND` overlay row is
  optional polish.
- **Verify:** unit test the gain-mapping function (pure); ear pass for the
  crossfades.

### H8. Gamepad support — **S–M**
*[Feature • demo ergonomics]*

**Why.** Keyboard-only is a deliberate philosophy that works at a desk; on the
couch/TV where the tour shines, a controller is the natural input.
`winit` doesn't do gamepads; **`gilrs`** is the standard pure-Rust,
cross-platform (macOS/Windows/Linux) answer — fits the dependency rules.

**What/How.** Poll `gilrs` in `about_to_wait` (it's event-pumped, cheap); map
sticks to pan/rotate-tilt (analog magnitudes feed the same code paths as keys —
generalize `Keys`' booleans to −1..1 axes internally, keys set ±1; this
refactor is the actual work and also cleans up the input model), triggers to
zoom, buttons to tour/teleport/menu. Feature-gate the dependency
(`features = ["gamepad"]`, default-on) so a build without it stays possible.
**Verify:** manual; the axis refactor gets a unit test (keys still produce
exactly ±1 behavior — camera feel must not change).

---

## I — Tooling & developer experience

### I1. Shader hot-reload (debug builds) — **S–M**
*[DX • force-multiplier for the entire G section]*

**Why.** Shaders are `include_str!`-embedded; every WGSL tweak is a full
rebuild + repackage + relaunch (+ cross-account copy when checking on the GUI
side) — minutes per iteration on work that wants dozens of iterations per hour.
G1–G7 are overwhelmingly shader work; this item pays for itself in the first
afternoon of G1.

**What.** Debug builds watch `src/shaders/`; on change, recompile the module
inside a validation error scope — success swaps the pipeline(s), failure logs
the naga error and keeps the old pipeline running. Release builds keep
`include_str!` exactly as today (self-contained property untouched).

**How.**
- `#[cfg(debug_assertions)]` path in `gfx`: shader sources resolve via a small
  `fn shader_source(name) -> Cow<'static, str>` — embedded in release, `fs::read`
  from a `SHADER_DEV_DIR` (env override, default relative `src/shaders`) in
  debug. Poll mtimes once a second in `frame()` (no new dependency; `notify`
  is overkill), rebuild affected pipelines via the same factory `make_pipeline`
  calls `assemble` uses (E1's split makes those callable piecemeal).
- Wrap rebuild in `device.push_error_scope(Validation)` → on error,
  `error!(target: "shader", …)` + keep old pipeline (degrade, don't panic —
  guidelines §2 applies to dev tools too).
- The dev loop then becomes: run the app on the GUI account once, edit WGSL
  from the SSH account, watch it live. (The shared-folder workflow finally
  points the right direction for iteration speed.)
- **Verify:** deliberately break a shader while running — app keeps rendering,
  log carries the naga diagnostics.

### I2. On-screen debug HUD (F3) — **M**
*[DX/observability]*

**Why.** The perf log is excellent for after-the-fact analysis but blind live;
the window title carries seven numbers at 0.4 s cadence. Tuning A3–A5, B5, G7
and the 5090 session (J1) all want *live* internals: frame-time sparkline,
draw/want/pending/uploads, resident bytes vs. budget, LOD histogram, and
visual debug modes (chunk-boundary tint, LOD-level coloring, frustum freeze).

**What/How.**
- `F3` toggles a debug overlay reusing the existing overlay pipeline (it's
  general instanced-quad text already; a second geometry buffer alongside the
  menu — `overlay.rs` grows a `panel` abstraction, which H2/H3's tabs want
  anyway). Frame-time graph = one quad per sample column (120 samples × 1
  quad ≈ nothing).
- Debug view modes as a `Globals` flag bit (`params` has spare lanes after C3
  moves altitude): terrain shader tints by `chunk-id hash` or LOD level
  (needs per-draw data — free after C2's per-chunk offset lane exists;
  before that, a cheap approximation: color by `floor(log2(quad_size))`
  derived from screen-space derivatives `fwidth(world)` — no plumbing at all).
- Frustum-freeze toggle stores the frozen planes in `App` and passes them to
  `select` (A5 lists this as its tuning tool).
- **Verify:** it's a tool — the acceptance test is using it during A5/B5.

### I3. Glyph-atlas text rendering — **S–M**
*[DX/perf hygiene • gated on I2/HUD growth]*

**Why.** Overlay text currently emits **one instanced quad per lit pixel per
glyph** (`overlay.rs::layout` walks `FONT8X8` bits) — the ESC menu is ~1,000+
quads rebuilt on every navigation keypress. Fine for a modal menu; wrong cost
model once I2 puts live text on screen every frame.

**What/How.** Bake `FONT8X8` into a 128×64 R8 texture at startup (one-time,
trivially derived from the same table — no new assets, still engine-free and
portable per the CLAUDE.md text-rendering rule); overlay instances become one
quad per *glyph* with a `uv_rect`, sampled with nearest filtering. `layout`
shrinks; per-pixel path deletes. ~64× fewer instances, and glyph scaling stops
being blocky-by-construction (still intentionally chunky at nearest — the
aesthetic survives). Do it when I2 lands, not before.

### I4. Headless benchmark mode (`--bench`) — **S–M**
*[Process • "measure before and after" made real]*

**Why.** The guidelines demand measurement around every perf change; the tools
today are the perf log (needs the GUI account running the app) and one ignored
micro-bench. Nearly every A/B/C item above says "verify via I4" — this is the
missing harness, and it's small because `--video` already built the hard parts
(headless renderer, fixed-timestep tour, settle loop).

**What.** `planet-explorer --bench [--seed N] [--preset P] [--seconds S]`:
run the deterministic tour headlessly (no encoder), fixed timestep, and report
a JSON line + human summary: avg/p99 frame CPU ms (sim+select+upload+record,
timed separately), chunk builds/sec, time-to-sharp after each teleport
(frames until `pending == 0`), peak resident bytes, total draws. Two runs of
the same build differ only by machine noise → diffable across commits and
across the two machines.

**How.** A `bench.rs` sibling of `video.rs` sharing `capture_frame`'s skeleton
(E1/E2 first); render into the offscreen target but skip readback (GPU timing
via timestamp queries is a stretch-goal — `wgpu::Features::TIMESTAMP_QUERY`
where available, reported as optional fields). Wire `just bench` (E6) and keep
a `bench-results/` log (gitignored) with commit hashes. **Verify:** run twice,
assert variance < a few percent; then use it to land A1 with numbers.

### I5. Golden-image visual regression harness — **M**
*[Process • shader safety net]*

**Why.** The B/C/G program rewrites every shader several times over; today the
only visual checks are two smoke-test brightness heuristics and eyeballs on
another account. The infrastructure for goldens exists (offscreen render →
PNG in three test modules); what's missing is fixed reference scenes and a
comparison gate.

**What.** A `goldens` test binary (or `--ignored` test set) rendering ~8 fixed
(seed, camera, time, preset) scenes chosen to cover the feature surface:
orbit-with-terminator, mountain grazing light, coastline sun-glint, forest
closeup, night limb, overlay menu, snow/ice, video-finale framing. Compare
against committed PNGs with a perceptual tolerance (mean ΔE or SSIM-lite;
per-pixel exact is too brittle across driver versions — record the tolerance
rationale in the harness doc comment). `just goldens` re-baselines
intentionally, and the diff images land in the temp dir for eyeballing.

**How.** Builds directly on E1's `TestGpu` + H4's capture path; store goldens
under `tests/goldens/` (small — 8 × ~500 KB PNGs; acceptable weight, or fold
into F3's LFS decision). Platform reality: goldens are per-GPU-family in the
worst case — start Mac-only (the machine that runs tests), add a
5090-generated set the first time J1 runs. GPU-less environments skip cleanly
like the smoke tests. **Verify:** flip `SUN_AMBIENT` by 0.01 → harness flags
exactly the lit scenes.

---

## J — Windows / RTX 5090 readiness

### J1. Backend/present-mode overrides + on-site validation checklist — **S**

**Why.** 5090 sessions are scarce and currently under-instrumented: wgpu
auto-picks the backend (DX12 by default on Windows — Vulkan never gets
compared), present mode is hard-coded `AutoVsync` (a 240 Hz G-Sync box can't
show what the renderer can actually do, and *every* perf number is
vsync-flattened), and there's no script for what to capture while there.

**What/How.**
- `PLANET_BACKEND=dx12|vulkan` env override (map to `wgpu::Backends` at
  instance creation; log the choice — the `gpu adapter selected` line already
  carries backend), `PLANET_PRESENT=fifo|mailbox|immediate` similarly (fall
  back with a warn if unsupported — capabilities are queried already). Env vars
  not settings rows: these are diagnostic knobs, and env parsing follows the
  existing tolerant style.
- A `WINDOWS_5090_CHECKLIST.md` (or section in BACKLOG): run `--bench` (I4) at
  every preset × both backends × vsync-off, capture the perf log + adapter
  lines, note GPU util vs. power (the original report's signature metric),
  re-run after each A6/J2 change. Ship the checklist *with* the next build
  handed to that machine.
- **Verify:** the next 5090 session produces comparable JSON artifacts instead
  of an anecdote.

### J2. Draw-call follow-ups, gated on measurement — *pointer*

The 5090's "100 % util / 50 % power" signature is submission-bound behavior;
the terrain arena attacked it, A6 (veg instance arena → indirect) is the
remaining planned work, and `SPLIT_MAX` retuning (BACKLOG) is the free knob.
**Do not** schedule indirect-draw work until J1 data shows the arena + A5
culling + A2 bucketing still leave the frame submission-bound — every one of
those may independently move the bottleneck to somewhere cheaper to fix.

### J3. Windows polish — **S**

Three small items, one trip: (a) `build.rs` shells `date` — absent on Windows,
so `BUILD_DATE` is empty in the overlay/startup log on the primary *other*
platform; replace with `std::time::SystemTime` + a tiny hand-rolled UTC
formatter (or the `jiff`/`time` crate if a dependency is acceptable — it isn't
needed; ~15 lines of math). (b) Log path on Windows is `%TEMP%` — surface it
prominently at startup (the console banner already prints it; verify the
console actually shows under a double-clicked exe — it won't; consider
`windows_subsystem = "windows"` implications *before* flipping it, since today
the console is the only place the banner goes). (c) Confirm the `.ico`
embed + taskbar identity look right at 125 %/150 % DPI scaling (winresource
already embeds multi-res).

---

## Sequencing — phases & dependencies

The phases respect the guideline hierarchy: measurement and correctness debt
first, then the perf floor, then visible richness, then scale investments.
Each phase is independently shippable; nothing in a later phase blocks an
earlier one.

**Phase 0 — Foundations & measurement (≈ a focused week).**
`E1` (splits + harness) → `E2` (globals builder) → `I4` (bench) → `A1`, `A2`,
`C3` (small wins with before/after numbers) → `D1` (**the seed epoch — do this
before anything user-facing depends on stability**) → `D3`, `E5`, `E6`, `F2`.
Exit criteria: bench JSON in hand on the Mac; goldens bootstrapped (`I5` can
start here with 3–4 scenes); all files < ~1,000 lines and shrinking.

**Phase 1 — The frame-path floor (1–2 weeks).**
`A3` → `A4` → `A5` (frustum) → `B1` (winding/cull) → `B2` (reversed-Z) →
`I1` (hot reload, before shader-heavy work) → `I2` (debug HUD). Then `J1` and
a 5090 session to decide `A6`/`J2`. Exit criteria: no hitch warns on teleport
at Ultra on the Mac; draw count at ground level ≤ ⅓ of today's; 5090 bench
artifacts captured.

**Phase 2 — The visible leap (2–3 weeks, shader-heavy).**
`G1` (terminator) → `G2` (sun/moon) → `C1` (ripple fix) → `G3` (flora look +
wind) → `B3` (MSAA/A2C) → `G5` (water v2, with B4's vertex-format change
landing together) → `G6` (atmosphere v2). `H4` (screenshots) early in the
phase — it feeds I5 as the scenes multiply. Exit criteria: a re-recorded
`--video` tour that is *obviously* a different-generation product; goldens
covering all new looks.

**Phase 3 — Scale & memory (2–3 weeks, measurement-gated).**
`F1` (asset pack) → `F3` → `B4` (remaining compression) → `A6` (veg arena; +
indirect only if J1 says so) → `A7` (octave LOD + veg reorder; the f32-noise
epoch only if worker throughput still binds) → `B5` stage 1 (mesh LOD).
Exit criteria: binary < 60 MB; startup < 300 ms; Ultra on the 5090 GPU-bound
at high power draw (the original report's signature inverted).

**Phase 4 — The world becomes a place (ongoing, feature-led).**
`G4` (LOD-stable vegetation — bundle its re-roll with any remaining epoch
break) → `H2` (bookmarks) → `H1` (walk mode) → `H3` (wonders) → `H7`
(ambient audio) → `H5` (day cycle) → `H6` (planet archetypes) → `H8`
(gamepad). `C2` (camera-relative + `MAX_LEVEL`) sits at the end of this phase
or the start of the next — after B2/B4 landed its prerequisites, before walk
mode's fidelity ceiling starts to chafe. `G7` (shadows) and `B5` stage 2
(impostors) schedule opportunistically once I1/I2/I5 make them tractable.

**Dependency spine (the edges that actually constrain order):**
`E1 → {E2, I1, I4, I5, D2, H4}` • `I4 → every A/B perf claim` •
`D1 → {A7-noise, G4, H2, H3}` (epoch discipline before stability consumers) •
`B2 → {C2, G7}` • `B4 ↔ G5` (one vertex-format change) • `B4 → C2` •
`C3 → {H1, G6-fog}` • `F1 → {B5-impostors, H7-assets}` • `A1 → {A5, H1, H3}`
(cheap height queries) • `G1 → {G2, G6, H5}`.

---

*Maintenance note: keep IDs stable when editing this file (strike items with
`~~` + a status line rather than renumbering, matching the review's ledger
style). When an item ships, move its one-line summary to `BACKLOG.md`'s "Done"
list and mark the entry here — this document is the plan of record until a
future full review supersedes it.*

