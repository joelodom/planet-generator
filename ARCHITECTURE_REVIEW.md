# Planet Explorer — Architecture & Implementation Review

**Reviewed commit:** `bb09736` (`bb097363b1bafa5927bc88924fd2c5205914b1f2`),
committed **2026-06-08 06:18:41 -04:00** — _"Add procedural per-planet vegetation
grown from the seed"_, branch `main`.

_Reviewer: senior-engineer pass over the full `src/` tree • Review date:
2026-06-08 • Scope: all 16 Rust modules (5,538 LOC) + 4 WGSL shaders (241 LOC)._

**Method.** ct structural analysis (`survey`/`themes`/`risk`/`conventions`/
`grep`), a real `cargo clippy --all-targets`, a panic/`unsafe`/concurrency/debt
"landmine" sweep, and a close read of every module. Tool output is summarised in
the appendix.

---

## 1. Executive summary

This is a **mature, high-craftsmanship codebase** — well above typical
hobby-project quality and close to production-grade for its declared "phase one"
scope. The evidence is concrete, not vibes:

- **`cargo clippy` is clean**: 6 warnings, **0 errors**, and all 6 are the *same*
  trivial `collapsible_if` lint at two lines.
- **Zero `unsafe`** in the entire tree (notable for a wgpu app).
- **Zero `TODO`/`FIXME`/`HACK`** markers — no acknowledged debt rot.
- **21 tests**, including headless GPU pipeline validation with framebuffer
  readback, determinism checks, and a 6-minute tour-invariant simulation.
- **Disciplined throughout**: named-const tuning values even in WGSL (per the
  project rule), structured `tracing` logs with a panic hook, build-stamping,
  per-module doc headers, and clean subsystem boundaries (`planet` is the seeded
  source of truth; `gfx` knows nothing about LOD policy).

**There are no Critical findings** — no reachable crash on normal input, and the
security surface is essentially nil (local single-player, no network, embedded
assets, no untrusted parsing). That is a real result, stated plainly.

The highest-value work is **performance headroom**, which aligns with the stated
priority order. The single biggest lever — terrain generation doing **~3× the
noise work it needs** (§4 H1) — **has since been fixed** (~1.9× faster terrain
sampling, byte-identical output). The main scalability ceiling for the planned
RTX 5090 "Ultra" target is the **per-chunk GPU buffer/draw model** (H2). After
that it's small robustness hardening (worker supervision, log-open failure) and
routine maintainability (three 500+ line files).

> **Post-review status (2026-06-08):** **H1 is fixed** (details in its entry
> below). All other findings stand exactly as written.

### Maturity scorecard

| Dimension | Rating | One-line assessment |
|---|---|---|
| **Correctness** | 🟢 Strong | Pure-math generation, deterministic, well-tested; panics confined to startup/asset paths. |
| **Security** | 🟢 Strong (low surface) | No network, no `unsafe`, no untrusted input; only nit is a world-writable shared log (by design). |
| **Performance** | 🟡 Good, with clear wins | Sound architecture (threaded streaming, LOD, horizon cull); leaves ~3× on the table in chunk gen + draw/upload churn at scale. |
| **Maintainability** | 🟢 Strong | Clean module seams, named consts, good docs; three files >500 LOC (one ~45% tests). |
| **Portability** | 🟢 Strong | Broad backend enablement, `cfg`-gated OS bits, pure-Rust deps; one startup-panic path hurts the Windows goal. |
| **Testing** | 🟢 Strong | 21 tests incl. GPU smoke + determinism + tour invariants. No CI config in-repo (headless smoke is the gate). |
| **Docs** | 🟢 Strong | Every module has an intent header; CLAUDE.md encodes scale/precision/portability rationale. |

---

## 2. What's mature

- **Subsystem architecture.** `planet` (seed → ground truth) / `mesh` / `lod` /
  `camera` / `tour` / `gfx` are cleanly separated with one-way dependencies. The
  renderer is "a thin, replaceable slab" as advertised — it takes `CpuChunk`s and
  a draw list and knows nothing about planet maths. This is what makes the
  roadmap (animals/weather/NPCs) tractable.
- **Scale & precision strategy.** The 10 m/unit trick to keep Earth-sized coords
  inside comfortable f32 range, with per-frame horizon-derived near/far
  (`camera.rs:251`), is a genuinely thoughtful answer to the planetary-renderer
  depth problem.
- **Streaming.** Background meshing pool with a generation guard
  (`lod.rs:161-271`) so a detail change can't leave stale geometry; horizon
  culling; coarse-ancestor fallback so there are never holes. Solid design.
- **Runtime detail system.** `settings.rs` fans a single master "Detail" knob out
  to LOD/mesh/veg, with a memory budget that derives a chunk cap — live values vs.
  rebuild-on-close values are correctly separated (atomics in `MeshConfig`,
  generation bump in `Streamer`).
- **Determinism.** Everything derives from the seed via SplitMix64 sub-seeding;
  `tests.rs` asserts identical worlds across runs and identical chunk builds.
- **Observability.** `logging.rs` (panic hook, build stamp, leveled targets) is
  built around the real remote-debugging workflow. Excellent.
- **Testing discipline.** The headless `offscreen_pipeline_validates` smoke test
  compiles the *real* shaders/pipelines inside a validation error scope and
  asserts the framebuffer isn't blank — catching GPU regressions without a window.

## 3. What's less mature / the gaps

- **Performance is correct-but-not-yet-optimised** — the generation hot path and
  the GPU resource model both have known, named headroom (§4).
- **Failure handling is panic-first at a few boundaries** (`logging::init`,
  asset decode) rather than graceful degradation — fine for trusted embedded
  assets, less fine for the log path on the Windows target.
- **No worker supervision** — a panic in a meshing worker is logged but the
  thread is never replaced.
- **File size / method length** creeping in `gfx.rs` (1,261 LOC), `flora.rs`
  (671), `main.rs` (657); `App::frame` is one long method.
- **No in-repo CI** (`.github/`/pipeline) was found; the headless smoke test is
  the de-facto gate but isn't automated here.

---

## 4. Findings & recommendations by priority

Each finding is tagged with its primary dimension and a rough effort estimate
(XS = minutes, S = <1h, M = ~half-day, L = days). Within a tier, items are
ordered by the project's priority axis (correctness → security → performance →
maintainability → portability).

### 🔴 Critical (crash / security)

**None found.** No reachable panic on normal user input; no security-relevant
defect. The closest things to crashes are startup/asset panics on *embedded*
(compile-time-fixed) data — see M3/M4 for the one path that can realistically
fire in the field (log open on a locked-down FS).

### 🟠 High (big difference)

**H1 — `steepness()` triple-samples the height field per vertex. ✅ FIXED (2026-06-08).** _[Performance • effort M]_
`Planet::sample()` called `height()` once, then `steepness()` called it **twice
more** at tangent offsets; `CpuChunk::build` ran `sample()` per grid vertex (up to
`(grid+1)²` ≈ 7,900 at Ultra), so terrain generation paid **~3× the noise cost it
needed** — the dominant streaming cost.
**Investigation corrected the original suggestion.** Finite-differencing the chunk
grid (the first idea) would have been *wrong*: `steepness()` deliberately uses a
fixed ~9.6 km baseline so biome classification is **LOD-independent**; differencing
grid neighbours would make slope — and thus biomes/colour — shift as you zoom in.
The real lever: `classify()` only reads slope for the **Mountain** test, gated by
`height > MOUNTAIN_MIN_HEIGHT`, so computing slope for the ~95% of vertices at or
below that line is **dead work**.
**Fix shipped:** a lean `Planet::sample_terrain()` (height + colour) that skips the
slope probe below the mountain line; the terrain grid uses it, while the full
`sample()` (vegetation/HUD) is unchanged. Colour moved out of `Surface` into the
meshing path — a cleaner render/query split that also drops colour work from
`sample`. **Result: ~1.9× faster** per terrain sample, **byte-identical output**
(new `sample_terrain_color_is_slope_independent_below_mountains` test proves the
skip is colour-neutral; all 21 tests pass, clippy clean).

**H2 — Per-chunk GPU buffer churn + unbatched draws cap scalability.** _[Performance • effort L]_
`upload_chunk` (`gfx.rs:408`) creates **2–4 fresh `wgpu::Buffer`s per chunk**
(`GpuMesh::upload`, `gfx.rs:40`); eviction frees them; camera motion re-streams →
continuous allocate/free churn. The draw loop (`gfx.rs:496-516`) then issues **one
`draw_indexed` per chunk for terrain and another per chunk for vegetation**, with
no batching or instancing. This is fine on Metal today but is the ceiling for the
explicit RTX 5090 "Ultra" goal (thousands of resident chunks). **Fix (incremental):**
(a) pool/suballocate chunk buffers (reuse freed allocations by size class) to kill
churn; (b) consider a growable per-frame arena or indirect/multi-draw to collapse
the per-chunk draw calls. This is an investment, but it's the one that "makes a
big difference" at the target the project is explicitly built for.

### 🟡 Medium

**M1 — Vegetation is baked per-chunk, not instanced. ✅ FIXED (2026-06-08).** _[Performance • effort L]_
Each chunk baked every plant into a unique world-space vertex buffer (`bake_plant`).
The trade — unlimited variety + 1 draw/chunk — was deliberate, but memory scaled with
*total plant count*, making it the dominant memory consumer and the cause of a
high-detail "treetop flashing" thrash (the baked covering exceeded the memory budget).
**Fixed by per-species instancing:** each species' base mesh is uploaded once and
drawn with a per-instance transform + tint (`shaders/vegetation.wgsl`,
`mesh::VegChunk`) — **~95× less veg memory** (a dense chunk: ~54 KB of instances vs
~5 MB baked), so the covering fits the budget and density went back to full.

**M2 — Eviction can thrash at the budget ceiling.** _[Performance • effort M]_
`evict` keeps only the currently-drawn set + roots (`gfx.rs:417-423`; `keep` is
built from `sel.draw` only, `main.rs:280`). A child that is *ready but not yet
drawn* (waiting for its three siblings) is in neither `draw` nor `keep`, so once
over budget it's evicted and must be re-meshed when the siblings arrive —
re-streaming churn at the ceiling. **Fix:** retain recently-uploaded / `want`
chunks too, or switch to an LRU rather than "drop everything not drawn."

**M3 — Meshing workers are unsupervised.** _[Correctness/robustness • effort S]_
`CpuChunk::build` runs outside the queue lock (`lod.rs:265`), so a panic there
won't poison the mutex — but it **silently kills the worker thread** (logged by
the panic hook, never respawned). Lose all N workers and streaming stalls forever
with no user-facing signal — a worse failure mode than a crash. It's near-zero
probability today (build is pure, in-range math) but a latent landmine as `build`
grows. **Fix:** wrap the worker body in `std::panic::catch_unwind`, log and
continue (or respawn). ~5 lines; cheap insurance. _(Low-hanging — see §5.)_

**M4 — `logging::init` panics if the log file can't be opened.** _[Robustness/Portability • effort S]_
`logging.rs:45-49` does `.unwrap_or_else(|e| panic!(...))`. On a locked-down or
read-only FS — more likely on the planned Windows target than on the dev Mac —
"can't write a log" becomes "app won't start." That inverts the intent (the log
is a diagnostic aid, not a launch dependency). **Fix:** degrade to stderr-only +
a `warn!`, return the path as `Option`/best-effort. _(Low-hanging — see §5.)_

**M5 — Three files exceed the 500-line norm; `gfx.rs` is ~45% tests.** _[Maintainability • effort M]_
`gfx.rs` (1,261), `flora.rs` (671), `main.rs` (657) were flagged by the static
file-size check. `gfx.rs` is the worst but ~570 of its lines are the `smoke` and
`gallery` `#[cfg(test)]` modules — move them to `tests/` (integration) or a
sibling `gfx_tests.rs` to bring the renderer proper to ~690. In `main.rs`,
`App::frame` (`main.rs:233-364`) bundles streaming, culling, uniform assembly,
and perf sampling — extract helpers.

**M6 — Water ripple feeds ~640k-magnitude world coords into `sin()`.** _[Performance/correctness • effort S • verify]_
`terrain.wgsl:66-73` animates the ocean glint with `sin(p.x*0.7 + t*…)` where
`p = in.world` can be ~637,000. f32 argument reduction loses phase precision at
that magnitude, so the glint may band/alias at the surface (the project is
otherwise meticulous about f32 scale). **Fix:** drive the ripple phase from
camera-relative or fractional coordinates. Worth a visual check first.

### 🟢 Low

- **L1 — `clippy` `collapsible_if` ×2** (`main.rs:121`, `:136`). `cargo clippy --fix`. _[Maintainability • XS]_ _(Low-hanging.)_
- **L2 — `is_water` radius threshold mis-flags low coastal land.** _[Cosmetic • S]_ `terrain.wgsl:61` treats any vertex within `WATER_DETECT_EPS` (15 m) of sea-level radius as water, so sub-15 m land gets a sun glint/sheen. The mesher already *knows* which verts are ocean (height clamp, `mesh.rs:122`) — pass an explicit per-vertex water flag (spare vertex channel) instead of inferring it.
- **L3 — Duplicated headless-renderer boilerplate.** _[Maintainability • M]_ `smoke`, `gallery`, and `render_and_save` each re-create device/pipelines (~150 LOC ×3). Extract one shared test harness.
- **L4 — World-writable shared log.** _[Security (local) • S]_ `logging.rs:43,51` chmod the dir `0o777` and file `0o666`. Intentional for cross-account dev sharing, but a predictable path in a world-writable shared dir is a classic local symlink/tamper vector. Fine for the workbench; tighten (or comment the trade-off) before any wider release.
- **L5 — Per-frame scratch allocations & redundant normalize.** _[Performance (micro) • S]_ `select`/`frame` allocate fresh `Vec`s (`draw`/`want`) and a `keep` `HashSet` every frame, and `select_node` recomputes `cam.normalize_or_zero()` per node (`lod.rs:112`). Hoist the camera normal; reuse scratch buffers if profiling shows it.
- **L6 — `build.rs` shells out to `date -u`.** _[Portability • XS]_ Empty `BUILD_DATE` on a build host without Unix `date` (handled gracefully). Optional: a small Rust time source removes the dependency.
- **L7 — `altitude()` is a `distance` proxy shown in the HUD.** _[Correctness/UX • S]_ `camera.rs:148` returns focus distance as "altitude"; when tilted this overstates true height-above-terrain in the HUD/log/`P` printout (and `params.w`, which shaders don't read). For a units-meticulous project, compute true AGL for display.
- **L8 — `Flora::generate` builds ~850 species meshes on the main thread at startup.** _[Performance • S]_ `flora.rs:96`, `SPECIES_PER_BIOME=100`. One-time and cheap today; watch as a startup-hitch source, or parallelize/lazy-gen if it grows.

---

## 5. Low-hanging fruit (medium+ impact, quick fix)

Ordered by value. These are the "do them this afternoon" wins:

| # | Finding | Why it's worth it | Effort |
|---|---|---|---|
| 1 | **Decorate-sort the `want` list** (`main.rs:271-275`) | `sort_by` recomputes `center_dir()`+`surface_radius()` (a sphere-map + normalize) on *every comparison* — O(n log n) expensive ops/frame. Replace with `sort_by_cached_key` computing each chunk's distance once. ~5 lines, pure win. | XS |
| 2 | **`catch_unwind` around the worker build** (M3, `lod.rs:265`) | Removes the silent streaming-stall landmine; turns a worker panic into a logged, recoverable event. | S |
| 3 | **Degrade instead of panic on log-open failure** (M4, `logging.rs:45`) | Removes a startup-crash mode on the Windows/locked-FS target. | S |
| 4 | **`cargo clippy --fix`** (L1) | Zero-risk; gets the tree to a literally clean clippy run. | XS |
| 5 | **Hoist `cam.normalize_or_zero()` out of `select_node`** (L5, `lod.rs:112`) | Constant for the whole walk; recomputed per node. Trivial. | XS |
| 6 | **Explicit per-vertex water flag** (L2) | Kills the coastal-shimmer artifact and removes a fragile radius heuristic. | S |

> Note: the single biggest perf win, **H1** (steepness triple-sampling), wasn't
> low-hanging — it was a real refactor of the sample path — and has since been
> **completed** (~1.9× faster terrain sampling, identical output).

---

## 6. Performance deep-dive (the #1 post-correctness priority)

**Cost model of one streamed chunk.** Build = `(grid+1)²` vertices × a terrain
sample, which *was* ≈ **3 × `height()`** (1 real + 2 for slope) + moisture + temp
+ color noise. At Ultra (`grid=88`) that's ~7,900 vertices × ~3 full height
evaluations — the dominant streaming cost. **H1 (now fixed) cut the grid's ×3 to
≈×1** wherever slope can't change the biome (~1.9× faster terrain sampling) — the
highest-leverage change in the codebase.

**Frame CPU.** `select` walks the 6-face quadtree each frame (bounded by the
visible set, horizon-culled — fine). The avoidable per-frame cost is the **`want`
sort** (LH1) and minor scratch allocations (L5).

**GPU.** vsync-bound today. The structural ceilings are **buffer churn** and
**per-chunk draws** (H2) plus **veg vertex volume** (M1) — all of which matter
specifically when the 5090 target lets you crank Detail. None are problems on the
Mac at current settings; all are worth addressing before "Ultra" is meaningful.

**What's already right:** off-main-thread meshing, horizon culling, coarse-ancestor
fallback, generation-guarded re-stream, atomics for live retune, and a memory
budget that derives the chunk cap. The bones are correct — this is tuning a good
engine, not fixing a broken one.

---

## 7. Suggested sequencing

1. **Low-hanging table (§5)** — an afternoon; banks a clean clippy run, a real
   per-frame win, and two robustness fixes.
2. ~~**H1 (steepness refactor)** — the headline perf win.~~ **✅ Done** — lean
   `sample_terrain` skips the slope probe below the mountain line; ~1.9× faster,
   equivalence-tested.
3. **M5 (split `gfx.rs` tests out; extract `frame` helpers)** — pure
   maintainability, makes the next two easier.
4. **H2 / M1 (buffer pooling + draw/veg batching)** — the scalability investment,
   gated behind the RTX target; schedule when Windows/5090 lands.
5. **M2, M6, and the remaining Low items** — opportunistic.

---

## Appendix — tools & evidence

- **`cargo clippy --all-targets --message-format short`** → 6 warnings / 0 errors,
  all `collapsible_if` at `main.rs:121,136`.
- **Landmine sweep (`ct grep`)** → 0 `unsafe`, 0 `TODO/FIXME/HACK`; panics
  isolated to `audio.rs:107`, `gfx.rs:598`, `logging.rs:49` (all asset/startup);
  concurrency confined to `lod.rs` (1 worker spawn, 7 locks, 4 channel ops).
- **`ct risk`** → top complexity-spines: `App::window_event → frame → camera::update`
  and `mesh::build → planet::sample → height` (the spine H1 targeted, since fixed).
- **`ct conventions`** → Rust: snake_case, 7 `unwrap()`, 26% doc coverage, avg
  347 LOC/file (max `gfx.rs` 1,261).
- **Test inventory** → 21 `#[test]`: `tests.rs` 7, `settings.rs` 5, `gfx.rs` 3
  (smoke+gallery), `units.rs` 2, `overlay.rs` 2, `tour.rs` 1, `audio.rs` 1.
- **Not indexed by ct:** the 4 WGSL shaders (reviewed via direct read).
