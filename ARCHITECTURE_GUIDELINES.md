# Planet Explorer — Architectural Guidelines (North Star)

This is the **standing standard** every non-trivial change is held to. It is
prescriptive and timeless; the point-in-time assessment of how well the code
currently meets it lives in [`ARCHITECTURE_REVIEW.md`](ARCHITECTURE_REVIEW.md).
Read this before designing a feature, and self-review against it before merging.

It complements `CLAUDE.md` (which owns the no-magic-numbers rule, the scale/
precision model, portability `cfg`-gating, asset embedding, logging, and ct
usage) — those rules are not repeated here, only referenced.

---

## 0. The priority hierarchy — the tie-breaker for every decision

When two pulls conflict, the **higher property wins**:

> **correctness → security → performance → maintainability → portability → polish**

Two corollaries:

1. **Never trade a higher property away for a lower one without a written
   reason.** A faster path that can corrupt state, or a tidier abstraction that
   forces a hot-path allocation, is the wrong trade by default.
2. **Don't sacrifice a lower property gratuitously either.** The hierarchy breaks
   ties; it is not a license to write unmaintainable code "for performance" when
   no measurement says you must. Most of this codebase achieves all six at once —
   that is the bar.

Each section below maps to one dimension and says how to keep scoring well on it.

---

## 1. Correctness & determinism — *the floor nothing else justifies breaking*

- **The seed is the single source of truth.** Everything observable about a world
  is a pure, deterministic function of `(seed, coordinates)` — terrain, biome,
  color, vegetation, sun, sky. The same inputs yield the same output on any
  thread, on any machine. _Exemplars: `Planet` (immutable after construction,
  `Arc`-shared), `splitmix64` sub-seeding, `ChunkKey::hash`, `flora::mix`._
- **No wall-clock, no thread-local RNG, no ambient state in generation.** If a
  result depends on *when* or *which thread* it ran, it's a bug. Seed every RNG
  explicitly from the world seed + coordinates.
- **Shared state is immutable or atomic.** Cross-thread data is either read-only
  behind `Arc` (`Planet`, `Flora`) or explicitly atomic/locked with a documented
  protocol (`MeshConfig` atomics, `Streamer`'s generation guard). No surprise
  interior mutability.
- **Guard degenerate inputs; prefer total functions.** Clamp, `normalize_or_zero`,
  `length_squared` epsilon checks, fallback axes at the poles — this defensive
  style is already pervasive (`camera::view`, `planet::tangent_basis`,
  `tour::slerp_dir`). Keep it. A surface sample or a camera update must never
  produce `NaN`.
- **Invariants are tested, not assumed.** New generation/maths ships with a
  determinism or bounds test (see §6). _Exemplars: `tests.rs`
  (`generation_is_deterministic`, `height_is_finite_and_bounded`), the tour
  smoothness test._

## 2. Robustness — *degrade, don't panic*

The app's job is to keep running and stay diagnosable. Crashing is a last resort.

- **Runtime/library code must not `panic!`/`unwrap`/`expect` on conditions the
  outside world can cause** — filesystem, audio/GPU devices, env vars, CLI args,
  surface loss. These **log a `warn!`/`error!` and degrade**. _Exemplars to copy:
  `Audio::start` (no device → `None`, runs silent), `Renderer::new` (no sRGB
  format → fall back), `supports_wireframe` feature-gating, `render()`'s
  surface-lost reconfigure._
- **`panic!`/`expect` is allowed only for:** (a) genuine programmer-invariant
  violations that mean *the code is wrong* (not the environment), and (b) decode
  of **compile-time-embedded** assets (`include_bytes!`), whose shape is fixed at
  build time. _Anti-pattern (see review M4): panicking because a **log file**
  can't be opened — that's an environment condition; degrade to stderr instead._
- **Background threads are supervised.** A worker must never silently vanish.
  Isolate its unit of work in `catch_unwind`, log the failure, and continue or
  respawn — one bad job must not kill a subsystem. _(Review M3: the meshing
  workers currently lack this.)_
- **Absence is a degraded mode, not death:** no audio device, no wireframe
  feature, no GPU adapter (tests skip cleanly) — all run, just with less.

## 3. Security & least surface

- **No `unsafe`** without a written safety justification and review. The current
  tree has zero; keep it that way.
- **Treat all external input as untrusted** — CLI args, env vars, and any future
  files/network. Parse with fallbacks, never panic. _Exemplars: `parse_seed`,
  `parse_units`, `env_filter` (empty `RUST_LOG` treated as unset)._
- **Least privilege.** Don't widen file permissions or use predictable paths in
  shared/world-writable locations beyond what a feature needs; if you must (e.g.
  the cross-account log), **document the trade-off** at the call site.
- **No network, telemetry, or phone-home** without an explicit, opt-in design
  decision. The app is self-contained by intent.

## 4. Performance — *the hot path is sacred* (top priority after correctness/security)

Know the two hot paths and treat them as load-bearing:

- **Per-frame:** `App::frame` → `lod::select` → `Renderer::render`.
- **Per-chunk (worker):** `CpuChunk::build` → `Planet::sample` → `Planet::height`.

Rules:

- **Never repeat expensive work.** Compute once, reuse. No expensive recomputation
  inside sort comparators (decorate/cache the key); hoist loop-invariants out of
  recursion; and **don't compute what the result can't depend on** (H1: terrain
  meshing skips the slope probe wherever it can't change the biome). Noise/`height()`
  is the most expensive primitive — budget
  it deliberately.
- **Budget allocations on hot paths.** Prefer reusable scratch buffers over
  per-frame / per-node heap churn. A `Vec`/`HashSet` allocated every frame is a
  code smell unless measured negligible.
- **Heavy generation stays off the main thread.** The main thread renders and
  feeds the GPU; it never blocks on geometry. New heavy producers go through (or
  alongside) the `Streamer` pool, never inline in `frame()`. The six root chunks
  built inline at startup/rebuild are the *only* sanctioned exception (so there's
  never a blank frame).
- **GPU resources: reuse and batch.** Pool/suballocate buffers rather than
  allocate-per-chunk-per-upload; batch or instance draws rather than one call per
  object. Mind churn from eviction/re-streaming. _(Review H2/M1.)_
- **Scale up behind config, not hardcoded assumptions.** Defaults that suit the
  Mac live in `settings.rs` and crank to the RTX-5090 "Ultra" target via the
  Detail/memory knobs. Don't bake a single device's budget into the code.
- **Measure before and after.** Use the perf log (`target=perf`) and `ct risk` to
  find hotspots; guard every optimization with a test or benchmark so you don't
  trade correctness for speed.

## 5. Module boundaries & dependency direction

- **One-way dependencies.** `planet` is the source of truth that everything reads;
  it depends on nothing above it. New systems (animals, weather, NPCs) **query
  `Planet` for ground truth** and must not reach sideways into rendering or each
  other.
- **`gfx` stays a thin, replaceable slab.** It consumes `CpuChunk`s + a draw list
  and knows nothing about LOD policy or planet maths. Keep rendering free of
  simulation logic.
- **The CPU/GPU seam is explicit.** Plain CPU data (`glam`, `MeshData`,
  `CpuChunk`) is produced by simulation; GPU types (`wgpu::Buffer`, `Globals`)
  live only in `gfx`. Don't leak `wgpu` types upward.

## 6. Maintainability

- **Name every tuning value** (see `CLAUDE.md` → no magic numbers) — Rust *and*
  WGSL. Comment the **why/units/intent**, not the obvious.
- **Files target < ~500 lines of non-test code**; split when they grow. Move large
  `#[cfg(test)]` modules into `tests/` or a sibling file rather than letting them
  inflate a source file. _(Review M5.)_
- **Functions do one thing.** When a method spans several concerns (streaming +
  culling + uniforms + perf sampling, à la `frame`), extract helpers.
- **Every module carries an intent doc header** explaining what it owns and why —
  this is a hard expectation, not a nicety. Match the comment density and idiom of
  the surrounding module.
- **Keep `cargo clippy` clean** and introduce no new `TODO`/`FIXME` without a
  tracked follow-up. The tree currently has zero debt markers.
- **Markdown docs are `ALL_CAPS.md`** (`README.md`, `BACKLOG.md`,
  `ARCHITECTURE_REVIEW.md`, …) — SCREAMING_SNAKE_CASE filenames; the only exceptions
  are tool-generated files we don't own (`.claude/scan-report*.md`).

## 7. Portability

Cross-platform is a standing constraint, not a port-later task (see `CLAUDE.md` →
Target platforms). In short: enable backends broadly, prefer winit/wgpu
abstractions over native calls, `cfg`-gate every OS-specific bit, keep
`required_limits` conservative, use only pure-Rust cross-platform crates, and gate
aggressive GPU defaults behind config. Anything that can fail per-platform
(paths, devices) follows §2 — degrade, don't panic.

## 8. Scale & precision

Honor the f32-in-10 m-units model (see `CLAUDE.md` → Scale & precision). One
addition surfaced by review: **f32 is safe for absolute *position*, but be wary
of feeding large absolute coordinates into transcendental/animation math** (e.g.
`sin(world_pos * k)` at ~640k magnitude loses phase precision). Drive periodic
effects from camera-relative or fractional coordinates. _(Review M6.)_

## 9. Testing & observability

- **New subsystems ship with tests.** At minimum: a determinism/invariant test for
  generation maths, and — for anything touching the GPU — a **headless validation
  test inside a `wgpu` error scope that skips cleanly when no adapter is present**
  (copy `gfx::smoke::offscreen_pipeline_validates`).
- **Log at the right level, with structured fields.** lifecycle = `INFO`,
  recurring per-frame data = `TRACE` (never `DEBUG` — it floods), user-visible
  state changes = `DEBUG`, recoverable trouble/hitches = `WARN`, failures/panics =
  `ERROR`. The log file is the primary remote diagnostic — keep it greppable
  (`debug!(field = x, "msg")`, not interpolated strings). See `CLAUDE.md` →
  Logging.

---

## 10. Pre-merge checklist

Before merging a non-trivial change, confirm:

- [ ] **Correctness:** deterministic from the seed; no `NaN`/degenerate path; new
      maths has an invariant/determinism test.
- [ ] **Robustness:** no runtime `unwrap`/`expect`/`panic` on environment
      conditions; new threads are `catch_unwind`-supervised.
- [ ] **Security:** no `unsafe` (or justified + reviewed); external input parsed
      with fallbacks; no privilege/path widening.
- [ ] **Performance:** no new expensive work on a per-frame/per-chunk path; no
      avoidable hot-path allocation; heavy work is off the main thread; checked
      `ct risk`/perf log if it touches a hotspot.
- [ ] **Boundaries:** dependency direction preserved; no `wgpu` types leaked above
      `gfx`.
- [ ] **Maintainability:** tuning values named; file/function didn't balloon past
      the size norms; module doc header present; `cargo clippy` clean.
- [ ] **Portability:** OS-specific bits `cfg`-gated; only cross-platform crates;
      device/path failures degrade.

---

_When this document and an expedient shortcut disagree, this document wins — or
the shortcut comes with a comment explaining which higher property justified it._
