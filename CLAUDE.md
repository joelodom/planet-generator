# planet-generator

## Architecture — north star — IMPORTANT

[`ARCHITECTURE_GUIDELINES.md`](ARCHITECTURE_GUIDELINES.md) is the standing
architectural standard — **read it before any non-trivial change** and hold new
code to it. It encodes the **priority hierarchy** that breaks every design tie:

> correctness → security → performance → maintainability → portability → polish

(higher wins on conflict; never trade a higher property away for a lower one
without a written reason). Headline rules: in runtime/library paths **degrade,
don't panic**; never repeat expensive work on a hot path (per-frame
`frame`/`select`/`render`, per-chunk `build`/`sample`); keep heavy generation off
the main thread and `gfx` a thin slab below the simulation; everything derives
deterministically from the seed. A point-in-time assessment against these
properties lives in [`ARCHITECTURE_REVIEW.md`](ARCHITECTURE_REVIEW.md).

## Running & deploying — IMPORTANT

**This repo is worked on over a headless SSH session to the `claude` macOS
account — there is no display here, so `cargo run` cannot open the app's window.**
The user runs the app from a **separate GUI account** on the same Mac. So
"produce a build the user can run/test" means **package it and drop it in the
shared folder** — never `cargo run`.

- **Build + deploy:** `./package_macos.sh`. It release-builds, assembles
  `dist/Planet Explorer.app`, **ad-hoc codesigns** it (so Gatekeeper doesn't flag
  it as "damaged" when launched from another account), makes it
  world-readable/executable, and copies it to **`/Users/Shared/Planet Explorer.app`**
  — the shared drop point the GUI account launches from. (There is no
  `/Users/Shared/Applications`; the `.app` lives directly in `/Users/Shared/`.)
- **Whenever the user will "run/test" a change, re-run `./package_macos.sh`** so
  the shared app reflects it. The bundle's version carries the git short hash with
  a `-dirty` suffix for uncommitted trees, so the user can confirm the build
  they're running (`".../planet-explorer" --version`, or the startup line in the
  log).
- **What you CAN verify headlessly here:** `cargo test` (the GPU smoke test runs
  if an adapter is present, else skips), `cargo clippy`, and `--version` (it exits
  before any window/GPU). Actual visuals are the user's to check from the GUI
  account; the shared log (see Logging) is how you observe that run from here.

## Coding conventions — IMPORTANT

**No magic numbers.** Tuning values and any non-obvious literal must be a
`const` with a `SCREAMING_SNAKE_CASE` name and a short comment on its
units/intent, hoisted near the top of its module (or the relevant `impl`). This
applies to Rust *and* the WGSL shaders (use `const NAME: T = ...;`). It matters
most for the tuning-heavy code: noise frequencies/weights and biome thresholds
(`planet.rs`), camera rates/clamps and tour phase durations/distances/speeds
(`camera.rs`/`tour.rs`), LOD factors (`lod.rs`), vegetation density/sizes and
skirt depth (`mesh.rs`), and lighting/fog/wave/atmosphere constants (shaders).

Obvious exceptions that may stay inline: identity/zero values (`0.0`, `1.0`),
halving/doubling (`0.5`, `2.0`), small structural integers (loop bounds, array
indices, `+ 1`), and standard math (`PI`, `255` for a byte). When in doubt, name
it. New code must follow this; don't reintroduce bare tuning literals.

## Target platforms — IMPORTANT

Developed on **macOS / Apple Silicon (Metal)**, but **will soon also run on
Windows with an RTX 5090 (DX12/Vulkan)**. Keep everything cross-platform as you
build:

- **GPU:** the wgpu instance enables `METAL | DX12 | VULKAN | GL` — don't assume
  Metal. The RTX 5090 has huge headroom, so LOD/draw budgets that are tuned for
  the Mac can be cranked up there later (gate aggressive defaults behind config,
  not hardcoded assumptions).
- **OS-specific bits must be `cfg`-gated.** Already done: the quit shortcut
  (Cmd-Q on macOS, Ctrl-Q elsewhere — see `main.rs`/`overlay.rs`) and the
  default log directory (`/Users/Shared/...` on macOS, OS temp dir elsewhere —
  see `logging.rs`). Never hardcode a `/Users/...` path in portable code.
- **Text rendering is intentionally engine-free and portable:** an embedded
  public-domain 8x8 bitmap font (`font8x8.rs`) drawn as instanced screen-space
  quads (`overlay.rs` + `shaders/overlay.wgsl`). No OS font APIs, no font files —
  it renders identically on every backend. Reuse this for any future on-screen
  text/HUD rather than pulling in a platform font stack.
- When adding features, prefer winit/wgpu abstractions over native calls; if you
  must go native, `cfg`-gate and provide a path for both macOS and Windows.

### Scale & precision — IMPORTANT

The planet is **Earth-sized**: `PLANET_RADIUS = 637_100` render units at
`METERS_PER_UNIT = 10` → 6,371 km. We render in "units" (10 m each), not real
metres, on purpose: it keeps absolute coordinates inside `f32`'s comfortable
range (~0.75 m precision at the surface) so the renderer needs **no
camera-relative / f64 path**. Heights carry a deliberate ~2x vertical
exaggeration (peaks ~+20 km) — at true Earth proportions an 8 km peak is only
0.13% of the radius and reads as flat, so the terrain is exaggerated (as Google
Earth / most renderers do) and the mountain noise is high-frequency enough to be
genuinely steep and snow-capped, while the planet still reads as huge.

Consequences to respect when changing things:
- Display real-world values via `src/units.rs` (`--units us` flag), never raw
  units, in any user-facing text (HUD title, `P` printout, perf log).
- Depth: a fixed far plane either clips the globe or z-fights near the ground, so
  `Camera::near_far` derives near/far from the eye's **horizon** each frame. Keep
  that if you touch projection.
- LOD floor: `MAX_LEVEL = 16` (~8 m quads) is about the f32 absolute-coordinate
  limit. Going finer (true sub-metre ground detail) would require camera-relative
  rendering (per-chunk origin + f64 chunk centers) — a deliberate future step,
  not a constant bump.

### Assets & audio

- Binary assets live in `assets/` and are **embedded** via `include_bytes!`
  (`assets/planet.png`, `assets/soundtrack.mp3`) so the app is self-contained and
  copyable between accounts. They're committed to the repo.
- **Audio** (`src/audio.rs`): `rodio` (cpal backend per-OS + pure-Rust symphonia
  decoders — cross-platform) plays an embedded **playlist** (`TRACKS`) — shuffled,
  played through, then reshuffled (never restarting on the just-played track).
  `App` calls `Audio::tick()` each frame to advance/reshuffle, and keeps the
  `Audio` handle alive. Add a song: drop an mp3 in `assets/` + one `TRACKS` line.
  Audio failures are non-fatal (runs silent).
- **planet.png** is also decoded (via `png`, pure Rust) into a texture and shown
  as a circular disc in the ESC help overlay (`shaders/image.wgsl`).
- **App icon:** `make_icon.py` (Pillow) builds a macOS squircle from planet.png;
  `package_macos.sh` copies `assets/AppIcon.icns` into the bundle. macOS-only —
  **a Windows build will need a `.ico`** (build script + `winresource`); track
  that when Windows lands.
- New crates must be cross-platform (rodio, png are). Verify before adding — the
  RTX 5090 / Windows target is coming.

<!-- ct-code-intelligence-start -->
## Code Intelligence — ct

This project is indexed by `ct`, an in-memory code intelligence daemon.
**PREFER ct over built-in Read, Grep, and Glob** — it's faster and returns
richer structural data (callers, callees, types, signatures).

**First thing every session:** run `ct --help` via the Bash tool. This is the
authoritative reference for all available commands and flags. ct evolves
frequently — do not assume you know what commands exist.

Run ct commands via the Bash tool (e.g. `ct lookup myFunc`, `ct grep "TODO"`,
`ct survey`). The CLI is always complete and current.

Fall back to built-in tools only for: binary/image files, files outside the
indexed project, or when the Edit tool requires a prior built-in `Read` call.
<!-- ct-code-intelligence-end -->

### Using ct on THIS project (Rust) — verified behavior

ct is enrolled and indexes all of `src/`. Prefer the `mcp__ct__ct_*` MCP tools
(structured JSON, no shell quoting) when available; the `ct <cmd>` CLI is the
same daemon. The notes below are empirically verified against this repo, not
copied from the help — Rust support here has specific quirks. Honor them.

**The one rule that matters most — methods are keyed `Type.method` (dot):**
- ✅ `lookup <bareName> --fuzzy` is the reliable way to fetch a method or any
  symbol you're unsure of. `ct lookup update --fuzzy` returns the full body,
  `package: Camera`, callers (`App.frame`) and callees.
- ❌ `lookup Type::method` / `callers Type::method` (colon form) return **no
  matches** here — never query the `Type::method` form. Use the bare name.
- A method/symbol lookup that "fails" almost always just needs `--fuzzy` or the
  bare name. Try that before falling back to Read/Grep.

**What works well — reach for these first:**
- **Orient:** `survey`, `map`, `themes`, `entry`; `outline <file>` to scan a
  file's symbols (with signatures + a forward `calls` list) without reading it.
- **Symbol:** `lookup <name>` (free fns: full body+doc+callers+callees; methods:
  add `--fuzzy`). `describe <Type>` is excellent — every field with its type,
  plus the impl locations (verified on `Planet`, `Camera`).
- **Find:** `grep` (exact/regex) · `search` (concept over bodies) ·
  `vsearch` (by name+docstring — best for "the function that does X").
- **Navigate:** `callers`/`callees`/`trace`/`spine` all work **by bare name**.
  `spine <fn>` gives a clean entry→leaf path; `references <name>` is the
  ground-truth impact list (e.g. 40 refs across 5 files for `ChunkKey`).
- **Edit in place (no Read+Edit):** `splice`, `delete-function`, `move-symbol`,
  `move-lines`, `extract-function` — always `--dry-run` first.

**Known false negatives — don't trust an empty result; cross-check:**
- `callers <freeFn>` can be **empty for a function only ever called
  module-path-qualified** (e.g. `planet::cube_to_sphere`). Verified: `callers
  cube_to_sphere` → none, but `references cube_to_sphere` finds the call sites.
  → When `callers` is empty, confirm with `references` before concluding it's dead.
- `path <a> <b>` is **unreliable here** (`path frame select` found no path though
  `frame` calls `select`). Prefer `spine`/`trace`/`references` to map flow.

**Memory & health:** `remember`/`recall`/`memories` persist across sessions;
`hot`/`risk`/`changed`/`owners`/`blame` for review targeting. `ct --help` is the
authoritative, current command list — re-check it if something seems missing.

## Logging (`src/logging.rs`)

The app uses **`tracing` + `tracing-subscriber`**. The whole point is remote
debugging: the app is usually launched on the user's GUI macOS account and we
(on the terminal-only account) read the log file to analyze bugs and perf. So
the log is the primary diagnostic artifact — keep it that way.

- **Where:** default `/Users/Shared/planet-explorer/planet-explorer.log`
  (world-readable so both accounts share it). Override with `$PLANET_LOG` (full
  path) or `$PLANET_LOG_DIR`. **Single file, always appended** across restarts —
  each run starts with an `INFO ... "planet-explorer starting"` line (seed,
  version, os/arch), so use that to find session boundaries.
- **Levels / default:** default filter is our crate at **DEBUG**, deps
  (`wgpu*`, `naga`, `winit`) at WARN to keep the log readable. Override with
  `RUST_LOG` (e.g. `RUST_LOG=planet_explorer=trace` for the per-frame/per-chunk
  firehose). Empty `RUST_LOG` is treated as unset (won't silence the log).
- **What's logged:** ERROR = failures + **panics** (a panic hook writes
  `target=panic` before the crash); WARN = recoverable trouble + frame hitches
  (`target=perf`, frames >120ms); INFO = lifecycle (startup, `gpu adapter
  selected`, teleport, location on `P`, exit); DEBUG = the **periodic perf
  sample** every ~2s (`target=perf`: `fps avg_ms max_ms alt lat lon biome chunks
  draw pending uploads`) + user actions (wireframe/free-look/speed/resize);
  TRACE = per-chunk uploads, per-frame skips.
- **Reading it:** `grep 'perf:' <log>` for the performance timeline;
  `grep -E 'WARN|ERROR|panic' <log>` for trouble; the `gpu adapter selected`
  line confirms which backend/GPU is in use.

When adding features, log at the right level: lifecycle=INFO, recurring
per-frame data=TRACE (never DEBUG — it floods), user-visible state changes=DEBUG,
anything that shouldn't happen=WARN/ERROR. Prefer structured fields
(`debug!(chunks = n, "...")`) over interpolated strings so the log stays greppable.

