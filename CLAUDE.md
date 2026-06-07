# planet-generator

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

