#!/usr/bin/env python3
"""
evolve.py — Autonomous Photorealism Optimizer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Drop this file into the root of your procedural 3D object generator project
and run:   python evolve.py

What it does:
  1. Scans the project to build context
  2. Launches Claude Code (headless) to set up the optimization infrastructure:
       • Adds a headless render-to-image CLI flag to the project
       • Creates tree_optimizer.py — the AI micro-optimization loop
       • Creates OPTIMIZATION_NOTES.md with documented strategy
  3. Owns and supervises tree_optimizer.py; restarts it automatically if it crashes
  4. Every 30 minutes, pauses the loop and runs a Claude Code strategic review,
     then restarts the loop
  5. If Claude Code improves THIS script, relaunches it automatically
  6. Monitors token usage and, if the 5-hour budget passes 95% or the weekly
     budget passes 90%, comes to a stopping point and WAITS for the budget to
     refresh, then resumes on its own — so you can truly walk away

Auth — NO API KEY NEEDED:
  Every model call (the bootstrap/review sessions AND the inner optimization
  loop) goes through the `claude` CLI, which uses your logged-in Claude Code
  subscription / OAuth token. To make that guarantee airtight, this script
  scrubs ANTHROPIC_API_KEY from every Claude Code subprocess's environment (see
  FORCE_SUBSCRIPTION_AUTH) so it can never silently fall back to metered API
  billing. Just make sure you're logged in first:  `claude` (then /login).

Requirements:
  • Claude Code installed & logged in:  npm install -g @anthropic-ai/claude-code
  • Python 3.8+   (no `anthropic` SDK, no API key)

Resumable: state is saved to evolve_state.json — re-run any time to continue.
"""

import datetime
import glob
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT     = Path(__file__).resolve().parent
SCRIPT_PATH      = Path(__file__).resolve()
STATE_FILE       = PROJECT_ROOT / "evolve_state.json"
LOG_FILE         = PROJECT_ROOT / "evolve_log.txt"
OPTIMIZER_SCRIPT = PROJECT_ROOT / "tree_optimizer.py"
# We deliberately do NOT touch the project's own CLAUDE.md (it may hold
# hand-authored project conventions). Our auto-generated context goes here, and
# the prompts tell Claude Code to read both files.
EVOLVE_CONTEXT   = PROJECT_ROOT / "EVOLVE_CONTEXT.md"
NOTES_FILE       = PROJECT_ROOT / "OPTIMIZATION_NOTES.md"  # living running-notes journal

BOOTSTRAP_TIMEOUT = 7200   # 2 hours — Claude Code sets up all infrastructure
REVIEW_TIMEOUT    = 3600   # 1 hour  — periodic strategic review
REVIEW_INTERVAL   = 1800   # run a review every 30 minutes
MONITOR_INTERVAL  = 60     # health-check the optimizer every 60 seconds

# Model alias passed to `claude --model` for every session (outer + inner loop).
CLAUDE_MODEL      = "opus"

# Force Claude Code to authenticate via the logged-in subscription / OAuth token
# by scrubbing ANTHROPIC_API_KEY from each child process's environment. This is
# what makes "no API key needed" a guarantee rather than a hope: even if a key
# is present in your shell, the model calls won't silently bill the metered API.
FORCE_SUBSCRIPTION_AUTH = True

# ── Token-budget monitoring ──────────────────────────────────────────────────
# Reuses the claude-monitor technique (github.com/joelodom/claude-monitor): read
# the Claude Code OAuth token and query the same usage endpoint that powers
# `/usage`. When a window crosses its stop threshold we come to a stopping point,
# stop the optimizer, and WAIT until the budget refreshes before resuming — so an
# unattended run never blows through your limits. Querying usage costs no tokens.
USAGE_URL          = "https://api.anthropic.com/api/oauth/usage"
USAGE_HTTP_HEADERS = {
    "anthropic-version": "2023-06-01",
    "anthropic-beta":    "oauth-2025-04-20",
    "User-Agent":        "evolve.py",
}
# macOS Keychain services that may hold the Claude Code credentials (tried in
# order); off macOS we fall back to the on-disk credentials file.
KEYCHAIN_SERVICES  = ("Claude Code-credentials", "Claude Code", "claude.ai")
CREDENTIALS_FILE   = Path.home() / ".claude" / ".credentials.json"

# Pause-and-wait thresholds (utilization is a 0–100 percentage from the API).
FIVE_HOUR_STOP_PCT = 95.0   # 5-hour rolling session budget
WEEKLY_STOP_PCT    = 90.0   # 7-day budget (all-models, and per-model caps)
# Windows we enforce → their stop threshold. Missing/null windows are ignored.
ENFORCED_WINDOWS   = {
    "five_hour":      FIVE_HOUR_STOP_PCT,
    "seven_day":      WEEKLY_STOP_PCT,
    "seven_day_opus": WEEKLY_STOP_PCT,
}
WINDOW_SHORT_NAME  = {
    "five_hour": "5h", "seven_day": "7d", "seven_day_opus": "7d-opus",
}
BUDGET_POLL_SECONDS  = 300  # while paused, re-check at least this often (heartbeat)
BUDGET_RESUME_BUFFER = 60   # wait this long past a reported reset before resuming
STATUS_LOG_SECONDS   = 300  # throttle the idle "still alive" status line to this


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def log(msg: str, level: str = "INFO") -> None:
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            log(f"Could not load state ({e}), starting fresh.", "WARN")
    return {
        "phase":         "bootstrap",
        "session":       0,
        "best_score":    0.0,
        "optimizer_pid": None,
        "last_review":   None,
        "history":       [],
    }


def save_state(state: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as e:
        log(f"Could not save state: {e}", "ERROR")


# ──────────────────────────────────────────────────────────────────────────────
# Self-modification detection
# ──────────────────────────────────────────────────────────────────────────────

def file_hash(path: Path) -> str:
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except Exception:
        return ""


def relaunch_if_modified(original_hash: str, state: dict) -> None:
    if file_hash(SCRIPT_PATH) != original_hash and original_hash:
        log("evolve.py was modified — relaunching with updated script...")
        save_state(state)
        time.sleep(1)
        os.execv(sys.executable, [sys.executable, str(SCRIPT_PATH)] + sys.argv[1:])


# ──────────────────────────────────────────────────────────────────────────────
# Project scanner  (writes EVOLVE_CONTEXT.md — NEVER touches the project CLAUDE.md)
# ──────────────────────────────────────────────────────────────────────────────

def scan_project() -> str:
    skip = {
        ".git", ".svn", ".hg", "__pycache__", "node_modules", ".DS_Store",
        "build", "dist", "out", "target", ".cache", ".idea", ".vscode",
        "snapshots", "renders",
    }

    lines = [f"Project root: {PROJECT_ROOT}", ""]

    # Top-level structure
    lines.append("Top-level contents:")
    for p in sorted(PROJECT_ROOT.iterdir()):
        if p.name not in skip and not p.name.startswith("."):
            tag = "[dir] " if p.is_dir() else "[file]"
            lines.append(f"  {tag} {p.name}")
    lines.append("")

    # Build system detection
    build_patterns = [
        "CMakeLists.txt", "Makefile", "makefile", "GNUmakefile",
        "*.sln", "*.vcxproj", "setup.py", "pyproject.toml",
        "package.json", "Cargo.toml", "build.gradle", "pom.xml",
        "SConstruct", "meson.build", "*.cabal",
    ]
    found_build = []
    for pat in build_patterns:
        found_build.extend(f.name for f in PROJECT_ROOT.glob(pat))
    if found_build:
        lines.append(f"Build system files detected: {', '.join(found_build)}")
        lines.append("")

    # Source file counts + tree-related file detection
    ext_labels = {
        ".cpp": "C++", ".cxx": "C++", ".cc": "C++",
        ".c":   "C",
        ".h":   "C/C++ header", ".hpp": "C++ header",
        ".py":  "Python",   ".rs": "Rust",
        ".cs":  "C#",       ".java": "Java",
        ".js":  "JavaScript", ".ts": "TypeScript",
        ".go":  "Go",       ".swift": "Swift",
        ".hlsl": "HLSL shader", ".glsl": "GLSL shader",
        ".frag": "Fragment shader", ".vert": "Vertex shader",
    }
    counts: dict = {}
    tree_files = []
    tree_kws = {
        "tree", "branch", "leaf", "foliage", "bark",
        "vegetation", "plant", "flora", "lod", "forest",
        "trunk", "twig", "canopy", "frond",
    }

    for f in PROJECT_ROOT.rglob("*"):
        if not f.is_file():
            continue
        if any(part in skip or part.startswith(".") for part in f.parts):
            continue
        ext = f.suffix.lower()
        if ext in ext_labels:
            lang = ext_labels[ext]
            counts[lang] = counts.get(lang, 0) + 1
        if any(kw in f.stem.lower() for kw in tree_kws):
            try:
                tree_files.append(str(f.relative_to(PROJECT_ROOT)))
            except ValueError:
                tree_files.append(str(f))

    if counts:
        lines.append("Source file counts by language:")
        for lang, n in sorted(counts.items()):
            lines.append(f"  {lang}: {n} file{'s' if n != 1 else ''}")
        lines.append("")

    if tree_files:
        lines.append("Files likely related to tree/vegetation generation:")
        for tf in tree_files[:25]:
            lines.append(f"  {tf}")
        if len(tree_files) > 25:
            lines.append(f"  … and {len(tree_files) - 25} more")
        lines.append("")

    context = "\n".join(lines)

    # Write EVOLVE_CONTEXT.md so Claude Code can pick up context. We do NOT
    # overwrite the project's own CLAUDE.md — it may contain hand-authored
    # conventions we must preserve. The prompts point Claude at both files.
    _write_evolve_context(context)

    return context


def _write_evolve_context(context: str) -> None:
    """Write/update EVOLVE_CONTEXT.md — referenced explicitly by our prompts."""
    content = f"""# Autonomous Photorealism Optimizer — Context (auto-generated by evolve.py)

This file is written by **evolve.py**. Do not hand-edit — it is regenerated each
run. The project's own `CLAUDE.md` (if present) is authoritative for project
conventions; always read it too and respect it.

evolve.py is an autonomous optimization system that uses Claude Code to
iteratively improve the photorealism of procedurally generated 3D objects.
Currently focused on **trees**; future objects: rocks, vegetation, terrain.

## What You Are Doing

You have been invoked by evolve.py to either:
- **Bootstrap**: build the optimization infrastructure (first run), OR
- **Review**: inspect progress and make strategic improvements (subsequent runs)

Read the specific task list in the prompt carefully.

## Auth — No API key

Every model call goes through the `claude` CLI on the logged-in subscription /
OAuth token. The inner loop (`tree_optimizer.py`) MUST also call the `claude`
CLI in headless `-p` mode — never the Anthropic SDK and never ANTHROPIC_API_KEY.

## Resilience — keep running notes

This system runs for hours and may be interrupted (Ctrl-C, crash, reboot) at any
moment, then resumed with `python evolve.py`. So treat `OPTIMIZATION_NOTES.md` as
a durable running journal: write down what you are about to do BEFORE you do it,
and the outcome AFTER, so a fresh session can read it and continue seamlessly
without repeating work. Never leave the project in a half-broken state — keep the
build green so an interrupted run can always resume from something that compiles.

## Evolve System Files

| File | Purpose |
|------|---------|
| `evolve.py` | Orchestrator (this launched you) — you may update it |
| `tree_optimizer.py` | AI micro-optimization loop (you create/maintain this) |
| `OPTIMIZATION_NOTES.md` | Living strategy doc + running journal (you maintain this) |
| `EVOLVE_CONTEXT.md` | This file — auto-generated context |
| `evolve_state.json` | Orchestrator state |
| `optimizer_state.json` | Inner loop state |
| `evolve_log.txt` | Orchestrator log |
| `optimizer_log.txt` | Per-iteration optimization log |
| `snapshots/best/` | Source snapshot of best-scoring version |
| `renders/best/` | Renders of best-scoring version |

## Project Scan (auto-generated)

{context}
"""
    try:
        EVOLVE_CONTEXT.write_text(content, encoding="utf-8")
    except Exception as e:
        log(f"Could not write EVOLVE_CONTEXT.md: {e}", "WARN")


# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

BOOTSTRAP_PROMPT = """\
You have been dropped into a procedural 3D object generator project. Your mission
is to build an autonomous AI optimization infrastructure that will continuously
improve the photorealism of generated objects — starting with trees.

This system will run unattended for hours. You are building an engine, not a
one-shot fix. Trees come first; the system is designed to eventually cover rocks,
ground vegetation, terrain features, and other natural objects.

It can be interrupted at ANY time (Ctrl-C, crash, reboot) and resumed later, so
build for resumability: keep the build green, persist state to disk, and keep a
running journal in OPTIMIZATION_NOTES.md (what you are about to do, then the
outcome) so a fresh session can pick up exactly where this one left off.

TOKEN BUDGET: evolve.py monitors usage and will STOP the optimizer when the
5-hour or weekly budget is nearly exhausted, then restart it after the budget
refreshes. So design tree_optimizer.py to be cheap per iteration and safe to
stop at any point — every iteration must end in a green, saved, revertible
state. Do NOT add your own budget logic; evolve.py owns that.

Read CLAUDE.md (project conventions — authoritative, respect it) AND
EVOLVE_CONTEXT.md (this system's context) in this directory before doing
anything else. Do not modify or overwrite CLAUDE.md.

AUTH — IMPORTANT: This whole system runs on a Claude Code subscription with NO
API key. Do NOT use the Anthropic Python SDK, do NOT import `anthropic`, and do
NOT read ANTHROPIC_API_KEY anywhere. Every model call — including the inner
optimization loop's image scoring and code editing — must go through the
`claude` command-line tool in headless mode (see Task 4).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLETE ALL TASKS IN ORDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK 1 — EXPLORE THE CODEBASE
Thoroughly read the relevant source files. You must understand:
  • Language(s), build system, exact commands to compile and run
  • Where tree/vegetation generation lives: geometry, branching algorithm,
    bark texture or shading, leaf/foliage placement, level-of-detail
  • How the program currently renders or outputs geometry
  • Existing CLI parameters
Do not guess or infer — actually read the files.

TASK 2 — ADD HEADLESS RENDER-TO-IMAGE OUTPUT
Add a CLI mode that renders a named procedural object to a PNG without
opening any window. The interface to implement:

    <program> --render-to-image --object tree --seed 42 \\
              --yaw 45 --pitch 30 --output render.png \\
              --width 512 --height 512

  --object   object type identifier (tree, rock, bush, …)
  --seed     integer seed for reproducible procedural generation
  --yaw      camera horizontal angle in degrees
  --pitch    camera vertical angle in degrees
  --output   output PNG path
  --width/--height  image size (default 512)

Implementation by tech stack (choose the right one for this project):
  C++/OpenGL  →  EGL offscreen context, or OSMesa software renderer
  Python      →  trimesh + pyrender,  open3d,  or pyvista
  No renderer →  export geometry to OBJ/PLY/glTF, render via Python helper
  Unreal/Unity →  command-line render target or offscreen headless mode

Run a test render after implementing and verify a valid, non-blank PNG is
produced before moving on to Task 3.

TASK 3 — CREATE OPTIMIZATION_NOTES.md
A living document in the project root. Write it thoughtfully — it guides
all future optimization. Include these exact sections, with Executive Summary
FIRST so a human can review status at a glance:

  ## Executive Summary
  ALWAYS the very first section, kept current. 5–10 lines, plain language, for a
  human skimming progress: current best photorealism score, whether the loop is
  running or paused (and why, e.g. waiting on token budget), the one-sentence
  current strategy, the most recent change and whether it helped, and the single
  next step. Update this every time you touch the file.

  ## Project Architecture
  Key files, data flow, how the tree generation pipeline works end-to-end.

  ## Build & Render Commands
  Exact shell commands to build the project and produce a render.

  ## Current Tree Quality Assessment
  Your honest appraisal of what makes the current trees look non-photorealistic.
  Be specific: uniform branching angles? flat shading? no bark texture variation?
  perfect bilateral symmetry? missing secondary detail? no subsurface scattering?

  ## Phase 1 Optimization Strategy
  What to improve first and why. Name the specific algorithmic changes that
  will have the most visible impact on photorealism. Be concrete.

  ## Phase 2+ Strategy
  What comes after Phase 1 for trees, and then the roadmap for:
  rocks and stones, ground vegetation (grass/ferns/bushes),
  terrain surface detail, and other natural environment objects.

  ## Assumptions
  Everything you inferred or could not verify directly from reading the code.

  ## Running Journal  (resume-from-here)
  A reverse-chronological log of sessions for crash/interrupt recovery. Append a
  dated entry whenever you start or finish meaningful work: what you were doing,
  what state things are in, and the single most useful "next step" so a fresh
  session can resume instantly. Keep the newest entry at the top. Write the
  "about to do X" line BEFORE doing X, and the outcome AFTER — that way an
  interruption mid-task still leaves a breadcrumb.

  ## Optimization Log
  (Leave blank — tree_optimizer.py will append per-iteration entries here.)

TASK 4 — CREATE tree_optimizer.py
The autonomous micro-optimization loop. It runs the model ONLY by shelling out
to the `claude` command-line tool in headless mode — NO Anthropic SDK, NO
ANTHROPIC_API_KEY. Runs indefinitely until interrupted.

  HOW TO CALL THE MODEL — a helper, used for BOTH scoring and code edits:

    import os, subprocess
    CLAUDE_MODEL = "opus"   # alias → latest Opus; uses the subscription login

    def claude_env():
        # Force the logged-in subscription / OAuth token; never the metered API.
        env = dict(os.environ)
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
        return env

    def run_claude(prompt, timeout):
        # Headless one-shot. Claude Code can Read PNGs (vision) and Edit files
        # itself, so we let it do the work directly rather than parsing replies.
        cmd = ["claude", "-p", prompt,
               "--model", CLAUDE_MODEL,
               "--dangerously-skip-permissions"]
        return subprocess.run(cmd, cwd=".", env=claude_env(), timeout=timeout,
                              capture_output=True, text=True)

  Find the `claude` binary the same way evolve.py does (PATH, then ~/.local/bin,
  /opt/homebrew/bin, etc.) and fail loudly with a clear message if it is missing.

  STATE FILE: optimizer_state.json
    {
      "iteration": 0,
      "best_score": 0.0,
      "best_snapshot": "snapshots/best/",
      "stagnation_count": 0,
      "current_strategy": "<your strategy from Tasks 1 & 3>",
      "tried_changes": [],
      "tree_source_files": ["<files found in Task 1>"],
      "build_command": "<exact build command>",
      "render_command": "<exact render command from Task 2>",
      "history": []
    }
  On startup: if optimizer_state.json exists, load it and resume from the best
  known state (restore files from snapshots/best/ if it exists). Make every step
  resumable — assume the process can die at any instant and be re-run.

  LOOP — repeat forever:

  1. RENDER
     Produce 4 PNGs at varied random yaw, pitch, and seed values.
     Save as renders/current/render_0.png … render_3.png.

  2. SCORE  (one headless `claude` session — it reads the images itself)
     Call run_claude(...) with a prompt that tells it to READ the four files
     renders/current/render_0.png … render_3.png and rate each on photorealism:
       "Read these four PNGs of a procedurally generated tree and rate EACH on
        photorealism from 0 to 100:
        0–20 = 1990s video-game quality; 21–40 = basic 3D, obviously synthetic;
        41–60 = decent CGI; 61–80 = good modern game quality;
        81–100 = photorealistic, indistinguishable from a photograph.
        Write ONLY valid JSON (no markdown fences) to the file
        renders/current/scores.json:
        {\\"scores\\": [n0,n1,n2,n3],
          \\"critiques\\": [\\"x\\",\\"y\\",\\"z\\"],
          \\"top_improvement\\": \\"<single most impactful algorithmic change>\\"}"
     Then read renders/current/scores.json. composite_score = mean of scores.
     If the file is missing/invalid, retry once, then skip the iteration.

  3. MODIFY  (one headless `claude` session — it edits the files directly)
     Call run_claude(...) with a prompt that includes:
       • The list of tree_source_files (paths) — let Claude Read them itself
       • composite_score + the critiques and top_improvement from scores.json
       • current_strategy and tried_changes from state
       • Last 5 entries from history
     Instruct it to make ONE specific, surgical change directly via its Edit
     tool that targets the most impactful issue, keep the change minimal, and
     write a one-line summary of what it changed to renders/current/change.txt.
     Read change.txt and append it to tried_changes.
     (Because Claude edits in place, snapshot BEFORE this step — see step 4.)

  4. SNAPSHOT-THEN-APPLY ordering
     BEFORE step 3, back up all tree_source_files to snapshots/iteration_N/.
     Step 3 applies the change in place. This ordering is what makes revert
     in step 7 possible.

  5. BUILD
     Run the build command. Capture stdout+stderr.
     On failure: restore from snapshots/iteration_N/, log BUILD_FAILED, and run
     another MODIFY session noting "previous change caused a build error, here
     is the compiler output, try a different approach." After 3 consecutive
     build failures, do a mini-strategy review before retrying. NEVER leave the
     tree broken — a reverted/green tree must be the resting state of every
     iteration so an interrupt can always resume from something that compiles.

  6. RE-RENDER & RE-SCORE
     Same angles and seeds as step 1. Same scoring method (step 2).

  7. COMPARE
     new_score > best_score:
       copy current files → snapshots/best/
       copy renders       → renders/best/
       update best_score, reset stagnation_count = 0
       log: IMPROVEMENT  (old_score → new_score, +delta)
     new_score ≤ best_score:
       restore tree_source_files from snapshots/iteration_N/
       stagnation_count += 1
       log: NO_IMPROVEMENT  (delta, description of what was tried)

  8. STAGNATION CHECK  (stagnation_count ≥ 5)
     Run a `claude` session with the last 10 history entries and ask:
       "My photorealism optimization strategy has stalled. Here are the recent
        attempts. Propose a fundamentally different strategy — a new algorithmic
        aspect to focus on, or a completely different approach. Be specific and
        actionable. Write it to renders/current/strategy.txt."
     Read strategy.txt → current_strategy, clear tried_changes, reset
     stagnation_count = 0. Log: STRATEGY_UPDATE with the new strategy.

  9. LOG & ITERATE
     Append to optimizer_log.txt:
       [TIMESTAMP] ITER N | score: X.X → Y.Y (Δ+/-Z.Z) | KEPT/REVERTED/FAILED
       Tried: <one-line description>
       Strategy: <first 100 chars of current_strategy>
     Also append a short dated line to the Running Journal in
     OPTIMIZATION_NOTES.md (newest at top) so progress survives interruption,
     and refresh the Executive Summary at the top (best score, last change and
     whether it helped, next step).
     Trim history to last 50 entries. Save state after EVERY iteration (write to
     a temp file and rename, so a crash mid-write can't corrupt it). Sleep 2s.

  ERROR HANDLING (subprocess-based — no anthropic exceptions exist here)
    `claude` returns a non-zero exit code or times out:
      log a warning, sleep 30s, retry the same step up to 3×, then skip the
      iteration and continue. Do not exit the loop for a single failed session.
    scores.json / change.txt / strategy.txt missing or invalid:
      retry the session once, else skip the iteration.
    KeyboardInterrupt:
      save state, print summary (iterations, best score), sys.exit(0)
    Build failure:
      revert files, log BUILD_FAILED, continue loop
    Any other Exception:
      log full traceback, revert any pending change, continue loop

TASK 5 — INSTALL, VERIFY (DO NOT LEAVE IT RUNNING)
  Install only what the RENDER step needs (e.g. `pip install Pillow` if you use
  it). Do NOT install `anthropic` and do NOT require ANTHROPIC_API_KEY.
  Confirm the `claude` CLI is on PATH and that a trivial `claude -p "say OK"`
  session succeeds (this proves subscription auth works) — if not, stop and
  write the problem to OPTIMIZATION_NOTES.md.
  Run one dry render to confirm the render pipeline works end-to-end, then run
  ONE full iteration of tree_optimizer.py to prove the loop works
  (render → score → modify → build → re-score → compare → log).
  IMPORTANT: do NOT leave tree_optimizer.py running in the background. evolve.py
  owns the optimizer process lifecycle and will launch and supervise it after
  you finish. Just verify it works, then stop it.

TASK 6 — OPTIONALLY UPDATE evolve.py
  evolve.py is the orchestrator that launched you. It lives at the project root.
  If you find improvements — better prompts, smarter project scanning, better
  phase or review logic — edit it. It detects modifications and relaunches
  itself automatically, so your improvements take effect on the next cycle.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PERMISSIONS: Read, create, modify, or delete any file. Run any shell command.
Install Python packages with pip. Make concrete, working changes — not outlines,
not stubs. Every task should leave the project in a measurably better state.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def build_review_prompt(state: dict) -> str:
    # Pull best score from optimizer_state.json if it's been updated
    opt_state_path = PROJECT_ROOT / "optimizer_state.json"
    best_score = state.get("best_score", 0.0)
    if opt_state_path.exists():
        try:
            opt = json.loads(opt_state_path.read_text(encoding="utf-8"))
            best_score = opt.get("best_score", best_score)
            state["best_score"] = best_score
        except Exception:
            pass

    history_lines = []
    for h in state.get("history", [])[-10:]:
        history_lines.append(
            f"  Session {h.get('session','?')}: phase={h.get('phase','?')} "
            f"at {str(h.get('ts',''))[:19]}"
        )
    history_block = (
        "\nRECENT EVOLVE SESSIONS:\n" + "\n".join(history_lines)
        if history_lines else ""
    )

    return f"""\
You are doing a periodic strategic review of an ongoing autonomous photorealism
optimization. Read CLAUDE.md (project conventions — respect them) AND
EVOLVE_CONTEXT.md (this system's context) in this directory first. Do not modify
CLAUDE.md.

AUTH: subscription only. Do not use the Anthropic SDK or ANTHROPIC_API_KEY; the
inner loop must keep calling the `claude` CLI in headless mode (see Task 4 of
the original bootstrap). If you find any SDK/API-key usage, convert it.

PROCESS OWNERSHIP: evolve.py exclusively owns the tree_optimizer.py process. It
has ALREADY STOPPED the optimizer for the duration of this review and will
restart it the moment you finish — so you can edit the tree source and
tree_optimizer.py freely without racing a running loop. Do NOT start, background,
or `python tree_optimizer.py` yourself; just leave it stopped and in a green,
buildable state.

CURRENT STATUS
  Evolve session:  {state['session']}
  Best score:      {best_score:.1f} / 100
  Current focus:   trees
{history_block}

TASKS

1. READ THE CURRENT STATE
   • optimizer_log.txt (last 50 lines)  — recent iteration outcomes
   • optimizer_state.json               — current score, strategy, stagnation
   • OPTIMIZATION_NOTES.md             — documented strategy and findings
   • The tree source files              — see the current state of the code

2. DIAGNOSE
   Is tree_optimizer.py making measurable progress?
   Is it stuck in stagnation or looping on build failures?
   Is the scoring rubric producing useful, discriminating feedback?
   Is the modification strategy targeting the right parts of the code?
   Are there recurring errors in the log?

3. MAKE CONCRETE IMPROVEMENTS — at least one of:
   • Fix crashes or recurring build failures in tree_optimizer.py
   • Improve the modification strategy if stagnated
   • Refine the vision scoring prompt for better discrimination
   • Add new tree source files to the optimization scope if they were missed
   • Try a fundamentally different algorithmic angle if deeply stagnated

4. LEAVE THE BUILD GREEN
   Whatever you change, end with a tree that compiles and renders. Do NOT start
   tree_optimizer.py — evolve.py restarts it automatically when you finish.

5. UPDATE OPTIMIZATION_NOTES.md
   Refresh the Executive Summary at the very top (current best score, running vs
   paused, one-line strategy, last change + whether it helped, next step), and
   append a dated entry to BOTH the Running Journal (newest at top: what you
   found, what you changed, the single best next step) and the Optimization Log,
   noting the current best score. This is the resume point if the run is
   interrupted before the next review.

6. OPTIONALLY UPDATE evolve.py
   If you see improvements to the orchestration logic (prompt quality,
   project scanning, review frequency, etc.), go ahead and update it.
   It will relaunch itself automatically after modification.

Make real, concrete changes. Leave everything better than you found it.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Claude Code runner
# ──────────────────────────────────────────────────────────────────────────────

def find_claude() -> Optional[str]:
    if shutil.which("claude"):
        return "claude"
    candidates = [
        "~/.local/bin/claude",
        "~/.npm-global/bin/claude",
        "~/.nvm/versions/node/*/bin/claude",
        "/usr/local/bin/claude",
        "/opt/homebrew/bin/claude",
    ]
    for pattern in candidates:
        expanded = os.path.expanduser(pattern)
        matches = glob.glob(expanded)
        for path in matches:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
    return None


def claude_env() -> dict:
    """Environment for Claude Code subprocesses.

    When FORCE_SUBSCRIPTION_AUTH is set we strip ANTHROPIC_API_KEY (and
    ANTHROPIC_AUTH_TOKEN) so Claude Code authenticates via the logged-in
    subscription / OAuth token and can never silently fall back to metered API
    billing. This is the mechanism behind "no API key needed".
    """
    env = dict(os.environ)
    if FORCE_SUBSCRIPTION_AUTH:
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
    return env


def preflight_auth() -> bool:
    """Verify the `claude` CLI exists and the subscription login works.

    Catches the common walk-away failure (not logged in) up front, before we
    burn into a multi-hour bootstrap. A trivial headless prompt round-trips the
    auth path; success means the subscription / OAuth token is good.
    """
    claude = find_claude()
    if not claude:
        log("claude CLI not found. Install + log in: "
            "npm install -g @anthropic-ai/claude-code, then `claude` and /login", "ERROR")
        return False
    try:
        result = subprocess.run(
            [claude, "-p", "Reply with the single word: READY",
             "--model", CLAUDE_MODEL],
            cwd=str(PROJECT_ROOT), env=claude_env(), timeout=120,
            capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        log("claude auth pre-flight timed out (120s).", "ERROR")
        return False
    except Exception as exc:
        log(f"claude auth pre-flight failed to run: {exc}", "ERROR")
        return False

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[:300]
        log(f"claude auth pre-flight failed (rc={result.returncode}). "
            f"Are you logged in? Run `claude` then /login. {detail}", "ERROR")
        return False
    log("Auth pre-flight OK — using the Claude Code subscription (no API key).")
    return True


def run_claude_code(prompt: str, timeout: int, label: str) -> Tuple[str, int]:
    claude = find_claude()
    if not claude:
        log("claude CLI not found. Install: npm install -g @anthropic-ai/claude-code", "ERROR")
        return "", 127

    session_log = PROJECT_ROOT / f".evolve_session_{label}.log"

    # Full tool access (Bash/Read/Write/Edit/Glob/Grep/…); permission prompts are
    # skipped for unattended operation. We do NOT pass --allowedTools — narrowing
    # it would block search tools the bootstrap/review sessions rely on.
    cmd = [
        claude,
        "-p", prompt,                      # -p = non-interactive print mode
        "--model", CLAUDE_MODEL,
        "--dangerously-skip-permissions",  # skip confirmation prompts (unattended)
    ]

    log(f"▶ Claude Code '{label}' | model: {CLAUDE_MODEL} | "
        f"prompt: {len(prompt):,} chars | timeout: {timeout//60}m")

    try:
        with open(session_log, "w", encoding="utf-8") as out_f:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=out_f,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                env=claude_env(),
            )
        output = session_log.read_text(encoding="utf-8") if session_log.exists() else ""
        log(f"◼ Session '{label}' done | rc={result.returncode} | {len(output):,} chars output")
        return output, result.returncode

    except subprocess.TimeoutExpired:
        output = session_log.read_text(encoding="utf-8") if session_log.exists() else ""
        log(f"⏱ Session '{label}' timed out after {timeout}s", "WARN")
        return output, -1

    except FileNotFoundError:
        log(f"Command not found: {claude}", "ERROR")
        return "", 127

    except Exception as exc:
        log(f"Error in Claude Code session: {exc}", "ERROR")
        return "", -1


# ──────────────────────────────────────────────────────────────────────────────
# Optimizer process management
# ──────────────────────────────────────────────────────────────────────────────

def process_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ProcessLookupError, ValueError, TypeError):
        return False


def ensure_optimizer_running(state: dict) -> dict:
    if not OPTIMIZER_SCRIPT.exists():
        return state  # Not created yet — bootstrap will handle it

    pid = state.get("optimizer_pid")
    if process_alive(pid):
        return state  # Already running fine

    log("Starting tree_optimizer.py...")
    try:
        out = open(PROJECT_ROOT / "tree_optimizer_stdout.log", "a", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, str(OPTIMIZER_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            stdout=out,
            stderr=subprocess.STDOUT,
            env=claude_env(),  # subscription auth flows down to the inner loop too
        )
        state["optimizer_pid"] = proc.pid
        log(f"tree_optimizer.py started (PID {proc.pid})")
    except Exception as exc:
        log(f"Could not start tree_optimizer.py: {exc}", "WARN")

    return state


def stop_optimizer(state: dict) -> dict:
    """Stop the optimizer and wait for it to exit.

    evolve.py is the sole owner of this process. We pause it during a strategic
    review so the review session can edit the tree source without racing a
    running loop, then restart it afterwards.
    """
    pid = state.get("optimizer_pid")
    if not process_alive(pid):
        state["optimizer_pid"] = None
        return state

    log(f"Pausing tree_optimizer.py (PID {pid}) for review...")
    try:
        os.kill(int(pid), signal.SIGTERM)
    except (OSError, ValueError, TypeError) as exc:
        log(f"Could not signal optimizer PID {pid}: {exc}", "WARN")

    # Wait up to ~10s for a graceful exit, then SIGKILL.
    for _ in range(20):
        if not process_alive(pid):
            break
        time.sleep(0.5)
    else:
        log(f"Optimizer PID {pid} did not exit; killing.", "WARN")
        try:
            os.kill(int(pid), signal.SIGKILL)
        except (OSError, ValueError, TypeError):
            pass

    state["optimizer_pid"] = None
    return state


# ──────────────────────────────────────────────────────────────────────────────
# Token-budget monitoring
# ──────────────────────────────────────────────────────────────────────────────

def _find_access_token(obj) -> Optional[str]:
    """Recursively pull the first `accessToken` string out of parsed JSON."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "accessToken" and isinstance(v, str) and v:
                return v
            found = _find_access_token(v)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_access_token(v)
            if found:
                return found
    return None


def _extract_token(raw: str) -> Optional[str]:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        tok = (data.get("claudeAiOauth") or {}).get("accessToken") if isinstance(data, dict) else None
        return tok or _find_access_token(data)
    except Exception:
        # The stored value might be the bare token itself.
        return raw if raw.startswith("sk-ant-oat") else None


def _oauth_token() -> Optional[str]:
    """Locate the Claude Code OAuth token (macOS Keychain, else credentials file)."""
    if sys.platform == "darwin":
        for svc in KEYCHAIN_SERVICES:
            try:
                out = subprocess.run(
                    ["security", "find-generic-password", "-s", svc, "-w"],
                    capture_output=True, text=True, timeout=10,
                )
            except Exception:
                continue
            if out.returncode == 0:
                tok = _extract_token(out.stdout)
                if tok:
                    return tok
    try:
        if CREDENTIALS_FILE.exists():
            return _extract_token(CREDENTIALS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _parse_reset(ts) -> Optional[datetime.datetime]:
    if not ts or not isinstance(ts, str):
        return None
    try:
        return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def fetch_usage() -> Optional[dict]:
    """Query the usage endpoint. Returns parsed JSON, or None on any failure.

    Prefers `curl` (uses the OS trust store — dodges the common "Python has no CA
    bundle" SSL failure, and matches how claude-monitor queries it). The bearer
    token is passed via a stdin config file, never on the argv, so it can't leak
    through `ps`. Falls back to urllib where curl is unavailable.
    """
    token = _oauth_token()
    if not token:
        return None
    headers = {"Authorization": f"Bearer {token}", **USAGE_HTTP_HEADERS}

    curl = shutil.which("curl")
    if curl:
        # curl config read from stdin (-K -): keeps the token out of argv.
        cfg_lines = [f'url = "{USAGE_URL}"']
        cfg_lines += [f'header = "{k}: {v}"' for k, v in headers.items()]
        try:
            out = subprocess.run(
                [curl, "-sS", "--max-time", "20", "-K", "-"],
                input="\n".join(cfg_lines), capture_output=True, text=True, timeout=30,
            )
            if out.returncode == 0 and out.stdout.strip():
                return json.loads(out.stdout)
            log(f"Usage query via curl failed (rc={out.returncode}); "
                "proceeding without budget gating.", "WARN")
        except Exception as exc:
            log(f"Usage query via curl errored ({exc}); trying urllib.", "WARN")

    import urllib.request
    req = urllib.request.Request(USAGE_URL, method="GET")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log(f"Usage query failed ({exc}); proceeding without budget gating.", "WARN")
        return None


def budget_status() -> Tuple[bool, str, Optional[datetime.datetime]]:
    """Check enforced windows.

    Returns (over_budget, human_summary, resume_at). resume_at is the soonest
    reset among breached windows (None if unknown). Fails OPEN — if usage can't
    be read we report "not over budget" so a monitoring glitch never deadlocks
    the run (degrade, don't panic).
    """
    data = fetch_usage()
    if not isinstance(data, dict):
        return False, "usage unavailable", None

    parts, breached, soonest = [], [], None
    for key, threshold in ENFORCED_WINDOWS.items():
        win = data.get(key)
        if not isinstance(win, dict) or win.get("utilization") is None:
            continue
        util = float(win["utilization"])
        parts.append(f"{WINDOW_SHORT_NAME.get(key, key)}={util:.0f}%")
        if util >= threshold:
            reset = _parse_reset(win.get("resets_at"))
            breached.append(key)
            if reset and (soonest is None or reset < soonest):
                soonest = reset
    summary = " ".join(parts) if parts else "no windows reported"
    return (len(breached) > 0), summary, soonest


def wait_for_budget(state: dict) -> dict:
    """If over budget, stop the optimizer and block until the budget refreshes.

    Always records the latest budget summary in state['budget'] so the status
    line can show it without a second query. Returns when back under budget.
    """
    over, summary, resume_at = budget_status()
    state["budget"] = summary
    if not over:
        if state.get("paused"):
            state["paused"] = False
        return state

    log(f"⏸  Token budget reached ({summary}) — coming to a stopping point "
        f"and waiting for a refresh.", "WARN")
    state = stop_optimizer(state)
    state["paused"] = True
    save_state(state)

    while True:
        now = datetime.datetime.now(datetime.timezone.utc)
        if resume_at:
            secs = (resume_at - now).total_seconds() + BUDGET_RESUME_BUFFER
            eta = resume_at.astimezone().strftime("%Y-%m-%d %H:%M")
        else:
            secs, eta = BUDGET_POLL_SECONDS, "unknown"
        secs = max(60.0, min(secs, float(BUDGET_POLL_SECONDS)))
        log(f"Paused for token budget ({summary}); soonest reset ~{eta}. "
            f"Re-checking in {int(secs)}s. Ctrl+C to stop.")
        time.sleep(secs)

        over, summary, resume_at = budget_status()
        state["budget"] = summary
        if not over:
            log(f"▶  Token budget refreshed ({summary}) — resuming.")
            state["paused"] = False
            save_state(state)
            return state


# ──────────────────────────────────────────────────────────────────────────────
# Initial README
# ──────────────────────────────────────────────────────────────────────────────

_README = """\
# Autonomous Photorealism Optimizer

An AI-driven system that continuously improves the photorealism of procedurally
generated 3D objects. Drop `evolve.py` in the project root and run it once.

## Architecture

```
python evolve.py               ← run once; keeps running in the background
  │
  ├─ Claude Code (bootstrap)   ← reads the codebase, adds render-to-image CLI,
  │                               creates tree_optimizer.py, writes strategy docs
  │
  └─ tree_optimizer.py         ← the continuous inner loop:
                                  render → score with Claude vision (via claude CLI)
                                  → modify by letting Claude Code edit the source
                                  → rebuild → compare → keep or revert
```

Every model call — bootstrap, review, and the inner loop's scoring and code
edits — goes through the `claude` CLI on your **Claude Code subscription**.
**No `ANTHROPIC_API_KEY` and no `anthropic` SDK are required or used.**

## Token Budget — Auto Pause / Resume

evolve.py monitors usage via the same endpoint that powers Claude Code's
`/usage` (it reads your OAuth token; the query itself costs no tokens). When the
**5-hour** budget passes **95%** or the **weekly** budget passes **90%**, it
comes to a stopping point — stops the optimizer — and **waits until the budget
refreshes**, then resumes automatically. The current `OPTIMIZATION_NOTES.md`
**Executive Summary** (top of file) always reflects status at a glance.

## Object Roadmap

- [x] **Trees** — current focus
- [ ] Rocks & stones
- [ ] Ground vegetation (grass, ferns, bushes)
- [ ] Terrain surface features

## Key Files Created by This System

| File | Description |
|------|-------------|
| `tree_optimizer.py` | AI optimization loop (created on first run) |
| `OPTIMIZATION_NOTES.md` | Strategy docs (created on first run) |
| `evolve_state.json` | Orchestrator state — delete to start fresh |
| `optimizer_state.json` | Inner loop state, best score, history |
| `evolve_log.txt` | Orchestrator log |
| `optimizer_log.txt` | Per-iteration optimization log |
| `snapshots/best/` | Source of the highest-scoring version so far |
| `renders/best/` | Renders of the best version |

## Monitoring

```bash
tail -f optimizer_log.txt      # live iteration feed
cat optimizer_state.json       # current best score and strategy
ls renders/best/               # renders of the best version
cat OPTIMIZATION_NOTES.md     # Claude's documented strategy
```

## Resuming After a Stop / Interruption

The system is built to be interrupted (Ctrl-C, crash, reboot) and resumed. State
is saved continuously to `evolve_state.json` / `optimizer_state.json`, and Claude
keeps a human-readable running journal in `OPTIMIZATION_NOTES.md` (newest entry
on top) so a fresh session can see exactly where it left off. Simply re-run:
```bash
python evolve.py
```

## Requirements

- Claude Code installed **and logged in** to your subscription:
  `npm install -g @anthropic-ai/claude-code`, then run `claude` once and `/login`.
- Python 3.8+ (no `anthropic` SDK, no `ANTHROPIC_API_KEY` — it runs entirely on
  your Claude Code subscription / OAuth token).
"""


def write_readme() -> None:
    readme = PROJECT_ROOT / "README_EVOLVE.md"
    if readme.exists():
        return
    try:
        readme.write_text(_README, encoding="utf-8")
        log("Created README_EVOLVE.md")
    except Exception as exc:
        log(f"Could not write README_EVOLVE.md: {exc}", "WARN")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log("=" * 60)
    log("  evolve.py — Autonomous Photorealism Optimizer")
    log("=" * 60)

    write_readme()

    if not preflight_auth():
        log("Aborting: Claude Code is not ready. Fix auth and re-run.", "ERROR")
        sys.exit(1)

    original_hash = file_hash(SCRIPT_PATH)
    state = load_state()
    log(f"Phase: {state['phase']} | Session: {state['session']} | "
        f"Best score: {state['best_score']:.1f}/100")

    log("Scanning project and writing EVOLVE_CONTEXT.md...")
    _context = scan_project()

    # ── Bootstrap phase ───────────────────────────────────────────────────────
    if state["phase"] == "bootstrap":
        state = wait_for_budget(state)  # don't start a 2h bootstrap if already capped
        log("Bootstrap — Claude Code will set up the optimization infrastructure.")
        log(f"This may take up to {BOOTSTRAP_TIMEOUT // 60} minutes. "
            "Output is being logged.")

        _, rc = run_claude_code(BOOTSTRAP_PROMPT, BOOTSTRAP_TIMEOUT, "bootstrap")

        state["session"] += 1
        state["history"].append({
            "session": state["session"],
            "phase":   "bootstrap",
            "rc":      rc,
            "ts":      datetime.datetime.now().isoformat(),
        })
        state["phase"] = "optimize"
        save_state(state)

        relaunch_if_modified(original_hash, state)
        original_hash = file_hash(SCRIPT_PATH)

        state = ensure_optimizer_running(state)
        # Count the bootstrap as the first "review" so we don't immediately run a
        # heavy strategic review on top of a just-finished 2-hour bootstrap.
        state["last_review"] = datetime.datetime.now().isoformat()
        save_state(state)
        log("Bootstrap complete. Entering optimization supervision loop.")

    # ── Optimization supervision loop ─────────────────────────────────────────
    log(f"Supervision active — reviews every {REVIEW_INTERVAL // 60}m, "
        f"health checks every {MONITOR_INTERVAL}s")
    log("Press Ctrl+C to stop gracefully.\n")

    last_review = None
    if state.get("last_review"):
        try:
            last_review = datetime.datetime.fromisoformat(state["last_review"])
        except Exception:
            pass

    last_status_at = None  # for throttling the idle status line

    while True:
        try:
            # Token-budget gate: if we're over the 5h/weekly limit this stops the
            # optimizer and blocks here until the budget refreshes, then returns.
            state = wait_for_budget(state)

            # Health check
            state = ensure_optimizer_running(state)
            save_state(state)

            now = datetime.datetime.now()
            elapsed = (now - last_review).total_seconds() if last_review else REVIEW_INTERVAL + 1

            if elapsed >= REVIEW_INTERVAL:
                log(f"Strategic review #{state['session'] + 1} starting...")

                # Pause the optimizer so the review can edit the tree source and
                # tree_optimizer.py without racing a running loop.
                state = stop_optimizer(state)
                save_state(state)

                prompt = build_review_prompt(state)
                original_hash = file_hash(SCRIPT_PATH)

                _, rc = run_claude_code(
                    prompt, REVIEW_TIMEOUT, f"review_{state['session'] + 1}"
                )

                now = datetime.datetime.now()
                state["session"] += 1
                state["last_review"] = now.isoformat()
                last_review = now
                state["history"].append({
                    "session": state["session"],
                    "phase":   "review",
                    "rc":      rc,
                    "ts":      now.isoformat(),
                })
                if len(state["history"]) > 100:
                    state["history"] = state["history"][-100:]
                save_state(state)

                relaunch_if_modified(original_hash, state)
                original_hash = file_hash(SCRIPT_PATH)

                # Restart the optimizer (the review may have modified it).
                state = ensure_optimizer_running(state)
                save_state(state)

            else:
                # Throttle the routine "still alive" line so stdout stays
                # readable over a long unattended run (transitions — reviews,
                # restarts, budget pauses — are always logged as they happen).
                now_mono = datetime.datetime.now()
                if last_status_at is None or \
                   (now_mono - last_status_at).total_seconds() >= STATUS_LOG_SECONDS:
                    mins_left = int((REVIEW_INTERVAL - elapsed) // 60)
                    pid = state.get("optimizer_pid")
                    status = f"PID {pid} ✓" if process_alive(pid) else "NOT RUNNING ✗"
                    log(f"Optimizer: {status} | "
                        f"Next review: ~{mins_left}m | "
                        f"Best: {state['best_score']:.1f}/100 | "
                        f"Tokens: {state.get('budget', 'n/a')}")
                    last_status_at = now_mono

            time.sleep(MONITOR_INTERVAL)

        except KeyboardInterrupt:
            log("\nStopped by user.")
            save_state(state)
            log(f"State saved. Best score so far: {state['best_score']:.1f}/100")
            log("Run 'python evolve.py' to resume.")
            sys.exit(0)

        except Exception as exc:
            import traceback
            log(f"Unexpected error in main loop: {exc}", "ERROR")
            log(traceback.format_exc(), "ERROR")
            save_state(state)
            time.sleep(30)


if __name__ == "__main__":
    main()
