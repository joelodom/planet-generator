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
       • Starts tree_optimizer.py running
  3. Monitors tree_optimizer.py; restarts it automatically if it crashes
  4. Every 30 minutes, runs a Claude Code strategic review session
  5. If Claude Code improves THIS script, relaunches it automatically

Requirements:
  • Claude Code installed:  npm install -g @anthropic-ai/claude-code
  • ANTHROPIC_API_KEY in your environment
  • Python 3.8+

Resumable: state is saved to evolve_state.json — re-run any time to continue.
"""

import datetime
import glob
import hashlib
import json
import os
import shutil
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
CLAUDE_MD        = PROJECT_ROOT / "CLAUDE.md"

BOOTSTRAP_TIMEOUT = 7200   # 2 hours — Claude Code sets up all infrastructure
REVIEW_TIMEOUT    = 3600   # 1 hour  — periodic strategic review
REVIEW_INTERVAL   = 1800   # run a review every 30 minutes
MONITOR_INTERVAL  = 60     # health-check the optimizer every 60 seconds


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
# Project scanner  (also updates CLAUDE.md so Claude Code auto-reads it)
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

    # Write CLAUDE.md so Claude Code picks up context automatically
    _write_claude_md(context)

    return context


def _write_claude_md(context: str) -> None:
    """Write/update CLAUDE.md — Claude Code reads this file automatically."""
    content = f"""# Autonomous Photorealism Optimizer — Project Context

This project is managed by **evolve.py**, an autonomous optimization system that
uses Claude Code to iteratively improve the photorealism of procedurally generated
3D objects. Currently focused on **trees**; future objects: rocks, vegetation, terrain.

## What You Are Doing

You have been invoked by evolve.py to either:
- **Bootstrap**: build the optimization infrastructure (first run), OR
- **Review**: inspect progress and make strategic improvements (subsequent runs)

Read the specific task list in the prompt carefully.

## Evolve System Files

| File | Purpose |
|------|---------|
| `evolve.py` | Orchestrator (this launched you) — you may update it |
| `tree_optimizer.py` | AI micro-optimization loop (you create/maintain this) |
| `OPTIMIZATION_NOTES.md` | Living strategy doc (you create/maintain this) |
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
        CLAUDE_MD.write_text(content, encoding="utf-8")
    except Exception as e:
        log(f"Could not write CLAUDE.md: {e}", "WARN")


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

Read CLAUDE.md in this directory for project context before doing anything else.

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
all future optimization. Include these exact sections:

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

  ## Optimization Log
  (Leave blank — tree_optimizer.py will append entries here.)

TASK 4 — CREATE tree_optimizer.py
The autonomous micro-optimization loop. Calls the Anthropic Python SDK
directly. Runs indefinitely until a fatal API error or keyboard interrupt.

  MODELS
    Vision scoring:   claude-opus-4-8
    Code modification: claude-opus-4-8
                       thinking={"type": "enabled", "budget_tokens": 20000}
    Strategy review:  claude-opus-4-8
                       thinking={"type": "enabled", "budget_tokens": 20000}

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
  On startup: if optimizer_state.json exists, load it and resume from the
  best known state (restore files from snapshots/best/ if it exists).

  LOOP — repeat forever:

  1. RENDER
     Produce 4 PNGs at varied random yaw, pitch, and seed values.
     Save as renders/current/render_0.png … render_3.png.

  2. SCORE  (one Claude vision API call per image)
     Encode each image as base64. Call claude-opus-4-8 with the image
     attached and this exact user message:
       "Rate this procedurally generated tree on photorealism from 0 to 100:
        0–20 = 1990s video-game quality; 21–40 = basic 3D, obviously synthetic;
        41–60 = decent CGI; 61–80 = good modern game quality;
        81–100 = photorealistic, indistinguishable from a photograph.
        Respond ONLY in valid JSON with no markdown fences:
        {\\"score\\": N, \\"critiques\\": [\\"x\\", \\"y\\", \\"z\\"],
         \\"top_improvement\\": \\"<single most impactful algorithmic change>\\"}"
     composite_score = mean of all image scores.

  3. MODIFY  (Claude with extended thinking)
     Send to claude-opus-4-8 with thinking enabled:
       • Full content of all tree_source_files (labelled with path)
       • composite_score + all critiques and top_improvement values from step 2
       • current_strategy and tried_changes from state
       • Last 5 entries from history
     Ask for ONE specific, surgical code change that targets the most
     impactful issue. Require the response as an explicit before/after
     replacement block: file path, exact original text, exact new text.
     Append a one-line description to tried_changes.

  4. APPLY
     Back up all tree_source_files to snapshots/iteration_N/.
     Apply the change via exact string replacement.

  5. BUILD
     Run the build command. Capture stdout+stderr.
     On failure: restore from snapshots/iteration_N/, log BUILD_FAILED,
     go back to step 3 noting "previous change caused build error, try
     a different approach". After 3 consecutive build failures, do a
     mini-strategy review before retrying.

  6. RE-RENDER & RE-SCORE
     Same angles and seeds as step 1. Same scoring method.

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
     Send last 10 history entries to claude-opus-4-8 with extended thinking.
     Ask: "My photorealism optimization strategy has stalled. Here are the
     recent attempts. Propose a fundamentally different strategy — a new
     algorithmic aspect to focus on, or a completely different approach.
     Be specific and actionable."
     Update current_strategy, clear tried_changes, reset stagnation_count = 0.
     Log: STRATEGY_UPDATE with the new strategy.

  9. LOG & ITERATE
     Append to optimizer_log.txt:
       [TIMESTAMP] ITER N | score: X.X → Y.Y (Δ+/-Z.Z) | KEPT/REVERTED/FAILED
       Tried: <one-line description>
       Strategy: <first 100 chars of current_strategy>
     Trim history to last 50 entries. Save state. Sleep 2s. Loop.

  ERROR HANDLING
    anthropic.RateLimitError:
      log warning, sleep 60s, retry same step
    anthropic.APIStatusError (status 529 or 500):
      sleep 30s, retry up to 3×, then skip iteration and continue
    anthropic.AuthenticationError:
      print "ANTHROPIC_API_KEY is missing or invalid", sys.exit(1)
    Any other anthropic.APIError:
      save state, print clean exit summary, sys.exit(0)
    KeyboardInterrupt:
      save state, print summary (iterations, best score), sys.exit(0)
    Build failure:
      revert files, log BUILD_FAILED, continue loop
    Any other Exception:
      log full traceback, revert any pending change, continue loop

TASK 5 — INSTALL, VERIFY, AND LAUNCH
  pip install anthropic Pillow
  Check ANTHROPIC_API_KEY is set — print a clear error and stop if not.
  Run one dry render to confirm the full pipeline works end-to-end.
  Launch: python tree_optimizer.py
  Watch it complete its first full iteration before considering yourself done
  (render → score → modify → build → re-score → compare → log).

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
optimization. Read CLAUDE.md in this directory first for project context.

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

4. ENSURE tree_optimizer.py IS RUNNING
   If it has stopped, diagnose why, fix the issue, and restart it.

5. UPDATE OPTIMIZATION_NOTES.md
   Append a dated entry to the Optimization Log section describing:
   what you found, what you changed, and the current best score.

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


def run_claude_code(prompt: str, timeout: int, label: str) -> Tuple[str, int]:
    claude = find_claude()
    if not claude:
        log("claude CLI not found. Install: npm install -g @anthropic-ai/claude-code", "ERROR")
        return "", 127

    session_log = PROJECT_ROOT / f".evolve_session_{label}.log"

    cmd = [
        claude,
        "-p", prompt,                     # -p = non-interactive print mode
        "--dangerously-skip-permissions",  # skip confirmation prompts
        "--allowedTools", "Bash,Read,Write,Edit",
    ]

    log(f"▶ Claude Code '{label}' | prompt: {len(prompt):,} chars | timeout: {timeout//60}m")

    try:
        with open(session_log, "w", encoding="utf-8") as out_f:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=out_f,
                stderr=subprocess.STDOUT,
                timeout=timeout,
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
        )
        state["optimizer_pid"] = proc.pid
        log(f"tree_optimizer.py started (PID {proc.pid})")
    except Exception as exc:
        log(f"Could not start tree_optimizer.py: {exc}", "WARN")

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
                                  render → score with Claude vision
                                  → modify with extended thinking
                                  → rebuild → compare → keep or revert
```

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

## Resuming After a Stop

State is saved continuously. Simply re-run:
```bash
python evolve.py
```

## Requirements

- Claude Code: `npm install -g @anthropic-ai/claude-code`
- `ANTHROPIC_API_KEY` in environment
- Python 3.8+, plus `pip install anthropic Pillow`
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

    original_hash = file_hash(SCRIPT_PATH)
    state = load_state()
    log(f"Phase: {state['phase']} | Session: {state['session']} | "
        f"Best score: {state['best_score']:.1f}/100")

    log("Scanning project and updating CLAUDE.md...")
    _context = scan_project()

    # ── Bootstrap phase ───────────────────────────────────────────────────────
    if state["phase"] == "bootstrap":
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

    while True:
        try:
            # Health check
            state = ensure_optimizer_running(state)
            save_state(state)

            now = datetime.datetime.now()
            elapsed = (now - last_review).total_seconds() if last_review else REVIEW_INTERVAL + 1

            if elapsed >= REVIEW_INTERVAL:
                log(f"Strategic review #{state['session'] + 1} starting...")

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

                # Re-check optimizer after review (it may have been modified)
                state["optimizer_pid"] = None
                state = ensure_optimizer_running(state)
                save_state(state)

            else:
                mins_left = int((REVIEW_INTERVAL - elapsed) // 60)
                pid = state.get("optimizer_pid")
                status = f"PID {pid} ✓" if process_alive(pid) else "NOT RUNNING ✗"
                log(f"Optimizer: {status} | "
                    f"Next review: ~{mins_left}m | "
                    f"Best score: {state['best_score']:.1f}/100")

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
