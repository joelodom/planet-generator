#!/usr/bin/env python3
"""
evolve.py — Autonomous Photorealism Optimizer (orchestrator)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Drop this file (plus its sibling `tree_optimizer.py`) into the root of the
procedural object generator and run:   python evolve.py

DESIGN PRINCIPLE — THE HARNESS NEVER CHANGES ITS OWN CODE
  evolve.py and tree_optimizer.py are FIXED, human-authored harness code. The
  optimization run never edits them, and never edits itself. The ONLY thing a run
  ever modifies is the TARGET tree source (src/flora.rs). To improve the
  optimizer, you read the notes/logs it writes and edit these scripts by hand —
  the system does not self-modify, so every run is reproducible and analyzable.

  (Earlier versions let Claude rewrite tree_optimizer.py and evolve.py during a
  periodic "review" and relaunch itself. That made runs non-deterministic and
  impossible to reason about, so it was removed.)

What it does:
  1. Scans the project to build context (EVOLVE_CONTEXT.md).
  2. ONE-TIME bootstrap (only if missing): launches Claude Code to add a headless
     `--render-to-image` CLI to the TARGET app (Rust). The bootstrap may touch
     ONLY the target source; any edit to a .py harness file is reverted.
  3. Owns and supervises tree_optimizer.py — restarts it if it crashes, and emits
     a fresh status snapshot every minute (to STATUS files and ~/Public) so you
     can watch progress live from another account.
  4. Monitors token usage; if the 5-hour budget passes 95% or the weekly budget
     passes 90% it stops the optimizer, WAITS for the budget to refresh, then
     resumes on its own — so you can walk away.

Auth — NO API KEY NEEDED:
  Every model call goes through the `claude` CLI on the logged-in Claude Code
  subscription / OAuth token. ANTHROPIC_API_KEY is scrubbed from every child
  process (FORCE_SUBSCRIPTION_AUTH) so it can never silently fall back to metered
  API billing. Just be logged in first:  `claude`  then  /login.

Requirements:
  • Claude Code installed & logged in:  npm install -g @anthropic-ai/claude-code
  • Python 3.8+   (no `anthropic` SDK, no API key)
  • tree_optimizer.py present next to this file (it ships with evolve.py)

Resumable: state is saved to evolve_state.json — re-run any time to continue.
"""

import datetime
import glob
import json
import os
import signal
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
# We deliberately do NOT touch the project's own CLAUDE.md (it holds hand-authored
# project conventions). Our auto-generated context goes here; prompts read both.
EVOLVE_CONTEXT   = PROJECT_ROOT / "EVOLVE_CONTEXT.md"
NOTES_FILE       = PROJECT_ROOT / "OPTIMIZATION_NOTES.md"  # owned by tree_optimizer.py
OPT_STATE_FILE   = PROJECT_ROOT / "optimizer_state.json"   # inner loop's state (read-only here)

# Supervisor status mirrors (so the GUI account can watch from elsewhere).
EVOLVE_STATUS    = PROJECT_ROOT / "EVOLVE_STATUS.md"
PUBLIC_DIR       = Path.home() / "Public" / "planet-explorer"
PUBLIC_SUPERVISOR = PUBLIC_DIR / "SUPERVISOR.md"

# Harness files the bootstrap session must never modify. If it does, we restore
# their exact bytes — "the harness never changes its own code".
PROTECTED_FILES  = ("evolve.py", "tree_optimizer.py")

BOOTSTRAP_TIMEOUT = 7200   # 2 hours — Claude Code adds the render CLI to the target
MONITOR_INTERVAL  = 60     # health-check + write a status snapshot every 60 seconds
STATUS_LINE_SECONDS = 120  # throttle the stdout/log status LINE (files update every tick)

# Model alias passed to `claude --model` for the bootstrap session.
CLAUDE_MODEL      = "opus"

# Force subscription / OAuth auth by scrubbing ANTHROPIC_API_KEY from each child
# process. Makes "no API key needed" a guarantee, not a hope.
FORCE_SUBSCRIPTION_AUTH = True

# ── Token-budget monitoring ──────────────────────────────────────────────────
# Reads the Claude Code OAuth token and queries the same usage endpoint that
# powers `/usage` (querying costs no tokens). When a window crosses its stop
# threshold we stop the optimizer and WAIT until the budget refreshes.
USAGE_URL          = "https://api.anthropic.com/api/oauth/usage"
USAGE_HTTP_HEADERS = {
    "anthropic-version": "2023-06-01",
    "anthropic-beta":    "oauth-2025-04-20",
    "User-Agent":        "evolve.py",
}
KEYCHAIN_SERVICES  = ("Claude Code-credentials", "Claude Code", "claude.ai")
CREDENTIALS_FILE   = Path.home() / ".claude" / ".credentials.json"

FIVE_HOUR_STOP_PCT = 95.0   # 5-hour rolling session budget
WEEKLY_STOP_PCT    = 90.0   # 7-day budget (all-models, and per-model caps)
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
            st = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            st.setdefault("phase", "bootstrap")
            st.setdefault("session", 0)
            st.setdefault("best_score", 0.0)
            st.setdefault("optimizer_pid", None)
            st.setdefault("history", [])
            return st
        except Exception as e:
            log(f"Could not load state ({e}), starting fresh.", "WARN")
    return {
        "phase":         "bootstrap",
        "session":       0,
        "best_score":    0.0,
        "optimizer_pid": None,
        "history":       [],
    }


def save_state(state: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as e:
        log(f"Could not save state: {e}", "ERROR")


# ──────────────────────────────────────────────────────────────────────────────
# Harness-protection guard (the bootstrap may only edit the TARGET, never a .py)
# ──────────────────────────────────────────────────────────────────────────────

def snapshot_protected() -> dict:
    out = {}
    for rel in PROTECTED_FILES:
        p = PROJECT_ROOT / rel
        try:
            out[rel] = p.read_bytes() if p.exists() else b""
        except Exception:
            out[rel] = b""
    return out


def restore_protected(snapshot: dict) -> None:
    """Restore exact pre-session bytes of any harness file a session changed."""
    for rel, original in snapshot.items():
        p = PROJECT_ROOT / rel
        try:
            if (p.read_bytes() if p.exists() else b"") != original:
                p.write_bytes(original)
                log(f"Reverted out-of-scope edit to harness file: {rel}", "WARN")
        except Exception as e:
            log(f"Could not restore protected file {rel}: {e}", "ERROR")


def render_harness_ready() -> bool:
    """True once the TARGET app exposes the headless `--render-to-image` CLI.

    This is the real gate for whether bootstrap is needed (robust to a reverted
    working tree or a stale phase in state): we just look for the flag in src/.
    """
    src = PROJECT_ROOT / "src"
    if not src.exists():
        return False
    try:
        for p in src.rglob("*.rs"):
            if "render-to-image" in p.read_text(encoding="utf-8", errors="ignore"):
                return True
    except Exception:
        pass
    return False


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

    lines.append("Top-level contents:")
    for p in sorted(PROJECT_ROOT.iterdir()):
        if p.name not in skip and not p.name.startswith("."):
            tag = "[dir] " if p.is_dir() else "[file]"
            lines.append(f"  {tag} {p.name}")
    lines.append("")

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
        ".wgsl": "WGSL shader",
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
    _write_evolve_context(context)
    return context


def _write_evolve_context(context: str) -> None:
    """Write/update EVOLVE_CONTEXT.md — referenced explicitly by the bootstrap prompt."""
    content = f"""# Autonomous Photorealism Optimizer — Context (auto-generated by evolve.py)

This file is written by **evolve.py**. Do not hand-edit — it is regenerated each
run. The project's own `CLAUDE.md` (if present) is authoritative for project
conventions; always read it too and respect it.

evolve.py + tree_optimizer.py form an autonomous loop that improves the
photorealism of a procedurally generated object. Current focus: **broadleaf
trees** (initial experiment). Future objects: other tree types, rocks,
vegetation, terrain.

## THE ONE RULE: the harness never changes its own code

`evolve.py` and `tree_optimizer.py` are FIXED, human-maintained harness code.
**You must NEVER modify them, or any other `.py` / `.md` / build file.** The only
file the optimization is allowed to change is the **target tree source**
(`src/flora.rs`). Anything you change outside the target is automatically
reverted. If you think the optimizer itself should change, do NOT change it —
instead it writes analyzable notes/logs (OPTIMIZATION_NOTES.md,
optimizer_history.jsonl) that a human reads to improve it by hand.

## Auth — No API key

Every model call goes through the `claude` CLI on the logged-in subscription /
OAuth token. Never use the Anthropic SDK and never read ANTHROPIC_API_KEY.

## Evolve System Files

| File | Purpose | May you edit it? |
|------|---------|------------------|
| `evolve.py` | Orchestrator (launched you) | NO — never |
| `tree_optimizer.py` | Inner optimization loop (ships fixed) | NO — never |
| `OPTIMIZATION_NOTES.md` | Living journal (written by tree_optimizer.py) | NO |
| `EVOLVE_CONTEXT.md` | This file (auto-generated) | NO |
| `src/flora.rs` | The TARGET tree source | only this, and only in the inner loop |
| `optimizer_state.json` / `evolve_state.json` | Loop / orchestrator state | NO |
| `~/Public/planet-explorer/` | Timestamped samples + STATUS for another account | written by the loop |

## Project Scan (auto-generated)

{context}
"""
    try:
        EVOLVE_CONTEXT.write_text(content, encoding="utf-8")
    except Exception as e:
        log(f"Could not write EVOLVE_CONTEXT.md: {e}", "WARN")


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap prompt — adds ONLY the target render CLI (never the .py harness)
# ──────────────────────────────────────────────────────────────────────────────

BOOTSTRAP_PROMPT = """\
You are setting up a headless render path in a procedural 3D object generator so
an external optimization loop can render its objects to images. This is a
ONE-TIME, NARROW setup task — not an open-ended project.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT SCOPE — THE HARNESS NEVER CHANGES ITS OWN CODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • You may edit ONLY the TARGET application source (the Rust code under `src/`,
    and `Cargo.toml` if a new dependency is truly required).
  • You must NEVER create or modify `evolve.py`, `tree_optimizer.py`, ANY `.py`
    file, or `CLAUDE.md`. They are fixed harness/convention files. (Edits to them
    are auto-reverted, so don't waste effort there.)
  • Do NOT create tree_optimizer.py — it already exists and ships with the system.
  • Do NOT start any optimization loop or long-running process. Set up the render
    CLI, verify it once, and stop.

Read `CLAUDE.md` (project conventions — authoritative, respect it, do NOT modify
it) AND `EVOLVE_CONTEXT.md` (this system's context) before doing anything else.
Respect CLAUDE.md's rules — especially "no magic numbers" (named SCREAMING_SNAKE
consts) and determinism (everything derives from the seed).

AUTH — IMPORTANT: NO API key. Do NOT use the Anthropic SDK, do NOT import
`anthropic`, do NOT read ANTHROPIC_API_KEY. (The optimization loop calls the
`claude` CLI itself; you don't need to wire that up.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK 1 — EXPLORE
  Read the relevant target source. Understand the build command, how the object
  geometry (trees/vegetation) is generated, how the program currently renders,
  and the existing CLI parsing. Do not guess — read the files.

TASK 2 — ADD A HEADLESS `--render-to-image` CLI TO THE TARGET APP
  Implement EXACTLY this interface (the optimizer depends on it verbatim):

      <program> --render-to-image --object broadleaf --seed 42 \\
                --yaw 45 --pitch 30 --output render.png \\
                --width 512 --height 512

    --object   object kind to render. MUST support at least "broadleaf" (a
               broadleaf tree). Also support a generic "tree" and the other plant
               forms if cheap, but "broadleaf" is REQUIRED — the current
               experiment renders broadleaf trees specifically.
    --seed     integer seed for reproducible procedural generation
    --yaw      camera horizontal angle in degrees
    --pitch    camera vertical angle in degrees
    --output   output PNG path
    --width / --height   image size in pixels (default 512)

  It must render ONE standalone object (framed to fill the view) to a PNG with NO
  window/GPU surface — a true offscreen/headless path — then exit before any
  windowing code. Reuse the project's real shaders/lighting so the look matches
  the app. Keep determinism: a given (object, seed) must always produce the same
  mesh. Keep the existing app behavior unchanged when the flag is absent.

  This is a Rust + wgpu project: add an offscreen render-to-texture path (no
  winit surface), parse the flag in main before the event loop, and write the PNG
  (a `png`/image encoder). Put the standalone-object mesh builder next to the
  existing vegetation/flora code so "broadleaf" maps to the existing broadleaf
  form. Add named consts for any new tuning values (no magic numbers).

TASK 3 — VERIFY ONCE, THEN STOP
  Build (release) and run ONE render to a temp PNG, e.g.:
      <build the project in release>
      <program> --render-to-image --object broadleaf --seed 42 \\
                --yaw 45 --pitch 30 --output /tmp/render_test.png \\
                --width 512 --height 512
  Confirm it exits 0 and writes a valid, non-blank PNG. Then STOP — do not loop,
  do not start tree_optimizer.py. evolve.py launches and supervises the optimizer
  after you finish.

  If you cannot make a valid render work, leave the build GREEN and write a short
  explanation of the blocker to OPTIMIZATION_NOTES.md (create it if absent) so a
  human can pick it up. Never leave the project in a non-compiling state.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PERMISSIONS: read/modify the TARGET source under src/ (and Cargo.toml if needed),
run shell commands, build the project. Make concrete, working changes — not stubs.
Do NOT touch any .py file or CLAUDE.md. Leave the build green.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
        for path in glob.glob(expanded):
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
    return None


def claude_env() -> dict:
    """Environment for Claude Code subprocesses — scrubs API keys for subscription auth."""
    env = dict(os.environ)
    if FORCE_SUBSCRIPTION_AUTH:
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
    return env


def preflight_auth() -> bool:
    """Verify the `claude` CLI exists and the subscription login works."""
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
                cmd, cwd=str(PROJECT_ROOT), stdout=out_f,
                stderr=subprocess.STDOUT, timeout=timeout, env=claude_env(),
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
        log("tree_optimizer.py is missing. It is fixed harness code that ships next "
            "to evolve.py — restore it before continuing.", "ERROR")
        return state

    pid = state.get("optimizer_pid")
    if process_alive(pid):
        return state

    log("Starting tree_optimizer.py...")
    try:
        out = open(PROJECT_ROOT / "tree_optimizer_stdout.log", "a", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, str(OPTIMIZER_SCRIPT)],
            cwd=str(PROJECT_ROOT), stdout=out, stderr=subprocess.STDOUT,
            env=claude_env(),  # subscription auth flows down to the inner loop too
        )
        state["optimizer_pid"] = proc.pid
        log(f"tree_optimizer.py started (PID {proc.pid})")
    except Exception as exc:
        log(f"Could not start tree_optimizer.py: {exc}", "WARN")
    return state


def stop_optimizer(state: dict) -> dict:
    """Stop the optimizer and wait for it to exit (used by the budget gate)."""
    pid = state.get("optimizer_pid")
    if not process_alive(pid):
        state["optimizer_pid"] = None
        return state

    log(f"Stopping tree_optimizer.py (PID {pid})...")
    try:
        os.kill(int(pid), signal.SIGTERM)
    except (OSError, ValueError, TypeError) as exc:
        log(f"Could not signal optimizer PID {pid}: {exc}", "WARN")

    for _ in range(20):  # up to ~10s for a graceful exit, then SIGKILL
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
        return raw if raw.startswith("sk-ant-oat") else None


def _oauth_token() -> Optional[str]:
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
    """Query the usage endpoint. Returns parsed JSON, or None on any failure."""
    token = _oauth_token()
    if not token:
        return None
    headers = {"Authorization": f"Bearer {token}", **USAGE_HTTP_HEADERS}

    curl = shutil.which("curl")
    if curl:
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
    """Check enforced windows. Fails OPEN if usage can't be read (degrade, don't panic)."""
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
    """If over budget, stop the optimizer and block until the budget refreshes."""
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
        write_status(state, note=f"PAUSED for token budget ({summary}); reset ~{eta}")
        time.sleep(secs)

        over, summary, resume_at = budget_status()
        state["budget"] = summary
        if not over:
            log(f"▶  Token budget refreshed ({summary}) — resuming.")
            state["paused"] = False
            save_state(state)
            return state


# ──────────────────────────────────────────────────────────────────────────────
# Status snapshot — written every health-check tick to files + ~/Public
# ──────────────────────────────────────────────────────────────────────────────

def read_optimizer_progress():
    """Read the inner loop's progress: (generation, last_verdict, loss_streak).

    generation = number of KEPT A/B improvements (the progress metric). None if the
    optimizer hasn't written state yet.
    """
    if OPT_STATE_FILE.exists():
        try:
            d = json.loads(OPT_STATE_FILE.read_text(encoding="utf-8"))
            return d.get("generation"), d.get("last_verdict", ""), d.get("loss_streak")
        except Exception:
            return None, "", None
    return None, "", None


def write_status(state: dict, note: str = "") -> None:
    """Refresh the supervisor status (EVOLVE_STATUS.md + ~/Public/SUPERVISOR.md)."""
    pid = state.get("optimizer_pid")
    alive = process_alive(pid)
    gen, verdict, streak = read_optimizer_progress()
    gen_txt = (f"{gen} kept improvement{'s' if gen != 1 else ''}"
               if gen is not None else "n/a (no head-to-head yet)")
    opt_status = f"RUNNING (PID {pid})" if alive else ("PAUSED" if state.get("paused") else "NOT RUNNING")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body = (
        f"# evolve.py — supervisor status\n\n"
        f"_Updated {ts}_\n\n"
        f"- **Optimizer:** {opt_status}\n"
        f"- **Progress (generation):** {gen_txt}\n"
        f"- **Last head-to-head:** {verdict or 'n/a'}"
        f"{f' (losing streak {streak})' if streak else ''}\n"
        f"- **Token budget:** {state.get('budget', 'n/a')}\n"
        f"- **Bootstrap:** {'done (render CLI present)' if render_harness_ready() else 'pending'}\n"
        f"{('- **Note:** ' + note + chr(10)) if note else ''}"
        f"\nThe inner loop writes the detailed A/B status and labeled before/after sample "
        f"PNGs to this same `~/Public/planet-explorer/` folder "
        f"(`STATUS.md`, `latest/`, `broadleaf_<timestamp>_*_A_best.png` / `_B_cand.png`).\n"
    )
    for target in (EVOLVE_STATUS, PUBLIC_SUPERVISOR):
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
            os.chmod(target, 0o644)
        except Exception:
            pass
    try:
        os.chmod(PUBLIC_DIR, 0o755)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Initial README
# ──────────────────────────────────────────────────────────────────────────────

_README = """\
# Autonomous Photorealism Optimizer

An AI-driven loop that continuously improves the photorealism of a procedurally
generated 3D object. Current experiment: **broadleaf trees**. Drop `evolve.py`
and its sibling `tree_optimizer.py` in the project root and run `python evolve.py`.

## The one rule: the harness never changes its own code

`evolve.py` and `tree_optimizer.py` are **fixed, human-maintained** code. A run
never edits them and never edits itself — the ONLY file an optimization run
changes is the target tree source (`src/flora.rs`). Anything a model session
touches outside the target is auto-reverted. To improve the optimizer, you read
the notes/logs it leaves and edit the scripts by hand. This keeps every run
reproducible and analyzable.

## Architecture

```
python evolve.py                ← orchestrator (fixed); supervises + budget-gates
  │
  ├─ Claude Code (bootstrap)    ← ONE-TIME, only if missing: adds a headless
  │                                `--render-to-image` CLI to the TARGET app.
  │                                May edit ONLY target source, never any .py.
  │
  └─ tree_optimizer.py          ← fixed inner A/B loop. Each iteration:
        A = current best tree, B = a new candidate the model just wrote
        → render A and B over the SAME fixed panel of seeds/angles (paired)
        → a JUDGE picks the more PHOTOREALISTIC of each A/B pair (no rubric —
          the AI decides what "better" is), order randomized to kill bias
        → B replaces A only if it wins the panel by a vote margin; else revert
        → the judge's written reason becomes FEEDBACK for the next rewrite, so
          the model iterates on real visual signal plus its own creative ideas
        → every few kept wins, re-judge the best vs the ORIGINAL (cumulative proof)
```

## Why the loop is built this way

We don't tell the model what a tree should look like, and we don't score on an
absolute rubric — an earlier absolute-scoring run re-scored the *same* tree from
24.5 to 28.5 and could never tell a real change from noise, so every attempt was
reverted. A direct "which of these two looks more like a real photo?" comparison
is far more reliable. The loop keeps a change only when the candidate wins the
paired panel by a margin, carries the judge's reasoning forward as feedback, and
periodically checks the best against the original so progress is cumulative, not
just local. Every vote and reason is logged (`optimizer_history.jsonl`).

## Watching it from another account

Every iteration drops timestamped sample renders + a `STATUS.md` into
`~/Public/planet-explorer/` (world-readable), and the supervisor refreshes
`SUPERVISOR.md` there every minute. Open that folder from your GUI account to
watch the trees evolve live. Locally: `optimizer_log.txt`, `OPTIMIZATION_NOTES.md`.

## Token Budget — Auto Pause / Resume

evolve.py reads your OAuth token to query the same usage endpoint that powers
`/usage` (the query costs no tokens). When the **5-hour** budget passes **95%** or
the **weekly** budget passes **90%**, it stops the optimizer and **waits for a
refresh**, then resumes automatically.

## Requirements

- Claude Code installed **and logged in**: `npm install -g @anthropic-ai/claude-code`,
  then run `claude` once and `/login`.
- Python 3.8+ (no `anthropic` SDK, no `ANTHROPIC_API_KEY`).
- `tree_optimizer.py` present next to `evolve.py` (ships with it).
"""


def write_readme() -> None:
    readme = PROJECT_ROOT / "README_EVOLVE.md"
    try:
        readme.write_text(_README, encoding="utf-8")
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

    # Harness integrity: tree_optimizer.py ships with evolve.py; we never generate it.
    if not OPTIMIZER_SCRIPT.exists():
        log("tree_optimizer.py is missing. It is fixed harness code that must ship "
            "next to evolve.py. Restore it and re-run.", "ERROR")
        sys.exit(1)

    if not preflight_auth():
        log("Aborting: Claude Code is not ready. Fix auth and re-run.", "ERROR")
        sys.exit(1)

    state = load_state()
    log(f"Phase: {state['phase']} | Session: {state['session']}")

    log("Scanning project and writing EVOLVE_CONTEXT.md...")
    scan_project()

    # ── One-time bootstrap: add the render CLI to the TARGET if it's not there ──
    # Gate on the actual capability (flag present in src/), not just stored phase —
    # robust to a reverted working tree.
    if not render_harness_ready():
        state = wait_for_budget(state)  # don't start a long bootstrap if already capped
        log("Bootstrap — adding the headless --render-to-image CLI to the target app.")
        log(f"This may take up to {BOOTSTRAP_TIMEOUT // 60} minutes. Output is logged.")
        write_status(state, note="bootstrap running (adding render CLI to target)")

        protected = snapshot_protected()
        _, rc = run_claude_code(BOOTSTRAP_PROMPT, BOOTSTRAP_TIMEOUT, "bootstrap")
        restore_protected(protected)  # the harness never changes its own code

        state["session"] += 1
        state["history"].append({
            "session": state["session"], "phase": "bootstrap",
            "rc": rc, "ts": datetime.datetime.now().isoformat(),
        })
        state["phase"] = "optimize"
        save_state(state)

        if not render_harness_ready():
            log("Bootstrap finished but no --render-to-image CLI is present in src/. "
                "Cannot render → cannot optimize. Check .evolve_session_bootstrap.log "
                "and OPTIMIZATION_NOTES.md, then re-run.", "ERROR")
            sys.exit(1)
        log("Bootstrap complete — render CLI present.")
    else:
        state["phase"] = "optimize"
        log("Render CLI already present — skipping bootstrap.")

    # ── Supervision loop (no strategic review; the harness never self-edits) ───
    log(f"Supervision active — health check + status every {MONITOR_INTERVAL}s. "
        "Samples + status go to ~/Public/planet-explorer/.")
    log("Press Ctrl+C to stop gracefully.\n")

    last_status_line_at = None
    while True:
        try:
            # Budget gate: stops the optimizer and blocks here if over the 5h/weekly
            # limit, returning once the budget refreshes.
            state = wait_for_budget(state)

            # Health check + status snapshot (files + ~/Public) every tick.
            state = ensure_optimizer_running(state)
            write_status(state)
            save_state(state)

            now = datetime.datetime.now()
            if last_status_line_at is None or \
               (now - last_status_line_at).total_seconds() >= STATUS_LINE_SECONDS:
                pid = state.get("optimizer_pid")
                status = f"PID {pid} ✓" if process_alive(pid) else "NOT RUNNING ✗"
                gen, verdict, _ = read_optimizer_progress()
                gen_txt = f"{gen} kept" if gen is not None else "n/a"
                log(f"Optimizer: {status} | Generation: {gen_txt} | "
                    f"Tokens: {state.get('budget', 'n/a')}")
                last_status_line_at = now

            time.sleep(MONITOR_INTERVAL)

        except KeyboardInterrupt:
            log("\nStopped by user.")
            save_state(state)
            write_status(state, note="stopped by user")
            gen, _, _ = read_optimizer_progress()
            log(f"State saved. Improvements kept so far: {gen}" if gen is not None
                else "State saved.")
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
