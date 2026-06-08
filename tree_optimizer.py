#!/usr/bin/env python3
"""
tree_optimizer.py — inner photorealism A/B loop (FIXED HARNESS CODE).

This file is OWNED BY A HUMAN, version-controlled, and NEVER modified by the
running system. evolve.py launches and supervises it; together they are the
optimizer. The loop only ever edits the TARGET tree source (src/flora.rs) — it
does not, and is not allowed to, edit itself or evolve.py. To improve the
optimizer, read the notes/logs it writes and edit THIS file by hand:

    OPTIMIZATION_NOTES.md     — living journal + executive summary (human-readable)
    optimizer_history.jsonl   — one JSON record per iteration (machine-analyzable)
    optimizer_log.txt         — terse per-iteration log lines
    ~/Public/planet-explorer/ — timestamped A/B sample renders + STATUS.md

How the loop works — A/B, judge-driven, with a feedback loop
------------------------------------------------------------
We do NOT score trees on an absolute rubric and we do NOT tell the model what a
good tree looks like. Each iteration is a head-to-head:

  A = the current best tree.   B = a candidate the model just produced.
  1. Render A and B over a FIXED PANEL of seeds/angles (paired, same camera).
  2. A JUDGE model looks at each A/B pair and picks the one that is MORE
     PHOTOREALISTIC — just "more like a real photograph of a real tree". No
     rubric, no checklist; the AI decides what "better" means. Presentation order
     is randomized per pair to cancel position bias; we tally the votes.
  3. B replaces A only if B wins the panel by a margin (more votes, by >= a
     threshold). Otherwise B is reverted.
  4. FEEDBACK LOOP: the judge also explains WHY the winner looked better and what
     the loser lacked. That explanation is fed into the next code-rewrite, so the
     improver iterates on real visual feedback plus its own creative ideas.

Periodically the current best is also judged against the ORIGINAL tree (the
"anchor") to prove cumulative progress, not just local wins.

Why A/B instead of absolute scores: an earlier absolute-scoring run re-scored the
*same* tree from 24.5 to 28.5 and could never tell a real change from noise — so
every attempt was reverted and nothing improved. A direct "which looks better"
comparison is far more reliable than calibrating an absolute number.

Auth: every model call shells out to the `claude` CLI on the Claude Code
subscription. No `anthropic` SDK, no ANTHROPIC_API_KEY (scrubbed from the child
env). evolve.py owns the token budget and will stop/restart this process; every
iteration ends in a green, saved, revertible state.
"""

import datetime
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Configuration  (named constants — no bare tuning literals, per CLAUDE.md spirit)
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent

# The ONLY file the optimization is permitted to edit. Everything else a model
# session touches is reverted (see guard_out_of_scope).
SRC_TARGET = "src/flora.rs"
ALLOWED_EDIT_FILES = (SRC_TARGET,)
# Harness files that must NEVER change while the loop runs. If a session edits one
# we restore its exact bytes and log it — "evolve never changes its own code".
PROTECTED_FILES = ("evolve.py", "tree_optimizer.py")

# ── Rendering ────────────────────────────────────────────────────────────────
BUILD_COMMAND  = ["cargo", "build", "--release"]
BUILD_TIMEOUT  = 1800  # seconds; release builds are slow but bounded
RENDER_BIN     = PROJECT_ROOT / "target" / "release" / "planet-explorer"
# Initial experiment focuses on ONE tree type: broadleaf. `--object broadleaf`
# routes straight to Form::Broadleaf regardless of seed, so every panel view is a
# broadleaf. Switch this string to retarget the experiment to another object.
RENDER_OBJECT  = "broadleaf"
RENDER_W       = 512   # sample image width  (px)
RENDER_H       = 512   # sample image height (px)
RENDER_TIMEOUT = 120   # seconds per single view

# Fixed evaluation panel: (seed, yaw_deg, pitch_deg). A and B are always rendered
# on THIS set, so the head-to-head is paired (same individual, same camera). The
# seeds give distinct broadleaf individuals; the angles vary so a change can't win
# by flattering a single pose.
SAMPLE_PANEL = (
    (42,    35, 18),
    (42,   205, 26),
    (7,     60, 14),
    (7,    290, 22),
    (108,  130, 20),
    (1235, 250, 12),
)

# ── A/B decision (the noise-robust core) ─────────────────────────────────────
# B replaces A only if B wins MORE panel views than A AND the net margin
# (B_votes - A_votes) is at least this. A direct pairwise vote is far less noisy
# than absolute scoring; the margin keeps us from banking a coin-flip win. Raise
# it if kept changes still look like noise; lower it if real wins keep losing.
VOTE_MARGIN = 2
# After this many KEPT improvements, re-judge the current best against the
# ORIGINAL tree (the anchor) to confirm cumulative — not just local — progress.
ANCHOR_EVERY = 5
# Consecutive losses after which the rewrite prompt nudges the model to be bolder.
BOLD_STREAK = 3

# ── Model / sessions ─────────────────────────────────────────────────────────
CLAUDE_MODEL   = "opus"   # alias → latest Opus on the subscription login
JUDGE_TIMEOUT  = 300      # seconds for one A/B judging session
MODIFY_TIMEOUT = 600      # seconds for one code-rewrite session
CLAUDE_RETRIES = 3        # attempts per model session before giving up the step
RETRY_SLEEP    = 30       # seconds between model retries
SKIP_SLEEP     = 30       # seconds to wait after skipping an iteration (avoid a hot loop)
ITER_SLEEP     = 2        # seconds between successful iterations

# ── Housekeeping ─────────────────────────────────────────────────────────────
HISTORY_KEEP   = 60   # iteration history / tried-changes entries retained in state
JOURNAL_KEEP   = 40   # running-journal entries retained in the notes file
SNAPSHOT_KEEP  = 8    # per-iteration source snapshots retained on disk
BUILD_FAIL_NOTE = "the previous change did not COMPILE"

# ── Paths ────────────────────────────────────────────────────────────────────
STATE_FILE    = PROJECT_ROOT / "optimizer_state.json"
LOG_FILE      = PROJECT_ROOT / "optimizer_log.txt"
JSONL_FILE    = PROJECT_ROOT / "optimizer_history.jsonl"
NOTES_FILE    = PROJECT_ROOT / "OPTIMIZATION_NOTES.md"
RENDERS       = PROJECT_ROOT / "renders"
SNAPSHOTS     = PROJECT_ROOT / "snapshots"
A_DIR         = RENDERS / "current_best"   # renders of A (current best)
B_DIR         = RENDERS / "candidate"      # renders of B (candidate)
ANCHOR_DIR    = RENDERS / "anchor"         # renders of the original tree
ANCHOR_CMP_DIR = RENDERS / "anchor_current"  # current-best renders for the anchor check
                                             # (separate dir so A_DIR stays valid for publish)
BEST_DIR      = RENDERS / "best"           # kept renders of the accepted best
SNAP_BEST     = SNAPSHOTS / "best"
SNAP_ANCHOR   = SNAPSHOTS / "anchor"
# Public drop point so the GUI account can watch progress from elsewhere.
PUBLIC_DIR    = Path.home() / "Public" / "planet-explorer"
PUBLIC_LATEST = PUBLIC_DIR / "latest"
PUBLIC_KEEP   = 80   # timestamped sample PNGs retained under ~/Public (then pruned oldest-first)


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%SZ")


def log(msg: str, level: str = "INFO") -> None:
    line = f"[{_ts()}] {msg}" if level == "INFO" else f"[{_ts()}] {level}: {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# claude CLI runner  (subscription auth; never the metered API)
# ──────────────────────────────────────────────────────────────────────────────

def find_claude():
    if shutil.which("claude"):
        return "claude"
    for pattern in ("~/.local/bin/claude", "~/.npm-global/bin/claude",
                    "~/.nvm/versions/node/*/bin/claude",
                    "/usr/local/bin/claude", "/opt/homebrew/bin/claude"):
        for path in glob.glob(os.path.expanduser(pattern)):
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
    return None


def claude_env() -> dict:
    """Force subscription / OAuth auth by scrubbing any API key from the child env."""
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    return env


def run_claude(prompt: str, timeout: int, label: str) -> bool:
    """Run one headless `claude -p` session. Returns True on a clean exit.

    Claude Code reads PNGs (vision) and edits files itself, so callers point it at
    paths and read the artifacts it writes rather than parsing stdout.
    """
    claude = find_claude()
    if not claude:
        log("claude CLI not found (install: npm install -g @anthropic-ai/claude-code, "
            "then `claude` and /login).", "ERROR")
        return False
    cmd = [claude, "-p", prompt, "--model", CLAUDE_MODEL,
           "--dangerously-skip-permissions"]
    for attempt in range(1, CLAUDE_RETRIES + 1):
        try:
            r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=claude_env(),
                               timeout=timeout, capture_output=True, text=True)
            if r.returncode == 0:
                return True
            detail = (r.stderr or r.stdout or "").strip()[:200]
            log(f"{label}: claude rc={r.returncode} (attempt {attempt}/{CLAUDE_RETRIES}) {detail}", "WARN")
        except subprocess.TimeoutExpired:
            log(f"{label}: claude timed out after {timeout}s (attempt {attempt}/{CLAUDE_RETRIES})", "WARN")
        except Exception as exc:
            log(f"{label}: claude error {exc} (attempt {attempt}/{CLAUDE_RETRIES})", "WARN")
        if attempt < CLAUDE_RETRIES:
            time.sleep(RETRY_SLEEP)
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Build + render (the Rust binary; no model tokens)
# ──────────────────────────────────────────────────────────────────────────────

def build():
    """Release-build the renderer. Returns (ok, tail_of_output)."""
    try:
        r = subprocess.run(BUILD_COMMAND, cwd=str(PROJECT_ROOT),
                           capture_output=True, text=True, timeout=BUILD_TIMEOUT)
        return r.returncode == 0, (r.stderr or r.stdout or "")[-4000:]
    except subprocess.TimeoutExpired:
        return False, f"build timed out after {BUILD_TIMEOUT}s"
    except Exception as exc:
        return False, str(exc)


def render_view(seed: int, yaw: int, pitch: int, out_path: Path) -> bool:
    cmd = [str(RENDER_BIN), "--render-to-image", "--object", RENDER_OBJECT,
           "--seed", str(seed), "--yaw", str(yaw), "--pitch", str(pitch),
           "--output", str(out_path), "--width", str(RENDER_W), "--height", str(RENDER_H)]
    try:
        r = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                           capture_output=True, text=True, timeout=RENDER_TIMEOUT)
        return r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0
    except Exception as exc:
        log(f"render failed (seed={seed} yaw={yaw} pitch={pitch}): {exc}", "WARN")
        return False


def render_panel(dest: Path):
    """Render the fixed SAMPLE_PANEL into `dest`. Returns [(path, (seed,yaw,pitch)), ...]."""
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for i, (seed, yaw, pitch) in enumerate(SAMPLE_PANEL):
        p = dest / f"view{i}_seed{seed}_y{yaw}_p{pitch}.png"
        if render_view(seed, yaw, pitch, p):
            out.append((p, (seed, yaw, pitch)))
    return out


def panel_from_dir(d: Path):
    """Reconstruct a panel list (path, meta) from a directory of view*.png files."""
    files = sorted(d.glob("view*.png"),
                   key=lambda p: int(re.match(r"view(\d+)", p.name).group(1)))
    if len(files) != len(SAMPLE_PANEL):
        return []
    return list(zip(files, SAMPLE_PANEL))


# ──────────────────────────────────────────────────────────────────────────────
# A/B judge — pairwise, no rubric, randomized order, with a "why" for feedback
# ──────────────────────────────────────────────────────────────────────────────

def _parse_judge(raw: str, n: int):
    txt = raw.strip()
    txt = re.sub(r"^```[a-zA-Z]*", "", txt).strip()
    txt = re.sub(r"```$", "", txt).strip()
    try:
        data = json.loads(txt)
    except Exception:
        return None
    verdicts = data.get("verdicts")
    if not isinstance(verdicts, list) or len(verdicts) != n:
        return None
    return data


def judge_ab(a_panel, b_panel, out_path: Path, salt: int):
    """Head-to-head: for each paired view decide which is more photorealistic.

    Presentation order (FIRST/SECOND) is randomized per pair (seeded by `salt` for
    reproducibility) so the judge can't win by position. Returns a dict with vote
    tallies, per-view winners ("A"/"B"/"tie"), and the judge's free-text `reason`
    (the feedback fed into the next rewrite). None if the session/parse fails.
    """
    if not a_panel or not b_panel or len(a_panel) != len(b_panel):
        return None
    rng = random.Random(salt)
    orient = []   # per pair: "A" if FIRST is the A-image else "B"
    lines = []
    for i, ((pa, _), (pb, _)) in enumerate(zip(a_panel, b_panel)):
        first_is_a = rng.random() < 0.5
        first, second = (pa, pb) if first_is_a else (pb, pa)
        orient.append("A" if first_is_a else "B")
        lines.append(f"Pair {i}:\n  FIRST  = {first}\n  SECOND = {second}")
    listing = "\n".join(lines)
    example = ('{"verdicts":[{"pair":0,"more_photorealistic":"first|second|tie"}, ...],'
               '"reason":"<what made the more realistic one look better, and what the other lacked>"}')
    if out_path.exists():
        try:
            out_path.unlink()
        except Exception:
            pass
    prompt = (
        f"Below are {len(orient)} PAIRS of rendered images. In each pair, FIRST and SECOND are two "
        f"versions of the SAME procedurally generated tree, from the SAME camera:\n\n{listing}\n\n"
        "Read every image. For each pair, decide which one looks MORE PHOTOREALISTIC — simply, more "
        "like an actual PHOTOGRAPH of a real tree. Use your own eye; there is NO rubric and no "
        "checklist — you decide what 'better' means. If a pair is genuinely indistinguishable you "
        "may answer 'tie', but prefer to pick one.\n\n"
        "Then, in 'reason', explain CONCRETELY and VISUALLY what made the more realistic version "
        "look better and what the other lacked — specific, actionable feedback a developer could use "
        "to push the tree further toward realism.\n\n"
        f"Write ONLY valid JSON (no markdown fences) to this exact path:\n  {out_path}\nSchema:\n  {example}\n"
        f"'verdicts' MUST have exactly {len(orient)} entries, one per pair, in order (pair 0..{len(orient)-1})."
    )
    for attempt in (1, 2):
        if run_claude(prompt, JUDGE_TIMEOUT, f"judge.{attempt}") and out_path.exists():
            data = _parse_judge(out_path.read_text(encoding="utf-8", errors="ignore"), len(orient))
            if data:
                return _tally(data, orient, a_panel)
        log(f"judge: missing/invalid verdict JSON (attempt {attempt})", "WARN")
    return None


def _tally(data, orient, a_panel):
    a_votes = b_votes = ties = 0
    per_view = []
    for i, v in enumerate(data["verdicts"]):
        choice = str(v.get("more_photorealistic", "tie")).strip().lower()
        if choice == "first":
            winner = orient[i]
        elif choice == "second":
            winner = "B" if orient[i] == "A" else "A"
        else:
            winner = "tie"
        if winner == "A":
            a_votes += 1
        elif winner == "B":
            b_votes += 1
        else:
            ties += 1
        seed, yaw, pitch = a_panel[i][1]
        per_view.append({"seed": seed, "yaw": yaw, "pitch": pitch, "winner": winner})
    return {
        "a_votes": a_votes, "b_votes": b_votes, "ties": ties,
        "per_view": per_view,
        "reason": str(data.get("reason", "")).strip(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Code-rewrite session — feedback-driven, NON-prescriptive (model decides "better")
# ──────────────────────────────────────────────────────────────────────────────

def modify(state, change_path: Path) -> bool:
    if change_path.exists():
        try:
            change_path.unlink()
        except Exception:
            pass

    fb = state.get("last_feedback", "").strip()
    verdict = state.get("last_verdict", "")
    if fb:
        feedback_block = (
            "FEEDBACK from the most recent head-to-head (a judge compared your previous attempt "
            f"against the current best, and the previous attempt was {verdict}):\n"
            f"\"\"\"\n{fb}\n\"\"\"\n"
            "Treat this as visual feedback to build on — but you are free to disagree and try your "
            "own idea.\n\n"
        )
    else:
        feedback_block = "This is the first attempt — there is no comparison feedback yet.\n\n"

    tried = state.get("tried_changes", [])[-12:]
    if tried:
        tried_block = "Changes already tried (newest last, with outcome — don't just repeat a "
        tried_block += "rejected one verbatim; iterate or try something new):\n"
        tried_block += "\n".join(
            f"  - [{t.get('result','?')}] {t.get('desc','')}" for t in tried) + "\n\n"
    else:
        tried_block = ""

    streak = state.get("loss_streak", 0)
    nudge = ("You have lost several head-to-heads in a row, so small tweaks aren't working — make a "
             "BOLDER, materially different change this time.\n\n") if streak >= BOLD_STREAK else ""

    prompt = (
        "You are iteratively improving a procedurally generated BROADLEAF tree in a Rust + wgpu "
        "renderer (planet-explorer) so it looks MORE PHOTOREALISTIC — just better, more like a real "
        "photograph of a real tree. YOU decide what 'better' means and how to get there; there is no "
        "checklist and nobody will tell you what a tree should look like. Be creative.\n\n"
        "STRICT SCOPE — READ CAREFULLY:\n"
        f"  * You may edit ONLY `{SRC_TARGET}`. Touch NO other file. The broadleaf tree is what gets\n"
        "    judged, so improve how IT looks (its builder and the leaf/branch/bark code it uses);\n"
        "    don't bother retuning other species.\n"
        "  * NEVER edit evolve.py or tree_optimizer.py (the optimizer harness), any other *.py,\n"
        "    any *.md, Cargo.*, or shaders. Edits outside the target are auto-reverted.\n"
        "  * Do NOT run cargo, git, package scripts, or tree_optimizer.py. Make the SOURCE change\n"
        "    only — the harness builds it, renders it, judges it against the current best, and keeps\n"
        "    or reverts it.\n\n"
        "RESPECT `CLAUDE.md`:\n"
        "  * No magic numbers — every new tuning literal must be a SCREAMING_SNAKE_CASE `const` with\n"
        "    a short unit/intent comment, hoisted near the top of its module/impl.\n"
        "  * Determinism — same seed must yield an identical mesh. Do NOT reorder or remove existing\n"
        "    RNG draws; only add new draws at the END of a builder (or derive from existing values).\n"
        "  * Stay under the 30,000-vertex-per-species cap. Lighting/shaders are fixed, so work in\n"
        "    geometry and vertex color.\n\n"
        f"{feedback_block}"
        f"{tried_block}"
        f"{nudge}"
        "Make ONE creative, surgical change to the broadleaf tree in "
        f"`{SRC_TARGET}` that you believe will make it look more photorealistic. Keep it minimal, "
        "self-contained, and green.\n\n"
        f"Finally, write to `{change_path}` a ONE-LINE plain-text summary of what you changed AND "
        "your hypothesis for why it will look more realistic."
    )
    return run_claude(prompt, MODIFY_TIMEOUT, "modify")


def _read_change(change_file: Path) -> str:
    try:
        txt = change_file.read_text(encoding="utf-8", errors="ignore").strip()
        return txt.splitlines()[0] if txt else "(no summary written)"
    except Exception:
        return "(no summary written)"


# ──────────────────────────────────────────────────────────────────────────────
# Scope guard — revert anything a session changed outside the allowed target
# ──────────────────────────────────────────────────────────────────────────────

def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except Exception:
        return b""


def guard_out_of_scope(protected_snapshot: dict):
    """Revert any change a model session made outside ALLOWED_EDIT_FILES.

    Restores exact pre-session bytes of the PROTECTED harness files, and
    git-reverts / deletes any other changed file under src/. Returns the list of
    reverted paths (empty == the session stayed in scope).
    """
    reverted = []
    for rel, original in protected_snapshot.items():
        p = PROJECT_ROOT / rel
        if _read_bytes(p) != original:
            try:
                p.write_bytes(original)
                reverted.append(rel)
            except Exception as exc:
                log(f"could not restore protected file {rel}: {exc}", "ERROR")
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "--", "src"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        for line in out.splitlines():
            if not line.strip():
                continue
            status, path = line[:2], line[3:].strip()
            if path in ALLOWED_EDIT_FILES:
                continue
            if "?" in status:  # untracked file the session created under src/
                try:
                    (PROJECT_ROOT / path).unlink()
                    reverted.append(path)
                except Exception:
                    pass
            else:
                subprocess.run(["git", "-C", str(PROJECT_ROOT), "checkout", "--", path],
                               capture_output=True, text=True, timeout=30)
                reverted.append(path)
    except Exception as exc:
        log(f"scope-guard git check failed: {exc}", "WARN")
    return reverted


# ──────────────────────────────────────────────────────────────────────────────
# Snapshots
# ──────────────────────────────────────────────────────────────────────────────

def snapshot_target(iteration: int) -> Path:
    d = SNAPSHOTS / f"iteration_{iteration}" / "src"
    d.mkdir(parents=True, exist_ok=True)
    snap = d / "flora.rs"
    shutil.copy2(PROJECT_ROOT / SRC_TARGET, snap)
    _prune_snapshots()
    return snap


def restore_from(snap: Path) -> None:
    if snap.exists():
        shutil.copy2(snap, PROJECT_ROOT / SRC_TARGET)


def save_best() -> None:
    (SNAP_BEST / "src").mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_ROOT / SRC_TARGET, SNAP_BEST / "src" / "flora.rs")


def restore_best() -> bool:
    p = SNAP_BEST / "src" / "flora.rs"
    if p.exists():
        shutil.copy2(p, PROJECT_ROOT / SRC_TARGET)
        return True
    return False


def save_anchor_source() -> None:
    (SNAP_ANCHOR / "src").mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_ROOT / SRC_TARGET, SNAP_ANCHOR / "src" / "flora.rs")


def _prune_snapshots() -> None:
    iters = sorted(SNAPSHOTS.glob("iteration_*"),
                   key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0)
    for old in iters[:-SNAPSHOT_KEEP]:
        shutil.rmtree(old, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────────────
# Publishing — A/B samples + status to ~/Public
# ──────────────────────────────────────────────────────────────────────────────

def ensure_public() -> None:
    try:
        PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
        os.chmod(PUBLIC_DIR, 0o755)
        PUBLIC_LATEST.mkdir(parents=True, exist_ok=True)
        os.chmod(PUBLIC_LATEST, 0o755)
    except Exception as exc:
        log(f"could not prepare ~/Public dir: {exc}", "WARN")


def _prune_public() -> None:
    # Two PNGs (A + B) and one Markdown record per iteration; keep a generous window.
    for pat, keep in ((f"{RENDER_OBJECT}_*.png", PUBLIC_KEEP),
                      (f"{RENDER_OBJECT}_*.md", PUBLIC_KEEP // 2)):
        for old in sorted(PUBLIC_DIR.glob(pat))[:-keep]:
            try:
                old.unlink()
            except Exception:
                pass


def _copy(src: Path, dst: Path) -> None:
    try:
        shutil.copy2(src, dst)
        os.chmod(dst, 0o644)
    except Exception as exc:
        log(f"publish copy failed ({dst.name}): {exc}", "WARN")


def publish(iteration, generation, a_panel, b_panel, verdict, kept, change_desc, anchor_note):
    """Every evaluation, drop into ~/Public: the A image, the B image, and a Markdown
    record of the comparison (code change + judge feedback). Also refresh the live
    STATUS.md and a side-by-side `latest/` panel. Logs exactly what it dropped."""
    ensure_public()
    # Side-by-side current panel (overwritten each iteration) for live viewing.
    for f in PUBLIC_LATEST.glob("*.png"):
        try:
            f.unlink()
        except Exception:
            pass
    for (pa, m), (pb, _) in zip(a_panel, b_panel):
        seed, yaw, pitch = m
        _copy(pa, PUBLIC_LATEST / f"A_best_seed{seed}_y{yaw}_p{pitch}.png")
        _copy(pb, PUBLIC_LATEST / f"B_candidate_seed{seed}_y{yaw}_p{pitch}.png")

    # Timestamped record: A image + B image + a Markdown writeup. Prefer a view B
    # won so the human sees a real before/after; the three files share a stem.
    idx = next((i for i, pv in enumerate(verdict["per_view"]) if pv["winner"] == "B"), 0)
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    seed = a_panel[idx][1][0]
    stem = f"{RENDER_OBJECT}_{ts}_iter{iteration:04d}_seed{seed}"
    a_name, b_name, md_name = f"{stem}_A_best.png", f"{stem}_B_candidate.png", f"{stem}.md"
    _copy(a_panel[idx][0], PUBLIC_DIR / a_name)
    _copy(b_panel[idx][0], PUBLIC_DIR / b_name)
    try:
        (PUBLIC_DIR / md_name).write_text(
            _comparison_md_text(iteration, generation, verdict, kept, change_desc,
                                anchor_note, a_name, b_name), encoding="utf-8")
        os.chmod(PUBLIC_DIR / md_name, 0o644)
    except Exception as exc:
        log(f"could not write comparison record {md_name}: {exc}", "WARN")
    _prune_public()

    # Live status snapshot (overwritten each iteration).
    status = _status_text(iteration, generation, verdict, kept, change_desc, anchor_note)
    for target in (PUBLIC_DIR / "STATUS.md", PROJECT_ROOT / "OPTIMIZER_STATUS.md"):
        try:
            target.write_text(status, encoding="utf-8")
            os.chmod(target, 0o644)
        except Exception as exc:
            log(f"could not write status to {target}: {exc}", "WARN")

    verb = "KEPT new best" if kept else "rejected"
    log(f"~/Public ← dropped A/B + record [{verb}, B {verdict['b_votes']}-{verdict['a_votes']} A]: "
        f"{a_name}, {b_name}, {md_name}")


def _status_text(iteration, generation, verdict, kept, change_desc, anchor_note) -> str:
    rows = "\n".join(
        f"| seed {pv['seed']} | yaw {pv['yaw']} | pitch {pv['pitch']} | "
        f"{'**B (candidate)**' if pv['winner']=='B' else ('A (best)' if pv['winner']=='A' else 'tie')} |"
        for pv in verdict["per_view"]
    )
    outcome = "B KEPT as new best" if kept else "B rejected — current best retained"
    return (
        "# Tree optimizer — status (A/B)\n\n"
        f"_Updated {_ts()}_\n\n"
        f"- **Improvements kept (generation):** {generation}\n"
        f"- **Iterations run:** {iteration}\n"
        f"- **Last head-to-head:** candidate B {verdict['b_votes']} vs best A {verdict['a_votes']} "
        f"(ties {verdict['ties']}) over {len(verdict['per_view'])} views, need net +{VOTE_MARGIN} "
        f"→ **{outcome}**\n"
        f"- **Last change:** {change_desc}\n"
        f"- **Judge's reason (feedback to next rewrite):** {verdict['reason'] or '(none)'}\n"
        f"{('- **Cumulative check vs original:** ' + anchor_note + chr(10)) if anchor_note else ''}"
        "\n## This head-to-head, per view\n\n"
        f"| seed | yaw | pitch | more photorealistic |\n|---|---|---|---|\n{rows}\n\n"
        f"`latest/` holds the current A (best) and B (candidate) renders side by side. Each "
        f"evaluation also drops a timestamped before/after pair "
        f"(`{RENDER_OBJECT}_<ts>_..._A_best.png` / `_B_candidate.png`) and a matching `.md` record.\n"
    )


def _comparison_md_text(iteration, generation, verdict, kept, change_desc, anchor_note,
                        a_name, b_name) -> str:
    """The per-iteration record dropped in ~/Public: code change + judge feedback."""
    rows = "\n".join(
        f"| seed {pv['seed']} | yaw {pv['yaw']} | pitch {pv['pitch']} | "
        f"{'**B (candidate)**' if pv['winner']=='B' else ('A (best)' if pv['winner']=='A' else 'tie')} |"
        for pv in verdict["per_view"]
    )
    outcome = "B KEPT as the new best ✅" if kept else "B rejected — current best retained ↩︎"
    net = verdict["b_votes"] - verdict["a_votes"]
    return (
        f"# Broadleaf A/B comparison — iteration {iteration}\n\n"
        f"_{_ts()}_\n\n"
        f"- **Outcome:** {outcome}\n"
        f"- **Vote:** candidate **B {verdict['b_votes']}** – **A {verdict['a_votes']}** "
        f"(ties {verdict['ties']}) over {len(verdict['per_view'])} views; needed net "
        f"+{VOTE_MARGIN}, got {net:+d}\n"
        f"- **Improvements kept so far (generation):** {generation}\n"
        f"{('- **Cumulative vs ORIGINAL:** ' + anchor_note + chr(10)) if anchor_note else ''}"
        f"\n## Code change tried (B vs the current best A)\n\n{change_desc}\n\n"
        "## Judge feedback — why one looked more photorealistic (this drives the next rewrite)\n\n"
        f"{verdict['reason'] or '(none given)'}\n\n"
        "## Per-view verdict\n\n"
        f"| seed | yaw | pitch | more photorealistic |\n|---|---|---|---|\n{rows}\n\n"
        "## Images in this folder\n\n"
        f"- **A** (current best): `{a_name}`\n- **B** (candidate): `{b_name}`\n"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Notes journal  (deterministic, harness-owned — no model call)
# ──────────────────────────────────────────────────────────────────────────────

NOTES_GUIDE = """\
## How this loop works (and how to improve IT)

This optimizer is **fixed harness code**: neither `tree_optimizer.py` nor
`evolve.py` is ever edited by the run. The loop only edits the target
**`src/flora.rs`**; any out-of-scope edit is auto-reverted. To improve the
*optimizer*, do NOT expect it to improve itself — read the data it leaves and edit
`tree_optimizer.py` by hand:

- **`optimizer_history.jsonl`** — one record per iteration: the change, the A/B
  vote tally, every per-view winner, and the judge's reason. Analyze this to ask:
  are candidates winning? is the vote margin right? is the judge consistent?
- **`optimizer_log.txt`** — terse per-iteration lines.
- **`~/Public/planet-explorer/`** — labeled A (best) vs B (candidate) sample PNGs +
  `STATUS.md`, so the trees can be eyeballed from another account while it runs.

**The method:** each iteration is a head-to-head. The current best (A) and a new
candidate (B) are rendered over the same fixed panel of seeds/angles; a judge
picks the more photorealistic of each pair (no rubric — it decides what "better"
is), with presentation order randomized to cancel position bias. B replaces A only
if it wins by `VOTE_MARGIN` net votes. The judge's written reason becomes the
**feedback** for the next rewrite, so the improver iterates on real visual signal
plus its own ideas. Every `ANCHOR_EVERY` kept improvements the best is also judged
against the ORIGINAL tree to confirm cumulative progress.

If real wins keep getting rejected, lower `VOTE_MARGIN` or add panel views; if kept
changes still look like noise, raise it or add a second judge vote.
"""


def write_notes(state, last_record):
    gen = state.get("generation", 0)
    streak = state.get("loss_streak", 0)
    fb = state.get("last_feedback", "").strip()
    verdict = state.get("last_verdict", "(none yet)")
    if last_record and "b_votes" in last_record:
        lr = (f"candidate {last_record['b_votes']}–{last_record['a_votes']} best "
              f"(ties {last_record.get('ties', 0)}) → {last_record['decision']} — "
              f"{last_record['change']}")
    elif last_record:
        lr = f"{last_record['decision']} — {last_record['change']}"
    else:
        lr = "(no iteration completed yet)"
    anchor_checks = state.get("anchor", {}).get("checks", [])
    if anchor_checks:
        ac = anchor_checks[0]
        anchor_line = (f"- **Cumulative vs ORIGINAL (gen {ac['generation']}):** current "
                       f"{ac['current_votes']}–{ac['anchor_votes']} original → "
                       f"{'PROGRESS' if ac['current_votes'] > ac['anchor_votes'] else 'no net gain'}\n")
    else:
        anchor_line = ""
    journal = "\n".join(f"- {e}" for e in state.get("journal", [])[:JOURNAL_KEEP]) or "- (none yet)"
    content = (
        "# Optimization Notes — Photorealism Engine (Broadleaf trees, A/B)\n\n"
        "## Executive Summary\n\n"
        f"- **Improvements kept (generation):** {gen}  |  **iterations:** {state.get('iteration', 0)}"
        f"  |  **current losing streak:** {streak}\n"
        f"- **Loop status:** RUNNING — fixed harness; A/B judged; only edits `src/flora.rs`. "
        "evolve.py supervises the token budget (auto pause/resume).\n"
        f"- **Most recent head-to-head:** {lr}\n"
        f"{anchor_line}"
        f"- **Latest judge feedback (drives the next rewrite):** {fb or '(none yet)'}\n"
        f"- _Updated {_ts()}._\n\n"
        f"{NOTES_GUIDE}\n"
        "## Running Journal (newest first)\n\n"
        f"{journal}\n"
    )
    try:
        NOTES_FILE.write_text(content, encoding="utf-8")
    except Exception as exc:
        log(f"could not write notes: {exc}", "WARN")


def append_jsonl(record):
    try:
        with open(JSONL_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        log(f"could not append jsonl: {exc}", "WARN")


# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────

def default_state() -> dict:
    return {
        "iteration": 0,
        "generation": 0,            # number of KEPT improvements (the progress metric)
        "loss_streak": 0,
        "focus": RENDER_OBJECT,
        "last_feedback": "",        # judge's reasoning carried into the next rewrite
        "last_verdict": "(none yet)",
        "tried_changes": [],        # [{desc, result}]
        "history": [],
        "journal": [],
        "anchor": {"captured": False, "since_check": 0, "checks": []},
        "tree_source_files": [SRC_TARGET],
        "build_command": " ".join(BUILD_COMMAND),
        "render_command": (f"{RENDER_BIN} --render-to-image --object {RENDER_OBJECT} "
                           f"--seed <S> --yaw <Y> --pitch <P> --output <OUT> "
                           f"--width {RENDER_W} --height {RENDER_H}"),
        "vote_margin": VOTE_MARGIN,
        "sample_panel": list(SAMPLE_PANEL),
    }


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            st = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            for k, v in default_state().items():
                st.setdefault(k, v)
            st.setdefault("anchor", {}).setdefault("captured", False)
            st["anchor"].setdefault("since_check", 0)
            st["anchor"].setdefault("checks", [])
            return st
        except Exception as exc:
            log(f"could not load state ({exc}); starting fresh", "WARN")
    return default_state()


def save_state(state: dict) -> None:
    """Atomic write (temp + rename) so a crash mid-write can't corrupt state."""
    tmp = STATE_FILE.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(tmp, STATE_FILE)
    except Exception as exc:
        log(f"could not save state: {exc}", "ERROR")


# ──────────────────────────────────────────────────────────────────────────────
# Iteration bookkeeping
# ──────────────────────────────────────────────────────────────────────────────

def _record(iteration, change, build_ok, decision, verdict, reverted):
    rec = {
        "iteration": iteration,
        "ts": datetime.datetime.now().isoformat(),
        "change": change,
        "build_ok": build_ok,
        "decision": decision,
        "out_of_scope_reverted": reverted or [],
    }
    if verdict:
        rec.update({
            "a_votes": verdict["a_votes"], "b_votes": verdict["b_votes"],
            "ties": verdict["ties"], "net_b_minus_a": verdict["b_votes"] - verdict["a_votes"],
            "vote_margin": VOTE_MARGIN, "per_view": verdict["per_view"],
            "judge_reason": verdict["reason"],
        })
    return rec


def _finish(state, rec, change_desc, kept):
    state["tried_changes"].append({"desc": change_desc, "result": rec["decision"]})
    state["tried_changes"] = state["tried_changes"][-HISTORY_KEEP:]
    state["history"].append(rec)
    state["history"] = state["history"][-HISTORY_KEEP:]
    if "b_votes" in rec:
        tally = f"B {rec['b_votes']}–{rec['a_votes']} A"
    else:
        tally = "build failed"
    tag = "KEPT (new best)" if kept else rec["decision"].replace("_", " ").lower()
    state["journal"].insert(
        0, f"{_ts()} (iter {rec['iteration']}, gen {state['generation']}): {tag} [{tally}] — {change_desc}")
    state["journal"] = state["journal"][:JOURNAL_KEEP]
    append_jsonl(rec)
    write_notes(state, rec)
    state["iteration"] += 1
    save_state(state)


# ──────────────────────────────────────────────────────────────────────────────
# Anchor check — cumulative progress vs the ORIGINAL tree
# ──────────────────────────────────────────────────────────────────────────────

def capture_anchor(a_panel) -> None:
    """Save the original tree's renders + source as the cumulative-progress anchor."""
    if ANCHOR_DIR.exists():
        shutil.rmtree(ANCHOR_DIR, ignore_errors=True)
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    for p, _ in a_panel:
        shutil.copy2(p, ANCHOR_DIR / p.name)
    save_anchor_source()
    log("captured ORIGINAL tree as the anchor (renders/anchor + snapshots/anchor)")


def anchor_check(state, current_panel) -> str:
    """Judge the current best against the original anchor. Returns a one-line note."""
    anchor_panel = panel_from_dir(ANCHOR_DIR)
    if not anchor_panel:
        return ""
    # A = anchor (original), B = current best → B-votes are "current beats original".
    verdict = judge_ab(anchor_panel, current_panel, RENDERS / "anchor_verdict.json",
                       salt=900000 + state["generation"])
    if not verdict:
        return ""
    rec = {
        "ts": datetime.datetime.now().isoformat(),
        "generation": state["generation"],
        "anchor_votes": verdict["a_votes"],
        "current_votes": verdict["b_votes"],
        "ties": verdict["ties"],
        "reason": verdict["reason"],
    }
    state["anchor"].setdefault("checks", []).insert(0, rec)
    state["anchor"]["checks"] = state["anchor"]["checks"][:20]
    append_jsonl({"anchor_check": rec})
    progress = "PROGRESS vs original" if verdict["b_votes"] > verdict["a_votes"] else "NO net gain vs original"
    log(f"ANCHOR CHECK (gen {state['generation']}): current {verdict['b_votes']}–{verdict['a_votes']} "
        f"original → {progress}. {verdict['reason'][:160]}")
    return f"current {verdict['b_votes']}–{verdict['a_votes']} original ({progress})"


# ──────────────────────────────────────────────────────────────────────────────
# One iteration (A/B head-to-head with feedback)
# ──────────────────────────────────────────────────────────────────────────────

def run_iteration(state: dict) -> dict:
    it = state["iteration"]
    log(f"=== ITER {it} begin (generation {state['generation']}, "
        f"loss_streak {state['loss_streak']}) ===")

    # 0. Build the current BEST (A) and render its panel.
    ok, err = build()
    if not ok:
        log("current best does not build; restoring snapshots/best", "ERROR")
        if not restore_best():
            log("no best snapshot to restore from; sleeping before retry", "ERROR")
            time.sleep(SKIP_SLEEP)
            return state
        build()
        time.sleep(SKIP_SLEEP)
        return state
    a_panel = render_panel(A_DIR)
    if len(a_panel) < len(SAMPLE_PANEL):
        log(f"A (best) render incomplete ({len(a_panel)}/{len(SAMPLE_PANEL)}); skipping", "WARN")
        time.sleep(SKIP_SLEEP)
        return state

    # Capture the original as the anchor once (cumulative-progress reference).
    if not state["anchor"].get("captured"):
        capture_anchor(a_panel)
        state["anchor"]["captured"] = True
        save_state(state)

    # 1. Snapshot A, then rewrite → B (feedback-driven; target-only, guarded).
    snap = snapshot_target(it)
    protected = {rel: _read_bytes(PROJECT_ROOT / rel) for rel in PROTECTED_FILES}
    CANDIDATE_CHANGE = B_DIR
    CANDIDATE_CHANGE.mkdir(parents=True, exist_ok=True)
    change_file = CANDIDATE_CHANGE / "change.txt"
    modify(state, change_file)
    reverted = guard_out_of_scope(protected)
    if reverted:
        log(f"scope guard reverted out-of-target edits: {reverted}", "WARN")
    change_desc = _read_change(change_file)

    # 2. Build B.
    ok, err = build()
    if not ok:
        restore_from(snap)
        state["loss_streak"] += 1
        last_err = err.strip().splitlines()[-1] if err.strip() else "unknown error"
        state["last_feedback"] = (f"Your previous change did not COMPILE: {last_err}. "
                                  "Make a simpler, self-contained change that builds cleanly.")
        state["last_verdict"] = "build failed"
        log(f"ITER {it}: BUILD_FAILED — reverted. {change_desc}", "WARN")
        rec = _record(it, change_desc, False, "BUILD_FAILED", None, reverted)
        _finish(state, rec, change_desc, kept=False)
        return state

    # 3. Render B.
    b_panel = render_panel(B_DIR)
    if len(b_panel) < len(SAMPLE_PANEL):
        restore_from(snap)
        state["loss_streak"] += 1
        log(f"ITER {it}: candidate render incomplete; reverted", "WARN")
        time.sleep(SKIP_SLEEP)
        return state

    # 4. A/B JUDGE (no rubric; randomized order; per-view vote + a written reason).
    verdict = judge_ab(a_panel, b_panel, B_DIR / "verdict.json", salt=it)
    if verdict is None:
        restore_from(snap)
        state["loss_streak"] += 1
        log(f"ITER {it}: judge failed; reverted", "WARN")
        time.sleep(SKIP_SLEEP)
        return state

    # 5. Decide: B replaces A only if it wins by VOTE_MARGIN net votes.
    net = verdict["b_votes"] - verdict["a_votes"]
    kept = verdict["b_votes"] > verdict["a_votes"] and net >= VOTE_MARGIN
    anchor_note = ""
    if kept:
        save_best()
        if BEST_DIR.exists():
            shutil.rmtree(BEST_DIR, ignore_errors=True)
        shutil.copytree(B_DIR, BEST_DIR)
        state["generation"] += 1
        state["loss_streak"] = 0
        state["anchor"]["since_check"] = state["anchor"].get("since_check", 0) + 1
        log(f"ITER {it}: B beats A {verdict['b_votes']}–{verdict['a_votes']} "
            f"(net {net:+d} ≥ {VOTE_MARGIN}) | KEPT — generation {state['generation']}. {change_desc}")
    else:
        restore_from(snap)
        state["loss_streak"] += 1
        log(f"ITER {it}: B vs A {verdict['b_votes']}–{verdict['a_votes']} "
            f"(net {net:+d}) | REJECTED. {change_desc}")

    # 6. Feedback loop: the judge's reasoning drives the next rewrite.
    state["last_feedback"] = verdict["reason"]
    state["last_verdict"] = "KEPT as the new best" if kept else "REJECTED (current best retained)"
    if verdict["reason"]:
        log(f"ITER {it}: judge feedback — {verdict['reason']}")

    # 7. Periodic cumulative check vs the original.
    if kept and state["anchor"].get("since_check", 0) >= ANCHOR_EVERY:
        build()  # ensure the binary is the new best before re-rendering it
        # Render into a SEPARATE dir so a_panel/b_panel (A_DIR/B_DIR) stay valid for publish().
        cur = render_panel(ANCHOR_CMP_DIR)
        if len(cur) == len(SAMPLE_PANEL):
            anchor_note = anchor_check(state, cur)
        state["anchor"]["since_check"] = 0

    rec = _record(it, change_desc, True, "KEPT" if kept else "REJECTED", verdict, reverted)
    _finish(state, rec, change_desc, kept)
    publish(it, state["generation"], a_panel, b_panel, verdict, kept, change_desc, anchor_note)
    return state


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    log(f"tree_optimizer starting (FIXED HARNESS, A/B); model={CLAUDE_MODEL}; "
        f"object={RENDER_OBJECT}; panel={len(SAMPLE_PANEL)} views; vote_margin={VOTE_MARGIN}")
    ensure_public()

    if not RENDER_BIN.exists():
        log("render binary missing — building once before the loop", "INFO")
        ok, err = build()
        if not ok or not RENDER_BIN.exists():
            log("initial build failed; cannot render. Is the --render-to-image CLI present? "
                f"Last output:\n{err[-800:]}", "ERROR")
            sys.exit(1)

    state = load_state()
    if not (SNAP_BEST / "src" / "flora.rs").exists():
        save_best()  # seed the best snapshot from the current (assumed-best) source
    write_notes(state, state["history"][-1] if state["history"] else None)

    while True:
        try:
            state = run_iteration(state)
            time.sleep(ITER_SLEEP)
        except KeyboardInterrupt:
            save_state(state)
            log(f"[interrupted] iterations: {state['iteration']}, "
                f"improvements kept (generation): {state['generation']}")
            sys.exit(0)
        except Exception as exc:
            import traceback
            log(f"unexpected error: {exc}\n{traceback.format_exc()}", "ERROR")
            restore_best()  # never leave the tree broken
            save_state(state)
            time.sleep(SKIP_SLEEP)


if __name__ == "__main__":
    main()
