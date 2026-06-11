#!/usr/bin/env python3
"""Generate photorealistic reference plates for the P0/P1 flora/object targets.

Pipeline context: these PNGs are the *reference images* fed to ../formcast to
produce 3D models that replace Planet Explorer's procedural flora. See
FLORA_MODEL_TARGETS.md for the full catalogue; this script covers the **P0** and
**P1** items (the prompts below are copied verbatim from that doc's
"Image-Generator Prompt" column). Each item carries its priority so you can
generate one tier at a time with --priority.

It calls OpenAI's **GPT Image 2** (`gpt-image-2`) — the current best image model
(reasoning/"thinking mode", up to 4K) — via the synchronous Images API. The call
blocks until the finished image returns, so there is nothing to poll; the only
loop here is a retry-with-backoff around transient failures (rate limits, 5xx).

The API key is read from a `.env` file sitting next to this script
(`flora-revamp/.env`, git-ignored), falling back to the OPENAI_API_KEY
environment variable if already set.

Usage:
    pip install openai
    # key lives in flora-revamp/.env as: OPENAI_API_KEY=sk-...
    python flora-revamp/generate_images.py                  # any MISSING plate (P0+P1)
    python flora-revamp/generate_images.py --priority P1     # only the P1 plates
    python flora-revamp/generate_images.py --force           # regenerate ALL plates
    python flora-revamp/generate_images.py --only barrel-cactus  # just one
    python flora-revamp/generate_images.py --list            # list item ids and exit
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

# --- Configuration ----------------------------------------------------------

MODEL = "gpt-image-2"  # OpenAI's latest/best image model (launched 2026-04-21)
QUALITY = "high"  # gpt-image-2 quality tier: low | medium | high | auto
N_PER_ITEM = 1  # one plate per target; formcast reconstructs a single object

# Standard gpt-image aspect presets. Tall subjects (trees, columnar cactus) get
# a portrait frame so the whole specimen fits without wasting pixels on the sides;
# broad/low subjects get a square frame.
SIZE_PORTRAIT = "1024x1536"
SIZE_SQUARE = "1024x1024"

# Retry/backoff for transient API errors (rate limit, 5xx, network blips). The
# generation call itself is synchronous, so this is failure-retry, not polling.
MAX_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 4.0  # first retry waits this long, then doubles
BACKOFF_MAX_SECONDS = 60.0  # cap per-attempt backoff
REQUEST_TIMEOUT_SECONDS = 300.0  # high-quality + thinking-mode can be slow

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR / ".env"
OUTPUT_DIR = SCRIPT_DIR / "images"

# --- The P0 targets (prompts copied verbatim from FLORA_MODEL_TARGETS.md) ----
# id == output filename stem (the link in the doc's "Image" column).

# Shared capture recipe. Every subject is ONE instance ("A single ..."): the
# renderer instances the model many times, so a cluster baked into the plate would
# become one un-repeatable mesh. See FLORA_MODEL_TARGETS.md "Prompt conventions".
_RECIPE = (
    " Photorealistic, single isolated specimen centered and fully in frame, "
    "plain seamless neutral mid-grey background, soft even diffuse lighting, "
    "no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, "
    "no text, no people, no watermark."
)

# --- P0: highest visual payoff (the iconic form of each dominant biome) ---
ITEMS = [
    {
        "id": "granite-boulder",
        "priority": "P0",
        "size": SIZE_SQUARE,
        "prompt": (
            "A single weathered grey granite boulder, rounded by glaciation with "
            "angular fractured faces and faint lichen flecks." + _RECIPE
        ),
    },
    {
        "id": "broadleaf-hardwood",
        "priority": "P0",
        "size": SIZE_PORTRAIT,
        "prompt": (
            "A single mature deciduous hardwood tree with a dense rounded full green "
            "canopy, a sturdy single trunk and a branching crown in summer foliage, "
            "sugar-maple form, whole tree shown." + _RECIPE
        ),
    },
    {
        "id": "spreading-oak",
        "priority": "P0",
        "size": SIZE_SQUARE,
        "prompt": (
            "A single large mature oak tree with a broad irregular spreading crown, a "
            "thick gnarled trunk and heavy lateral limbs, dense dark-green summer "
            "foliage, whole tree shown." + _RECIPE
        ),
    },
    {
        "id": "spruce-spire-conifer",
        "priority": "P0",
        "size": SIZE_PORTRAIT,
        "prompt": (
            "A single tall narrow spruce tree, steeply conical with a pointed top and "
            "dense dark blue-green needled branches drooping slightly, straight trunk, "
            "whole tree shown." + _RECIPE
        ),
    },
    {
        "id": "tropical-emergent-tree",
        "priority": "P0",
        "size": SIZE_PORTRAIT,
        "prompt": (
            "A single towering rainforest kapok tree with a tall straight pale trunk "
            "and a wide flat umbrella-shaped crown of green foliage high above, "
            "emergent giant, whole tree shown." + _RECIPE
        ),
    },
    {
        "id": "feather-frond-palm",
        "priority": "P0",
        "size": SIZE_PORTRAIT,
        "prompt": (
            "A single tall coconut palm with a slender slightly curved trunk topped by "
            "a crown of long arching pinnate green fronds, whole tree shown." + _RECIPE
        ),
    },
    {
        "id": "bunchgrass-tussock",
        "priority": "P0",
        "size": SIZE_SQUARE,
        "prompt": (
            "A single dense tussock of tall bunchgrass, fine arching golden-green "
            "blades radiating from one tight base." + _RECIPE
        ),
    },
    {
        "id": "savanna-acacia",
        "priority": "P0",
        "size": SIZE_SQUARE,
        "prompt": (
            "A single umbrella acacia tree with a clear trunk and a high wide "
            "flat-topped canopy of fine green foliage, classic savanna form, whole "
            "tree shown." + _RECIPE
        ),
    },
    {
        "id": "columnar-cactus",
        "priority": "P0",
        "size": SIZE_PORTRAIT,
        "prompt": (
            "A single tall saguaro cactus, a ribbed green column with two raised "
            "curving arms and rows of spines along the ridges, whole plant shown."
            + _RECIPE
        ),
    },
    # --- P1: variety + realism (build after the P0 layer reads well) -------
    {
        "id": "rock-outcrop",
        "priority": "P1",
        "size": SIZE_SQUARE,
        "prompt": (
            "A single low outcrop of fractured stratified sandstone bedrock, "
            "horizontal layered beds with cracked weathered edges, sun-bleached tan "
            "and grey." + _RECIPE
        ),
    },
    {
        "id": "fallen-log",
        "priority": "P1",
        "size": SIZE_SQUARE,
        "prompt": (
            "A single fallen tree trunk lying horizontally, bark mostly stripped to "
            "bare weathered grey wood, split and cracked along its length with a "
            "broken root flare at one end." + _RECIPE
        ),
    },
    {
        "id": "white-birch",
        "priority": "P1",
        "size": SIZE_PORTRAIT,
        "prompt": (
            "A single tall slender birch tree with smooth white papery bark marked "
            "with dark scars and a light airy canopy of small green leaves, whole "
            "tree shown." + _RECIPE
        ),
    },
    {
        "id": "understory-shrub",
        "priority": "P1",
        "size": SIZE_SQUARE,
        "prompt": (
            "A single rounded leafy deciduous shrub about waist-to-head height, dense "
            "green foliage on several woody stems rising from the base, hazel form."
            + _RECIPE
        ),
    },
    {
        "id": "dense-fir",
        "priority": "P1",
        "size": SIZE_PORTRAIT,
        "prompt": (
            "A single full conical fir tree with soft dense deep-green foliage in "
            "layered branches and a slightly rounded profile, balsam-fir form, whole "
            "tree shown." + _RECIPE
        ),
    },
    {
        "id": "long-needle-pine",
        "priority": "P1",
        "size": SIZE_PORTRAIT,
        "prompt": (
            "A single open irregular pine tree with long green needles in tufts, a "
            "visible reddish-brown trunk and bare lower branches, Scots-pine form, "
            "whole tree shown." + _RECIPE
        ),
    },
    {
        "id": "tree-fern",
        "priority": "P1",
        "size": SIZE_PORTRAIT,
        "prompt": (
            "A single tree fern with a short slender fibrous brown trunk topped by a "
            "rosette of large lacy arching green fronds, whole plant shown." + _RECIPE
        ),
    },
    {
        "id": "broadleaf-understory",
        "priority": "P1",
        "size": SIZE_SQUARE,
        "prompt": (
            "A single banana-like understory plant with very large broad paddle-shaped "
            "bright-green leaves rising and arching from the base." + _RECIPE
        ),
    },
    {
        "id": "hanging-liana",
        "priority": "P1",
        "size": SIZE_PORTRAIT,
        "prompt": (
            "A single woody jungle liana vine, twisting and looping with sparse green "
            "leaves." + _RECIPE
        ),
    },
    {
        "id": "sagebrush",
        "priority": "P1",
        "size": SIZE_SQUARE,
        "prompt": (
            "A single low silvery sagebrush shrub, a woody base with soft grey-green "
            "aromatic foliage in a rounded sparse form." + _RECIPE
        ),
    },
    {
        "id": "meadow-wildflower",
        "priority": "P1",
        "size": SIZE_PORTRAIT,
        "prompt": (
            "A single tall prairie wildflower plant, a green stem topped with bright "
            "red and yellow blooms." + _RECIPE
        ),
    },
    {
        "id": "sandstone-hoodoo",
        "priority": "P1",
        "size": SIZE_PORTRAIT,
        "prompt": (
            "A single eroded red-orange sandstone hoodoo, a wind-carved layered rock "
            "spire with horizontal striations and a rounded balanced cap." + _RECIPE
        ),
    },
    {
        "id": "barrel-cactus",
        "priority": "P1",
        "size": SIZE_SQUARE,
        "prompt": (
            "A single squat barrel cactus, a ribbed green sphere with dense radiating "
            "spines." + _RECIPE
        ),
    },
    {
        "id": "creosote-shrub",
        "priority": "P1",
        "size": SIZE_SQUARE,
        "prompt": (
            "A single sparse wiry creosote bush, thin woody branches with tiny "
            "dark-green leaves in an open airy desert shrub form." + _RECIPE
        ),
    },
    {
        "id": "dwarf-shrub",
        "priority": "P1",
        "size": SIZE_SQUARE,
        "prompt": (
            "A single low ground-hugging dwarf willow plant, small woody stems with "
            "tiny green leaves spreading flat and wind-pruned." + _RECIPE
        ),
    },
    {
        "id": "cotton-grass",
        "priority": "P1",
        "size": SIZE_SQUARE,
        "prompt": (
            "A single tussock cottongrass plant, a tuft of fine green sedge blades "
            "topped with fluffy white seed-heads." + _RECIPE
        ),
    },
    {
        "id": "lichen-boulder",
        "priority": "P1",
        "size": SIZE_SQUARE,
        "prompt": (
            "A single grey granite boulder crusted with orange, pale-green and grey "
            "lichen blotches, weathered tundra rock." + _RECIPE
        ),
    },
    {
        "id": "leaning-palm",
        "priority": "P1",
        "size": SIZE_PORTRAIT,
        "prompt": (
            "A single coconut palm with a curved wind-leaned trunk and a crown of long "
            "arching green fronds, tropical shoreline form, whole tree shown." + _RECIPE
        ),
    },
    {
        "id": "dune-grass",
        "priority": "P1",
        "size": SIZE_SQUARE,
        "prompt": (
            "A single tuft of tall wispy marram dune grass, thin pale-green blades "
            "rising from one sandy base." + _RECIPE
        ),
    },
    {
        "id": "talus-rock",
        "priority": "P1",
        "size": SIZE_SQUARE,
        "prompt": (
            "A single angular broken grey rock fragment with sharp fractured edges, "
            "alpine scree stone." + _RECIPE
        ),
    },
    {
        "id": "krummholz-conifer",
        "priority": "P1",
        "size": SIZE_SQUARE,
        "prompt": (
            "A single stunted wind-bent krummholz conifer, a low gnarled twisted trunk "
            "with foliage flagged to one side, treeline dwarf, whole plant shown."
            + _RECIPE
        ),
    },
    {
        "id": "snow-laden-conifer",
        "priority": "P1",
        "size": SIZE_PORTRAIT,
        "prompt": (
            "A single spruce tree heavily laden with snow, dark green branches bowed "
            "under thick white snow caps, conical winter form, whole tree shown."
            + _RECIPE
        ),
    },
]


# --- Helpers ----------------------------------------------------------------


def log(msg: str) -> None:
    """Timestamped, immediately-flushed stdout line.

    The explicit flush matters: Python block-buffers stdout when it is NOT a
    terminal (redirected to a file, piped to `tee`, run under a wrapper), so
    without this you'd see no progress for minutes and then a burst at the end.
    flush=True keeps every line live regardless of where stdout goes.
    """
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def load_env_key() -> str:
    """Return the OpenAI API key, preferring the process env, then .env."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()

    if ENV_PATH.is_file():
        for raw in ENV_PATH.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() == "OPENAI_API_KEY":
                # strip optional surrounding quotes
                return value.strip().strip('"').strip("'")

    sys.exit(
        f"error: no OPENAI_API_KEY found. Put it in {ENV_PATH} as\n"
        f"    OPENAI_API_KEY=sk-...\n"
        f"or export it into the environment."
    )


def generate_one(client, item: dict) -> bytes:
    """Generate a single plate, retrying transient failures with backoff."""
    last_err: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client.images.generate(
                model=MODEL,
                prompt=item["prompt"],
                size=item["size"],
                quality=QUALITY,
                n=N_PER_ITEM,
            )
            datum = resp.data[0]
            # gpt-image models return base64 by default; tolerate a URL too.
            if getattr(datum, "b64_json", None):
                return base64.b64decode(datum.b64_json)
            url = getattr(datum, "url", None)
            if url:
                import urllib.request

                with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as r:
                    return r.read()
            raise RuntimeError("response contained neither b64_json nor url")
        except Exception as err:  # noqa: BLE001 — retry only transient failures
            last_err = err
            # Retry transient failures only: 429 (rate limit), 5xx (server), or no
            # HTTP status at all (connection/timeout). A 4xx like 401/400/404 is a
            # permanent client error — retrying just wastes time and floods the log.
            status = getattr(err, "status_code", None)
            retryable = status is None or status == 429 or status >= 500
            if not retryable:
                log(f"        non-retryable error (HTTP {status}); not retrying")
                break
            if attempt == MAX_ATTEMPTS:
                break
            wait = min(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), BACKOFF_MAX_SECONDS)
            log(f"        attempt {attempt}/{MAX_ATTEMPTS} failed: {err}; retrying in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"giving up after {attempt} attempt(s): {last_err}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="regenerate plates that already exist")
    parser.add_argument("--only", metavar="ID", help="generate only this item id (see --list)")
    parser.add_argument("--priority", choices=["P0", "P1"], help="generate only items of this priority tier")
    parser.add_argument("--list", action="store_true", help="print the item ids and exit")
    args = parser.parse_args()

    if args.list:
        for item in ITEMS:
            print(f"{item['id']:28} {item['priority']}  {item['size']}")
        return 0

    items = ITEMS
    if args.only:
        items = [it for it in ITEMS if it["id"] == args.only]
        if not items:
            sys.exit(f"error: no item with id '{args.only}'. Run --list to see ids.")
    if args.priority:
        items = [it for it in items if it["priority"] == args.priority]
        if not items:
            sys.exit(f"error: no items with priority '{args.priority}'.")

    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("error: the 'openai' package is not installed. Run: pip install openai")

    client = OpenAI(api_key=load_env_key(), timeout=REQUEST_TIMEOUT_SECONDS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(items)
    log(f"model={MODEL} quality={QUALITY} | {total} item(s) -> {OUTPUT_DIR}")
    generated = skipped = failed = 0
    for idx, item in enumerate(items, 1):
        tag = f"[{idx}/{total}] {item['id']}"
        out_path = OUTPUT_DIR / f"{item['id']}.png"
        if out_path.exists() and not args.force:
            log(f"{tag}: skip (exists; use --force to regenerate)")
            skipped += 1
            continue
        log(f"{tag}: sending [{item['size']}] -> GPT Image 2")
        log(f"        prompt: {item['prompt']}")
        start = time.perf_counter()
        try:
            png = generate_one(client, item)
        except Exception as err:  # noqa: BLE001
            log(f"{tag}: FAIL after {time.perf_counter() - start:.1f}s: {err}")
            failed += 1
            continue
        out_path.write_bytes(png)
        rel = out_path.relative_to(SCRIPT_DIR.parent)
        log(f"{tag}: saved {rel} ({len(png) // 1024} KB, {time.perf_counter() - start:.1f}s)")
        generated += 1

    log(f"done: {generated} generated, {skipped} skipped, {failed} failed -> {OUTPUT_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
