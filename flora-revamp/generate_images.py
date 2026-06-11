#!/usr/bin/env python3
"""Generate photorealistic reference plates for the P0 flora/object targets.

Pipeline context: these PNGs are the *reference images* fed to ../formcast to
produce 3D models that replace Planet Explorer's procedural flora. See
FLORA_MODEL_TARGETS.md for the full catalogue; this script covers only the
**P0** items (the prompts below are copied verbatim from that doc's
"Image-Generator Prompt" column).

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
    python flora-revamp/generate_images.py             # generate any MISSING P0 plates
    python flora-revamp/generate_images.py --force      # regenerate ALL P0 plates
    python flora-revamp/generate_images.py --only columnar-cactus   # just one
    python flora-revamp/generate_images.py --list       # list item ids and exit
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

_RECIPE = (
    " Photorealistic, single isolated {framing} centered and fully in frame, "
    "plain seamless neutral mid-grey background, soft even diffuse lighting, "
    "no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, "
    "no text, no people, no watermark."
)

ITEMS = [
    {
        "id": "granite-boulder-cluster",
        "size": SIZE_SQUARE,
        "prompt": (
            "A cluster of three or four weathered grey granite boulders grouped "
            "together, rounded by glaciation with angular fractured faces and faint "
            "lichen flecks." + _RECIPE.format(framing="group")
        ),
    },
    {
        "id": "broadleaf-hardwood",
        "size": SIZE_PORTRAIT,
        "prompt": (
            "A mature deciduous hardwood tree with a dense rounded full green canopy, "
            "a single sturdy trunk and a branching crown in summer foliage, sugar-maple "
            "form, whole tree shown." + _RECIPE.format(framing="specimen")
        ),
    },
    {
        "id": "spreading-oak",
        "size": SIZE_SQUARE,
        "prompt": (
            "A large mature oak tree with a broad irregular spreading crown, a thick "
            "gnarled trunk and heavy lateral limbs, dense dark-green summer foliage, "
            "whole tree shown." + _RECIPE.format(framing="specimen")
        ),
    },
    {
        "id": "spruce-spire-conifer",
        "size": SIZE_PORTRAIT,
        "prompt": (
            "A tall narrow spruce tree, steeply conical with a pointed top and dense "
            "dark blue-green needled branches drooping slightly, straight trunk, whole "
            "tree shown." + _RECIPE.format(framing="specimen")
        ),
    },
    {
        "id": "tropical-emergent-tree",
        "size": SIZE_PORTRAIT,
        "prompt": (
            "A towering rainforest kapok tree with a tall straight pale trunk and a "
            "wide flat umbrella-shaped crown of green foliage high above, emergent "
            "giant, whole tree shown." + _RECIPE.format(framing="specimen")
        ),
    },
    {
        "id": "feather-frond-palm",
        "size": SIZE_PORTRAIT,
        "prompt": (
            "A tall coconut palm with a slender slightly curved trunk topped by a crown "
            "of long arching pinnate green fronds, whole tree shown."
            + _RECIPE.format(framing="specimen")
        ),
    },
    {
        "id": "bunchgrass-tussock",
        "size": SIZE_SQUARE,
        "prompt": (
            "A dense clump of tall bunchgrass, fine arching golden-green blades "
            "radiating from a tight base, prairie tussock."
            + _RECIPE.format(framing="specimen")
        ),
    },
    {
        "id": "savanna-acacia",
        "size": SIZE_SQUARE,
        "prompt": (
            "A lone umbrella acacia tree with a clear trunk and a high wide flat-topped "
            "canopy of fine green foliage, classic savanna form, whole tree shown."
            + _RECIPE.format(framing="specimen")
        ),
    },
    {
        "id": "columnar-cactus",
        "size": SIZE_PORTRAIT,
        "prompt": (
            "A tall saguaro cactus, a single ribbed green column with two raised curving "
            "arms and rows of spines along the ridges, whole plant shown."
            + _RECIPE.format(framing="specimen")
        ),
    },
]


# --- Helpers ----------------------------------------------------------------


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
        except Exception as err:  # noqa: BLE001 — retry anything transient-looking
            last_err = err
            if attempt == MAX_ATTEMPTS:
                break
            wait = min(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), BACKOFF_MAX_SECONDS)
            print(f"    attempt {attempt}/{MAX_ATTEMPTS} failed: {err}; retrying in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"giving up after {MAX_ATTEMPTS} attempts: {last_err}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="regenerate plates that already exist")
    parser.add_argument("--only", metavar="ID", help="generate only this item id (see --list)")
    parser.add_argument("--list", action="store_true", help="print the item ids and exit")
    args = parser.parse_args()

    if args.list:
        for item in ITEMS:
            print(f"{item['id']:28} {item['size']}")
        return 0

    items = ITEMS
    if args.only:
        items = [it for it in ITEMS if it["id"] == args.only]
        if not items:
            sys.exit(f"error: no item with id '{args.only}'. Run --list to see ids.")

    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("error: the 'openai' package is not installed. Run: pip install openai")

    client = OpenAI(api_key=load_env_key(), timeout=REQUEST_TIMEOUT_SECONDS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated = skipped = failed = 0
    for item in items:
        out_path = OUTPUT_DIR / f"{item['id']}.png"
        if out_path.exists() and not args.force:
            print(f"skip   {item['id']} (exists; use --force to regenerate)")
            skipped += 1
            continue
        print(f"gen    {item['id']} [{item['size']}] ...")
        try:
            png = generate_one(client, item)
        except Exception as err:  # noqa: BLE001
            print(f"FAIL   {item['id']}: {err}")
            failed += 1
            continue
        out_path.write_bytes(png)
        print(f"  ->   {out_path.relative_to(SCRIPT_DIR.parent)} ({len(png) // 1024} KB)")
        generated += 1

    print(f"\ndone: {generated} generated, {skipped} skipped, {failed} failed -> {OUTPUT_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
