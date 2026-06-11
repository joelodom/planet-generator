# Formcast bake plan — P0 model library

Plan for turning the 9 **P0 reference plates** (`flora-revamp/images/*.png`, see
[`FLORA_MODEL_TARGETS.md`](FLORA_MODEL_TARGETS.md)) into a starter library of 3D
models using **`../formcast`** — **3 seed-varied variants per item** — placed in
a new **`flora-revamp/models/`** folder. These models will later be imported
into Planet Explorer as per-species instanced meshes (import work is out of
scope here).

## What formcast gives us (verified against formcast 1.0.0)

- `python formcast.py bake <photo> --out-dir D --count 3` bakes **3 seed-varied
  `.glb` models** (seeds 0..2) from one reference image: believable variations
  of the archetype, not copies. Exactly what an instanced species library wants.
- Output models are **+Y-up, in meters, base on the ground plane**, within
  per-density triangle budgets — a sane, consistent import contract.
- Each `.glb` embeds its own provenance (description, source photo, exact
  generator script) — readable later via `formcast.py inspect`, re-bakeable.
- **Filenames use formcast's own classifier stem** (e.g. `saguaro-cactus-00.glb`),
  *not* our input filename — so a rename step is needed to keep the library keyed
  by our catalogue ids.
- `bake` drives the local `claude` CLI (default `--model opus`, Read-tool-only)
  through ~3 authoring passes + 1 refine round, then executes the generated
  Python locally to bake. Budget roughly **10–25 min per item ⇒ ~2–4 h for all
  9**, run **sequentially** (each bake is itself multi-process; don't parallelize
  Claude sessions).
- `view --save` works **headless** (software-renderer fallback) — so preview
  contact sheets can be produced from this SSH account without a display.

## Target layout

One subfolder per archetype, stems renamed to match the catalogue id
(collision-proof regardless of what stem formcast picks, and the natural shape
for a species library):

```
flora-revamp/models/
  granite-boulder/
    granite-boulder-00.glb
    granite-boulder-01.glb
    granite-boulder-02.glb
    preview.png              # 3-up contact sheet (headless render)
  broadleaf-hardwood/
    ...
```

## The 9 items

Same ids as `images/` and the catalogue (the bake input is
`flora-revamp/images/<id>.png` in every case):

| # | id | formcast object class (expected) |
|---|----|-----------------------------------|
| 1 | granite-boulder | rock (noise-displaced hull) |
| 2 | broadleaf-hardwood | tree (foliage envelope + leaf-card clumps) |
| 3 | spreading-oak | tree |
| 4 | spruce-spire-conifer | tree (conical) |
| 5 | tropical-emergent-tree | tree (umbrella crown) |
| 6 | feather-frond-palm | tree (frond crown) |
| 7 | bunchgrass-tussock | grass/shrub |
| 8 | savanna-acacia | tree (flat-topped) |
| 9 | columnar-cactus | plant (ribbed column) |

## Steps

### 0. Preflight (one-time)

1. `pip install -r /Users/claude/formcast/requirements.txt` (numpy, trimesh, …).
2. Confirm the `claude` CLI is on PATH and authenticated (`claude --version`).
3. `mkdir -p /Users/claude/planet-explorer/flora-revamp/models`.
4. Sanity-check formcast end-to-end is healthy: `python formcast.py --version`
   from `/Users/claude/formcast` (already verified: 1.0.0).

### 1. Bake (per item, sequential)

Run from `/Users/claude/formcast` (its `formcast.log` stays in its own repo,
which gitignores it). Keep the defaults — `--density high`, `--refine 1`,
`--model opus` — for the first pass; tune only if results disappoint.

```bash
cd /Users/claude/formcast
IMAGES=/Users/claude/planet-explorer/flora-revamp/images
MODELS=/Users/claude/planet-explorer/flora-revamp/models
for id in granite-boulder broadleaf-hardwood spreading-oak \
          spruce-spire-conifer tropical-emergent-tree feather-frond-palm \
          bunchgrass-tussock savanna-acacia columnar-cactus; do
  # resumable: skip items that already have their 3 variants
  if [ "$(ls "$MODELS/$id"/*.glb 2>/dev/null | wc -l)" -ge 3 ]; then
    echo "== $id: already baked, skipping"; continue
  fi
  echo "== baking $id"
  python formcast.py bake "$IMAGES/$id.png" --out-dir "$MODELS/$id" --count 3 \
    || { echo "== $id FAILED — continuing with the rest"; continue; }
done
```

A failed item must **not** abort the run — finish the other items, then report
which failed (with the relevant `formcast.log` excerpt) and retry just those.

### 2. Normalize filenames

Inside each `models/<id>/`, rename `<formcast-stem>-NN.glb` → `<id>-NN.glb`
(skip if the stems already match). The embedded metadata keeps each file
self-describing, so renames are harmless. After this every folder holds exactly
`<id>-00.glb`, `<id>-01.glb`, `<id>-02.glb`.

### 3. Preview contact sheets (headless)

```bash
cd /Users/claude/formcast
python formcast.py view "$MODELS/$id"/*.glb --save "$MODELS/$id/preview.png"
```

One 3-up sheet per item so the results are reviewable at a glance (from the GUI
account, or attached in conversation).

### 4. Verify

Per item: exactly 3 `.glb` files; `formcast.py inspect` opens each and shows
real geometry + provenance; `preview.png` exists and shows the archetype
(compare against `images/<id>.png` — same kind of thing, plausible variation).
Then check the library's total size (`du -sh flora-revamp/models`) and report it.

### 5. Report & review gate

Summarize per item: baked/failed, file sizes, anything that looks off in the
previews (wrong scale, missing foliage, classifier misreads). **Do not commit**
— `.glb` binaries may be sizeable; whether the library is committed (this repo
does commit binary assets) is the user's call after reviewing the previews.

## Knobs deliberately left for later

- **`--lods`** — formcast can bake a high/med/low LOD chain per variant; Planet
  Explorer will want this for distance LOD, but it triples bake output. Add when
  the import design exists.
- **`--count` > 3** and the P1/P2 catalogue items — scale up after the P0
  pipeline is proven.
- **`--refine 2+`** or `--density med` — quality/time/poly-budget tuning per
  item if any first-pass result disappoints.
- **Scale calibration** — formcast bakes in meters; Planet Explorer renders in
  10 m units with its own per-species size constants. Real-world height
  calibration per species belongs to the import step, not the bake.
