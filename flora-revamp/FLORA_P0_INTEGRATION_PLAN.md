# P0 model integration plan — replace procedural flora

Executable plan: replace the procedurally-generated flora (`src/flora.rs`'s
~850 built-at-startup species) with the **9 P0 formcast models** in
[`models/`](models/). One archetype per species, instanced exactly as today —
the placement/clustering/instancing machinery survives unchanged; only the
*source of species meshes* and the *vegetation vertex format* change.

Read `ARCHITECTURE_GUIDELINES.md` first (correctness → … → polish; no magic
numbers; heavy work off the main thread; everything deterministic from the
seed). Models are static assets, identical on every machine, so determinism is
preserved by construction.

## 0. Facts the plan is built on (verified)

- **Engine veg pipeline today:** `Flora::generate` builds `Species { mesh:
  MeshData, scale_min, scale_max, cluster_radius }` per biome
  (`flora.rs`); `mesh.rs::place_vegetation` scatters instances by per-species
  presence fields + `biome_coverage(biome)` lushness; `gfx.rs::VegBaseMeshes`
  concatenates all species meshes into ONE vertex/index buffer drawn instanced
  per chunk per species by `base_vertex`/`first_index`. `Vertex` is
  `{pos,normal,color}` — **no UVs, no textures**.
- **The models:** glTF 2.0 `.glb`, **+Y-up, meters, base at y=0**, semantic
  submeshes (`trunk`, `canopy`, …), each submesh has a **baseColorTexture +
  UVs**. Foliage/spines use **alphaMode=MASK** (alpha-cutout cards) —
  rendering them opaque would show rectangles, so the pipeline must gain
  textures + alpha-test. Trunk/rock UVs **tile beyond [0,1]** (up to ~21×), so
  pack textures as a **texture2d array** (repeat sampler per layer), NOT an
  atlas. Totals across all 9 models: **23 materials** (mix of 512²/1024²),
  **~143k verts** — trivial under instancing.
- Models embed provenance metadata (source photo etc.) in glTF `extras`/buffers
  (~2 MB each); loaders ignore it.

## 1. Assets — copy these 9 files

Copy (don't move) the `-00` variant of each archetype into `assets/models/`,
dropping the `-00` suffix. (The three variants are near-identical; revisit
-01/-02 later.)

| Copy from `flora-revamp/models/…` | To `assets/models/` |
|---|---|
| `granite-boulder/granite-boulder-00.glb` | `granite-boulder.glb` |
| `broadleaf-hardwood/broadleaf-hardwood-00.glb` | `broadleaf-hardwood.glb` |
| `spreading-oak/spreading-oak-00.glb` | `spreading-oak.glb` |
| `spruce-spire-conifer/spruce-spire-conifer-00.glb` | `spruce-spire-conifer.glb` |
| `tropical-emergent-tree/tropical-emergent-tree-00.glb` | `tropical-emergent-tree.glb` |
| `feather-frond-palm/feather-frond-palm-00.glb` | `feather-frond-palm.glb` |
| `bunchgrass-tussock/bunchgrass-tussock-00.glb` | `bunchgrass-tussock.glb` |
| `savanna-acacia/savanna-acacia-00.glb` | `savanna-acacia.glb` |
| `columnar-cactus/columnar-cactus-00.glb` | `columnar-cactus.glb` |

Embed via `include_bytes!` like `planet.png`/the mp3s (self-contained app,
copyable between accounts). Total ~45 MB of binary growth — acceptable;
optionally strip the embedded reference-photo metadata first (each `.glb`
carries ~2 MB of provenance) if binary size becomes a complaint, but do NOT
block on it.

## 2. Biome assignment & density

Keep `biome_coverage()` (the lushness probabilities) **unchanged** — it already
encodes per-biome density and the area-proportional `veg_attempts` cap keeps it
LOD-independent. Replace each biome's species list with weighted archetypes
(weights play the role of today's `Profile.forms` weights — they set the mix,
coverage sets the amount):

| Biome | Archetype (weight) |
|---|---|
| TemperateForest | broadleaf-hardwood (0.40), spreading-oak (0.25), spruce-spire-conifer (0.20), granite-boulder (0.15) |
| BorealForest | spruce-spire-conifer (0.80), granite-boulder (0.20) |
| TropicalForest | tropical-emergent-tree (0.45), feather-frond-palm (0.40), granite-boulder (0.15) |
| Grassland | bunchgrass-tussock (0.75), savanna-acacia (0.12), granite-boulder (0.13) |
| Desert | columnar-cactus (0.60), granite-boulder (0.40) |
| Beach | feather-frond-palm (0.50), bunchgrass-tussock (0.35), granite-boulder (0.15) |
| Tundra | bunchgrass-tussock (0.55), granite-boulder (0.45) |
| Mountain | spruce-spire-conifer (0.45), granite-boulder (0.55) |
| Ocean / PolarIce / Snow | none (unchanged — placement already skips them) |

Notes: bunchgrass stands in for dune grass (Beach) and tussock sedge (Tundra)
until those P1 models exist; spruce stands in for krummholz on Mountain. The
granite boulder is the cross-biome object from `FLORA_MODEL_TARGETS.md` — it
appears everywhere. P0-only means some biomes get sparser *variety* than today
(e.g. no desert shrubs); that's expected, P1 fills it.

**Per-(biome,archetype) scale** so the same mesh can be dwarfed where it makes
sense: keep `scale_min`/`scale_max` on `Species` and let multiple `Species`
entries share one mesh (see §3.4). Mountain spruce ×0.5, Tundra/Beach grass
×0.8; all others ×1.0.

## 3. Engineering steps

### 3.1 Dependency

Add `gltf` (pure-Rust glTF loader; verify it builds on macOS now — it's
cross-platform, no native deps, fine for the Windows/5090 target). Enable
`import` so buffers + PNG images decode.

### 3.2 New module `src/models.rs` — load the archetype library

A loader that turns the 9 embedded `.glb`s into engine data, run **once at
startup** (it replaces `Flora::generate`'s mesh building; ~1 s budget, log an
INFO line with counts):

- Parse each `.glb`; for each primitive (submesh) emit vertices as the **new
  vegetation vertex** (§3.3) carrying that material's **texture-array layer
  index**; concatenate submeshes into one mesh per archetype.
- **Scale to the world.** Models are in meters; world units are 10 m
  (`METERS_PER_UNIT`). Don't trust raw model height (terrain is ~2× vertically
  exaggerated, so true-scale flora reads tiny). Instead normalize each model's
  bounding height to 1.0 and let per-archetype `TARGET_HEIGHT` consts (units;
  the analogue of today's `form_height`) set the world size:

  | Archetype | target height (units, min–max) |
  |---|---|
  | granite-boulder | 0.10–0.45 |
  | broadleaf-hardwood | 1.8–3.6 |
  | spreading-oak | 1.5–3.0 |
  | spruce-spire-conifer | 2.0–4.4 |
  | tropical-emergent-tree | 3.4–6.0 |
  | feather-frond-palm | 2.0–4.2 |
  | bunchgrass-tussock | 0.10–0.26 |
  | savanna-acacia | 1.4–2.6 |
  | columnar-cactus | 0.5–1.6 |

  Map the min–max range onto `Species::scale_min/scale_max` (the existing
  per-instance jitter + reverse-J age multiplier then works untouched).
- **Cluster radius per archetype** (drives `species_presence`; reuse the
  existing km→units conversion): trees 1–4 km, bunchgrass 0.5–2 km, cactus
  1–3 km, boulder 2–8 km (broad scatter). Pick deterministically from the
  planet seed exactly as `form_cluster_km` does today.
- **Texture array.** Decode every material's baseColorTexture, resize to
  `VEG_TEX_LAYER_SIZE` (const, **512** default — 23 layers × 512² RGBA ≈ 24 MB;
  a settings preset can raise it for the 5090), generate a simple CPU mip chain
  (box filter) so distant foliage doesn't shimmer, upload one
  `texture_2d_array`. Store `layer` per vertex.

### 3.3 Vegetation vertex + shader

- New `VegVertex { pos: [f32;3], normal: [f32;3], uv: [f32;2], layer: u32 }`
  (36 B — same size as the old vertex; per-instance `tint` already exists and
  replaces the old per-vertex color). Terrain keeps the old `Vertex` — only the
  vegetation pipeline changes.
- `shaders/vegetation.wgsl`: sample
  `textureSample(veg_tex, veg_sampler, uv, layer)` × instance tint × existing
  lighting; **`discard` when `alpha < VEG_ALPHA_CUTOFF`** (const 0.45 — the
  models were authored for MASK cutoff 0.4–0.5). Opaque materials have α=1 so
  one unconditional alpha-test path covers both.
- Pipeline: sampler = repeat addressing (the tiling trunk UVs), **cull_mode =
  None** for vegetation (leaf cards are double-sided).
- `VegBaseMeshes` gains the texture array + sampler in its bind group (bound
  once per frame — binds stay O(1), preserving the arena win).

### 3.4 Rewire `Flora`

- `Flora::generate(seed)` keeps its signature and the `Species`/`by_biome`
  shape, but: load the 9 archetype meshes via `models.rs` (mesh stored once),
  then build per-biome `Species` entries from the §2 table — each entry is
  `{ mesh_index, scale_min, scale_max, cluster_radius }` referencing a shared
  mesh. `VegBaseMeshes` maps species → (base_vertex, first_index) through
  `mesh_index`, so several species reuse one mesh range.
- Weighted biome mix: reuse the existing weighted-pick (weights from §2) when
  building each biome's species list, or simpler — emit each archetype once
  per biome and fold the weight into the presence sum. Choose whichever keeps
  `place_vegetation` untouched; the goal is NO changes to placement logic.
- Delete the procedural builders (`build_conifer` … `build_vine`, the
  colour/palette helpers) once nothing references them; keep
  `species_presence` clustering as-is. `SPECIES_PER_BIOME` shrinks to the
  table's row counts.

### 3.5 Tests & gallery

- `gfx.rs` gallery/smoke tests build species meshes and render via the terrain
  shader — update them to the new `VegVertex` (the closeup/gallery should
  render through the **vegetation** pipeline now, which `BACKLOG.md` already
  wants). Keep the GPU smoke test passing headless.
- `cargo test` + `cargo clippy` clean; determinism test still passes (assets
  are static; placement RNG unchanged).

### 3.6 Verify & ship

- Log at INFO: archetypes loaded, total verts, texture layers (startup line).
- **`./package_macos.sh` after every build** — the user tests from the GUI
  account; check the shared log's startup + perf lines (`grep 'perf:'`) after
  their run. Watch `draw` counts (unchanged species-per-chunk draws) and frame
  times — 143k verts of base mesh and 24 MB of texture are well inside budget.

## Out of scope (deliberate)

- P1/P2 archetypes, the `-01`/`-02` variants, LOD chains (`--lods` rebake),
  impostors, wind sway, per-biome seasonal tinting — all later passes.
- Snow/PolarIce boulders (would need enabling vegetation in biomes that
  currently return `None` — small, but a separate decision).
- Mip-less quality shortcuts: if CPU mip generation turns out fiddly, ship
  without mips behind a TODO rather than blocking — but expect shimmer.
