# Flora Photorealism — Plan

A staged plan to make Planet Explorer's vegetation read as *more photorealistic*,
**with memory and compute cost quantified for every idea**. The goal is "better,
not perfect": land the high-leverage cues cheaply, defer the expensive pipeline
work, and never let polish blow the resource budget.

Every item names the files it touches, its **resource factor** (against the pool it
actually grows), and how it stays inside [`ARCHITECTURE_GUIDELINES.md`](ARCHITECTURE_GUIDELINES.md).

> Priority hierarchy (the tie-breaker): **correctness → security → performance →
> maintainability → portability → polish.** Photorealism is *polish* — it never
> justifies breaking determinism, a hot path, or portability, and it must respect
> the memory budget the user sets in the GRAPHICS menu.

---

## TL;DR

1. **Fix density first — it's the keystone.** Today density is *attempts per chunk*,
   identical at every LOD level, so the near field is up to ~256× denser per unit
   area than the far field and the user-visible "too dense." Switching to
   **area-proportional density with a per-chunk cap**, tuned to **ecological
   plants/hectare per biome**, plus a **layered-strata + reverse-J size
   distribution**, makes it look right *and* **shrinks** the one veg cost that
   scales with the world. This is free realism — it costs negative memory.
2. **Most shading/geometry ideas are nearly memory-free.** They touch the
   *base meshes* (uploaded **once**, ~1.6 MB total) or the *existing* per-instance
   matrix+tint (80 B/plant, unchanged). Per-instance memory factor ≈ **1.0×** for
   Tier 0, 1a, 2.
3. **The only genuinely new memory is fixed VRAM** — leaf/bark **textures**
   (~5–50 MB, one-time) and **MSAA** render targets (~120 MB at 1440p, resolution-
   scaled). Both are independent of plant count and **gated behind the graphics
   preset**.

So the resource story is reassuring: **vegetation is already a *minority* of
geometry memory (terrain dominates ~5:1), and the photoreal work barely moves the
veg pool — while the density fix actively reduces it.**

---

## Resource model — the three pools (know which one each idea grows)

Confirmed from the code (`gfx.rs` `mesh_bytes`/byte accounting, `mesh.rs` sizes,
`settings.rs` presets):

| Pool | What | Current size (High preset) | Scales with |
|---|---|---|---|
| **A — base meshes** | One mesh per species, uploaded once (`Renderer.species_meshes`) | ~180 species × ~250 verts × 36 B ≈ **1.6 MB** | species count × poly/species (one-time) |
| **B — per-instance working set** | `VegInstance` (64 B model + 16 B tint = **80 B**) per planted plant, per resident chunk | tens of MB to ~**200 MB** worst case | **plant count = density × resident area** ← the lever |
| **C — fixed VRAM** | Textures, MSAA targets | **0** for veg today | resolution / atlas size (one-time) |

Anchor numbers (High, `grid=63`, budget 8 GB):

- **Terrain per chunk** ≈ 4096 verts·36 B + 23.8 k idx·4 B ≈ **~240 KB**.
- **Veg per dense chunk** ≈ 675 plants · 80 B ≈ **~54 KB** (≈ 18 % of that chunk;
  most land chunks are far sparser, ocean/ice = 0).
- **Working set** ≈ 3,800 chunks → terrain ≈ **~0.9 GB**, veg ≈ **tens of MB–200 MB**.

**The single most important fact:** only **Pool B** grows with the world, and it is
`80 B × plants`. Reduce plants (density) and you reduce the only scaling veg cost —
linearly — while improving realism. Everything else is one-time and small.

---

## Where flora lives today (pipeline)

| Stage | File | What |
|---|---|---|
| Species library | `flora.rs` | `Flora::generate(seed)` grows ≤`SPECIES_PER_BIOME` low-poly meshes per vegetated biome from `segment`/`cone`/`ellipsoid`/`frond`. Per-vertex colour. |
| Placement | `mesh.rs::place_vegetation` | Per-chunk, chunk-key-seeded scatter; `density` **attempts/chunk** (not area-scaled), gated by `biome_coverage` × per-species clustering presence. Emits `VegInstance`s grouped by species. |
| Render | `shaders/vegetation.wgsl` | Instanced; Lambert + flat ambient + tiny sky-fill + fog. No specular/translucency/AO/shadow/wind. |
| Evaluate | `gfx.rs::mod gallery` | Offscreen PNGs. ⚠️ Renders flora via the **terrain** shader (baked to world space), not `vegetation.wgsl`. |

---

## Diagnosis — why current flora reads as "CG"

1. **Density is wrong and view-dependent** (see next section) — the biggest single tell.
2. **Blobby/geometric silhouette** (`ellipsoid` 3×5 rings, clean `cone` tiers).
3. **No translucency** — Lambert makes leaves look like opaque plastic.
4. **Harsh terminator + dead shade side** (`max(dot(n,l),0)` on smooth blobs).
5. **Flat, *random* colour** — `FOLIAGE_VERT_JITTER=0.05`; hue picked by RNG, not
   tied to climate, so stands aren't cohesive.
6. **"Clone army"** — every instance is the same mesh; only uniform scale + near-
   white tint vary; all perfectly upright.
7. **Uniform sizes** — no age/size distribution; a stand is all one size.
8. **Pasted-on** — no contact shadow/AO, no leaf litter, no basal grass.
9. **Dead still** — no wind.
10. **Aliasing** — `sample_count: 1`; thin trunks/fronds/leaf edges crawl in motion.

---

## Density & ecological realism (the keystone — do this first)

### The current model and why it's "too dense"

`place_vegetation` makes a **fixed `density` attempts per chunk regardless of the
chunk's LOD level or ground area** (`veg_density` ≈ 498 at High, max 750). Because
the LOD system draws *fine* chunks near the camera and *coarse* chunks far away:

- A level-16 (near) chunk ≈ **3.8 ha**; ~675 plants ⇒ ~**180 plants/ha**.
- A level-12 (far) chunk ≈ **970 ha**; ~675 plants ⇒ ~**0.7 plants/ha**.

So **the same ground is ~256× denser per unit area when you stand on it than when
you see it from altitude** (4× per level, levels 12→16). Walking toward a forest, it
*thickens* unnaturally; the near field becomes a wall — especially with tropical
trees at `size_scale=1.6` (heights ~50–75 m) spaced ~7 m apart, so canopies fully
overlap. That's the "too dense in general."

It also makes **total veg memory LOD-dependent and unpredictable**: zoom in and Pool
B balloons because each fine chunk independently plants ~`density`.

### The fix: area-proportional density, capped, ecological targets

1. **Make density physical — plants per unit area, not per chunk.** Expected plants
   for a chunk = `D_biome × chunk_ground_area` (Bernoulli/Poisson so fractional
   expectations work at fine levels). Now ground density is **constant across LOD** —
   the forest looks the same near and far, and **total Pool B ≈ `D̄ × resident_area ×
   80 B`, decoupled from LOD** (predictable, budgetable).
2. **Cap per chunk** (`VEG_MAX_INSTANCES_PER_CHUNK`). Pure area-proportional would
   make a far level-12 chunk want `D × 970 ha` = tens of thousands of plants (a
   single 7 MB instance buffer!). Cap it: far/coarse chunks come out *sparse but
   present and bounded*; the deficit is invisible at distance (and is exactly where
   **impostors**, Tier 3c, take over later). The cap is the Pool-B safety rail.
3. **Tune `D_biome` to ecology** (1 unit = 10 m ⇒ 1 ha = 100 unit² ⇒ *100 plants/ha =
   1 plant/unit²*). Earth anchors for the **canopy layer**:
   - Tropical / temperate forest: ~**60–150 trees/ha** (was effectively ~180 near-field, *and* now constant) → spacing ~8–13 m, canopies touch but don't merge into a wall.
   - Boreal: ~**100–200/ha** (denser stems, smaller crowns).
   - Savanna / grassland trees: ~**5–40/ha** (open, scattered).
   - Desert woody: ~**5–60/ha**; tundra dwarf shrubs sparse.
4. **Layer the strata** instead of one scatter. Real ecosystems stack
   canopy → understory → shrub → ground/herb. Plant a *sparse* big-tree canopy
   (numbers above) **plus** a denser low layer (grass/forb/shrub at high count but
   *tiny* meshes). Scenes read lush at the ground while the canopy stays open — the
   opposite failure mode from today. *(Tiny ground plants are cheap geometry and the
   per-instance cost is still 80 B each, so keep the ground-layer cap sane.)*
5. **Reverse-J size distribution.** Draw per-instance scale from a power law (many
   small/young, few large/mature) rather than uniform `0.82–1.20`. Kills the "all one
   size" tell, ZERO memory cost (it only reshapes the existing scale RNG).
6. **Competition spacing + environmental gradients.** A minimum-spacing/Poisson-disk
   reject so plants don't interpenetrate; density gradients with moisture/altitude
   (lush near water, thinning up mountains / into arid edges). Mostly placement-logic,
   ~0 memory.

**Where:** `mesh.rs::place_vegetation` (area-scaled attempt count + cap + strata +
size law + spacing), `flora.rs` `biome_profile`/new per-biome `D_biome`, `settings.rs`
(density knob becomes an areal target; add the cap). Keep determinism (chunk-key
seeded) and keep it O(attempts) per chunk.

**Resource impact:** **Pool B near-field ×~0.3–0.5** (fewer near plants), total Pool B
**bounded and LOD-independent**; **CPU placement ×~0.3–0.6** (fewer attempts where it
was densest); GPU veg vertex/draw load **down**. Pools A and C unchanged. *Net: a
realism win that also reclaims memory and CPU.*

### Per-biome density calibration (empirical — 2026-06-08)

After the area-proportional redesign shipped, in-world feedback: **Tundra reads about
right; the dense biomes (tropical/temperate/grassland/boreal) and the cold/high ground
around snow read too dense.** First correction applied — a **global ~50% cut**:
`VEG_REFERENCE_AREA` 1100→2200 (halves the area-proportional attempt rate) and
`VEG_MAX_ATTEMPTS` 384→192 (halves the cap), so **every LOD level drops ~50%
uniformly** while the per-biome *relative* mix (via `biome_coverage`) is preserved.

Future per-biome tuning, if the dense biomes still run heavy relative to Tundra (0.40)
after the global cut: scale the high-coverage biomes down via `biome_coverage` (today
Tropical 0.95 / Temperate 0.85 / Grassland 0.80 / Boreal 0.70), keeping the ecological
ordering (rainforest > tundra). Note the **Snow biome itself grows no flora**
(`biome_profile` returns `None`); "snow looks too dense" is the sparse Tundra/Boreal/
Mountain plants reading against snowy terrain — the global cut thins those. The
calibration knob of first resort stays `VEG_REFERENCE_AREA` (raise = sparser).

---

## Day/night terminator — HIGH PRIORITY (whole-planet lighting)

**Problem.** The lit hemisphere looks good (nice sun glint), but the planet has **no
real night side**: terrain/veg shading is `color · (amb + diff·(1-amb) + sky_fill)`
with a *flat* `amb ≈ 0.42` applied everywhere, so the hemisphere facing away from the
sun is still ~42 %-lit — a uniformly-lit globe with no terminator, which reads wrong
(the far side has no business being that bright).

**Goal.** A genuine **lit day side and dark night side** with a soft terminator: the
sun's diffuse contribution falls to ~0 past the terminator, the day side stays fully
lit, and the night side drops to a low **twilight floor** — not pure black, so it stays
navigable and atmospheric.

**Approach (shaders + globals; cross-cutting — `terrain.wgsl`, `vegetation.wgsl`,
`sky.wgsl`):**
- Drive ambient from the **sun geometry**, not a constant: scale it by
  `dot(surface_up, sun_dir)` (the sun's elevation at that point) so it's high in
  daylight and decays through the terminator to a small `NIGHT_FLOOR`.
- **Soft terminator band** via a `smoothstep` around the horizon so the day/night line
  is a gradient (atmospheric scatter, not a hard edge), tinted warm (sunset) in the band.
- **Night fill:** the low ambient floor + the existing starfield + a faint
  atmospheric/moonlight term (cool tint) so the dark side is dark-but-readable.
- The sky's rim already brightens on the sunlit side (`sky.wgsl` `sun_face`) — extend
  that day/night consistency to the surface terminator.

**Design caveats.** The guided tour and free-fly can sit on the night side — keep
`NIGHT_FLOOR` high enough to see terrain and tour-visible vegetation, or bias the tour
toward the lit hemisphere. Day/night stays **deterministic** from the (per-world,
constant) sun direction — no wall-clock.

**Resource impact:** shader-only, a few ALU ops per fragment. **Pools A/B/C ×1.0**, no
new memory. The cost is purely tuning the terminator/twilight. Synergises with the
Tier-0 foliage shading below (those terms are already sun-relative, so they darken
correctly once the ambient floor drops at night).

---

## Finer branches & leaves — breaking the blob silhouette

**Why trees still read as blobs (geometry, not shading).** A canopy is 3–7 `ellipsoid`
blobs (`BLOB_RINGS=3`, `BLOB_SECTORS=5` → smooth 24-vert spheres) over a *single* level
of `segment` limbs. So at any distance the silhouette is a few overlapping balls with a
smooth edge — no twig- or leaf-scale structure, no recursive woody ramification. Tier-0
shading makes the blobs read as foliage-coloured *volume*, but the **silhouette** is
still a blob; only geometry fixes that.

**Approach (a Tier-1 geometry expansion; baked into the species mesh = Pool A, once):**
1. **Recursive branching (L-system-lite).** Replace the one limb level with 2–3 levels
   of ramification — trunk → boughs → branches → twigs — each child shorter/thinner at a
   branch angle, with droop and per-branch RNG, so crowns sit on visible structure.
2. **Leaf clusters, not solid blobs.** Break each canopy blob into many small foliage
   elements so the edge goes lacy:
   - **Geometry leaves** (no texture): many small displaced quads/tris — cheapest path,
     no new pipeline, just more polys.
   - **Alpha-tested leaf cards** (Tier 3a): textured alpha-masked quads — far fewer tris
     for the same richness, but needs the texture/alpha pipeline (Pool C + fragment
     overdraw). The real photoreal path; geometry leaves are the stepping stone.
3. **LOD discipline.** Finer geometry only pays off up close — pair with vegetation LOD
   impostors (Tier 3c) or raise `veg_min_level` so heavy meshes are near-only.

**Resource impact.** Pool A ×~2–4 (a broadleaf ~200 → ~600–1200 verts; total ~1.6 MB →
~4–6 MB — still trivial). **Pool B (per-instance) ×1.0** — instances are unchanged. The
real cost is **GPU vertex throughput on drawn veg ×~2–4**, mitigated by the just-shipped
~50 %-lower density, impostors, and a higher `veg_min_level`. Leaf cards add Pool C
(textures ~5–50 MB fixed) + fragment overdraw. *This is the highest-impact remaining
silhouette fix, but the GPU-heaviest flora item — gate poly density behind the preset
and lean on impostors.*

---

## The tiers (each with its resource factor)

### Tier 0 — Shading & light (biggest ratio; ~memory-free)

Give vegetation its **own vertex attribute** packing a `material` flag (wood vs leaf)
+ a baked `ao` term, then upgrade `vegetation.wgsl`:

- **0a. Leaf translucency / back-light transmission** — back-lit canopies glow
  (the single biggest "plastic → leaves" cue). Foliage-gated.
- **0b. Half-Lambert wrap** — softens the terminator, lifts the shade side naturally.
- **0c. Weak leaf sheen** — broad, low specular for waxy gloss under sun.
- **0d. Baked AO + foliage-normal break-up** — darken canopy interiors/undersides &
  trunk bases at *generation* time; jitter foliage normals so a blob stops shading
  like a balloon.

**Where:** `vegetation.wgsl`; a veg-specific vertex layout (`VegVertex` or +1 attr on
the base mesh); `flora.rs` builders tag material/AO.
**Resource:** Pool A only — `Vertex` 36 B → 40 B *on base meshes* (+~0.2 MB total).
**Pool B ×1.0**, Pool C +0. GPU: a few fragment ALU ops (negligible). CPU: gen-time
AO, off the main thread. **Memory factor ≈ 1.00×.**

### Tier 1 — Cheap geometry & per-instance variation

- **1a. Per-instance variation** — small lean (tilt quat), gentle non-uniform scale,
  wider tint, all in the *already-emitted* model matrix + tint. Kills the clone army.
  **Pool B ×1.0** (no new bytes), CPU ~×1.0. *(Caveat: keep non-uniform scale modest
  or pass a normal matrix — the shader assumes rotation×uniform scale for normals.)*
- **1b. N mesh variants per species** — pick per instance for silhouette variety.
  **Pool A ×N** (e.g. ×3 → ~5 MB, still trivial); **Pool B ×1.0**; cost is **draw
  calls** (up to ×N species-runs/chunk — weigh against `BACKLOG.md` "Tame `split_factor`").
- **1c. Richer canopy silhouette** — more, smaller, more-displaced blobs; jittered
  cone rims; needle fronds. **Pool A ×~2–3** (→ ~3–5 MB); **Pool B ×1.0**; real cost
  is **GPU vertex/triangle throughput ×~2–3** on drawn veg (fine on a 5090; measure on
  Mac; gate poly density behind the preset).
- **1d. Branch structure** for broadleaf/snag (a second branching level). Same profile
  as 1c (Pool A modest, Pool B ×1.0, GPU vtx ↑).
- **1e. Basal grass collar** — a few tiny ground plants around big plants. This is part
  of the strata work above; **Pool B** grows only by the (capped) ground-layer count.

### Tier 2 — Wind / motion

- **2a. Wind sway** in `vegetation.wgsl` `vs` — amplitude scales up the plant (base
  fixed, tips move), via the `material`/local-Y weight, a wind vector, time
  (`camera_pos.w`), and a per-instance phase.
- **⚠️ Precision (ARCH §8 / M6):** do **not** phase off absolute world position
  (~637 k units loses `sin` phase). Use a **per-instance phase scalar** or local/
  fractional coords.
**Resource:** **Pool B ×1.0** (derive phase from existing data) **or ×1.05** (add a
4 B phase float). GPU: a few vertex ops (negligible). Render-time only — world stays
deterministic (like the existing water animation).

### Tier 3 — The photoreal leap (highest effort; the only real new memory)

- **3a. Alpha-tested leaf cards + textures.** Clusters of textured, **alpha-clipped**
  quads instead of solid blobs — how real-time engines get photoreal canopies.
  Order-independent (clip in the fragment; works with the current opaque+depth pass).
  **Resource:** **Pool C +5–50 MB** (one shared leaf/bark atlas, e.g. 1–2 k² RGBA +
  mips; *fixed, independent of plant count*); **Pool A ×~1.5–4** (UVs: `Vertex` +8 B;
  more card geometry — still single-digit MB); **Pool B ×1.0**. GPU: **fragment
  overdraw + `discard`** is the real cost (alpha-tested foliage is fill-heavy) — pairs
  naturally with impostors (3c) and a density that isn't a wall. Texture source:
  procedural at startup (seed-deterministic, self-contained) or a small embedded PNG.
- **3b. Bark / ground detail textures** — same pipeline, lower priority. Pool C +few MB.
- **3c. Vegetation LOD impostors** — far plants → one billboard; full mesh only near.
  **Resource: REDUCES** Pool B and GPU in the far field; small impostor atlas in Pool
  C. This is what lets a low `veg_min_level` (far-field veg) coexist with a sane
  near-field density and the per-chunk cap. **A memory/compute *saver*.**
- **3d. MSAA (or FXAA)** — fixes aliasing (#10) for *all* geometry, not just veg.
  **Resource:** **Pool C + render-target VRAM × sample_count** — ~**120 MB at 1440p /
  ~330 MB at 4K for 4×** (color+depth+resolve), *resolution-scaled, fixed*; GPU fill
  ×~2–4. Gate sample count behind `settings.rs` (5090 eats 4–8×; Mac fine at 4×).
  FXAA is a cheaper alternative (~one extra full-screen pass, no MSAA VRAM).

---

## Per-idea resource impact — at a glance

Factors are against the pool the idea actually grows (✱ = the dominant/limiting cost).

| Idea | Pool A (base, 1.6 MB) | Pool B (80 B/plant) ✱ | Pool C (fixed VRAM) | CPU place | GPU |
|---|---|---|---|---|---|
| **Density redesign** | ×1.0 | **×0.3–0.5, capped** | 0 | **×0.3–0.6** | ↓ |
| 0 Shading + material/AO | +0.2 MB | ×1.0 | 0 | ×1.0 | frag +ε |
| 1a Per-instance variation | ×1.0 | ×1.0 | 0 | ×1.0 | ×1.0 |
| 1b N variants (×3) | ×3 (~5 MB) | ×1.0 | 0 | ×1.0 | +draw calls |
| 1c/1d Richer geometry | ×2–3 (~5 MB) | ×1.0 | 0 | ×1.0 | **vtx ×2–3** |
| 2 Wind | ×1.0 | ×1.0–1.05 | 0 | ×1.0 | vtx +ε |
| 3a Leaf cards + textures | ×1.5–4 (~6 MB) | ×1.0 | **+5–50 MB** | ×1.0 | **frag overdraw** |
| 3c Impostors | ×1.0 | **↓** | +small | ↓ | ↓ |
| 3d MSAA 4× | ×1.0 | ×1.0 | **+120 MB @1440p** | ×1.0 | **fill ×2–4** |

**Reading it:** nothing here multiplies the world-scaling pool (B) except the density
redesign, which *divides* it. New memory is one-time fixed VRAM (textures, MSAA),
budget-gated. Compute risk concentrates in richer geometry (vertex throughput) and
alpha-tested foliage (fragment overdraw) — both mitigated by impostors + sane density.

---

## More photorealism from Earth-flora thinking

Beyond shading/geometry, what actually separates real vegetation from CG — most of it
**near-zero memory**:

- **Climate-driven colour (not RNG).** Tie foliage hue/value/saturation to the
  sample's **temperature & moisture** (the planet already computes both) instead of
  `green_foliage`'s pure random pick: deep saturated greens in warm-wet, olive/straw
  in arid, blue-green needles in cold, autumn where seasonal. Stands become *cohesive*
  and place-appropriate. **Pool B/A ×1.0** — it only reseeds existing colour choices.
- **Allometric size distribution (reverse-J)** — covered under density #5; free.
- **Layered strata** — covered under density #4.
- **Textured leaves/bark + translucent leaf masks** — Tier 3a/3b; the leap that needs
  Pool C.
- **Competition spacing, clearings, riparian/altitude gradients** — density #6; free.
- **Dead matter & ground variation** — stumps, fallen logs, bare soil patches, rock;
  breaks uniformity. Cheap geometry (reuse `Snag`), Pool B by the (small) count.
- **Contact shadow / soil darkening under canopy** — a subtle darkening of terrain
  vertices (or a decal) beneath big plants; grounds them. Modest terrain-side work;
  Pool A/B ×1.0.
- **Aerial-perspective desaturation** — slight extra desaturation with distance on top
  of fog; a couple of shader ops, free.

---

## Evaluation workflow (study output headlessly)

The `gallery` module is the tight loop: `flora_gallery_renders` →
`<tempdir>/planet_flora_gallery.png`, `terrain_closeup_renders` →
`planet_flora_closeup.png` (`--ignored`, slow). **Fix first:** it renders flora through
the **terrain** shader (baked to world space), so Tier-0 veg-shader work won't show —
extend it to render via the **vegetation pipeline**, add a **back-lit** view (to show
translucency) and a **wide stand** view (to judge *density* before/after). Run
`cargo test --release flora_gallery -- --nocapture`, copy the PNG to `/Users/Shared/`
to view from the GUI account. Interactive builds still ship via `package_macos.sh`.

Also lean on the existing **`mem_mb` HUD/log readout** (`BACKLOG.md`) to measure Pool B
before/after each change — the density redesign should visibly *drop* it.

---

## Recommended sequencing

1. **Density redesign** (area-proportional + cap + ecological targets + strata + size
   law). Highest realism impact, *reduces* memory/CPU, no new assets. Measure Pool B
   drop via `mem_mb`. ← do first.
2. **Tier 0 shading overhaul + 1a per-instance variation + climate-driven colour.**
   ~memory-free, transforms the look. Extend the gallery to the veg shader. ← package
   & study.
3. **Tier 1c/1d silhouette + branches, then Tier 2 wind.** Watch GPU vertex load.
4. **Tier 3 leaf cards + impostors + MSAA** — the real leap; the only items that add
   (fixed, gated) VRAM. Impostors offset the leaf-card fill cost.

---

## Guardrails (hold every slice to these)

- **Respect the memory budget.** Pool B stays bounded by the density cap; Pool C
  (textures, MSAA samples) is **gated behind `settings.rs` presets**, never a
  hardcoded per-device assumption (the 5090 cranks; the laptop preset stays lean).
  Re-check `mem_mb` after any change that touches placement or adds VRAM.
- **Determinism.** Generation seeds from existing sub-seeds (`flora::mix`,
  `ChunkKey::hash`); animation (wind) is render-time only, like water.
- **Hot paths.** `place_vegetation` (per-chunk worker) stays O(attempts), no new
  per-call heap churn; shader additions are cheap per-vertex/fragment; no new
  per-frame main-thread allocation.
- **`gfx` stays thin.** Policy in `flora`/`mesh`; `gfx` grows only a texture/bind
  group if Tier 3 lands. No `wgpu` types leaked upward.
- **Portability.** Plain WGSL, `wgpu` textures, pure-Rust cross-platform crates.
- **No magic numbers.** Every new tuning value (densities, caps, atlas sizes, sway
  constants, AO weights) is a named `SCREAMING_SNAKE_CASE` `const` with a units/intent
  comment — Rust *and* WGSL.
- **Tests.** New generation maths ships a determinism/bounds test (e.g. *areal density
  is LOD-independent*, *per-chunk instances ≤ cap*); keep `offscreen_pipeline_validates`
  green; use the gallery for the visual record.

## Out of scope (deferred)

Real-time cast shadows / shadow maps; global illumination; volumetric lighting;
full PBR material authoring; seasonal simulation. Large cross-cutting renderer
projects — revisit only after Tier 3, and only if the payoff justifies the weight and
the VRAM.
