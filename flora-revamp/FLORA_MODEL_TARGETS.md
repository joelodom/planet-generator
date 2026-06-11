# Flora & Object Model Targets

A prioritized catalogue of Earth-flora **archetypes** and **scene objects** to
build as 3D models for Planet Explorer's vegetation pass. The intended pipeline
is: pull real-world exemplar imagery from the internet → feed it to
[`../formcast`](../formcast) to generate a 3D model → import as a per-species
mesh that replaces today's procedurally-generated flora.

This document is **only the target list** — it does not change any code. It is
organised by the 11 biomes the planet actually classifies (see
`Biome` in `src/planet.rs`), each driven by the height/temperature/moisture/slope
rules in `classify()`. The goal is believable, place-appropriate stands per
biome, not botanical completeness.

## How to read this

- **Object Type** — the short archetype name the model will be filed under (the
  kind of label a species slot would carry, e.g. "spire conifer", "boulder").
- **Description** — what the form looks like and *why it belongs in this biome*,
  with a real-world representative species to drive the formcast exemplar search.
- **Priority** — global build order across the whole list:
  - **P0** — highest visual payoff. Either a dominant, frequently-seen biome
    (temperate / boreal / tropical forest, grassland, desert) **or** the single
    most iconic silhouette of a biome. Build these first.
  - **P1** — important for variety and realism; build after the P0 layer reads well.
  - **P2** — sparse, rare, or polish; fills out thin/barren biomes last.

Priority is **global**, so a forest's P0 tree outranks a polar-ice P2 even though
both are "top of their biome." Build all P0s, then all P1s, then all P2s.

> **Climate cross-check.** A biome here only ever appears where `classify()` puts
> it: forests need moisture; grassland/desert split on moisture at warm temps;
> tundra/boreal split on moisture at cold temps; snow/mountain override by
> altitude+slope. Keep model choices consistent with that — e.g. desert flora must
> survive low moisture, boreal flora must survive cold.

---

## Cross-biome objects (build once, tint/scale per biome)

These are not tied to one biome — the same base mesh gets recoloured and rescaled
wherever it's placed, so they earn their build cost across the whole planet.

| Object Type | Description | Priority |
|---|---|---|
| Granite boulder cluster | A few rounded-to-angular grey boulders grouped as one prop. The single highest-value object: rock reads believable everywhere and breaks up flat ground in *every* land biome. Representative: weathered granite glacial erratics. | **P0** |
| Weathered rock outcrop / fractured slab | A low slab of fractured, stratified bedrock poking through soil. Anchors mid-slope terrain and forest floors. Representative: exposed sandstone/limestone shelf. | **P1** |
| Fallen log / deadwood | A toppled, barkless trunk lying on the ground, optionally split. Adds life-cycle realism to any forested or formerly-forested biome. Representative: fallen pine/oak bole. | **P1** |
| Standing dead snag | A bare, branchless or broken-topped dead trunk left standing. Cheap silhouette variety in forest and boreal stands. Representative: fire-killed conifer snag. | **P2** |

---

## P0 biomes (dominant, highest payoff)

### Temperate Forest
_Mild temps, high moisture. One of the most-seen vegetated biomes — invest heavily._

| Object Type | Description | Priority |
|---|---|---|
| Rounded broadleaf hardwood | Dense, rounded deciduous canopy on a moderate trunk — the default "leafy tree" silhouette. Representative: sugar maple / European beech. | **P0** |
| Spreading-crown oak | Tall, sturdy hardwood with a broad, somewhat irregular spreading crown and heavier branching than the maple form. Representative: English/white oak. | **P0** |
| Slender white-barked birch | Tall, thin, pale-trunked tree with a light open canopy; reads great in clumps for tonal contrast against darker hardwoods. Representative: paper birch / aspen. | **P1** |
| Leafy understory shrub | Knee-to-head-height rounded bush filling the forest floor between trunks. Representative: hazel / dogwood thicket. | **P1** |
| Fern cluster | Low, lacy frond groundcover for damp shaded floor. Representative: bracken / lady fern. | **P2** |
| Wildflower clump | Small mixed-colour flowering groundcover for clearings and edges. Representative: woodland aster / foxglove patch. | **P2** |

### Boreal Forest
_Cold but moist. The classic dark northern conifer band._

| Object Type | Description | Priority |
|---|---|---|
| Tall spire conifer (spruce) | Narrow, steeply conical evergreen with a pointed top — the defining boreal silhouette. Representative: black/white spruce. | **P0** |
| Soft dense fir | Fuller, softer conical conifer with denser foliage and a slightly rounder profile than spruce. Representative: balsam/Douglas fir. | **P1** |
| Open long-needle pine | More open, irregular conifer with longer needles and visible trunk/branch structure. Representative: Scots/jack pine. | **P1** |
| Low juniper / scrub conifer | Sprawling low evergreen shrub for the forest floor and clearings. Representative: common juniper mat. | **P2** |
| Moss/lichen-covered rock | Boulder variant draped in green moss and pale lichen for the damp boreal floor (or reuse the cross-biome boulder with a boreal tint). Representative: mossy granite. | **P2** |

### Tropical Forest
_Warm, high moisture. Lush, tall, structurally varied — needs the most distinct forms._

| Object Type | Description | Priority |
|---|---|---|
| Emergent broad-canopy tree | A towering rainforest giant with a wide, flat, umbrella-like crown breaking above the canopy. The iconic jungle silhouette. Representative: kapok / ceiba. | **P0** |
| Feather-frond palm | Tall slender trunk topped with a crown of arching pinnate fronds. Representative: coconut / royal palm. | **P0** |
| Tree fern | A short fibrous trunk crowned with a rosette of large lacy fronds — distinctly prehistoric, fills the mid-story. Representative: Cyathea tree fern. | **P1** |
| Broad-leaf understory plant | Clustered very large paddle leaves at ground-to-waist height. Representative: banana / heliconia / wild ginger. | **P1** |
| Hanging liana / vine | Woody vines draping and looping between canopy layers; adds vertical tangle. Representative: rattan / jungle liana. | **P1** |
| Buttress-root tree | A straight-boled tree with dramatic flared buttress roots at its base. Representative: strangler fig / Pterocarpus. | **P2** |
| Bromeliad / epiphyte clump | A rosette of stiff leaves perched on trunks/branches, often vividly tinted. Representative: tank bromeliad. | **P2** |

### Grassland
_Warm/temperate, drier than forest. Open ground with sparse standout features._

| Object Type | Description | Priority |
|---|---|---|
| Bunchgrass tussock | A dense clump of tall blades — the staple groundcover that *makes* a grassland read as grass, placed at high density. Representative: bluestem / fescue tussock. | **P0** |
| Flat-topped savanna tree | A lone tree with a high, wide, umbrella-flat crown on a clear trunk — the unmistakable savanna icon. Representative: umbrella acacia. | **P0** |
| Sagebrush / scattered shrub | Low silvery-green woody shrub dotted across open ground. Representative: big sagebrush / rabbitbrush. | **P1** |
| Wildflower meadow clump | Mixed flowering forbs in colourful patches for prairie variety. Representative: coneflower / poppy meadow. | **P1** |
| Grassland rock outcrop | Low rocks rising from open plain (or reuse cross-biome outcrop with a sun-bleached tint). | **P2** |

### Desert
_Warm, very low moisture. Sparse, spiky, sun-bleached — striking against bare sand._

| Object Type | Description | Priority |
|---|---|---|
| Columnar cactus | A tall ribbed column with raised arms — the signature desert silhouette. Representative: saguaro. | **P0** |
| Eroded sandstone formation | A wind-carved, layered rock spire/mesa fragment; the desert's main vertical mass where plants are scarce. Representative: red sandstone hoodoo. | **P1** |
| Barrel / pad cactus cluster | A low grouping of squat barrel cacti and flat prickly-pear pads. Representative: barrel cactus / prickly pear. | **P1** |
| Creosote / desert shrub | Sparse, wiry low shrub with tiny leaves, widely spaced. Representative: creosote bush. | **P1** |
| Agave / yucca rosette | A rosette of stiff sword-leaves, sometimes with a tall flower spike. Representative: agave / Joshua-tree yucca. | **P2** |
| Dry dead bush | A small leafless tangled shrub / tumbleweed form for desolation. Representative: dead sagebrush / tumbleweed. | **P2** |

---

## P1 biomes (important, build after the P0 layer)

### Tundra
_Cold and dry-to-moderate. Low, hardy, ground-hugging plants and lichened rock._

| Object Type | Description | Priority |
|---|---|---|
| Dwarf shrub mat | A low, ground-hugging woody mat (knee-high at most), wind-pruned. Representative: dwarf willow / dwarf birch. | **P1** |
| Tussock sedge / cotton grass | Clumped sedge tufts, some with white seed-head tufts, dotting boggy flats. Representative: tussock cottongrass. | **P1** |
| Lichen-crusted boulder | Rock blotched orange/grey/green with crustose lichen — a defining tundra texture (or cross-biome boulder, tundra tint). Representative: lichen-covered granite. | **P1** |
| Cushion plant mound | A tight low dome of tiny leaves/flowers hugging the ground. Representative: moss campion. | **P2** |

### Beach
_Warm coastal sand at the shoreline. Salt-tolerant, wind-shaped forms._

| Object Type | Description | Priority |
|---|---|---|
| Leaning coastal palm | A palm with a curved, wind-leaned trunk — the iconic shoreline tree. Representative: coconut palm. | **P1** |
| Dune grass tuft | Tall wispy salt-tolerant grass clumps anchoring the sand. Representative: marram / sea oats. | **P1** |
| Driftwood log | A bleached, smoothed, often root-tangled beached log. Representative: weathered driftwood. | **P2** |
| Coastal scrub shrub | Low salt-pruned shrub at the back of the beach. Representative: bayberry / sea grape. | **P2** |
| Sea stack / tide boulder | A wet, dark, surf-rounded rock or emergent stack at the waterline. Representative: coastal sea stack. | **P2** |

### Mountain
_High, steep, bare rock (overrides biome by altitude+slope). Mostly geology, sparse hardy plants._

| Object Type | Description | Priority |
|---|---|---|
| Talus / scree pile | A heap of angular broken rock fragments on a slope — the dominant high-mountain ground feature. Representative: alpine talus. | **P1** |
| Krummholz conifer | A stunted, flagged, wind-bent dwarf conifer clinging to the treeline edge. Representative: wind-pruned krummholz spruce/pine. | **P1** |
| Jagged rock spire | A sharp, fractured vertical rock tooth/pinnacle for ridgelines. Representative: granite aiguille. | **P2** |
| Alpine cushion / hardy tussock | A tiny tight cushion plant or hardy grass tuft wedged among rocks. Representative: alpine cushion plant. | **P2** |

---

## P2 biomes (sparse / barren — fill out last)

### Snow
_Cold high or polar ground under snow cover. Mostly white with a few capped forms._

| Object Type | Description | Priority |
|---|---|---|
| Snow-laden conifer | A spire/fir conifer heavy with snow on its branches — the one strong silhouette in a snowfield. Representative: snow-capped spruce. | **P1** |
| Snow-covered boulder | A rounded white-capped rock mound (cross-biome boulder under snow). | **P2** |
| Bare snag in snow | A dark leafless dead trunk standing out against white. Representative: dead snag. | **P2** |

### Polar Ice
_Coldest, effectively barren. Geology/ice only — minimal targets._

| Object Type | Description | Priority |
|---|---|---|
| Ice hummock / pressure ridge | A buckled mound or ridge of fractured ice slabs. Representative: sea-ice pressure ridge. | **P2** |
| Glacial erratic on ice | A lone dark boulder stranded on the ice sheet (cross-biome boulder, polar tint). Representative: glacial erratic. | **P2** |

---

## Ocean

Ocean is a terrain classification (sea floor) rendered as an animated water
surface — there is **no above-surface flora** to model. Shoreline interest is
covered by the **Beach** rows (leaning palm, dune grass, sea stack / tide
boulder). If submerged or floating detail is ever wanted (kelp, floating ice
floes), add it here as a future section.

---

## Summary counts

| Priority | Count |
|---|---|
| **P0** (build first) | 9 |
| **P1** | 22 |
| **P2** | 20 |
| **Total target models** | **51** |

| Section | Count |
|---|---|
| Cross-biome objects | 4 |
| Temperate Forest | 6 |
| Boreal Forest | 5 |
| Tropical Forest | 7 |
| Grassland | 5 |
| Desert | 6 |
| Tundra | 4 |
| Beach | 5 |
| Mountain | 4 |
| Snow | 3 |
| Polar Ice | 2 |

Build order: all **P0** rows (cross-biome boulder + the iconic forms of the five
dominant biomes) → all **P1** rows → all **P2** rows.
