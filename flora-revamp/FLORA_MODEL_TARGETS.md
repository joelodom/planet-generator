# Flora & Object Model Targets

A prioritized catalogue of Earth-flora **archetypes** and **scene objects** to
build as 3D models for Planet Explorer's vegetation pass. The intended pipeline
is: generate a photorealistic reference image (OpenAI image generator, using the
**Image-Generator Prompt** column below) → feed it to
[`../formcast`](../formcast) to generate a 3D model → import as a per-species
mesh that replaces today's procedurally-generated flora.

This document is **only the target list + prompts** — it does not change any
code. It is organised by the 11 biomes the planet actually classifies (see
`Biome` in `src/planet.rs`), each driven by the height/temperature/moisture/slope
rules in `classify()`. The goal is believable, place-appropriate stands per
biome, not botanical completeness.

## How to read this

- **Object Type** — the short archetype name the model will be filed under (the
  kind of label a species slot would carry, e.g. "spire conifer", "boulder").
- **Image** — link to the generated reference plate in [`images/`](images/), once
  produced. Generate the **P0** plates with [`generate_images.py`](generate_images.py)
  (uses GPT Image 2 and the prompts below); rows without an image yet show `—`.
- **Description** — what the form looks like and *why it belongs in this biome*,
  with a real-world representative species to anchor the look.
- **Image-Generator Prompt** — a ready-to-send prompt that produces a single
  photorealistic specimen **isolated on a plain background**, framed so formcast
  can segment and reconstruct it cleanly. Each prompt is self-contained — copy one
  cell, send it, done.
- **Priority** — global build order across the whole list:
  - **P0** — highest visual payoff. Either a dominant, frequently-seen biome
    (temperate / boreal / tropical forest, grassland, desert) **or** the single
    most iconic silhouette of a biome. Build these first.
  - **P1** — important for variety and realism; build after the P0 layer reads well.
  - **P2** — sparse, rare, or polish; fills out thin/barren biomes last.

Priority is **global**, so a forest's P0 tree outranks a polar-ice P2 even though
both are "top of their biome." Build all P0s, then all P1s, then all P2s.

### Prompt conventions (why every prompt ends the same way)

Each prompt pairs a specific **subject** with a fixed **capture recipe** chosen to
maximize formcast's image-to-3D success:

- **One isolated specimen, centered, fully in frame** — no scene, no companions, no
  ground plane, nothing cropped. formcast reconstructs *one* object.
- **Plain seamless neutral mid-grey background (~50% grey)** — a deliberate mid-grey
  (not white) so it contrasts cleanly with *both* white subjects (snow, birch bark,
  ice) and dark subjects (snags, wet rock), giving clean segmentation either way.
- **Soft even diffuse lighting, no cast shadows** — flat, shadow-free lighting avoids
  baking directional light/occlusion into the model's albedo.
- **Slight 3/4 viewing angle, sharp focus, high detail, no text/people/watermark** —
  a 3/4 angle reveals form (depth + silhouette) better than a flat elevation; the
  rest keeps the plate clean.

If you tune the recipe, change it once here and re-derive — keep all 51 prompts on
the same recipe so formcast gets consistent inputs.

> **Climate cross-check.** A biome here only ever appears where `classify()` puts
> it: forests need moisture; grassland/desert split on moisture at warm temps;
> tundra/boreal split on moisture at cold temps; snow/mountain override by
> altitude+slope. Keep model choices consistent with that — e.g. desert flora must
> survive low moisture, boreal flora must survive cold.

---

## Cross-biome objects (build once, tint/scale per biome)

These are not tied to one biome — the same base mesh gets recoloured and rescaled
wherever it's placed, so they earn their build cost across the whole planet.

| Object Type | Image | Description | Image-Generator Prompt | Priority |
|---|---|---|---|---|
| Granite boulder cluster | [granite-boulder-cluster.png](images/granite-boulder-cluster.png) | A few rounded-to-angular grey boulders grouped as one prop. The single highest-value object: rock reads believable everywhere and breaks up flat ground in *every* land biome. Representative: weathered granite glacial erratics. | A cluster of three or four weathered grey granite boulders grouped together, rounded by glaciation with angular fractured faces and faint lichen flecks. Photorealistic, single isolated group centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P0** |
| Weathered rock outcrop / fractured slab | — | A low slab of fractured, stratified bedrock poking through soil. Anchors mid-slope terrain and forest floors. Representative: exposed sandstone/limestone shelf. | A low outcrop of fractured stratified sandstone bedrock, horizontal layered beds with cracked weathered edges, sun-bleached tan and grey. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P1** |
| Fallen log / deadwood | — | A toppled, barkless trunk lying on the ground, optionally split. Adds life-cycle realism to any forested or formerly-forested biome. Representative: fallen pine/oak bole. | A single fallen tree trunk lying horizontally, bark mostly stripped to bare weathered grey wood, split and cracked along its length with a broken root flare at one end. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P1** |
| Standing dead snag | — | A bare, branchless or broken-topped dead trunk left standing. Cheap silhouette variety in forest and boreal stands. Representative: fire-killed conifer snag. | A bare standing dead tree snag, branchless weathered silver-grey trunk with a broken jagged top and remnants of cracked bark, no leaves. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P2** |

---

## P0 biomes (dominant, highest payoff)

### Temperate Forest
_Mild temps, high moisture. One of the most-seen vegetated biomes — invest heavily._

| Object Type | Image | Description | Image-Generator Prompt | Priority |
|---|---|---|---|---|
| Rounded broadleaf hardwood | [broadleaf-hardwood.png](images/broadleaf-hardwood.png) | Dense, rounded deciduous canopy on a moderate trunk — the default "leafy tree" silhouette. Representative: sugar maple / European beech. | A mature deciduous hardwood tree with a dense rounded full green canopy, a single sturdy trunk and a branching crown in summer foliage, sugar-maple form, whole tree shown. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P0** |
| Spreading-crown oak | [spreading-oak.png](images/spreading-oak.png) | Tall, sturdy hardwood with a broad, somewhat irregular spreading crown and heavier branching than the maple form. Representative: English/white oak. | A large mature oak tree with a broad irregular spreading crown, a thick gnarled trunk and heavy lateral limbs, dense dark-green summer foliage, whole tree shown. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P0** |
| Slender white-barked birch | — | Tall, thin, pale-trunked tree with a light open canopy; reads great in clumps for tonal contrast against darker hardwoods. Representative: paper birch / aspen. | A tall slender birch tree with smooth white papery bark marked with dark scars and a light airy canopy of small green leaves, single trunk, whole tree shown. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P1** |
| Leafy understory shrub | — | Knee-to-head-height rounded bush filling the forest floor between trunks. Representative: hazel / dogwood thicket. | A rounded leafy deciduous shrub about waist-to-head height, dense green foliage on multiple woody stems rising from the base, hazel form. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P1** |
| Fern cluster | — | Low, lacy frond groundcover for damp shaded floor. Representative: bracken / lady fern. | A cluster of bracken ferns, lacy arching green fronds radiating from a central base as a low clump. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P2** |
| Wildflower clump | — | Small mixed-colour flowering groundcover for clearings and edges. Representative: woodland aster / foxglove patch. | A small clump of mixed woodland wildflowers, slender green stems topped with purple and white blossoms in a loose natural bunch. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P2** |

### Boreal Forest
_Cold but moist. The classic dark northern conifer band._

| Object Type | Image | Description | Image-Generator Prompt | Priority |
|---|---|---|---|---|
| Tall spire conifer (spruce) | [spruce-spire-conifer.png](images/spruce-spire-conifer.png) | Narrow, steeply conical evergreen with a pointed top — the defining boreal silhouette. Representative: black/white spruce. | A tall narrow spruce tree, steeply conical with a pointed top and dense dark blue-green needled branches drooping slightly, straight trunk, whole tree shown. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P0** |
| Soft dense fir | — | Fuller, softer conical conifer with denser foliage and a slightly rounder profile than spruce. Representative: balsam/Douglas fir. | A full conical fir tree with soft dense deep-green foliage in layered branches and a slightly rounded profile, balsam-fir form, whole tree shown. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P1** |
| Open long-needle pine | — | More open, irregular conifer with longer needles and visible trunk/branch structure. Representative: Scots/jack pine. | An open irregular pine tree with long green needles in tufts, a visible reddish-brown trunk and bare lower branches, Scots-pine form, whole tree shown. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P1** |
| Low juniper / scrub conifer | — | Sprawling low evergreen shrub for the forest floor and clearings. Representative: common juniper mat. | A low sprawling juniper shrub, a spreading mat of blue-green needled evergreen foliage close to the ground on a woody base. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P2** |
| Moss/lichen-covered rock | — | Boulder variant draped in green moss and pale lichen for the damp boreal floor (or reuse the cross-biome boulder with a boreal tint). Representative: mossy granite. | A granite boulder draped in thick green moss and pale grey-green lichen patches, damp surface. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P2** |

### Tropical Forest
_Warm, high moisture. Lush, tall, structurally varied — needs the most distinct forms._

| Object Type | Image | Description | Image-Generator Prompt | Priority |
|---|---|---|---|---|
| Emergent broad-canopy tree | [tropical-emergent-tree.png](images/tropical-emergent-tree.png) | A towering rainforest giant with a wide, flat, umbrella-like crown breaking above the canopy. The iconic jungle silhouette. Representative: kapok / ceiba. | A towering rainforest kapok tree with a tall straight pale trunk and a wide flat umbrella-shaped crown of green foliage high above, emergent giant, whole tree shown. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P0** |
| Feather-frond palm | [feather-frond-palm.png](images/feather-frond-palm.png) | Tall slender trunk topped with a crown of arching pinnate fronds. Representative: coconut / royal palm. | A tall coconut palm with a slender slightly curved trunk topped by a crown of long arching pinnate green fronds, whole tree shown. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P0** |
| Tree fern | — | A short fibrous trunk crowned with a rosette of large lacy fronds — distinctly prehistoric, fills the mid-story. Representative: Cyathea tree fern. | A tree fern with a short slender fibrous brown trunk topped by a rosette of large lacy arching green fronds, whole plant shown. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P1** |
| Broad-leaf understory plant | — | Clustered very large paddle leaves at ground-to-waist height. Representative: banana / heliconia / wild ginger. | A clustered banana-like understory plant with very large broad paddle-shaped bright-green leaves rising and arching from the base. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P1** |
| Hanging liana / vine | — | Woody vines draping and looping between canopy layers; adds vertical tangle. Representative: rattan / jungle liana. | A tangle of woody jungle lianas, thick twisting vines looping and hanging with sparse green leaves, as one draped mass. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P1** |
| Buttress-root tree | — | A straight-boled tree with dramatic flared buttress roots at its base. Representative: strangler fig / Pterocarpus. | A tall straight tropical rainforest tree with dramatic flared buttress roots fanning out at its base, smooth grey trunk and a high green canopy, whole tree shown. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P2** |
| Bromeliad / epiphyte clump | — | A rosette of stiff leaves perched on trunks/branches, often vividly tinted. Representative: tank bromeliad. | A tank bromeliad, a rosette of stiff arching strap-shaped leaves with red-and-green coloration and a bright central bloom. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P2** |

### Grassland
_Warm/temperate, drier than forest. Open ground with sparse standout features._

| Object Type | Image | Description | Image-Generator Prompt | Priority |
|---|---|---|---|---|
| Bunchgrass tussock | [bunchgrass-tussock.png](images/bunchgrass-tussock.png) | A dense clump of tall blades — the staple groundcover that *makes* a grassland read as grass, placed at high density. Representative: bluestem / fescue tussock. | A dense clump of tall bunchgrass, fine arching golden-green blades radiating from a tight base, prairie tussock. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P0** |
| Flat-topped savanna tree | [savanna-acacia.png](images/savanna-acacia.png) | A lone tree with a high, wide, umbrella-flat crown on a clear trunk — the unmistakable savanna icon. Representative: umbrella acacia. | A lone umbrella acacia tree with a clear trunk and a high wide flat-topped canopy of fine green foliage, classic savanna form, whole tree shown. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P0** |
| Sagebrush / scattered shrub | — | Low silvery-green woody shrub dotted across open ground. Representative: big sagebrush / rabbitbrush. | A low silvery sagebrush shrub, a woody base with soft grey-green aromatic foliage in a rounded sparse form. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P1** |
| Wildflower meadow clump | — | Mixed flowering forbs in colourful patches for prairie variety. Representative: coneflower / poppy meadow. | A clump of mixed prairie wildflowers, tall green stems topped with red, yellow and purple blooms in a natural bunch. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P1** |
| Grassland rock outcrop | — | Low rocks rising from open plain (or reuse cross-biome outcrop with a sun-bleached tint). | A low cluster of sun-bleached grey rocks with weathered rounded tops, a few dry grass blades at the base. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P2** |

### Desert
_Warm, very low moisture. Sparse, spiky, sun-bleached — striking against bare sand._

| Object Type | Image | Description | Image-Generator Prompt | Priority |
|---|---|---|---|---|
| Columnar cactus | [columnar-cactus.png](images/columnar-cactus.png) | A tall ribbed column with raised arms — the signature desert silhouette. Representative: saguaro. | A tall saguaro cactus, a single ribbed green column with two raised curving arms and rows of spines along the ridges, whole plant shown. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P0** |
| Eroded sandstone formation | — | A wind-carved, layered rock spire/mesa fragment; the desert's main vertical mass where plants are scarce. Representative: red sandstone hoodoo. | An eroded red-orange sandstone hoodoo, a wind-carved layered rock spire with horizontal striations and a rounded balanced cap. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P1** |
| Barrel / pad cactus cluster | — | A low grouping of squat barrel cacti and flat prickly-pear pads. Representative: barrel cactus / prickly pear. | A low cluster of squat barrel cacti and flat oval prickly-pear pads, ribbed green surfaces with dense spines. Photorealistic, single isolated group centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P1** |
| Creosote / desert shrub | — | Sparse, wiry low shrub with tiny leaves, widely spaced. Representative: creosote bush. | A sparse wiry creosote bush, thin woody branches with tiny dark-green leaves in an open airy desert shrub form. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P1** |
| Agave / yucca rosette | — | A rosette of stiff sword-leaves, sometimes with a tall flower spike. Representative: agave / Joshua-tree yucca. | An agave rosette of thick stiff blue-green sword-shaped leaves with sharp tips, radiating symmetrically from the center. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P2** |
| Dry dead bush | — | A small leafless tangled shrub / tumbleweed form for desolation. Representative: dead sagebrush / tumbleweed. | A small leafless tangled dead shrub, dry grey woody twigs in a rounded brittle tumbleweed mass. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P2** |

---

## P1 biomes (important, build after the P0 layer)

### Tundra
_Cold and dry-to-moderate. Low, hardy, ground-hugging plants and lichened rock._

| Object Type | Image | Description | Image-Generator Prompt | Priority |
|---|---|---|---|---|
| Dwarf shrub mat | — | A low, ground-hugging woody mat (knee-high at most), wind-pruned. Representative: dwarf willow / dwarf birch. | A low ground-hugging dwarf willow mat, small woody stems with tiny green leaves spreading flat and wind-pruned. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P1** |
| Tussock sedge / cotton grass | — | Clumped sedge tufts, some with white seed-head tufts, dotting boggy flats. Representative: tussock cottongrass. | A clump of tussock cottongrass, tufts of fine green sedge blades topped with fluffy white seed-heads. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P1** |
| Lichen-crusted boulder | — | Rock blotched orange/grey/green with crustose lichen — a defining tundra texture (or cross-biome boulder, tundra tint). Representative: lichen-covered granite. | A grey granite boulder crusted with orange, pale-green and grey lichen blotches, weathered tundra rock. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P1** |
| Cushion plant mound | — | A tight low dome of tiny leaves/flowers hugging the ground. Representative: moss campion. | A tight low dome-shaped moss campion cushion plant, a dense mound of tiny green leaves studded with small pink flowers. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P2** |

### Beach
_Warm coastal sand at the shoreline. Salt-tolerant, wind-shaped forms._

| Object Type | Image | Description | Image-Generator Prompt | Priority |
|---|---|---|---|---|
| Leaning coastal palm | — | A palm with a curved, wind-leaned trunk — the iconic shoreline tree. Representative: coconut palm. | A coconut palm with a curved wind-leaned trunk and a crown of long arching green fronds, tropical shoreline form, whole tree shown. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P1** |
| Dune grass tuft | — | Tall wispy salt-tolerant grass clumps anchoring the sand. Representative: marram / sea oats. | A tuft of tall wispy marram dune grass, thin pale-green blades clumped together rising from a sandy base. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P1** |
| Driftwood log | — | A bleached, smoothed, often root-tangled beached log. Representative: weathered driftwood. | A bleached smooth driftwood log, pale silver-grey weathered wood smoothed by surf with a tangle of worn roots at one end. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P2** |
| Coastal scrub shrub | — | Low salt-pruned shrub at the back of the beach. Representative: bayberry / sea grape. | A low salt-pruned coastal shrub, a dense rounded mound of small waxy green leaves, sea-grape form. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P2** |
| Sea stack / tide boulder | — | A wet, dark, surf-rounded rock or emergent stack at the waterline. Representative: coastal sea stack. | A dark wet surf-rounded boulder, smooth rock streaked with damp and a band of barnacles near the base. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P2** |

### Mountain
_High, steep, bare rock (overrides biome by altitude+slope). Mostly geology, sparse hardy plants._

| Object Type | Image | Description | Image-Generator Prompt | Priority |
|---|---|---|---|---|
| Talus / scree pile | — | A heap of angular broken rock fragments on a slope — the dominant high-mountain ground feature. Representative: alpine talus. | A pile of angular broken grey rock fragments heaped together, sharp-edged alpine talus scree. Photorealistic, single isolated pile centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P1** |
| Krummholz conifer | — | A stunted, flagged, wind-bent dwarf conifer clinging to the treeline edge. Representative: wind-pruned krummholz spruce/pine. | A stunted wind-bent krummholz conifer, a low gnarled twisted trunk with foliage flagged to one side, treeline dwarf, whole plant shown. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P1** |
| Jagged rock spire | — | A sharp, fractured vertical rock tooth/pinnacle for ridgelines. Representative: granite aiguille. | A sharp fractured granite rock spire, a vertical jagged pinnacle with cracked angular faces in grey alpine stone. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P2** |
| Alpine cushion / hardy tussock | — | A tiny tight cushion plant or hardy grass tuft wedged among rocks. Representative: alpine cushion plant. | A small alpine cushion plant, a tight low mound of tiny green leaves and minute flowers wedged on a small rock. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P2** |

---

## P2 biomes (sparse / barren — fill out last)

### Snow
_Cold high or polar ground under snow cover. Mostly white with a few capped forms._

| Object Type | Image | Description | Image-Generator Prompt | Priority |
|---|---|---|---|---|
| Snow-laden conifer | — | A spire/fir conifer heavy with snow on its branches — the one strong silhouette in a snowfield. Representative: snow-capped spruce. | A spruce tree heavily laden with snow, dark green branches bowed under thick white snow caps, conical winter form, whole tree shown. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P1** |
| Snow-covered boulder | — | A rounded white-capped rock mound (cross-biome boulder under snow). | A rounded boulder topped with a smooth cap of white snow, grey rock showing on the lower sides. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P2** |
| Bare snag in snow | — | A dark leafless dead trunk standing out against white. Representative: dead snag. | A dark bare leafless dead tree snag with weathered grey wood and broken branches, its base sitting in a mound of white snow. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P2** |

### Polar Ice
_Coldest, effectively barren. Geology/ice only — minimal targets._

| Object Type | Image | Description | Image-Generator Prompt | Priority |
|---|---|---|---|---|
| Ice hummock / pressure ridge | — | A buckled mound or ridge of fractured ice slabs. Representative: sea-ice pressure ridge. | A buckled ridge of fractured pale-blue sea ice, jagged tilted ice slabs piled into a pressure ridge. Photorealistic, single isolated formation centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P2** |
| Glacial erratic on ice | — | A lone dark boulder stranded on the ice sheet (cross-biome boulder, polar tint). Representative: glacial erratic. | A lone dark grey glacial erratic boulder, rounded and weathered, resting on a small patch of white ice. Photorealistic, single isolated specimen centered and fully in frame, plain seamless neutral mid-grey background, soft even diffuse lighting, no cast shadows, slight 3/4 viewing angle, sharp focus, high detail, no text, no people, no watermark. | **P2** |

---

## Ocean

Ocean is a terrain classification (sea floor) rendered as an animated water
surface — there is **no above-surface flora** to model. Shoreline interest is
covered by the **Beach** rows (leaning palm, dune grass, sea stack / tide
boulder). If submerged or floating detail is ever wanted (kelp, floating ice
floes), add it here as a future section.

---

## Generating the reference images

The P0 plates are generated by [`generate_images.py`](generate_images.py), which
calls OpenAI's **GPT Image 2** (`gpt-image-2`) with the exact prompts above and
writes PNGs into [`images/`](images/) under the filenames in the **Image** column.

```bash
pip install openai                 # OpenAI Python SDK
export OPENAI_API_KEY=sk-...        # your key
python flora-revamp/generate_images.py          # generate any missing P0 plates
python flora-revamp/generate_images.py --force  # regenerate all P0 plates
python flora-revamp/generate_images.py --only columnar-cactus   # just one
```

Only the **P0** items are wired up so far. Add P1/P2 items to the `ITEMS` list in
the script (and fill in their **Image** links above) when you're ready to extend.

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
