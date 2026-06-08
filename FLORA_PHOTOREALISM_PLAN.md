# Flora Photorealism — Plan

A staged plan to make Planet Explorer's vegetation read as *more photorealistic*.
The goal is "better, not perfect": land the high-leverage cues first, defer the
expensive pipeline work. Every item names the files/functions it touches, its
cost, and how it stays inside [`ARCHITECTURE_GUIDELINES.md`](ARCHITECTURE_GUIDELINES.md)
(determinism, the per-frame/per-chunk hot paths, `gfx` stays a thin slab,
cross-platform, **no magic numbers**).

> Priority hierarchy (the tie-breaker): **correctness → security → performance →
> maintainability → portability → polish.** Photorealism is *polish* — it must
> never buy looks by breaking determinism, the hot path, or portability.

---

## TL;DR — what to do first

**Slice 1 (recommended first cut — no new assets, low risk, deterministic, and it
visibly transforms the look):**

1. **Veg shading overhaul** in `shaders/vegetation.wgsl`: leaf **translucency /
   back-light transmission**, a **half-Lambert wrap** to soften the terminator,
   and a weak **leaf sheen**. Driven by a new per-vertex **material/AO** attribute
   so leaves are lit differently from wood.
2. **Baked ambient occlusion** into the foliage at generation time (`flora.rs`
   builders) — canopy interiors/undersides and trunk bases darken, tops/outers
   brighten. Gives forms volume without real-time AO.
3. **Per-instance variation** in `mesh.rs::place_vegetation`: small lean, gentle
   non-uniform scale, and richer per-instance tint — kills the "clone army" look.
4. **Extend the gallery harness** to render through `vegetation.wgsl` (it currently
   uses the terrain shader), plus a back-lit view, so we can *see* 1–3 in a headless
   PNG and iterate.

That slice is the one to build, package, and study. Everything below it (silhouette
geometry, wind, leaf cards, MSAA) is sequenced after.

---

## Where flora lives today (the pipeline)

| Stage | File | What it does |
|---|---|---|
| Species library | `src/flora.rs` | Per-planet, per-biome procedural plants. `Flora::generate(seed)` grows up to `SPECIES_PER_BIOME` meshes per vegetated biome from low-poly primitives: `segment` (tapered tube — trunks/limbs), `cone` (conifer tiers), `ellipsoid` (lumpy blob — canopy/shrub/flower heads), `frond` (flat drooping strip — grass/palm). Colour baked per vertex. |
| Placement | `src/mesh.rs` → `place_vegetation` | Per-chunk, chunk-key-seeded scatter. Linear-decay clustering per species; emits `VegInstance { model: Mat4, tint }` (~80 B) grouped by species. |
| Render | `src/shaders/vegetation.wgsl` | Instanced draw, one per species per chunk. Lambert diffuse + flat ambient + tiny sky fill + distance fog. **No** specular/translucency/AO/shadows/wind. |
| Evaluate | `src/gfx.rs` → `mod gallery` | Offscreen PNGs: `flora_gallery_renders` → `planet_flora_gallery.png`, `terrain_closeup_renders` → `planet_flora_closeup.png`. ⚠️ Renders flora via the **terrain** shader (baked to world space), not `vegetation.wgsl`. |

Lighting context (`gfx.rs` Globals): a single directional sun (`sun_dir.xyz`), a
flat ambient (`sun_dir.w`, ~0.42), atmosphere tint for fog/sky. Opaque pass is
`sample_count: 1` (no MSAA), blend `REPLACE`, `cull_mode: None`.

---

## Diagnosis — why the current flora reads as "CG"

1. **Blobby/geometric silhouette.** Canopies are smooth lumpy spheres
   (`ellipsoid`, `BLOB_RINGS=3`, `BLOB_SECTORS=5`); conifers are clean stacked
   cones. Real foliage has a broken, near-fractal, semi-transparent edge.
2. **No translucency.** Pure Lambert makes leaves look like opaque painted
   plastic. Real leaves transmit and scatter light; back-lit canopies glow.
3. **Harsh terminator + dead shade side.** `max(dot(n,l),0)` on a sphere gives a
   hard light/dark split; the shadow side is lifted only by flat ambient. Real
   canopies have strong inter-leaf bounce, so the falloff is soft.
4. **Flat, low-variance colour.** Base colour ± small jitter (`FOLIAGE_VERT_JITTER
   = 0.05`); little top/bottom or sun/shade variation, no per-leaf hue spread.
5. **"Clone army."** Every instance of a species is the *same* mesh; only uniform
   scale (`0.82–1.20`) and a near-white tint vary. Identical trees in identical
   upright poses read as instanced CG.
6. **Pasted-on, no ground integration.** Plants are lone objects on the terrain
   (sunk by `VEG_SINK`). No contact shadow/AO, no leaf litter, no basal grass — so
   they look like decals floating on the ground.
7. **Dead still.** No wind. Static vegetation is an instant tell.
8. **Aliasing.** `sample_count: 1`: thin trunks, fronds, and leaf edges shimmer and
   crawl, especially in motion (the tour).

---

## The plan, in tiers

Each tier is roughly *payoff ÷ effort* descending. Tier 0–1 are the "better, not
perfect" core; Tier 3 is the genuine photoreal leap when we want it.

### Tier 0 — Shading & light (biggest ratio; no new assets)

A single cohesive change: give vegetation its **own vertex attribute** carrying a
`material` flag (wood vs leaf) and a baked `ao` term, then upgrade
`vegetation.wgsl`. Today terrain and veg share `vertex_layout()`; this introduces a
veg-specific layout so the shader can treat foliage ≠ bark without touching terrain.

- **0a. Leaf translucency / back-light transmission.** Add a transmission lobe:
  light arriving from *behind* a leaf surface (`dot(-n, l)` style, modulated by
  view) adds a warm, brightened tint of the leaf colour. Foliage only (gate on
  `material`). *Why:* the single biggest "plastic → leaves" cue.
- **0b. Half-Lambert wrap.** Replace `max(dot(n,l),0)` for foliage with a wrapped
  diffuse (`dot*0.5+0.5`), softening the terminator and lifting the shade side
  naturally instead of with flat ambient. *Why:* fixes tell #3.
- **0c. Weak leaf sheen.** A broad, low-strength specular (half-vector, like the
  water glint but much softer) so canopies catch a waxy highlight under sun.
- **0d. Baked AO + foliage normal break-up.** In `flora.rs` builders, compute a
  per-vertex AO (canopy interior/underside and trunk base darker; tops/outers
  brighter — generalising the conifer's existing tier ramp) and store it in the new
  attribute. Optionally jitter foliage normals (we already jitter position+colour,
  not normals) so lighting breaks up across a blob instead of reading as a balloon.
- **Where:** `shaders/vegetation.wgsl` (`vs`/`fs`); `mesh.rs` `Vertex`/veg vertex
  layout (or a new `VegVertex`); `flora.rs` `vert`/`ellipsoid`/`cone`/`segment`/`frond`
  tag material + AO.
- **Cost:** per-fragment math (cheap); one extra vertex attribute; gen-time AO.
- **Arch:** deterministic (baked from the seed at build time), off the main thread
  (in flora build), `gfx` unchanged in spirit (just richer shader). All new tuning
  values become named `const`s in WGSL/Rust.
- **Test:** keep `offscreen_pipeline_validates` green; extend gallery (below) to
  show it.

### Tier 1 — Cheap geometry & per-instance variation (generation side)

- **1a. Per-instance variation.** In `place_vegetation`, enrich the per-plant
  transform/tint that's *already emitted*: a small random **lean** (compose a tilt
  quaternion — trees aren't all plumb), gentle **non-uniform scale** (squash/stretch
  in Vec3), and a wider **tint** spread (hue/brightness, not just near-white). *Why:*
  kills the clone-army look (#5) at ~zero cost — no extra geometry or memory.
  *Caveat:* `vegetation.wgsl` assumes the instance upper-3×3 is rotation×uniform
  scale for normals; keep the anisotropy modest, or pass a per-instance normal
  matrix. Flag in the shader comment.
- **1b. A few mesh variants per species.** Build N≈3 variants per species and pick
  per instance. *Why:* silhouette variety within a species. *Cost:* tiny extra
  base-mesh memory (uploaded once) but **more draw calls** — see `BACKLOG.md`
  "Tame `split_factor`"; weigh against that.
- **1c. Richer canopy silhouette.** More, smaller overlapping foliage blobs with
  stronger displacement (raise/za second octave on `BLOB_LUMP`), jittered cone rims
  so conifer tiers aren't perfect cones, optional drooping needle fronds at tier
  edges. *Why:* attacks tell #1 directly. *Cost:* more polys per species (still
  tiny memory; watch CPU/visual only, per `BACKLOG.md`).
- **1d. Branch structure.** A second branching level for `build_broadleaf` /
  `build_snag` so crowns sit on visible boughs and bare/winter frames read as real
  wood.
- **1e. Basal integration (grass collar).** Bias a few small grass/forb instances
  around the base of larger plants in `place_vegetation`. *Why:* stops the pasted-on
  look (#6). (Contact shadows on the terrain itself are harder — defer to Tier 3.)
- **Arch:** all seeded from existing per-chunk/per-species sub-seeds → deterministic;
  all in worker-side generation → off the main thread. Keep per-plant work O(1).

### Tier 2 — Wind / motion

- **2a. Wind sway.** Animate foliage vertices in `vegetation.wgsl` `vs`: sway
  amplitude scales up the plant (trunk base fixed → canopy/frond/grass tips move
  most) and with the `material`/local-Y weight, driven by a wind vector + time
  (`camera_pos.w`) + a per-instance phase. *Why:* the biggest "alive" boost; static
  plants are an instant tell. Precedent: the water already animates in `terrain.wgsl`.
- **⚠️ Precision landmine (ARCH §8 / review M6).** Do **not** phase the sway off
  absolute world position — at ~637 k-unit magnitude `sin(world_pos·k)` loses phase
  precision. Drive it from a **per-instance phase scalar** (add to the instance
  buffer) or local/fractional coordinates. Call this out at the call site.
- **Arch:** render-time only (like water ripples) — the *world* mesh/instances stay
  deterministic; nothing in generation depends on wall-clock. Named consts for
  amplitude/frequency/stiffness.

### Tier 3 — The photoreal leap (highest effort, highest payoff; later)

- **3a. Alpha-tested leaf cards.** Replace/augment solid foliage blobs with clusters
  of textured, **alpha-tested** quads (crossed quads or camera-facing billboards)
  carrying a leaf-cluster texture — how most real-time engines get photoreal
  canopies. Needs: UVs on veg vertices; a texture+sampler bind group (veg is
  vertex-colour only today); **alpha-test/clip** in the fragment (order-independent —
  works with the current opaque+depth-write pass, no sorting). Texture source:
  procedurally generated at startup (stays self-contained & seed-deterministic, in
  keeping with the project) or a small embedded PNG (`assets/` + `include_bytes!`,
  like `planet.png`).
- **3b. Bark / ground textures.** Same pipeline, lower priority than leaves.
- **3c. Vegetation LOD impostors.** Far plants → a single billboard impostor; full
  mesh only up close. Hooks into the existing chunk LOD (`veg_min_level`). Lets near
  plants get much richer (Tier 1c/3a) without paying for it at distance, and cuts
  distant thin-geometry aliasing.
- **3d. MSAA (or FXAA).** Enable multisampling on the opaque pass (resolve target),
  or a cheap post AA. *Why:* fixes tell #8 for *all* geometry, not just veg — the
  tour especially. Cross-cutting (touches every pipeline + the render targets); gate
  the sample count behind the graphics preset (`settings.rs`) — the RTX 5090 eats
  4–8×, the Mac is fine at 4×.

### Cross-cutting — ecology & distribution

Cheap realism via `place_vegetation`: denser ground cover *under* canopy
(undergrowth layering), a more natural size histogram (mix saplings + mature — falls
out of 1a's scale variation), and continued species mixing at biome edges (already
dithered via `sample_blended`). Lower priority; fold into Tier 1.

---

## Evaluation workflow (how to study output headlessly)

The `gallery` module is the tight loop — no GUI needed:

- `flora_gallery_renders` → `<tempdir>/planet_flora_gallery.png` (species catalogue).
- `terrain_closeup_renders` → `<tempdir>/planet_flora_closeup.png` (in-situ forest;
  `--ignored`, slow).

**Fix first:** the gallery renders flora through the **terrain** shader (baked to
world space), so Tier-0 veg-shader work won't show there. Extend it to render via the
**vegetation pipeline** (instanced), and add a **back-lit** camera/sun setup so
translucency is visible. Then the before/after of every slice is a PNG diff.

Run: `cargo test --release flora_gallery -- --nocapture` (add `--ignored` for the
closeup), then copy the PNG into `/Users/Shared/` to view from the GUI account. The
existing `package_macos.sh` flow remains how the *interactive* build is shipped.

---

## Recommended sequencing

1. **Slice 1 — Tier 0 shading overhaul + Tier 1a per-instance variation + 1e grass
   collar + gallery-through-veg-shader.** No assets, low risk, deterministic, and it
   visibly changes everything. ← build, package, study.
2. **Slice 2 — Tier 1c/1d silhouette + branches, then Tier 2 wind.**
3. **Slice 3 — Tier 3 leaf cards + MSAA + LOD impostors** (the real leap, when we
   commit the pipeline work).

---

## Guardrails (hold every slice to these)

- **Determinism.** Generation changes seed from existing sub-seeds
  (`flora::mix`, `ChunkKey::hash`). Animation (wind) is render-time only, like water —
  the world mesh/instances stay a pure function of the seed.
- **Hot paths.** `place_vegetation` is per-chunk (worker): keep added work O(1) per
  plant, no new per-call heap churn. Shader additions are per-vertex/fragment: keep
  them cheap; no new per-frame allocations on the main thread.
- **`gfx` stays thin.** Policy lives in `flora`/`mesh`; `gfx` only grows a texture/
  bind group if we take Tier 3. Don't leak `wgpu` types upward.
- **Portability.** Plain WGSL, `wgpu` textures, pure-Rust cross-platform crates only;
  `cfg`-gate nothing device-specific into portable code. Gate heavier defaults
  (variant count, MSAA samples, density) behind `settings.rs` presets, not hardcoded
  per-device assumptions.
- **No magic numbers.** Every new tuning value is a named `SCREAMING_SNAKE_CASE`
  `const` with a units/intent comment — Rust *and* WGSL.
- **Tests.** New generation maths ships a determinism/bounds test; keep
  `offscreen_pipeline_validates` green; use the gallery for the visual record.

## Out of scope (deferred deliberately)

Real-time cast shadows / shadow maps; global illumination; volumetric/God-ray
lighting; physically-based material authoring; seasonal simulation. These are large,
cross-cutting renderer projects — revisit only after Tier 3 lands and only if the
payoff justifies the weight.
