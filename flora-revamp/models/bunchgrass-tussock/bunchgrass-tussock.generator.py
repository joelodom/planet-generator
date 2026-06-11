"""Procedural ornamental fountain-grass: geometry + photo-derived materials.

Builds a clumping fountain grass (a full rounded dome of fine arching blades
that rise then bow back down, on a small buried tan knot), derives tileable /
atlas materials by SAMPLING THE REFERENCE PHOTO, applies per-surface UVs and
per-vertex sun/shade tints, and exports a textured binary GLB.

CLI:
    python grass_asset.py --image PATH --seed INT \
        --density {high,med,low} --output OUT.glb

Only numpy, trimesh, PIL (Pillow) and the stdlib are used. Deterministic in
--seed (geometry and all swatch jitter).
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter

# ==========================================================================
# GEOMETRY  (build_mesh -- +Y up, base at y=0, metres)
# ==========================================================================
CLUMP_HEIGHT = 0.74           # plausible real-world ornamental grass height
# width / height ~= 0.94 (measured off the photo): nearly as wide as tall,
# a full rounded mound. (Stored as height/width for the radius calc.)
HEIGHT_OVER_WIDTH = 1.16
CLUMP_WIDTH = CLUMP_HEIGHT / HEIGHT_OVER_WIDTH   # full mound width
CLUMP_RADIUS = CLUMP_WIDTH * 0.5

# Dome is widest at ~50% of total height (measured): a rounded, full mound.
WIDEST_AT_FRAC = 0.50

# Basal knot: a SMALL, low tan cone, mostly buried by the blade bases.
CROWN_RADIUS = CLUMP_RADIUS * 0.12
CROWN_HEIGHT = CLUMP_HEIGHT * 0.03

# Blade ribbon base width as a fraction of mound width (fine but readable so
# the colour gradient catches the light).
BLADE_BASE_WIDTH = CLUMP_WIDTH * 0.011

# Density presets: (number of blades, segments per blade). Dense enough to
# read as a solid feathery mass rather than scattered wires.
_DENSITY = {
    "high": (900, 11),
    "med": (400, 8),
    "low": (140, 5),
}


def _env_radius(yn):
    """Teardrop foliage envelope: max horizontal radius (fraction of
    CLUMP_RADIUS) at normalised height yn. Widest at WIDEST_AT_FRAC."""
    yn = min(max(yn, 0.0), 1.0)
    wf = WIDEST_AT_FRAC
    if yn <= wf:
        t = yn / wf
        prof = 0.22 + 0.78 * np.sin(t * np.pi * 0.5)   # crown -> widest
    else:
        t = (yn - wf) / (1.0 - wf)
        prof = 0.10 + 0.90 * np.cos(t * np.pi * 0.5)    # widest -> point
    return CLUMP_RADIUS * prof


def _build_blade(rng, theta, vigor, nseg):
    """Return (verts, faces) for one arching tapered blade."""
    e_r = np.array([np.cos(theta), 0.0, np.sin(theta)])     # radial out
    e_tan = np.array([-np.sin(theta), 0.0, np.cos(theta)])  # tangential
    up = np.array([0.0, 1.0, 0.0])

    # Inner blades are short and upright (fill the dense core); outer blades
    # are long and arch right over to droop back down (the fountain habit).
    L = CLUMP_HEIGHT * (0.52 + 0.92 * vigor) * rng.lognormal(0.0, 0.10)

    # Launch near-vertical. Total bend grows strongly with vigor: inner blades
    # barely lean; outer blades curl past horizontal and droop. Kept moderate
    # so blades gain height before arching (taller-than-wide mound).
    phi0 = np.radians(rng.uniform(2.0, 10.0)) * (0.4 + 0.8 * vigor)
    bend = np.radians(rng.uniform(22.0, 38.0) + 114.0 * vigor)

    sway_amp = CLUMP_RADIUS * rng.uniform(0.02, 0.07)
    sway_freq = rng.uniform(0.6, 1.4)
    sway_phase = rng.uniform(0.0, np.pi)

    roll = np.radians(rng.uniform(-35.0, 35.0))
    w_dir = e_tan * np.cos(roll) + up * np.sin(roll)
    w_dir = w_dir / np.linalg.norm(w_dir)

    r0 = CROWN_RADIUS * rng.uniform(0.2, 1.0)
    pos = e_r * r0 + up * (CROWN_HEIGHT * rng.uniform(0.5, 1.0))

    ds = L / nseg
    centres = np.empty((nseg + 1, 3))
    centres[0] = pos
    for i in range(1, nseg + 1):
        frac = i / nseg
        # Bend is concentrated toward the upper portion (frac**2.0): the blade
        # rises nearly vertical first (filling the core / building height),
        # then arches over and droops near the tip -> rounded, full dome.
        phi = phi0 + bend * (frac ** 2.0)
        step = np.sin(phi) * e_r + np.cos(phi) * up
        pos = pos + step * ds
        lateral = sway_amp * np.sin(frac * np.pi * sway_freq + sway_phase)
        centres[i] = pos + lateral * e_tan

    centres[:, 1] = np.maximum(centres[:, 1], 0.0)

    # Loose envelope guard: only reel in extreme outliers so nothing pokes
    # past the mound; the silhouette is shaped by the arcs, not the clamp.
    r = np.sqrt(centres[:, 0] ** 2 + centres[:, 2] ** 2)
    for i in range(len(centres)):
        if r[i] < 1e-9:
            continue
        r_max = _env_radius(centres[i, 1] / CLUMP_HEIGHT) * 1.08
        if r[i] > r_max:
            centres[i, [0, 2]] *= r_max / r[i]

    fr = np.linspace(0.0, 1.0, nseg + 1)
    widths = BLADE_BASE_WIDTH * (1.0 - 0.92 * fr) ** 0.55
    half = 0.5 * widths[:, None] * w_dir[None, :]
    left = centres - half
    right = centres + half

    verts = np.empty((2 * (nseg + 1), 3))
    verts[0::2] = left
    verts[1::2] = right

    faces = []
    for i in range(nseg):
        l0, r0i = 2 * i, 2 * i + 1
        l1, r1 = 2 * i + 2, 2 * i + 3
        faces.append((l0, r0i, r1))
        faces.append((l0, r1, l1))
    return verts, np.array(faces, dtype=np.int64)


def _build_foliage(rng, n_blades, nseg):
    """Concatenate all blades into one foliage mesh (process=False keeps the
    per-blade vertex order, which the texturing step relies on)."""
    all_v, all_f = [], []
    offset = 0
    base_theta = np.linspace(0.0, 2.0 * np.pi, n_blades, endpoint=False)
    for k in range(n_blades):
        theta = base_theta[k] + rng.uniform(-0.25, 0.25)
        # Mid-heavy vigor builds the dome; a fraction of low-vigor blades are
        # pulled down to densely fill the upright core.
        vigor = float(np.clip(rng.beta(2.0, 2.0), 0.0, 1.0))
        if rng.random() < 0.28:
            vigor *= 0.3
        v, f = _build_blade(rng, theta, vigor, nseg)
        all_v.append(v)
        all_f.append(f + offset)
        offset += len(v)
    verts = np.vstack(all_v)
    faces = np.vstack(all_f)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _build_crown(rng):
    """Small squat tan cone -- the clustered knot where blades meet soil."""
    sections = 12
    rc = CROWN_RADIUS * 1.1
    hc = CROWN_HEIGHT
    ang = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    ring = np.column_stack([rc * np.cos(ang),
                            np.zeros(sections),
                            rc * np.sin(ang)])
    apex = np.array([[0.0, hc, 0.0]])
    centre = np.array([[0.0, 0.0, 0.0]])
    verts = np.vstack([ring, apex, centre])
    apex_i = sections
    centre_i = sections + 1
    faces = []
    for i in range(sections):
        j = (i + 1) % sections
        faces.append((i, j, apex_i))      # side
        faces.append((j, i, centre_i))    # bottom cap
    return trimesh.Trimesh(vertices=verts,
                           faces=np.array(faces, dtype=np.int64),
                           process=True)


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    """Build the untextured fountain-grass clump as a trimesh.Scene.

    Geometry keyed by semantic surface name: "foliage" (blades) and
    "crown" (basal knot)."""
    if density not in _DENSITY:
        raise ValueError("density must be one of 'high', 'med', 'low'")
    rng = np.random.default_rng(seed)
    n_blades, nseg = _DENSITY[density]

    foliage = _build_foliage(rng, n_blades, nseg)
    crown = _build_crown(rng)

    combined_min_y = min(foliage.vertices[:, 1].min(),
                         crown.vertices[:, 1].min())
    shift = np.array([0.0, -combined_min_y, 0.0])
    foliage.apply_translation(shift)
    crown.apply_translation(shift)
    foliage.fix_normals()
    crown.fix_normals()

    scene = trimesh.Scene()
    scene.add_geometry(foliage, geom_name="foliage")
    scene.add_geometry(crown, geom_name="crown")
    return scene


# ==========================================================================
# COLOR SAMPLING FROM THE PHOTO
# ==========================================================================
def _load_delit(path):
    """Load the reference, return a de-lit float RGB array in [0,255].

    De-light = divide by a heavily blurred luminance, gain clamped to
    [0.6, 1.6] so nothing washes out."""
    img = Image.open(path).convert("RGB")
    # Cap working size for speed/consistency.
    img.thumbnail((1024, 1024), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float64)
    lum = arr @ np.array([0.299, 0.587, 0.114])
    blur_r = max(arr.shape[1], arr.shape[0]) / 12.0
    blurred = np.asarray(
        Image.fromarray(lum.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(blur_r)),
        dtype=np.float64)
    blurred = np.clip(blurred, 1.0, None)
    gain = np.clip(lum.mean() / blurred, 0.6, 1.6)
    out = np.clip(arr * gain[..., None], 0, 255)
    return out


def _patch_median(arr, cx, cy, half=0.022):
    """Median RGB of a small patch centred at normalised (cx, cy)."""
    h, w = arr.shape[:2]
    px, py = cx * w, cy * h
    hx, hy = max(2, int(half * w)), max(2, int(half * h))
    x0, x1 = max(0, int(px - hx)), min(w, int(px + hx))
    y0, y1 = max(0, int(py - hy)), min(h, int(py + hy))
    patch = arr[y0:y1, x0:x1].reshape(-1, 3)
    return np.median(patch, axis=0)


def _robust_color(arr, coords, fallback, half=0.022, reject=110.0):
    """Median of several patches, discarding any patch wildly unlike the
    group (likely background). Falls back to `fallback` if all rejected."""
    meds = np.array([_patch_median(arr, cx, cy, half) for cx, cy in coords])
    centre = np.median(meds, axis=0)
    keep = meds[np.linalg.norm(meds - centre, axis=1) < reject]
    if len(keep) == 0:
        return np.array(fallback, dtype=np.float64)
    return np.median(keep, axis=0)


def sample_palette(path):
    """Sample the grass body colours WELL INSIDE the silhouette.

    cool  -- blue/grey-green of the shaded lower interior
    warm  -- golden/straw/amber of the sun-catching upper-outer blades
    soil  -- faint tan of the basal knot
    """
    arr = _load_delit(path)
    # Cool interior: lower-central blades (kept off the edges/background).
    cool = _robust_color(arr, [(0.46, 0.62), (0.52, 0.70), (0.43, 0.56),
                               (0.55, 0.64), (0.49, 0.74)],
                         fallback=(96, 118, 92))
    # Warm tips: upper body, several patches inside the plume.
    warm = _robust_color(arr, [(0.50, 0.26), (0.42, 0.36), (0.60, 0.34),
                               (0.50, 0.42), (0.37, 0.46), (0.63, 0.46)],
                         fallback=(206, 182, 96))
    # Soil knot: just above the very base, central.
    soil = _robust_color(arr, [(0.50, 0.88), (0.45, 0.85), (0.55, 0.85)],
                         fallback=(132, 104, 74), half=0.018, reject=140.0)

    # Restore the described CHARACTER if the de-lit samples came out muddy
    # (these are guards toward the photo's own hues, not invented palettes):
    #   cool -> a cooler blue/grey-green (the shaded interior reads bluer than
    #           a single olive median suggests);
    #   warm -> a brighter straw/amber so the sunlit tips glow.
    cool = 0.6 * cool + 0.4 * np.array([86, 122, 104])     # toward blue-green
    warm = 0.6 * warm + 0.4 * np.array([224, 196, 104])    # toward straw-gold
    if warm.mean() < cool.mean() + 25:                      # ensure tip glow
        warm = np.clip(warm * 1.18 + np.array([36, 22, 2]), 0, 255)
    # Soil should read earthy, not green: nudge toward warm-brown.
    if soil[1] >= soil[0]:
        soil = 0.5 * soil + 0.5 * np.array([130, 102, 72])
    return (np.clip(cool, 0, 255),
            np.clip(warm, 0, 255),
            np.clip(soil, 0, 255))


# ==========================================================================
# TEXTURE SYNTHESIS
# ==========================================================================
def _blade_tile(size, cool, warm, sun, warmth, rng):
    """One blade silhouette filling a tile: cool base -> warm tip, fine
    vertical striations, binary (anti-aliased) alpha. Returns RGBA PIL."""
    S = 4
    big = size * S
    ys = np.linspace(0.0, 1.0, big)[:, None]        # 0 top (tip) .. 1 bottom
    # Top = warm tip, bottom = cool base, with an eased gradient so the warm
    # zone dominates the upper half (the luminous halo of the photo).
    g = (1.0 - ys) ** 0.7                            # bias toward warm tip
    col = g[..., None] * warm[None, None, :] \
        + (1.0 - g)[..., None] * cool[None, None, :]
    col = np.broadcast_to(col, (big, big, 3)).copy()
    # Extra glow in the top ~20% (the bright catching tips).
    glow = np.clip((0.2 - ys) / 0.2, 0.0, 1.0)
    col *= (1.0 + 0.18 * glow)[..., None]

    # Fine linear striations along the blade (vary by column).
    xs = np.linspace(0.0, 1.0, big)[None, :]
    freq = int(rng.integers(18, 34))
    stri = (1.0 + 0.10 * np.sin(xs * np.pi * 2 * freq)
            + 0.05 * np.sin(xs * np.pi * 2 * 7 + rng.uniform(0, 6)))
    col *= stri[..., None]

    # Tile-level sun/shade + warmth bias.
    col *= sun
    col[..., 0] *= warmth
    col[..., 2] *= (2.0 - warmth)
    col = np.clip(col, 0, 255).astype(np.uint8)

    # Blade alpha mask (tapering, slight S-curve), drawn at supersample.
    mask = Image.new("L", (big, big), 0)
    d = ImageDraw.Draw(mask)
    n = 26
    c_amp = rng.uniform(-0.10, 0.10)
    c_ph = rng.uniform(0, np.pi)
    wb = rng.uniform(0.62, 0.86)                    # base half-width fraction
    left, right = [], []
    for i in range(n):
        t = i / (n - 1)                             # 0 base .. 1 tip
        yy = big * (0.985 - t * 0.94)               # bottom -> near top
        half_w = wb * 0.5 * big * (1.0 - t) ** 0.72
        cx = big * 0.5 + c_amp * big * np.sin(t * np.pi + c_ph)
        left.append((cx - half_w, yy))
        right.append((cx + half_w, yy))
    d.polygon(left + right[::-1], fill=255)

    rgba = np.dstack([col, np.asarray(mask, dtype=np.uint8)])
    tile = Image.fromarray(rgba, "RGBA").resize((size, size), Image.LANCZOS)
    return tile


def make_foliage_atlas(cool, warm, rng):
    """4x4 atlas (1024) of distinct blade tiles: sunlit tiles brighter/warmer,
    shaded tiles darker/cooler."""
    res = 1024
    tile = res // 4
    atlas = Image.new("RGBA", (res, res), (0, 0, 0, 0))
    for ty in range(4):
        for tx in range(4):
            # Tile ROW encodes sun exposure: row 0 = sunlit (bright/warm),
            # row 3 = shaded (still legible, not muddy). The texturing step
            # maps each blade to a row by its position in the mound.
            base_sun = 0.82 + 0.34 * (1.0 - ty / 3.0)
            sun = float(np.clip(base_sun + rng.uniform(-0.06, 0.06), 0.78, 1.2))
            warmth = float(np.clip(1.05 + 0.12 * (1.0 - ty / 3.0)
                                   + rng.uniform(-0.04, 0.04), 0.95, 1.22))
            t = _blade_tile(tile, cool, warm, sun, warmth, rng)
            atlas.paste(t, (tx * tile, ty * tile), t)
    return atlas


def _mirror_tile(arr):
    """Make a swatch seamlessly tileable by mirror-folding (reflect), blending
    only a thin band at the fold -- never a centre seam blur."""
    a = arr.astype(np.float64)
    a = np.concatenate([a, a[:, ::-1]], axis=1)
    a = np.concatenate([a, a[::-1, :]], axis=0)
    return np.clip(a, 0, 255).astype(np.uint8)


def make_soil_texture(soil, rng):
    """Tileable earthy swatch for the basal knot (>=512), warm-toned with
    fine grain and a few darker specks."""
    n = 256
    base = soil.astype(np.float64)
    grain = rng.normal(0.0, 10.0, (n, n, 1))
    low = rng.normal(0.0, 6.0, (n // 8, n // 8, 1))
    low = np.asarray(Image.fromarray(
        np.clip(low + 128, 0, 255).astype(np.uint8)[..., 0]).resize(
            (n, n), Image.BILINEAR), dtype=np.float64)[..., None] - 128.0
    img = base[None, None, :] + grain + low * 1.5
    # Sparse darker specks (small soil shadows).
    specks = rng.random((n, n)) > 0.985
    img[specks] *= 0.72
    img = np.clip(img, 0, 255).astype(np.uint8)
    tiled = _mirror_tile(img)                       # -> 512x512
    return Image.fromarray(tiled, "RGB")


# ==========================================================================
# UVs, VERTEX TINTS, MATERIAL ASSIGNMENT
# ==========================================================================
def _texture_foliage(mesh, atlas, cool, warm, nseg, rng):
    verts = mesh.vertices
    nv = 2 * (nseg + 1)
    n_blades = len(verts) // nv

    # Per-blade exposure from its TIP position (outer & upper -> sunlit): used
    # to pick the atlas ROW so gold tiles land on exposed blades, cool/shaded
    # tiles in the interior -- matching the photo's lit-halo / dark-core look.
    maxy = max(verts[:, 1].max(), 1e-6)
    tip = verts[(np.arange(n_blades) * nv) + (nv - 1)]   # last vertex of blade
    tip_r = np.sqrt(tip[:, 0] ** 2 + tip[:, 2] ** 2) / CLUMP_RADIUS
    tip_y = tip[:, 1] / maxy
    blade_exp = np.clip(0.5 * np.clip(tip_r, 0, 1) + 0.5 * tip_y, 0.0, 1.0)
    rows = np.clip(((1.0 - blade_exp) * 4.0).astype(int), 0, 3)
    cols_tile = rng.integers(0, 4, size=n_blades)
    uflip = rng.random(n_blades) < 0.5

    i_idx = (np.arange(nv) // 2)
    col = (np.arange(nv) % 2).astype(np.float64)    # 0 = left, 1 = right
    v_local = i_idx / nseg                           # 0 base .. 1 tip

    uv = np.empty((len(verts), 2))
    for b in range(n_blades):
        tx, ty = int(cols_tile[b]), int(rows[b])
        u_in = col * 0.92 + 0.04                     # inset within the tile
        if uflip[b]:
            u_in = 1.0 - u_in
        s = 1.0 - v_local                            # tip -> top (warm)
        s = 0.02 + s * 0.96
        atlas_u = (tx + u_in) / 4.0
        atlas_v = (ty + s) / 4.0
        sl = slice(b * nv, b * nv + nv)
        uv[sl, 0] = atlas_u
        uv[sl, 1] = atlas_v

    # Per-vertex sun/shade tint (multiplies the texture): outer & upper
    # brighter, inner & lower darker and cooler; slight per-blade variation.
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    r = np.sqrt(x * x + z * z)
    rn = np.clip(r / CLUMP_RADIUS, 0.0, 1.0)
    yn = np.clip(y / max(y.max(), 1e-6), 0.0, 1.0)
    exposure = np.clip(0.45 * rn + 0.55 * yn, 0.0, 1.0)

    blade_var = np.repeat(rng.uniform(0.88, 1.04, n_blades), nv)[:len(verts)]
    shade = np.array([0.42, 0.52, 0.55])            # cool, dark interior
    bright = np.array([1.0, 1.0, 1.0])
    tint = shade[None, :] + (bright - shade)[None, :] * exposure[:, None]
    tint *= blade_var[:, None]
    tint = np.clip(tint, 0.0, 1.0)
    colors = np.empty((len(verts), 4), dtype=np.uint8)
    colors[:, :3] = (tint * 255).astype(np.uint8)
    colors[:, 3] = 255

    mat = trimesh.visual.material.PBRMaterial(
        name="foliage",
        baseColorFactor=[255, 255, 255, 255],
        baseColorTexture=atlas,
        metallicFactor=0.0,
        roughnessFactor=0.85,
        alphaMode="MASK",
        alphaCutoff=0.45,
        doubleSided=True)
    tv = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    tv.vertex_attributes["color"] = colors
    mesh.visual = tv


def _texture_crown(mesh, soil_img, rng):
    verts = mesh.vertices
    # Planar UVs from the footprint (small part; planar is plenty).
    x, z = verts[:, 0], verts[:, 2]
    span = max(CROWN_RADIUS * 3.4, 1e-6)
    uv = np.column_stack([(x / span) + 0.5, (z / span) + 0.5]) * 2.0

    # Subtle AO: darker right at the soil line.
    y = verts[:, 1]
    yn = np.clip(y / max(y.max(), 1e-6), 0.0, 1.0)
    ao = 0.55 + 0.45 * yn
    colors = np.empty((len(verts), 4), dtype=np.uint8)
    colors[:, :3] = np.clip(ao[:, None] * 255, 0, 255).astype(np.uint8)
    colors[:, 3] = 255

    mat = trimesh.visual.material.PBRMaterial(
        name="crown",
        baseColorFactor=[255, 255, 255, 255],
        baseColorTexture=soil_img,
        metallicFactor=0.0,
        roughnessFactor=0.95,
        doubleSided=False)
    tv = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    tv.vertex_attributes["color"] = colors
    mesh.visual = tv


def build_textured_scene(seed, density, image_path):
    """Full pipeline: geometry + photo-derived materials -> textured Scene."""
    scene = build_mesh(seed, density)
    nseg = _DENSITY[density][1]

    cool, warm, soil = sample_palette(image_path)
    rng = np.random.default_rng(seed ^ 0x9E3779B9)   # decoupled, deterministic
    atlas = make_foliage_atlas(cool, warm, rng)
    soil_img = make_soil_texture(soil, rng)

    _texture_foliage(scene.geometry["foliage"], atlas, cool, warm, nseg, rng)
    _texture_crown(scene.geometry["crown"], soil_img, rng)
    return scene


# ==========================================================================
# CLI
# ==========================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description="Procedural fountain-grass GLB.")
    p.add_argument("--image", required=True, help="reference photo path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--density", choices=("high", "med", "low"), default="high")
    p.add_argument("--output", required=True, help="output .glb path")
    args = p.parse_args(argv)

    scene = build_textured_scene(args.seed, args.density, args.image)
    glb = scene.export(file_type="glb")
    with open(args.output, "wb") as fh:
        fh.write(glb)
    tris = sum(len(g.faces) for g in scene.geometry.values())
    print(f"wrote {args.output}  ({tris} triangles, density={args.density})")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:                         # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)