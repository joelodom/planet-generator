"""
Standalone procedural asset script: a broad-leaf basal-rosette foliage plant
(banana/canna-like) -- an UPRIGHT fountain of large, broad lance-to-paddle
leaves erupting from a short, fibrous, clustered crown.

It builds geometry, derives tileable materials from a reference photo, applies
per-surface UVs, attaches PBR materials + COLOR_0 vertex tints, and exports a
textured binary GLB.

Surfaces:
    "foliage" -- large curved leaf blades (green, midrib + lateral veins)
    "crown"   -- squat, ridged, fibrous bundle of petiole bases

CLI:
    python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb

Only numpy, trimesh, PIL (Pillow) and the Python stdlib are used.
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter
from trimesh.visual.material import PBRMaterial


# ============================================================================
# MEASURED PROPORTIONS (read by eye off reference.png)
# ============================================================================
# The plant is an UPRIGHT fountain: height ~= width (photo aspect ~1.01).
PLANT_HEIGHT        = 0.85    # m, plausible real potted/garden specimen
HEIGHT_OVER_WIDTH   = 0.98    # height / full canopy width (~1.0 -> near square)
CROWN_HEIGHT_FRAC   = 0.07    # crown height as fraction of total height
CROWN_R             = 0.045   # m, radius of the bundled petiole cluster
LEAF_LEN_MAX        = 0.82    # m, tallest central (near-vertical) leaf
LEAF_LEN_MIN        = 0.55    # m, shortest outer leaf (still fairly upright)
LEAF_BLADE_W_FRAC   = 0.20    # blade half-width as fraction of length (BROAD paddle)
LEAF_STALK_FRAC     = 0.13    # fraction of leaf that is the narrow petiole base
STALK_HALF_W        = 0.009   # m, half-width of the narrow petiole

# Leaf-shape character knobs
CUP_DEPTH           = 0.26    # cross-section channel depth (frac of local width)
WAVE_AMP            = 0.06    # wavy-margin amplitude (fraction of local width)
WAVE_CYCLES         = 2.4     # number of margin undulations along a leaf
TWIST_MAX           = 0.30    # rad, max gentle twist about the leaf's long axis

# Posture: how upright / how much the leaves arch (keeps the fountain tall).
ELEV_BASE_DEG       = 55.0    # outer-leaf base elevation (upright-ish)
ELEV_SPAN_DEG       = 30.0    # extra elevation for central leaves
ARCH_OUTER          = 0.40    # outer leaves arch the most
ARCH_CENTRAL        = 0.20    # central leaves stay near-vertical

# Crown character
CROWN_RIDGES        = 9       # fibrous vertical ridges around the bundle
CROWN_RIDGE_AMP     = 0.14    # ridge relief as fraction of crown radius

# Texture sizes
ATLAS_PX            = 1024    # foliage atlas (4x4 leaf tiles)
TILE_PX             = ATLAS_PX // 4
CROWN_PX            = 512
SS                  = 4       # supersample factor for silhouette drawing


# ============================================================================
# Density presets (counts chosen BEFORE building)
# ============================================================================
_DENSITY = {
    "high": dict(n_leaves=18, n_len=28, n_wid=10, crown_theta=24, crown_h=8),
    "med":  dict(n_leaves=14, n_len=18, n_wid=7,  crown_theta=16, crown_h=6),
    "low":  dict(n_leaves=10, n_len=10, n_wid=4,  crown_theta=10, crown_h=4),
}


# ============================================================================
# Mesh helpers
# ============================================================================
def _grid_faces(n_rows, n_cols):
    """Triangulate an (n_rows x n_cols) vertex grid; index = i*n_cols + j."""
    i = np.arange(n_rows - 1)[:, None]
    j = np.arange(n_cols - 1)[None, :]
    a = (i * n_cols + j).ravel()
    b = (i * n_cols + (j + 1)).ravel()
    c = ((i + 1) * n_cols + j).ravel()
    d = ((i + 1) * n_cols + (j + 1)).ravel()
    f1 = np.stack([a, c, d], axis=1)
    f2 = np.stack([a, d, b], axis=1)
    return np.concatenate([f1, f2], axis=0)


def _half_width(s, blade_half_w):
    """Leaf half-width profile along normalized length s in [0,1].

    Broad, rounded paddle: low exponent keeps the blade wide over most of its
    length, narrowing to a soft point at the tip and a slim petiole at base.
    """
    blade = blade_half_w * np.sin(np.pi * s) ** 0.45         # broad paddle
    petiole = STALK_HALF_W * np.clip(1.0 - s / LEAF_STALK_FRAC, 0.0, 1.0)
    return np.maximum(blade, petiole)


def _build_leaf(n_len, n_wid, length, blade_half_w, elev0, arch, twist,
                wave_phase, azimuth, base_y, base_r, tile, flip_u, flip_v, bf):
    """Build one curved, cupped, wavy leaf grid with UVs + COLOR_0 tints."""
    R = n_len + 1
    Ck = n_wid + 1
    s = np.linspace(0.0, 1.0, R)

    alpha = elev0 * (1.0 - arch * s)
    alpha = np.maximum(alpha, np.radians(-8.0))             # only a slight droop
    ds = length / n_len
    dr = np.cos(alpha) * ds
    dy = np.sin(alpha) * ds
    r = base_r + np.cumsum(np.concatenate([[0.0], dr[:-1]]))
    y = base_y + np.cumsum(np.concatenate([[0.0], dy[:-1]]))

    P = np.stack([np.zeros_like(r), y, r], axis=1)

    sa, ca = np.sin(alpha), np.cos(alpha)
    N0 = np.stack([np.zeros_like(sa), ca, -sa], axis=1)
    Xax = np.array([1.0, 0.0, 0.0])

    tw = twist * s
    C = np.cos(tw)[:, None] * Xax + np.sin(tw)[:, None] * N0
    N = -np.sin(tw)[:, None] * Xax + np.cos(tw)[:, None] * N0

    w = _half_width(s, blade_half_w)
    u = np.linspace(-1.0, 1.0, Ck)

    wave = 1.0 + WAVE_AMP * np.abs(u)[None, :] * np.sin(
        2.0 * np.pi * WAVE_CYCLES * s[:, None] + wave_phase)
    w_eff = w[:, None] * wave
    uu = u[None, :] * np.ones((R, 1))
    cup = CUP_DEPTH * w[:, None] * (uu ** 2)

    pos = (P[:, None, :]
           + (uu * w_eff)[..., None] * C[:, None, :]
           + cup[..., None] * N[:, None, :])
    pos = pos.reshape(-1, 3)

    cphi, sphi = np.cos(azimuth), np.sin(azimuth)
    x, z = pos[:, 0].copy(), pos[:, 2].copy()
    pos[:, 0] = x * cphi + z * sphi
    pos[:, 2] = -x * sphi + z * cphi

    faces = _grid_faces(R, Ck)

    # ---- UVs: map into one 0.25-size atlas tile, with a small inset --------
    col_t, row_t = tile % 4, tile // 4
    a_w = (np.arange(Ck) / (Ck - 1.0))[None, :] * np.ones((R, 1))
    a_l = s[:, None] * np.ones((1, Ck))
    if flip_u:
        a_w = 1.0 - a_w
    if flip_v:
        a_l = 1.0 - a_l
    m = 0.03
    a_w = m + a_w * (1.0 - 2.0 * m)
    a_l = m + a_l * (1.0 - 2.0 * m)
    U = (col_t + a_w) * 0.25
    V = (row_t + a_l) * 0.25
    uv = np.stack([U.ravel(), V.ravel()], axis=1)

    # ---- COLOR_0 tint: near-neutral sun/shade (tip & sunlit brighter) ------
    val = np.clip(0.62 + 0.45 * s + 0.25 * (bf - 0.85), 0.5, 1.12)
    val = val[:, None] * np.ones((1, Ck))
    rgb = np.empty((R, Ck, 3))
    rgb[..., 0] = np.clip(val * 230 + s[:, None] * 12, 0, 255)
    rgb[..., 1] = np.clip(val * 240 + s[:, None] * 6, 0, 255)
    rgb[..., 2] = np.clip(val * 214 - s[:, None] * 8, 0, 255)
    rgba = np.concatenate(
        [rgb.reshape(-1, 3), np.full((R * Ck, 1), 255.0)], axis=1)

    return pos, faces, rgba.astype(np.uint8), uv


def _build_crown(rng, n_theta, n_h):
    """Squat, ridged, fibrous petiole bundle with cylindrical UVs + tints."""
    crown_h = PLANT_HEIGHT * CROWN_HEIGHT_FRAC
    hs = np.linspace(0.0, 1.0, n_h)
    yv = hs * crown_h
    rprof = CROWN_R * np.interp(hs, [0.0, 0.5, 1.0], [0.6, 1.0, 0.82])

    nc = n_theta + 1                                        # duplicate seam column
    theta = np.linspace(0.0, 2.0 * np.pi, nc)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    ridge = 1.0 + CROWN_RIDGE_AMP * np.sin(CROWN_RIDGES * theta + phase)
    rr = rprof[:, None] * ridge[None, :]

    x = rr * np.cos(theta)[None, :]
    z = rr * np.sin(theta)[None, :]
    yg = yv[:, None] * np.ones((1, nc))
    verts = np.stack([x, yg, z], axis=2).reshape(-1, 3)

    U = (theta / (2.0 * np.pi))[None, :] * np.ones((n_h, 1))
    V = hs[:, None] * np.ones((1, nc))
    uv = np.stack([U.ravel(), V.ravel()], axis=1)

    faces = list(_grid_faces(n_h, nc))

    verts = list(verts)
    uv = list(uv)
    bottom_c = len(verts); verts.append([0.0, 0.0, 0.0]);     uv.append([0.5, 0.0])
    top_c = len(verts);    verts.append([0.0, crown_h, 0.0]); uv.append([0.5, 1.0])
    top_row = (n_h - 1) * nc
    for j in range(n_theta):
        faces.append([bottom_c, j, j + 1])
        faces.append([top_c, top_row + j + 1, top_row + j])

    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    uv = np.asarray(uv, dtype=float)

    hfull = np.concatenate([V.ravel(), [0.0, 1.0]])
    f = np.clip(0.55 + 0.5 * hfull, 0.5, 1.1)
    rgba = np.stack([f * 235, f * 225, f * 210, np.full_like(f, 255.0)], axis=1)
    return verts, faces, rgba.astype(np.uint8), uv


# ============================================================================
# Geometry entry point
# ============================================================================
def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    if density not in _DENSITY:
        density = "high"
    cfg = _DENSITY[density]
    rng = np.random.default_rng(seed)

    n_leaves = cfg["n_leaves"]
    crown_h = PLANT_HEIGHT * CROWN_HEIGHT_FRAC

    base_az = (np.arange(n_leaves) * (2.0 * np.pi * 0.61803)) % (2.0 * np.pi)
    az_jit = rng.uniform(-0.30, 0.30, n_leaves)

    fv, ff, fc, fuv = [], [], [], []
    voff = 0
    for k in range(n_leaves):
        u = rng.random()                                   # rank: 1 -> central/upright
        length = (LEAF_LEN_MIN + (LEAF_LEN_MAX - LEAF_LEN_MIN) * u) * rng.uniform(0.94, 1.05)
        blade_half_w = length * LEAF_BLADE_W_FRAC * rng.uniform(0.9, 1.12)
        # Upright fountain: high base elevation, modest arch -> tall, not splayed.
        elev0 = np.radians(ELEV_BASE_DEG + ELEV_SPAN_DEG * u) + np.radians(rng.uniform(-4.0, 4.0))
        arch = (ARCH_OUTER - (ARCH_OUTER - ARCH_CENTRAL) * u) * rng.uniform(0.9, 1.1)
        twist = rng.uniform(-1.0, 1.0) * TWIST_MAX
        wave_phase = rng.uniform(0.0, 2.0 * np.pi)
        azimuth = base_az[k] + az_jit[k]
        base_r = CROWN_R * rng.uniform(0.25, 0.55)
        base_y = crown_h * rng.uniform(0.55, 0.95)

        tile = int(rng.integers(16))
        flip_u = bool(rng.random() < 0.5)
        flip_v = bool(rng.random() < 0.5)
        bf = float(np.clip(0.8 + 0.35 * u + rng.uniform(-0.05, 0.05), 0.6, 1.2))

        v, f, c, uvv = _build_leaf(cfg["n_len"], cfg["n_wid"], length, blade_half_w,
                                   elev0, arch, twist, wave_phase, azimuth,
                                   base_y, base_r, tile, flip_u, flip_v, bf)
        fv.append(v); ff.append(f + voff); fc.append(c); fuv.append(uvv)
        voff += len(v)

    fol_v = np.concatenate(fv, axis=0)
    fol_f = np.concatenate(ff, axis=0)
    fol_c = np.concatenate(fc, axis=0)
    fol_uv = np.concatenate(fuv, axis=0)

    cr_v, cr_f, cr_c, cr_uv = _build_crown(rng, cfg["crown_theta"], cfg["crown_h"])

    all_v = np.concatenate([fol_v, cr_v], axis=0)
    min_y = all_v[:, 1].min()
    cx = 0.5 * (all_v[:, 0].min() + all_v[:, 0].max())
    cz = 0.5 * (all_v[:, 2].min() + all_v[:, 2].max())
    shift = np.array([cx, min_y, cz])
    fol_v = fol_v - shift
    cr_v = cr_v - shift

    foliage = trimesh.Trimesh(vertices=fol_v, faces=fol_f, process=False)
    foliage.metadata["uv"] = fol_uv
    foliage.metadata["vcolor"] = fol_c

    crown = trimesh.Trimesh(vertices=cr_v, faces=cr_f, process=False)
    crown.metadata["uv"] = cr_uv
    crown.metadata["vcolor"] = cr_c

    scene = trimesh.Scene()
    scene.add_geometry(foliage, geom_name="foliage")
    scene.add_geometry(crown, geom_name="crown")
    return scene


# ============================================================================
# Photo sampling
# ============================================================================
def _patch_median(arr, cx, cy, frac):
    h, w, _ = arr.shape
    pw = max(2, int(frac * w)); ph = max(2, int(frac * h))
    x0 = int(np.clip(cx * w - pw / 2, 0, w - pw))
    y0 = int(np.clip(cy * h - ph / 2, 0, h - ph))
    patch = arr[y0:y0 + ph, x0:x0 + pw].reshape(-1, 3)
    return np.median(patch, axis=0)


def _sample_foliage(arr):
    coords = [(0.50, 0.32), (0.38, 0.46), (0.62, 0.42), (0.50, 0.55),
              (0.42, 0.28), (0.60, 0.60), (0.50, 0.45), (0.33, 0.58)]
    kept = []
    for cx, cy in coords:
        c = _patch_median(arr, cx, cy, 0.04)
        r, g, b = c
        if g > r and g > b and (c.max() - c.min()) > 18:
            kept.append(c)
    if not kept:
        return np.array([54.0, 112.0, 42.0])
    return np.median(np.array(kept), axis=0)


def _sample_crown(arr):
    coords = [(0.50, 0.90), (0.45, 0.88), (0.55, 0.90), (0.50, 0.86)]
    kept = []
    for cx, cy in coords:
        c = _patch_median(arr, cx, cy, 0.03)
        r, g, b = c
        if r >= g >= b and (c.max() - c.min()) > 12 and r > 50:
            kept.append(c)
    if not kept:
        return np.array([120.0, 82.0, 46.0])
    return np.median(np.array(kept), axis=0)


# ============================================================================
# Texture synthesis
# ============================================================================
def _delight(a):
    a = a.astype(float)
    lum = a @ np.array([0.299, 0.587, 0.114])
    blur = np.asarray(
        Image.fromarray(lum.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=max(a.shape[0], a.shape[1]) / 6.0))
    ).astype(float) + 1e-3
    gain = np.clip(lum.mean() / blur, 0.6, 1.6)
    return np.clip(a * gain[..., None], 0, 255)


def _mirror_tile(a):
    top = np.concatenate([a, a[:, ::-1]], axis=1)
    return np.concatenate([top, top[::-1]], axis=0)


def _build_foliage_atlas(base_green, rng):
    """4x4 atlas of full-bleed leaf-blade tiles with SUBTLE midrib + veins."""
    base = np.asarray(base_green, dtype=float)
    deep = np.clip(base * np.array([0.58, 0.62, 0.56]), 0, 255)
    light = np.clip(base * 1.12 + np.array([18, 22, -12]), 0, 255)   # chartreuse tip
    # Veins are only slightly lighter than the blade -- not bright lime.
    vein = tuple(int(v) for v in np.clip(base * 1.16 + np.array([12, 16, 0]), 0, 255))
    midrib = tuple(int(v) for v in np.clip(base * 1.24 + np.array([20, 26, -2]), 0, 255))

    atlas = Image.new("RGBA", (ATLAS_PX, ATLAS_PX), (0, 0, 0, 0))
    T = TILE_PX * SS

    for ti in range(16):
        col_t, row_t = ti % 4, ti // 4
        sun = float(rng.uniform(0.82, 1.14))
        warm = float(rng.uniform(-0.04, 0.07))

        yy = np.linspace(0.0, 1.0, T)[:, None]             # 0 top(tip) .. 1 bottom(base)
        grad = light[None, :] * (1.0 - yy) + deep[None, :] * yy
        grad = grad * sun
        grad[:, 0] *= (1.0 + warm)
        grad[:, 2] *= (1.0 - warm)
        grad = np.clip(grad, 0, 255)
        color = np.repeat(grad[:, None, :], T, axis=1)

        # broad blade silhouette (full width, soft pointed tip), gentle margins
        ph = rng.uniform(0, 2 * np.pi)
        edge_ys = np.linspace(0.12, 1.0, 28)
        left = [(0.5 * T, 0.0)]
        right = []
        for ey in edge_ys:
            wob = 0.015 * np.sin(5.0 * ey + ph)
            left.append(((0.06 + wob) * T, ey * T))
            right.append(((0.94 + wob) * T, ey * T))
        poly = left + right[::-1]
        mask_img = Image.new("L", (T, T), 0)
        ImageDraw.Draw(mask_img).polygon(poly, fill=255)
        mask = np.asarray(mask_img).astype(float) / 255.0

        cimg = Image.fromarray(color.astype(np.uint8), "RGB")
        d = ImageDraw.Draw(cimg)
        # subtle midrib
        d.line([(0.5 * T, 0.0), (0.5 * T, T)], fill=midrib, width=max(2, int(0.010 * T)))
        # sparse, thin pinnate veins
        step = int(0.11 * T)
        off = int(0.045 * T)
        vw = max(1, int(0.0035 * T))
        for y in range(int(0.14 * T), int(0.96 * T), step):
            d.line([(0.5 * T, y), (0.12 * T, y - off)], fill=vein, width=vw)
            d.line([(0.5 * T, y), (0.88 * T, y - off)], fill=vein, width=vw)

        # ease vein contrast by blending the drawn lines back toward the blade
        cnp = 0.6 * np.asarray(cimg).astype(float) + 0.4 * color
        rgb = (cnp * mask[..., None]).astype(np.uint8)
        alpha = (mask * 255).astype(np.uint8)

        rgb_s = Image.fromarray(rgb, "RGB").resize((TILE_PX, TILE_PX), Image.LANCZOS)
        a_s = Image.fromarray(alpha, "L").resize((TILE_PX, TILE_PX), Image.LANCZOS)
        tile = Image.merge("RGBA", (*rgb_s.split(), a_s))
        atlas.paste(tile, (col_t * TILE_PX, row_t * TILE_PX))

    return atlas


def _normal_from_albedo(rgb_arr, strength=2.0):
    lum = (rgb_arr @ np.array([0.299, 0.587, 0.114])) / 255.0
    height = 1.0 - lum
    gy, gx = np.gradient(height)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(height)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    out = np.stack([(nx / norm * 0.5 + 0.5),
                    (ny / norm * 0.5 + 0.5),
                    (nz / norm * 0.5 + 0.5)], axis=2)
    return Image.fromarray((out * 255).astype(np.uint8), "RGB")


def _build_crown_texture(img, arr, crown_col, rng):
    h, w, _ = arr.shape
    pw = max(8, int(0.16 * w)); ph = max(8, int(0.16 * h))
    x0 = int(np.clip(0.5 * w - pw / 2, 0, w - pw))
    y0 = int(np.clip(0.88 * h - ph / 2, 0, h - ph))
    patch = arr[y0:y0 + ph, x0:x0 + pw].astype(float)

    patch = _delight(patch)
    pmean = patch.reshape(-1, 3).mean(axis=0) + 1e-3
    retint = np.clip(np.asarray(crown_col) / pmean, 0.4, 2.2)
    patch = np.clip(patch * retint[None, None, :], 0, 255)

    tile = _mirror_tile(patch)
    base = np.asarray(
        Image.fromarray(tile.astype(np.uint8)).resize((CROWN_PX, CROWN_PX), Image.LANCZOS)
    ).astype(float)

    cols = rng.uniform(0.82, 1.16, CROWN_PX)
    k = np.ones(9) / 9.0
    cols = np.convolve(np.concatenate([cols[-9:], cols, cols[:9]]), k, mode="same")[9:-9]
    fib = cols[None, :, None]
    for _ in range(7):
        gc = int(rng.integers(0, CROWN_PX))
        gw = int(rng.integers(2, 6))
        base[:, max(0, gc - gw):gc + gw, :] *= 0.72
    base = np.clip(base * fib, 0, 255)

    rgb = base.astype(np.uint8)
    crown_img = Image.fromarray(rgb, "RGB")
    normal_img = _normal_from_albedo(base, strength=2.5)
    return crown_img, normal_img


# ============================================================================
# Apply materials + UVs to the scene
# ============================================================================
def texture_scene(scene, image_path, seed):
    rng = np.random.default_rng(seed + 911)
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img).astype(float)

    fol_green = _sample_foliage(arr)
    crown_col = _sample_crown(arr)

    atlas = _build_foliage_atlas(fol_green, rng)
    crown_tex, crown_norm = _build_crown_texture(img, arr, crown_col, rng)

    for name, geom in scene.geometry.items():
        uv = geom.metadata["uv"]
        vcolor = geom.metadata["vcolor"]
        if name == "foliage":
            mat = PBRMaterial(
                name="foliage",
                baseColorTexture=atlas,
                baseColorFactor=[255, 255, 255, 255],
                metallicFactor=0.0,
                roughnessFactor=0.55,           # glossy-to-semi-matte sheen
                alphaMode="MASK",
                alphaCutoff=0.45,
                doubleSided=True,
            )
        else:
            mat = PBRMaterial(
                name="crown",
                baseColorTexture=crown_tex,
                normalTexture=crown_norm,
                baseColorFactor=[255, 255, 255, 255],
                metallicFactor=0.0,
                roughnessFactor=0.9,
                doubleSided=True,
            )
        vis = trimesh.visual.TextureVisuals(uv=uv, material=mat)
        vis.vertex_attributes["color"] = vcolor.astype(np.uint8)
        geom.visual = vis

    return scene


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Procedural broad-leaf rosette plant -> textured GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        scene = build_mesh(args.seed, args.density)
        scene = texture_scene(scene, args.image, args.seed)
        glb = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(glb)
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1
    print("wrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())