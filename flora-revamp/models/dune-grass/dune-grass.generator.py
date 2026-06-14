#!/usr/bin/env python3
"""
Procedural ornamental tussock-grass: geometry + photo-derived materials,
textured GLB export.

A tight basal crown from which dozens of fine, thread-thin blades launch
nearly vertically and arch outward into a broad, feathery vase-to-fountain.
Light cool sage/blue-green body warming to bright straw/gold at the wispy
tips; a low, dark congested crown hidden among the blade bases.

  python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb

Only numpy / trimesh / PIL / stdlib. +Y up, base at y=0, meters.
Deterministic given --seed.
"""

import argparse
import sys
import traceback

import numpy as np
import trimesh
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial
from PIL import Image, ImageDraw, ImageFilter


# ==========================================================================
#  GEOMETRY  (build_mesh -- +Y up, base at y=0, meters)
# ==========================================================================
OVERALL_HEIGHT = 0.62          # m, plausible real-world ornamental grass tuft
# Photo front-view aspect ~0.88 -> width ~0.88*height; broad open fountain.
HEIGHT_OVER_WIDTH = 1.14       # h/w of the whole clump (widened to match photo)
CROWN_HALF_WIDTH = (OVERALL_HEIGHT / HEIGHT_OVER_WIDTH) / 2.0  # ~0.272 m reach

BASE_TUFT_RADIUS = 0.042       # m, radius over which blades emerge
MOUND_RADIUS = 0.030           # m, small crown mound (kept inside the blades)
MOUND_HEIGHT = 0.018           # m, low/flat so it never reads as a "bulb"
BLADE_BASE_WIDTH = 0.0035      # m, ~3.5 mm ribbon at base, tapers to a point

# Foliage envelope guard: nothing protrudes past the shell by > ~3%.
ENVELOPE_MARGIN = 1.03

# Color character (RGB 0..1) -- cool sage body warming to dry straw tips.
COL_GREEN = np.array([0.46, 0.55, 0.42])   # light blue/grey-green body
COL_STRAW = np.array([0.84, 0.76, 0.47])   # sun-bleached straw/gold tips
COL_BASE = np.array([0.22, 0.25, 0.16])    # dark, dense crown


# Per-density element counts -- chosen BEFORE building (generate at target).
_DENSITY = {
    "high": dict(n_blades=900, n_stations=12, base_sub=2),  # ~18.9k tris
    "med":  dict(n_blades=380, n_stations=9,  base_sub=1),  # ~5.8k tris
    "low":  dict(n_blades=110, n_stations=6,  base_sub=0),  # ~1.0k tris
}


def _norm(v):
    """Row-wise normalize an (N,3) array, safe against zero-length rows."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n < 1e-9, 1.0, n)
    return v / n


def _build_blade(rng, n_stations):
    """
    Build one grass blade as a thin tapered ribbon (a strip of quads that
    collapses to a single point at the tip).

    Returns (verts (V,3), faces (F,3) local indices, colors (V,3) float).
    """
    S = n_stations
    t = np.linspace(0.0, 1.0, S)
    up = np.array([0.0, 1.0, 0.0])

    # --- per-blade character parameters ---------------------------------
    # splay biased outward (**0.75) -> fuller, broader fountain like the photo
    s = float(rng.random()) ** 0.75         # splay: 0 inner/vertical, 1 outer/arching
    lj = float(np.exp(rng.normal(0.0, 0.13)))  # length jitter (log-normal)
    rj = float(np.exp(rng.normal(0.0, 0.18)))  # radial reach jitter
    r0 = float(rng.uniform(0.0, BASE_TUFT_RADIUS))  # base radius in the tuft
    phi0 = float(rng.uniform(0.0, 2.0 * np.pi))     # azimuth of emergence

    H = OVERALL_HEIGHT
    # radial reach grows strongly with splay; clamp inside the envelope shell.
    r_end = r0 + (0.12 + 0.95 * s) * CROWN_HALF_WIDTH * rj
    r_end = min(r_end, CROWN_HALF_WIDTH * ENVELOPE_MARGIN)
    # control point sits low in radius -> a near-vertical launch off the crown.
    r_ctrl = r0 + (0.20 + 0.10 * s) * (r_end - r0)
    # heights: tall launch for all; outer blades end lower (drooping tips).
    y_ctrl = min(H * (0.95 - 0.12 * s) * lj, H * ENVELOPE_MARGIN)
    y_end = min(H * (0.92 - 0.62 * s) * lj, H * ENVELOPE_MARGIN)

    # gentle out-of-plane bow + slight azimuthal curl (more for outer blades)
    bow_amp = float(rng.uniform(-1.0, 1.0)) * 0.050 * (0.25 + 0.75 * s)
    dphi = float(rng.uniform(-1.0, 1.0)) * 0.35 * s

    # --- centerline: quadratic Bezier in (radial, vertical) -------------
    omt = 1.0 - t
    r_t = omt * omt * r0 + 2.0 * omt * t * r_ctrl + t * t * r_end
    y_t = 2.0 * omt * t * y_ctrl + t * t * y_end          # P0 height = 0
    bow_t = bow_amp * np.sin(np.pi * t)

    phi = phi0 + dphi * t
    cosp = np.cos(phi)
    sinp = np.sin(phi)
    zeros = np.zeros_like(cosp)
    radd = np.stack([cosp, zeros, sinp], axis=1)          # radial dir (S,3)
    tangd = np.stack([-sinp, zeros, cosp], axis=1)         # tangential dir (S,3)

    C = radd * r_t[:, None] + up * y_t[:, None] + tangd * bow_t[:, None]

    # --- ribbon frame: width direction perpendicular to the tangent -----
    T = _norm(np.gradient(C, axis=0))
    alpha = float(rng.uniform(0.0, 2.0 * np.pi))
    # which way the thin blade faces (horizontal), varied per blade
    d0 = np.cos(alpha) * tangd[0] + np.sin(alpha) * radd[0]
    raw = d0[None, :] - (T @ d0)[:, None] * T
    rn = np.linalg.norm(raw, axis=-1, keepdims=True)
    alt = np.cross(T, up)
    wdir = _norm(np.where(rn < 1e-6, alt, raw))

    # --- widths taper to a point ----------------------------------------
    wmul = float(np.exp(rng.normal(0.0, 0.18)))
    m = S - 1                                              # number of rings
    widths = BLADE_BASE_WIDTH * wmul * (1.0 - t[:m]) ** 0.55
    half = 0.5 * widths[:, None]

    left = C[:m] + half * wdir[:m]
    right = C[:m] - half * wdir[:m]
    tip = C[-1]

    ring = np.empty((2 * m, 3))
    ring[0::2] = left
    ring[1::2] = right
    verts = np.vstack([ring, tip[None, :]])
    tip_idx = 2 * m

    # --- colors: sage body warming to straw tips ------------------------
    straw_bias = float(rng.uniform(-0.05, 0.30))
    bright = float(rng.uniform(0.90, 1.10))
    hj = rng.normal(0.0, 0.025, 3)
    w = np.clip(0.12 + 0.70 * t ** 0.9 + (0.45 * s + straw_bias) * t, 0.0, 1.0)
    cols_st = (COL_GREEN * (1.0 - w)[:, None] + COL_STRAW * w[:, None]) * bright + hj
    cols_st = np.clip(cols_st, 0.0, 1.0)
    ring_cols = np.empty((2 * m, 3))
    ring_cols[0::2] = cols_st[:m]
    ring_cols[1::2] = cols_st[:m]
    colors = np.vstack([ring_cols, cols_st[-1][None, :]])

    # --- faces ----------------------------------------------------------
    faces = []
    for i in range(m - 1):
        Li, Ri = 2 * i, 2 * i + 1
        Lj, Rj = 2 * (i + 1), 2 * (i + 1) + 1
        faces.append([Li, Ri, Rj])
        faces.append([Li, Rj, Lj])
    faces.append([2 * (m - 1), 2 * (m - 1) + 1, tip_idx])  # collapse to tip
    faces = np.array(faces, dtype=np.int64)

    return verts, faces, colors


def _build_base(rng, base_sub):
    """A small, low, dark ellipsoid mound for the congested crown."""
    mound = trimesh.creation.icosphere(subdivisions=base_sub, radius=1.0)
    rx = rz = MOUND_RADIUS
    ry = MOUND_HEIGHT * 0.5
    v = mound.vertices * np.array([rx, ry, rz])
    v[:, 1] += ry                      # sit bottom exactly on y=0
    mound.vertices = v
    cols = np.clip(COL_BASE + rng.normal(0.0, 0.02, (len(v), 3)), 0.0, 1.0)
    rgba = np.empty((len(v), 4), dtype=np.uint8)
    rgba[:, :3] = (cols * 255).astype(np.uint8)
    rgba[:, 3] = 255
    mound.visual.vertex_colors = rgba
    mound.fix_normals()
    return mound


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    """Procedurally generate a tussock-grass instance as a trimesh.Scene."""
    rng = np.random.default_rng(seed)
    cfg = _DENSITY.get(density, _DENSITY["high"])
    n_blades = cfg["n_blades"]
    n_stations = max(4, cfg["n_stations"])

    all_v = []
    all_f = []
    all_c = []
    voff = 0
    for _ in range(n_blades):
        v, f, c = _build_blade(rng, n_stations)
        all_v.append(v)
        all_f.append(f + voff)
        all_c.append(c)
        voff += len(v)

    V = np.vstack(all_v)
    F = np.vstack(all_f)
    C = np.vstack(all_c)

    # center in X/Z (mean of all geometry) so the clump sits over the origin
    cx, cz = V[:, 0].mean(), V[:, 2].mean()
    V[:, 0] -= cx
    V[:, 2] -= cz

    rgba = np.empty((len(V), 4), dtype=np.uint8)
    rgba[:, :3] = (np.clip(C, 0.0, 1.0) * 255).astype(np.uint8)
    rgba[:, 3] = 255

    blades = trimesh.Trimesh(vertices=V, faces=F, vertex_colors=rgba,
                             process=False)

    base = _build_base(rng, cfg["base_sub"])
    base.vertices[:, 0] -= cx
    base.vertices[:, 2] -= cz

    scene = trimesh.Scene()
    scene.add_geometry(blades, geom_name="blades")
    scene.add_geometry(base, geom_name="base")
    return scene


# ==========================================================================
#  PHOTO SAMPLING  (de-light, background rejection, plant-color medians)
# ==========================================================================
def _load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float64) / 255.0


def _delight(img):
    """Divide out a heavily blurred luminance; gain clamped to [0.6, 1.6]."""
    H, W, _ = img.shape
    lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    limg = Image.fromarray((np.clip(lum, 0, 1) * 255).astype(np.uint8))
    rad = max(8, int(min(H, W) / 8))
    bl = np.asarray(limg.filter(ImageFilter.GaussianBlur(rad)))
    bl = np.clip(bl.astype(np.float64) / 255.0, 1e-3, None)
    gain = np.clip(lum.mean() / bl, 0.6, 1.6)
    return np.clip(img * gain[..., None], 0.0, 1.0)


def _bg_estimate(img):
    """Median color of the four image corners (assumed background)."""
    H, W, _ = img.shape
    cs = max(4, int(min(H, W) * 0.06))
    corners = [img[:cs, :cs], img[:cs, -cs:], img[-cs:, :cs], img[-cs:, -cs:]]
    meds = [np.median(c.reshape(-1, 3), axis=0) for c in corners]
    return np.median(np.array(meds), axis=0)


def _gather_color(img, bbox, bg, default):
    """
    Median of small patches inside `bbox` (fractions of H,W) that look like
    the plant: far enough from the background color, not specular-bright.
    Falls back to `default` only if nothing plant-like is found.
    """
    H, W, _ = img.shape
    y0, y1 = int(H * bbox[0]), int(H * bbox[1])
    x0, x1 = int(W * bbox[2]), int(W * bbox[3])
    ps = max(3, int(min(H, W) * 0.012))
    step = max(2, ps // 2)
    keep = []
    for thr in (0.08, 0.05, 0.03):
        keep = []
        yy = y0
        while yy < y1 - ps:
            xx = x0
            while xx < x1 - ps:
                patch = img[yy:yy + ps, xx:xx + ps].reshape(-1, 3)
                med = np.median(patch, axis=0)
                d = np.linalg.norm(med - bg)
                if d > thr and med.max() < 0.92:
                    keep.append(med)
                xx += step
            yy += step
        if len(keep) >= 5:
            return np.median(np.array(keep), axis=0)
    if keep:
        return np.median(np.array(keep), axis=0)
    return np.array(default, dtype=float)


# ==========================================================================
#  TEXTURE SYNTHESIS
# ==========================================================================
ATLAS_RES = 1024     # foliage atlas (4x4 tiles of distinct blade clusters)
GRID = 4
TILE = ATLAS_RES // GRID
SS = 4               # supersample for crisp, near-binary silhouette alpha
BASE_RES = 512       # base-mound swatch


def _build_blade_atlas(green, straw, rng):
    """
    1024x1024 4x4 atlas. Each tile: a centered tapered blade silhouette
    (PIL polygon, supersampled then LANCZOS-downscaled to near-binary alpha)
    over a vertical green(base)->straw(tip) gradient with fine longitudinal
    striations. Top rows sunlit (brighter/warmer), bottom rows shaded.
    Brightened/warmed so the body reads light like the photo.
    """
    green = np.clip(np.asarray(green, float), 0, 1)
    straw = np.clip(np.asarray(straw, float), 0, 1)

    color = np.zeros((ATLAS_RES, ATLAS_RES, 3))
    yy, xx = np.mgrid[0:TILE, 0:TILE]
    gy = yy / (TILE - 1.0)            # 0 at top (tip) -> 1 at bottom (base)
    xn = xx / (TILE - 1.0)

    for r in range(GRID):
        sun = 1.0 - r / (GRID - 1.0)            # row 0 sunlit, last shaded
        bf = 0.92 + 0.30 * sun                  # lighter overall
        warm = 0.07 * sun
        for c in range(GRID):
            hj = rng.normal(0.0, 0.02, 3)
            body = np.clip(green * bf +
                           np.array([warm * 0.4, warm * 0.2, -warm * 0.2]) + hj, 0, 1)
            tip = np.clip(straw * (0.92 + 0.30 * sun) +
                          np.array([warm, warm * 0.5, 0.0]) + hj, 0, 1)
            grad = body[None, None, :] * gy[..., None] + \
                tip[None, None, :] * (1.0 - gy)[..., None]
            f = rng.uniform(6.0, 12.0)
            ph = rng.uniform(0.0, 2.0 * np.pi)
            stri = 1.0 + 0.10 * np.sin(2.0 * np.pi * f * xn + ph)
            stri = stri + rng.normal(0.0, 0.02, (TILE, TILE))
            til = np.clip(grad * stri[..., None], 0, 1)
            color[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = til

    # silhouette alpha (supersampled polygons -> LANCZOS for AA-only edges)
    big = Image.new("L", (ATLAS_RES * SS, ATLAS_RES * SS), 0)
    draw = ImageDraw.Draw(big)
    for r in range(GRID):
        for c in range(GRID):
            cx = (c + 0.5) * TILE * SS
            ybot = (r + 0.96) * TILE * SS
            ytop = (r + 0.03) * TILE * SS
            lean = rng.uniform(-0.06, 0.06) * TILE * SS
            halfw = 0.17 * TILE * SS
            n = 12
            pts = []
            for i in range(n):                          # up the left edge
                tt = i / (n - 1.0)
                wv = max(1.0, halfw * (1.0 - tt) ** 0.6)
                pts.append((cx - wv + lean * tt, ybot + (ytop - ybot) * tt))
            for i in range(n - 1, -1, -1):              # down the right edge
                tt = i / (n - 1.0)
                wv = max(1.0, halfw * (1.0 - tt) ** 0.6)
                pts.append((cx + wv + lean * tt, ybot + (ytop - ybot) * tt))
            draw.polygon(pts, fill=255)
    alpha = np.asarray(big.resize((ATLAS_RES, ATLAS_RES), Image.LANCZOS))
    alpha = alpha.astype(np.float64) / 255.0

    rgba = np.zeros((ATLAS_RES, ATLAS_RES, 4), dtype=np.uint8)
    rgba[..., :3] = (np.clip(color, 0, 1) * 255).astype(np.uint8)
    rgba[..., 3] = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def _mirror_tile(patch):
    """Mirror-fold a patch into a seamless tile (reflect on both axes)."""
    top = np.concatenate([patch, patch[:, ::-1]], axis=1)
    full = np.concatenate([top, top[::-1, :]], axis=0)
    return np.clip(full, 0, 1)


def _build_base_swatch(img, base_color, rng):
    """
    Tileable swatch for the crown mound: pick the most plant-like crop in the
    base region, mirror-fold it, bias strongly toward the dark sampled base
    color, add fibrous striations and grain. Falls back procedurally. Kept
    dark so the mound never reads as a bright bulb.
    """
    base_color = np.clip(np.asarray(base_color, float), 0, 1)
    H, W, _ = img.shape
    y0, y1 = int(H * 0.80), int(H * 0.99)
    x0, x1 = int(W * 0.40), int(W * 0.62)
    cs = max(8, int(min(H, W) * 0.06))

    best, bestd = None, 1e9
    yy = y0
    while yy < max(y0 + 1, y1 - cs):
        xx = x0
        while xx < max(x0 + 1, x1 - cs):
            patch = img[yy:yy + cs, xx:xx + cs]
            if patch.shape[0] >= 4 and patch.shape[1] >= 4:
                med = np.median(patch.reshape(-1, 3), axis=0)
                d = np.linalg.norm(med - base_color)
                if d < bestd:
                    bestd, best = d, patch.copy()
            xx += max(2, cs // 2)
        yy += max(2, cs // 2)

    if best is None:
        arr = np.ones((BASE_RES, BASE_RES, 3)) * base_color[None, None, :]
        arr = arr + rng.normal(0.0, 0.03, (BASE_RES, BASE_RES, 3))
        return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

    full = _mirror_tile(best)
    pil = Image.fromarray((full * 255).astype(np.uint8)).resize(
        (BASE_RES, BASE_RES), Image.LANCZOS)
    arr = np.asarray(pil).astype(np.float64) / 255.0
    # bias strongly toward the dark sampled crown color, then darken a touch
    arr = (0.45 * arr + 0.55 * base_color[None, None, :]) * 0.85
    yy, xx = np.mgrid[0:BASE_RES, 0:BASE_RES]
    fib = 1.0 + 0.06 * np.sin(2.0 * np.pi * 9.0 * (xx / BASE_RES) +
                              rng.uniform(0.0, 2.0 * np.pi))
    arr = arr * fib[..., None]
    arr = arr + rng.normal(0.0, 0.02, (BASE_RES, BASE_RES, 3))
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


# ==========================================================================
#  UV ASSIGNMENT + MATERIAL BINDING
# ==========================================================================
def _blade_uvs(blade, n_stations, rng):
    """
    Per-blade atlas mapping. Each blade ribbon maps into one 4x4 tile with
    base->tip running bottom->top (so it samples the green->straw gradient
    and the alpha taper at the tip). Random tile + horizontal flip + jitter.
    """
    m = n_stations - 1
    vpb = 2 * m + 1                       # vertices per blade
    N = len(blade.vertices)
    n_blades = N // vpb
    ts = 1.0 / GRID

    tcol = rng.integers(0, GRID, n_blades)
    trow = rng.integers(0, GRID, n_blades)
    flip = rng.integers(0, 2, n_blades)
    ujit = rng.uniform(-0.03, 0.03, n_blades) * ts

    uv = np.zeros((N, 2), dtype=np.float64)
    ks = np.arange(m)
    f = ks / float(m)                     # length fraction at each ring
    half = 0.13 * ts
    for b in range(n_blades):
        base_i = b * vpb
        u0 = tcol[b] * ts
        v0 = trow[b] * ts
        uc = u0 + 0.5 * ts + ujit[b]
        uL, uR = uc - half, uc + half
        if flip[b]:
            uL, uR = uR, uL
        v_ring = v0 + ts * (0.04 + (1.0 - f) * 0.92)
        uv[base_i + 2 * ks, 0] = uL
        uv[base_i + 2 * ks, 1] = v_ring
        uv[base_i + 2 * ks + 1, 0] = uR
        uv[base_i + 2 * ks + 1, 1] = v_ring
        uv[base_i + 2 * m, 0] = uc                 # tip
        uv[base_i + 2 * m, 1] = v0 + ts * 0.04
    return uv


def _base_uvs(base):
    """Cylindrical/spherical projection for the small crown mound."""
    v = base.vertices
    ry = MOUND_HEIGHT * 0.5
    u = 0.5 + np.arctan2(v[:, 2], v[:, 0]) / (2.0 * np.pi)
    vv = np.clip(v[:, 1] / (2.0 * ry), 0.0, 1.0)
    return np.stack([u * 1.5, vv], axis=1)


def _color0_blades(blade):
    """
    Per-vertex COLOR_0: LIGHT value/warmth tints (so they multiply the
    texture without muddying it). Cool light body low/inner, warm straw and
    brighter toward the top -- the sun/shade gradient. Slight darkening only
    at the very base.
    """
    y = blade.vertices[:, 1]
    hf = np.clip(y / OVERALL_HEIGHT, 0.0, 1.0)
    body_tint = np.array([0.82, 0.88, 0.80])   # cool, light
    tip_tint = np.array([1.05, 0.97, 0.66])    # warm straw
    sw = np.clip(hf * 1.15, 0.0, 1.0)
    tint = body_tint[None, :] * (1.0 - sw)[:, None] + tip_tint[None, :] * sw[:, None]
    sun = (0.84 + 0.28 * hf)[:, None]          # brighter toward the top
    rgb = np.clip(tint * sun, 0.0, 1.0) * 255.0
    out = np.empty((len(y), 4), dtype=np.uint8)
    out[:, :3] = rgb.astype(np.uint8)
    out[:, 3] = 255
    return out


def _attach_visual(mesh, uv, material, color0):
    """Build TextureVisuals and attach per-vertex COLOR_0 (exports to glTF)."""
    tv = TextureVisuals(uv=uv, material=material)
    tv.vertex_attributes["color"] = color0
    mesh.visual = tv


def build_textured_scene(seed, density, image_path):
    """Geometry + photo-derived materials -> textured trimesh.Scene."""
    scene = build_mesh(seed, density)
    cfg = _DENSITY.get(density, _DENSITY["high"])
    n_stations = max(4, cfg["n_stations"])

    # --- sample colors from WELL INSIDE the silhouette ------------------
    img = _delight(_load_image(image_path))
    bg = _bg_estimate(img)
    green = _gather_color(img, (0.40, 0.85, 0.32, 0.68), bg, COL_GREEN)
    straw = _gather_color(img, (0.08, 0.40, 0.30, 0.70), bg, COL_STRAW)
    basec = _gather_color(img, (0.80, 0.99, 0.38, 0.62), bg, COL_BASE)

    # gentle grade: lift/cool the body, brighten/warm the tips, keep base dark
    green = np.clip(green * 1.12, 0.0, 1.0)
    straw = np.clip(straw * 1.10 + np.array([0.05, 0.03, -0.02]), 0.0, 1.0)
    basec = np.clip(basec * 0.75, 0.0, 1.0)

    rng_tex = np.random.default_rng(seed + 101)
    atlas_img = _build_blade_atlas(green, straw, rng_tex)
    base_img = _build_base_swatch(img, basec, rng_tex)

    # --- blades: foliage atlas, masked & double-sided -------------------
    blade = scene.geometry["blades"]
    blade_col0 = _color0_blades(blade)
    blade_uv = _blade_uvs(blade, n_stations, rng_tex)
    blade_mat = PBRMaterial(
        name="blades",
        baseColorTexture=atlas_img,
        metallicFactor=0.0,
        roughnessFactor=0.8,
        alphaMode="MASK",
        alphaCutoff=0.45,
        doubleSided=True,
    )
    _attach_visual(blade, blade_uv, blade_mat, blade_col0)

    # --- base mound: tileable dark crown swatch -------------------------
    base = scene.geometry["base"]
    base_col0 = np.asarray(base.visual.vertex_colors).copy().astype(np.uint8)
    base_uv = _base_uvs(base)
    base_mat = PBRMaterial(
        name="base",
        baseColorTexture=base_img,
        metallicFactor=0.0,
        roughnessFactor=0.95,
        doubleSided=False,
    )
    _attach_visual(base, base_uv, base_mat, base_col0)

    return scene


# ==========================================================================
#  CLI
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="Procedural tussock-grass GLB.")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", choices=("high", "med", "low"), default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        scene = build_textured_scene(args.seed, args.density, args.image)
        glb = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(glb)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    print("wrote {} ({} bytes)".format(args.output, len(glb)))


if __name__ == "__main__":
    main()