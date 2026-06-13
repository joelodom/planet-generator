#!/usr/bin/env python3
"""Procedural sparse multi-stemmed shrub: geometry + photo-derived materials.

Builds a small, airy, vase-to-fountain shrub (several slender stems that fan
upward into a fine twiggy crown with sparse, clumped foliage), derives tileable
bark + a 4x4 foliage-card atlas from a reference photo, applies per-surface UVs
and PBR materials, and exports a textured binary GLB.

Surfaces:
    "branches" -- woody stems / twigs, cylindrical UVs, bark texture
    "canopy"   -- clumped flat leaf cards, atlas tiles, MASK alpha

CLI:
    python shrub_tex.py --image PATH --seed INT --density {high,med,low} \
        --output OUT.glb

Only numpy + trimesh + PIL + stdlib.  Deterministic given --seed.
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter


# ----------------------------------------------------------------------------
# Measured proportions (ratios read by eye off the reference image)
# ----------------------------------------------------------------------------
HEIGHT = 2.4                      # overall height in METERS (plausible shrub)
HEIGHT_OVER_WIDTH = 0.90          # nearly as wide as tall (photo aspect ~0.98)
WIDTH = HEIGHT / HEIGHT_OVER_WIDTH

CROWN_BASE_FRAC = 0.30            # crown starts ~30% up; lower part is bare stem
CROWN_PEAK_FRAC = 0.72           # broadest a little above mid (broad fountain)
BASE_GATHER_FRAC = 0.05          # stems converge into a narrow ~5%-wide base

STEM_BASE_RADIUS = 0.034          # radius (m) of a stem at the ground
MIN_RADIUS = 0.0018              # let twigs persist finer -> dense fine network

EPS = 1e-9

# Fallback palettes (used only if the photo yields too few clean samples).
_FALLBACK_WOOD_DARK = np.array([0.30, 0.25, 0.20])
_FALLBACK_WOOD_LIGHT = np.array([0.63, 0.57, 0.46])
_FALLBACK_FOLIAGE = np.array([0.44, 0.51, 0.32])

_LUM = np.array([0.299, 0.587, 0.114])


# ============================================================================
# small vector helpers
# ============================================================================
def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > EPS else v


def _perp_basis(d):
    a = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(_norm(d), a)) > 0.95:
        a = np.array([1.0, 0.0, 0.0])
    u = _norm(np.cross(d, a))
    v = _norm(np.cross(d, u))
    return u, v


def _rotate(v, axis, ang):
    axis = _norm(axis)
    c, s = np.cos(ang), np.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


# ============================================================================
# generalized tapered tube -> (verts, faces, uvs) with cylindrical UVs
# ============================================================================
def _tube(points, radii, sides, v_scale):
    points = np.asarray(points, float)
    radii = np.asarray(radii, float)
    n = len(points)
    if n < 2:
        return None

    seg = points[1:] - points[:-1]
    seg = np.array([_norm(s) for s in seg])
    tan = np.empty((n, 3))
    tan[0] = seg[0]
    tan[-1] = seg[-1]
    for i in range(1, n - 1):
        tan[i] = _norm(seg[i - 1] + seg[i])

    normals = np.empty((n, 3))
    u0, _ = _perp_basis(tan[0])
    normals[0] = u0
    for i in range(1, n):
        t0, t1 = tan[i - 1], tan[i]
        axis = np.cross(t0, t1)
        s = np.linalg.norm(axis)
        if s < EPS:
            normals[i] = normals[i - 1]
        else:
            ang = np.arctan2(s, np.dot(t0, t1))
            normals[i] = _norm(_rotate(normals[i - 1], axis / s, ang))

    # cumulative arc length -> v coordinate (bark runs along the stem)
    seglen = np.linalg.norm(points[1:] - points[:-1], axis=1)
    arclen = np.concatenate([[0.0], np.cumsum(seglen)])
    vcoord = arclen * v_scale

    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    cos, sin = np.cos(ang), np.sin(ang)
    ucoord = np.linspace(0.0, 1.0, sides, endpoint=False)

    verts = np.empty((n * sides, 3))
    uvs = np.empty((n * sides, 2))
    for i in range(n):
        ni = normals[i]
        bi = _norm(np.cross(tan[i], ni))
        ring = (cos[:, None] * ni[None, :] + sin[:, None] * bi[None, :]) * radii[i]
        verts[i * sides:(i + 1) * sides] = points[i] + ring
        uvs[i * sides:(i + 1) * sides, 0] = ucoord
        uvs[i * sides:(i + 1) * sides, 1] = vcoord[i]

    faces = []
    for i in range(n - 1):
        a0 = i * sides
        a1 = (i + 1) * sides
        for k in range(sides):
            k1 = (k + 1) % sides
            faces.append([a0 + k, a0 + k1, a1 + k1])
            faces.append([a0 + k, a1 + k1, a1 + k])

    base_c = len(verts)
    apex = points[-1] + tan[-1] * radii[-1] * 1.2
    verts = np.vstack([verts, points[0][None, :], apex[None, :]])
    uvs = np.vstack([uvs, [0.5, vcoord[0]], [0.5, vcoord[-1]]])
    apex_i = base_c + 1
    for k in range(sides):
        k1 = (k + 1) % sides
        faces.append([base_c, k1, k])
        faces.append([apex_i, (n - 1) * sides + k, (n - 1) * sides + k1])
    return verts, np.asarray(faces, np.int64), uvs


# ============================================================================
# crown envelope
# ============================================================================
def _envelope_radius(y, lobes):
    f = y / HEIGHT
    if f <= BASE_GATHER_FRAC:
        return WIDTH * 0.03
    t = (f - BASE_GATHER_FRAC) / (1.0 - BASE_GATHER_FRAC)
    peak = (CROWN_PEAK_FRAC - BASE_GATHER_FRAC) / (1.0 - BASE_GATHER_FRAC)
    if t <= peak:
        prof = (t / peak) ** 0.7
    else:
        prof = 1.0 - 0.35 * ((t - peak) / (1.0 - peak)) ** 1.5
    base = WIDTH * 0.5 * max(prof, 0.03)
    lobe = 1.0 + 0.12 * np.sin(lobes[0] + 3.0 * f) + 0.08 * np.cos(lobes[1] + 5.0 * f)
    return base * lobe


# ============================================================================
# skeleton growth
# ============================================================================
def _grow(cfg, rng):
    branches = []
    tips = []
    lobes = (rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi))
    max_depth = cfg["max_depth"]
    nodes = cfg["nodes"]
    cap = cfg["branch_cap"]

    n_stems = cfg["stems"]
    stack = []
    for i in range(n_stems):
        phi = 2 * np.pi * i / n_stems + rng.uniform(-0.3, 0.3)
        r = WIDTH * BASE_GATHER_FRAC * rng.uniform(0.1, 0.6)
        p0 = np.array([r * np.cos(phi), 0.0, r * np.sin(phi)])
        out = _norm(np.array([np.cos(phi), 0.0, np.sin(phi)]))
        d0 = _norm(np.array([0, 1, 0]) * 2.4 + out * rng.uniform(0.4, 0.95))
        stack.append((p0, d0, STEM_BASE_RADIUS * rng.uniform(0.85, 1.0), 0))

    count = 0
    while stack:
        p, d, radius, depth = stack.pop()
        if count >= cap:
            break
        count += 1

        seg_len = HEIGHT * (0.32 if depth == 0 else 0.22) * (0.82 ** depth)
        seg_len *= rng.uniform(0.8, 1.2)

        pts = [p.copy()]
        rads = [radius]
        cur = p.copy()
        cdir = d.copy()
        r = radius
        droop = 0.04 + 0.05 * depth
        for j in range(nodes - 1):
            radial = np.array([cur[0], 0.0, cur[2]])
            radial = _norm(radial) if np.linalg.norm(radial) > EPS else np.zeros(3)
            cdir = _norm(
                cdir
                + radial * 0.30                       # stronger outward fan
                + np.array([0.0, -1.0, 0.0]) * droop
                + rng.normal(0, 0.12, 3)
            )
            step = seg_len / (nodes - 1)
            nxt = cur + cdir * step
            er = _envelope_radius(nxt[1], lobes)
            hr = np.hypot(nxt[0], nxt[2])
            if hr > er * 1.03 and hr > EPS:
                k = (er * 1.03) / hr
                nxt[0] *= k
                nxt[2] *= k
            cur = nxt
            r *= rng.uniform(0.88, 0.94)
            pts.append(cur.copy())
            rads.append(max(r, MIN_RADIUS * 0.6))

        branches.append((np.array(pts), np.array(rads)))
        tips.append((cur.copy(), cur[1]))

        if depth < max_depth and r > MIN_RADIUS and count < cap:
            n_child = 2 if (depth >= 1 and rng.random() < 0.55) else 3
            child_r = r / np.sqrt(n_child)
            u, v = _perp_basis(cdir)
            for c in range(n_child):
                phi = 2 * np.pi * (c + rng.uniform(-0.2, 0.2)) / n_child
                axis = np.cos(phi) * u + np.sin(phi) * v
                theta = np.radians(rng.uniform(24, 50))     # wider fork angle
                cd = _norm(_rotate(cdir, axis, theta) + np.array([0, 1, 0]) * 0.14)
                stack.append((cur.copy(), cd, child_r * rng.uniform(0.85, 1.05), depth + 1))

    return branches, tips, lobes


# ============================================================================
# foliage cards -> (verts, faces, uvs, tints) ; each quad maps to an atlas tile
# ============================================================================
ATLAS_TILES = 4          # 4x4 atlas
ATLAS_SIZE = 1024


def _build_canopy(cfg, tips, lobes, rng):
    n_clumps = cfg["clumps"]
    cards_per = max(1, cfg["cards_total"] // n_clumps)
    crown_w = WIDTH
    half_base = 0.024 * crown_w          # small leaf tufts (sparse, airy look)
    clump_rad = 0.06 * crown_w           # tight tufts scattered at twig tips

    cand = [t for t in tips if t[1] > HEIGHT * (CROWN_BASE_FRAC + 0.03)]
    if len(cand) < n_clumps:
        cand = tips if tips else [(np.array([0.0, HEIGHT * 0.7, 0.0]), HEIGHT * 0.7)]
    heights = np.array([t[1] for t in cand])
    w = (heights - heights.min() + 0.05) ** 1.5
    w = w / w.sum()
    idx = rng.choice(len(cand), size=n_clumps, replace=len(cand) < n_clumps, p=w)
    centers = [cand[i][0] for i in idx]

    crown_cy = HEIGHT * 0.66
    pad = 2.0 / ATLAS_SIZE
    step = 1.0 / ATLAS_TILES

    verts, faces, uvs, tints = [], [], [], []
    base_i = 0
    for c in centers:
        clump_var = rng.uniform(-0.06, 0.06)   # subtle per-clump value shift
        for _ in range(cards_per):
            off = rng.normal(0, 1, 3)
            off = off / (np.linalg.norm(off) + EPS) * clump_rad * rng.uniform(0.0, 1.0)
            pos = c + off

            out = pos - np.array([0.0, crown_cy, 0.0])
            out = _norm(out) if np.linalg.norm(out) > EPS else np.array([0.0, 1.0, 0.0])
            jit_ax = _norm(rng.normal(0, 1, 3))
            out = _norm(_rotate(out, jit_ax, np.radians(rng.uniform(-25, 25))))

            u, v = _perp_basis(out)
            roll = rng.uniform(0, 2 * np.pi)
            u2 = np.cos(roll) * u + np.sin(roll) * v
            v2 = -np.sin(roll) * u + np.cos(roll) * v

            scale = np.exp(rng.normal(0, 0.3))
            hu = half_base * scale
            hv = half_base * 0.7 * scale
            quad = np.array([
                pos - u2 * hu - v2 * hv,
                pos + u2 * hu - v2 * hv,
                pos + u2 * hu + v2 * hv,
                pos - u2 * hu + v2 * hv,
            ])
            verts.append(quad)
            faces.append([base_i, base_i + 1, base_i + 2])
            faces.append([base_i, base_i + 2, base_i + 3])

            # --- UV: pick a random atlas tile + 0/90/180/270 rotation --------
            tx = int(rng.integers(0, ATLAS_TILES))
            ty = int(rng.integers(0, ATLAS_TILES))
            u_lo, u_hi = tx * step + pad, (tx + 1) * step - pad
            v_lo, v_hi = ty * step + pad, (ty + 1) * step - pad
            corners = [(u_lo, v_lo), (u_hi, v_lo), (u_hi, v_hi), (u_lo, v_hi)]
            k = int(rng.integers(0, 4))
            corners = corners[k:] + corners[:k]
            uvs.extend(corners)

            # --- per-card sun/shade tint (outer + higher = brighter) ---------
            hf = np.clip(pos[1] / HEIGHT, 0.0, 1.0)
            outward = np.clip(np.hypot(pos[0], pos[2]) / (0.5 * crown_w), 0.0, 1.0)
            bright = np.clip(0.86 + 0.16 * hf + 0.10 * outward + clump_var, 0.78, 1.1)
            warm = np.array([1.04, 1.0, 0.94]) if hf > 0.6 else np.array([0.98, 1.0, 1.02])
            tint = np.clip(bright * warm, 0.0, 1.0)
            tints.extend([tint] * 4)
            base_i += 4

    verts = np.vstack(verts)
    faces = np.asarray(faces, np.int64)
    uvs = np.asarray(uvs, float)
    tints = np.asarray(tints, float)
    return verts, faces, uvs, tints


# ============================================================================
# vertex colours (COLOR_0 tints that multiply the albedo texture)
# ============================================================================
def _wood_colors(verts):
    # darker grey-brown low down, bleaching to pale tan at the twig tips;
    # a touch of AO darkening right at the ground.
    f = np.clip(verts[:, 1] / HEIGHT, 0.0, 1.0)
    low = np.array([0.55, 0.50, 0.44])    # tints are near-white so the bark
    high = np.array([1.00, 0.97, 0.90])   # texture shows through; tips bleach
    rgb = low[None, :] * (1 - f[:, None]) + high[None, :] * f[:, None]
    ao = np.clip(0.7 + 1.6 * f, 0.0, 1.0)[:, None]   # slightly darker near base
    rgb = np.clip(rgb * ao, 0.0, 1.0)
    col = np.empty((len(verts), 4))
    col[:, :3] = rgb
    col[:, 3] = 1.0
    return (col * 255.0).astype(np.uint8)


# ============================================================================
# ---- photo sampling -------------------------------------------------------
# ============================================================================
def _delight(arr):
    """Divide by a heavily blurred luminance, gain clamped to [0.6, 1.6]."""
    h, w = arr.shape[:2]
    lum = arr @ _LUM
    pil = Image.fromarray((np.clip(lum, 0, 1) * 255).astype(np.uint8))
    rad = max(8, int(min(h, w) / 8))
    blur = np.asarray(pil.filter(ImageFilter.GaussianBlur(rad)), float) / 255.0
    blur = np.maximum(blur, 1e-3)
    target = float(np.median(lum))
    gain = np.clip(target / blur, 0.6, 1.6)
    return np.clip(arr * gain[..., None], 0.0, 1.0)


def sample_palette(image_path):
    """Return a palette dict sampled from inside the object's silhouette.

    Masks the pale background out, then separates green foliage flecks from the
    darker woody stems and reads representative colors from each."""
    pal = dict(
        wood_dark=_FALLBACK_WOOD_DARK.copy(),
        wood_light=_FALLBACK_WOOD_LIGHT.copy(),
        foliage=_FALLBACK_FOLIAGE.copy(),
        foliage_light=np.clip(_FALLBACK_FOLIAGE * 1.2, 0, 1),
        foliage_dark=_FALLBACK_FOLIAGE * 0.7,
    )
    if not image_path:
        return pal
    try:
        im = Image.open(image_path).convert("RGB")
    except Exception:
        return pal

    im.thumbnail((512, 512), Image.LANCZOS)
    arr = np.asarray(im, float) / 255.0
    dl = _delight(arr)
    h, w = arr.shape[:2]

    # background = median of a border frame (the pale, uniform surround)
    bw = max(2, int(0.06 * min(h, w)))
    border = np.concatenate([
        arr[:bw].reshape(-1, 3), arr[-bw:].reshape(-1, 3),
        arr[:, :bw].reshape(-1, 3), arr[:, -bw:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)

    diff = np.linalg.norm(arr - bg[None, None, :], axis=2)
    green = arr[..., 1] - 0.5 * (arr[..., 0] + arr[..., 2])

    obj = diff > 0.06
    foliage_mask = obj & (green > 0.015)
    wood_mask = obj & (~foliage_mask) & (diff > 0.10)

    wood_px = dl[wood_mask]
    fol_px = dl[foliage_mask]

    if len(wood_px) >= 40:
        dark = np.percentile(wood_px, 22, axis=0)
        light = np.percentile(wood_px, 82, axis=0)
        # keep a visible albedo value range (lightest ~2-3x the darkest)
        ld, ll = float(dark @ _LUM), float(light @ _LUM)
        if ll < ld * 1.8 + 1e-3:
            light = np.clip(light * (ld * 2.2 + 0.02) / (ll + 1e-3), 0, 1)
        # warm grey-brown bias
        pal["wood_dark"] = np.clip(dark, 0.08, 0.9)
        pal["wood_light"] = np.clip(light, 0.12, 0.95)

    if len(fol_px) >= 20:
        med = np.median(fol_px, axis=0)
        pal["foliage"] = np.clip(med, 0.08, 0.85)
        pal["foliage_light"] = np.clip(np.percentile(fol_px, 75, axis=0) * 1.05, 0, 0.95)
        pal["foliage_dark"] = np.clip(np.percentile(fol_px, 25, axis=0) * 0.9, 0.04, 0.8)

    return pal


# ============================================================================
# ---- texture synthesis ----------------------------------------------------
# ============================================================================
def _tileable_noise(size, fx_lo, fx_hi, fy_lo, fy_hi, n_waves, rng):
    """Sum of integer-frequency sinusoids -> seamlessly tileable field [0,1]."""
    lin = np.linspace(0.0, 2.0 * np.pi, size, endpoint=False)
    x, y = np.meshgrid(lin, lin)
    out = np.zeros((size, size))
    for _ in range(n_waves):
        fx = int(rng.integers(fx_lo, fx_hi + 1))
        fy = int(rng.integers(fy_lo, fy_hi + 1))
        ph = rng.uniform(0, 2 * np.pi)
        amp = 1.0 / (1.0 + fx + fy)
        out += amp * np.sin(fx * x + fy * y + ph)
    p = np.ptp(out)
    return (out - out.min()) / (p + 1e-9)


def make_bark(size, dark, light, rng):
    """Tileable bark albedo: vertical grain streaks over soft patches, with a
    warm grey-brown palette and grain-scale value variation."""
    streak = _tileable_noise(size, 3, 9, 0, 2, 14, rng)    # vertical streaks
    patch = _tileable_noise(size, 1, 3, 1, 3, 8, rng)      # broad mottling
    fine = _tileable_noise(size, 8, 16, 6, 16, 10, rng)    # fine speckle
    g = np.clip(0.55 * streak + 0.32 * patch + 0.13 * fine, 0, 1)
    g = g ** 1.1
    dark = np.asarray(dark)
    light = np.asarray(light)
    rgb = dark[None, None, :] + (light - dark)[None, None, :] * g[..., None]
    # gentle warm undertone in the mid values
    warm = (0.5 - np.abs(g - 0.5))[..., None] * np.array([0.04, 0.015, -0.01])
    rgb = np.clip(rgb + warm, 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8), "RGB")


def make_normal(albedo_img, strength=1.4):
    """Tangent-space normal map from albedo luminance (tileable via wrap)."""
    arr = np.asarray(albedo_img, float) / 255.0
    h = 1.0 - (arr @ _LUM)               # height = inverse luminance
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * strength
    gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * strength
    nz = np.ones_like(h)
    n = np.stack([-gx, -gy, nz], axis=-1)
    n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)
    return Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8), "RGB")


def _leaf_polygon(draw, cx, cy, ang, length, width, color):
    ca, sa = np.cos(ang), np.sin(ang)
    ts = np.linspace(0.0, 1.0, 8)
    right, left = [], []
    for t in ts:
        hw = width * (np.sin(np.pi * t) ** 0.6)
        ax = t * length
        for sign, lst in ((1.0, right), (-1.0, left)):
            px, py = ax, sign * hw
            rx = cx + px * ca - py * sa
            ry = cy + px * sa + py * ca
            lst.append((rx, ry))
    draw.polygon(right + left[::-1], fill=color)


def _leaf_tile(tile_px, base, bright, warm, rng):
    """One atlas tile: a loose tuft of small simple leaves on transparent bg."""
    ss = 4
    S = tile_px * ss
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = S * 0.5, S * 0.5
    n_leaves = int(rng.integers(5, 9))            # fewer leaves -> airier tuft
    base_ang = rng.uniform(0, 2 * np.pi)
    for _ in range(n_leaves):
        ang = base_ang + rng.uniform(-1.4, 1.4) + rng.normal(0, 0.3)
        length = S * rng.uniform(0.16, 0.30)      # small, simple leaves
        width = length * rng.uniform(0.16, 0.26)
        # start point near the cluster centre so leaves stay connected
        rad = S * rng.uniform(0.0, 0.12)
        sx = cx + rad * np.cos(rng.uniform(0, 2 * np.pi))
        sy = cy + rad * np.sin(rng.uniform(0, 2 * np.pi))
        jit = rng.normal(0, 0.04, 3)
        col = np.clip(base * bright + warm + jit, 0.04, 0.96)
        rgb = tuple(int(v * 255) for v in col)
        _leaf_polygon(draw, sx, sy, ang, length, width, rgb + (255,))
    return img.resize((tile_px, tile_px), Image.LANCZOS)


def make_foliage_atlas(pal, rng):
    """4x4 atlas of distinct leaf-cluster tiles; sunlit tiles warmer/brighter,
    shaded tiles cooler/darker.  Binary alpha with anti-aliased edges only."""
    tile_px = ATLAS_SIZE // ATLAS_TILES
    atlas = Image.new("RGBA", (ATLAS_SIZE, ATLAS_SIZE), (0, 0, 0, 0))
    # lighten the sampled olive toward a dusty, desaturated early-season sage
    base = np.clip(0.55 * np.asarray(pal["foliage"]) + 0.45 * np.array([0.62, 0.66, 0.47]), 0, 1)
    for ty in range(ATLAS_TILES):
        for tx in range(ATLAS_TILES):
            sun = 1.0 - ty / (ATLAS_TILES - 1)      # top rows = sunlit
            bright = 0.96 + 0.26 * sun              # keep foliage light overall
            warm = (np.array([0.05, 0.03, -0.03]) * sun
                    + np.array([-0.02, 0.0, 0.03]) * (1 - sun))
            tile = _leaf_tile(tile_px, base, bright, warm, rng)
            atlas.paste(tile, (tx * tile_px, ty * tile_px), tile)
    # binarize alpha (keep only anti-aliased edge softness)
    a = np.asarray(atlas)[..., 3]
    hard = np.where(a > 128, 255, 0).astype(np.uint8)
    # blend a hint of AA back at edges so silhouettes are not jagged
    aa = np.asarray(atlas.split()[3])
    edge = (aa > 16) & (aa <= 200)
    final_a = hard.copy()
    final_a[edge] = aa[edge]
    out = np.dstack([np.asarray(atlas)[..., :3], final_a]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


# ============================================================================
# entry point
# ============================================================================
_DENSITY = {
    "high": dict(stems=7, max_depth=7, nodes=4, sides=5,
                 branch_cap=1100, clumps=60, cards_total=1500),
    "med":  dict(stems=6, max_depth=6, nodes=3, sides=4,
                 branch_cap=450,  clumps=30, cards_total=650),
    "low":  dict(stems=5, max_depth=5, nodes=3, sides=4,
                 branch_cap=150,  clumps=12, cards_total=220),
}

_BARK_SIZE = 512
_V_SCALE = 1.0 / 0.12     # bark repeats every ~0.12 m of stem length


def build_mesh(seed: int, density: str = "high", image_path: str = None) -> trimesh.Scene:
    if density not in _DENSITY:
        density = "high"
    cfg = _DENSITY[density]
    rng = np.random.default_rng(seed)

    palette = sample_palette(image_path)

    # --- woody skeleton with cylindrical UVs --------------------------------
    branches, tips, lobes = _grow(cfg, rng)
    sides = cfg["sides"]
    wv, wf, wuv = [], [], []
    off = 0
    for pts, rads in branches:
        res = _tube(pts, rads, sides, _V_SCALE)
        if res is None:
            continue
        v, f, uv = res
        wv.append(v)
        wf.append(f + off)
        wuv.append(uv)
        off += len(v)
    wood_v = np.vstack(wv)
    wood_f = np.vstack(wf)
    wood_uv = np.vstack(wuv)

    # --- foliage cards ------------------------------------------------------
    leaf_v, leaf_f, leaf_uv, leaf_tint = _build_canopy(cfg, tips, lobes, rng)

    # --- ground + normalise to measured proportions -------------------------
    min_y = min(wood_v[:, 1].min(), leaf_v[:, 1].min())
    cx = np.concatenate([wood_v[:, 0], leaf_v[:, 0]]).mean()
    cz = np.concatenate([wood_v[:, 2], leaf_v[:, 2]]).mean()
    shift = np.array([cx, min_y, cz])
    wood_v -= shift
    leaf_v -= shift

    all_y = np.concatenate([wood_v[:, 1], leaf_v[:, 1]])
    all_h = np.concatenate([np.hypot(wood_v[:, 0], wood_v[:, 2]),
                            np.hypot(leaf_v[:, 0], leaf_v[:, 2])])
    sy = HEIGHT / max(all_y.max(), EPS)
    sxz = (WIDTH * 0.5) / max(all_h.max(), EPS)
    scl = np.array([sxz, sy, sxz])
    wood_v *= scl
    leaf_v *= scl

    # --- textures -----------------------------------------------------------
    bark_img = make_bark(_BARK_SIZE, palette["wood_dark"], palette["wood_light"], rng)
    bark_normal = make_normal(bark_img)
    atlas_img = make_foliage_atlas(palette, rng)

    # --- branches mesh ------------------------------------------------------
    bark_mat = trimesh.visual.material.PBRMaterial(
        name="bark",
        baseColorTexture=bark_img,
        baseColorFactor=np.array([255, 255, 255, 255], np.uint8),
        metallicFactor=0.0,
        roughnessFactor=0.9,
    )
    try:
        bark_mat.normalTexture = bark_normal
    except Exception:
        pass
    branches_mesh = trimesh.Trimesh(vertices=wood_v, faces=wood_f, process=False)
    branches_mesh.visual = trimesh.visual.TextureVisuals(uv=wood_uv, material=bark_mat)
    branches_mesh.visual.vertex_attributes["color"] = _wood_colors(wood_v)

    # --- canopy mesh --------------------------------------------------------
    leaf_mat = trimesh.visual.material.PBRMaterial(
        name="foliage",
        baseColorTexture=atlas_img,
        baseColorFactor=np.array([255, 255, 255, 255], np.uint8),
        metallicFactor=0.0,
        roughnessFactor=0.8,
        alphaMode="MASK",
        alphaCutoff=0.45,
        doubleSided=True,
    )
    canopy_mesh = trimesh.Trimesh(vertices=leaf_v, faces=leaf_f, process=False)
    canopy_mesh.visual = trimesh.visual.TextureVisuals(uv=leaf_uv, material=leaf_mat)
    leaf_col = np.empty((len(leaf_v), 4), np.uint8)
    leaf_col[:, :3] = np.clip(leaf_tint * 255.0, 0, 255).astype(np.uint8)
    leaf_col[:, 3] = 255
    canopy_mesh.visual.vertex_attributes["color"] = leaf_col

    scene = trimesh.Scene()
    scene.add_geometry(branches_mesh, geom_name="branches")
    scene.add_geometry(canopy_mesh, geom_name="canopy")
    return scene


def main():
    ap = argparse.ArgumentParser(description="Procedural shrub -> textured GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()
    try:
        scene = build_mesh(args.seed, args.density, args.image)
        glb = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(glb)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())