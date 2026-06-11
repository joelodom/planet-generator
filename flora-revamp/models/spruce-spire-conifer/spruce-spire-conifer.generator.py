#!/usr/bin/env python3
"""
Procedural spire-conifer ("blue-spruce" archetype) -> textured GLB.

Builds a tall conical evergreen (clumped foliage leaf-cards + slender tapered
trunk), derives tileable materials by SAMPLING COLORS from a reference photo,
applies per-surface UVs (atlas-mapped cards, cylindrical bark), and exports a
binary .glb with embedded textures and COLOR_0 vertex tints.

CLI:
    python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial


# =============================================================================
# GEOMETRY
# =============================================================================

TREE_HEIGHT = 5.0                       # meters; plausible ornamental spruce
HEIGHT_OVER_WIDTH = 2.05                # broad spruce cone (front aspect ~0.49)
CROWN_BASE_DIAMETER = TREE_HEIGHT / HEIGHT_OVER_WIDTH
CROWN_BASE_RADIUS = CROWN_BASE_DIAMETER * 0.5   # widest foliage radius

BARE_TRUNK_FRAC = 0.05                  # short exposed trunk at very bottom (~5%)
APEX_BARE_FRAC = 0.025                  # thin bare spindle tip at crown
CONE_POWER = 1.05                       # ~straight skirt profile (full cone)
N_WHORL_TIERS = 9.0                     # subtle tiered branch ripples up the cone
TRUNK_BASE_FRAC = 0.018                 # trunk radius as frac of height (slender)
TRUNK_FLARE = 1.45                      # basal flare multiplier
TRUNK_FLARE_FRAC = 0.06                 # over bottom ~6% of trunk

CANOPY_RGB = np.array([96, 128, 112], dtype=np.uint8)    # dusty blue-green
TRUNK_RGB = np.array([104, 82, 66], dtype=np.uint8)      # grey-brown bark


def _density_params(density: str):
    # cards*2 triangles: high 12k, med 4.8k, low 1.4k (+ small trunk) -> in budget
    table = {
        "high": dict(n_cards=6000, n_clumps=40, trunk_sides=14, trunk_levels=14),
        "med":  dict(n_cards=2400, n_clumps=28, trunk_sides=10, trunk_levels=10),
        "low":  dict(n_cards=700,  n_clumps=16, trunk_sides=6,  trunk_levels=6),
    }
    if density not in table:
        density = "high"
    return table[density]


def _make_lobes(rng):
    n = int(rng.integers(3, 7))
    modes = rng.integers(2, 6, size=n)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n)
    amps = rng.uniform(0.025, 0.06, size=n)   # gentler lobes -> cleaner outline
    return modes, phases, amps


def _radius_at(u, theta, lobes):
    modes, phases, amps = lobes
    u = np.clip(u, 0.0, 1.0)
    base = CROWN_BASE_RADIUS * np.power(np.maximum(1.0 - u, 0.0), CONE_POWER)
    tier = 1.0 + 0.030 * np.sin(u * N_WHORL_TIERS * 2.0 * np.pi)
    bulge = np.ones_like(np.atleast_1d(theta), dtype=float)
    for m, p, a in zip(modes, phases, amps):
        bulge = bulge + a * np.sin(m * theta + p)
    bulge = bulge.reshape(np.shape(theta))
    return base * tier * bulge


def _build_quads(centers, normals, u_axes, v_axes, hu, hv):
    n = len(centers)
    du = (u_axes * hu[:, None])
    dv = (v_axes * hv[:, None])
    c0 = centers - du - dv
    c1 = centers + du - dv
    c2 = centers + du + dv
    c3 = centers - du + dv
    verts = np.empty((n * 4, 3), dtype=np.float64)
    verts[0::4] = c0
    verts[1::4] = c1
    verts[2::4] = c2
    verts[3::4] = c3
    base = np.arange(n) * 4
    faces = np.empty((n * 2, 3), dtype=np.int64)
    faces[0::2] = np.stack([base, base + 1, base + 2], axis=1)
    faces[1::2] = np.stack([base, base + 2, base + 3], axis=1)
    return verts, faces


def _build_canopy(rng, p):
    y0 = BARE_TRUNK_FRAC * TREE_HEIGHT
    crown_h = TREE_HEIGHT * (1.0 - BARE_TRUNK_FRAC) - APEX_BARE_FRAC * TREE_HEIGHT
    lobes = _make_lobes(rng)

    n_clumps = p["n_clumps"]
    n_cards = p["n_cards"]
    weights = rng.uniform(0.7, 1.3, size=n_clumps)
    counts = np.maximum(1, np.round(weights / weights.sum() * n_cards)).astype(int)

    crown_w = 2.0 * CROWN_BASE_RADIUS
    clump_r = 0.13 * crown_w
    card_h_mean = 0.055 * crown_w
    cone_slope = CROWN_BASE_RADIUS / crown_h

    all_c, all_n, all_u, all_v, all_hu, all_hv = [], [], [], [], [], []

    for i in range(n_clumps):
        m = int(counts[i])
        # spread clump heights across the whole cone so the trunk is covered
        u_c = float(np.clip((i + rng.uniform(0.0, 1.0)) / n_clumps, 0.0, 0.985))
        theta_c = float(rng.uniform(0.0, 2.0 * np.pi))
        interior = rng.random() < 0.18
        shell_frac = rng.uniform(0.45, 0.72) if interior else rng.uniform(0.9, 1.05)

        r_shell = float(_radius_at(u_c, np.array([theta_c]), lobes)[0])
        y_c = y0 + u_c * crown_h
        r_c = r_shell * shell_frac
        center0 = np.array([r_c * np.cos(theta_c), y_c, r_c * np.sin(theta_c)])

        out = np.array([np.cos(theta_c), cone_slope, np.sin(theta_c)])
        out /= np.linalg.norm(out)

        off = rng.normal(0.0, 1.0, size=(m, 3))
        off[:, 1] *= 1.05
        centers = center0[None, :] + off * clump_r
        ang = np.arctan2(centers[:, 2], centers[:, 0])
        rad = np.hypot(centers[:, 0], centers[:, 2])
        uu = np.clip((centers[:, 1] - y0) / crown_h, 0.0, 1.0)
        r_target = _radius_at(uu, ang, lobes) * np.clip(shell_frac, 0.0, 1.05)
        rad = 0.4 * rad + 0.6 * r_target
        centers[:, 0] = rad * np.cos(ang)
        centers[:, 2] = rad * np.sin(ang)

        # per-card normals: clump outward + modest jitter (plush, not wispy)
        nz = out[None, :] + rng.normal(0.0, 0.22, size=(m, 3))
        nz /= np.linalg.norm(nz, axis=1, keepdims=True)
        ref = np.tile(np.array([0.0, 1.0, 0.0]), (m, 1))
        ua = np.cross(nz, ref)
        bad = np.linalg.norm(ua, axis=1) < 1e-6
        ua[bad] = np.cross(nz[bad], np.array([1.0, 0.0, 0.0]))
        ua /= np.linalg.norm(ua, axis=1, keepdims=True)
        va = np.cross(nz, ua)
        va /= np.linalg.norm(va, axis=1, keepdims=True)
        va[:, 1] -= 0.12                      # slight downward sweep only
        va /= np.linalg.norm(va, axis=1, keepdims=True)

        hu = card_h_mean * rng.lognormal(0.0, 0.22, size=m)
        hv = card_h_mean * rng.lognormal(0.0, 0.22, size=m)

        all_c.append(centers)
        all_n.append(nz)
        all_u.append(ua)
        all_v.append(va)
        all_hu.append(hu)
        all_hv.append(hv)

    centers = np.concatenate(all_c)
    normals = np.concatenate(all_n)
    u_axes = np.concatenate(all_u)
    v_axes = np.concatenate(all_v)
    hu = np.concatenate(all_hu)
    hv = np.concatenate(all_hv)

    ang = np.arctan2(centers[:, 2], centers[:, 0])
    uu = np.clip((centers[:, 1] - y0) / crown_h, 0.0, 1.0)
    r_max = _radius_at(uu, ang, lobes) * 1.03
    rad = np.hypot(centers[:, 0], centers[:, 2])
    scale = np.where(rad > r_max, r_max / np.maximum(rad, 1e-9), 1.0)
    centers[:, 0] *= scale
    centers[:, 2] *= scale

    verts, faces = _build_quads(centers, normals, u_axes, v_axes, hu, hv)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.fix_normals()
    return mesh


def _build_trunk(rng, p):
    sides = p["trunk_sides"]
    levels = p["trunk_levels"]
    base_r = TRUNK_BASE_FRAC * TREE_HEIGHT

    ys = np.linspace(0.0, TREE_HEIGHT, levels)
    t = ys / TREE_HEIGHT
    radii = base_r * np.power(np.maximum(1.0 - t, 0.0), 0.9) + 0.0015
    flare_mask = t < TRUNK_FLARE_FRAC
    fr = 1.0 + (TRUNK_FLARE - 1.0) * (1.0 - t[flare_mask] / TRUNK_FLARE_FRAC)
    radii[flare_mask] *= fr

    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    verts = []
    for j in range(levels):
        wob = 1.0 + rng.normal(0.0, 0.05, size=sides)
        ring = np.stack([radii[j] * ca * wob, np.full(sides, ys[j]),
                         radii[j] * sa * wob], axis=1)
        verts.append(ring)
    verts = np.concatenate(verts)

    faces = []
    for j in range(levels - 1):
        a = j * sides
        b = (j + 1) * sides
        for k in range(sides):
            k2 = (k + 1) % sides
            faces.append([a + k, b + k, b + k2])
            faces.append([a + k, b + k2, a + k2])
    bc = len(verts)
    verts = np.vstack([verts, [0.0, 0.0, 0.0]])
    for k in range(sides):
        k2 = (k + 1) % sides
        faces.append([bc, k2, k])

    mesh = trimesh.Trimesh(vertices=np.asarray(verts),
                           faces=np.asarray(faces), process=True)
    mesh.fix_normals()
    return mesh


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    """Build a procedural spire-conifer instance as a named-geometry Scene."""
    rng = np.random.default_rng(seed)
    p = _density_params(density)
    trunk = _build_trunk(rng, p)
    canopy = _build_canopy(rng, p)
    scene = trimesh.Scene()
    scene.add_geometry(trunk, geom_name="trunk")
    scene.add_geometry(canopy, geom_name="canopy")
    return scene


# =============================================================================
# COLOR SAMPLING FROM THE PHOTO
# =============================================================================

def _load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64)


def _crop_frac(img, x0, x1, y0, y1):
    h, w = img.shape[:2]
    xa, xb = int(x0 * w), int(x1 * w)
    ya, yb = int(y0 * h), int(y1 * h)
    xa, xb = max(0, xa), min(w, max(xa + 1, xb))
    ya, yb = max(0, ya), min(h, max(ya + 1, yb))
    return img[ya:yb, xa:xb]


def _delight(crop):
    """Divide out a heavily blurred luminance; clamp gain to [0.6, 1.6]."""
    pim = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8))
    lum = np.asarray(pim.convert("L").filter(
        ImageFilter.GaussianBlur(radius=max(crop.shape[:2]) / 6.0)),
        dtype=np.float64)
    mean_l = max(1.0, float(lum.mean()))
    gain = np.clip(mean_l / np.maximum(lum, 1.0), 0.6, 1.6)
    out = np.clip(crop * gain[..., None], 0, 255)
    return out


def _looks_like_foliage(med):
    r, g, b = med
    bright = med.mean()
    return (g >= r - 6.0) and (bright < 200.0) and (bright > 25.0)


def _looks_like_bark(med):
    r, g, b = med
    bright = med.mean()
    return (r > b + 4.0) and (bright < 190.0) and (bright > 20.0)


def _sample_foliage(img, rng):
    """Return (swatch_flat (P,3) uint8 color source, body median rgb)."""
    crop = _crop_frac(img, 0.40, 0.60, 0.40, 0.62)   # well inside the cone
    crop = _delight(crop)
    med = np.median(crop.reshape(-1, 3), axis=0)
    if not _looks_like_foliage(med):
        base = CANOPY_RGB.astype(np.float64)
        noise = rng.normal(0.0, 12.0, size=(4096, 3))
        flat = np.clip(base[None, :] + noise, 0, 255).astype(np.uint8)
        return flat, base
    # lift toward the bright frosted body color (avoid muddy shadow median)
    crop = np.clip(crop * 1.12, 0, 255)
    flat = crop.reshape(-1, 3).astype(np.uint8)
    return flat, med


def _sample_bark(img):
    best = None
    for (x0, x1, y0, y1) in [(0.485, 0.515, 0.82, 0.86),
                             (0.49, 0.51, 0.84, 0.88),
                             (0.48, 0.52, 0.83, 0.87)]:
        crop = _crop_frac(img, x0, x1, y0, y1)
        med = np.median(crop.reshape(-1, 3), axis=0)
        if _looks_like_bark(med):
            best = med if best is None else (best + med) * 0.5
    if best is None:
        return TRUNK_RGB.astype(np.float64)
    return np.clip(best, 0, 255)


# =============================================================================
# TILEABLE NOISE + SWATCH HELPERS
# =============================================================================

def _value_noise(h, w, cy, cx, rng):
    low = (rng.random((cy, cx)) * 255).astype(np.uint8)
    im = Image.fromarray(low).resize((w, h), Image.BICUBIC)
    return np.asarray(im, dtype=np.float64) / 255.0


def _mirror_tile(arr, out_size):
    a = np.concatenate([arr, arr[:, ::-1]], axis=1)
    a = np.concatenate([a, a[::-1, :]], axis=0)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    im = im.resize((out_size, out_size), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64)


def _normal_from_albedo(arr, strength=2.2):
    lum = arr.mean(axis=2) / 255.0
    height = 1.0 - lum
    gy, gx = np.gradient(height)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(nx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    out = np.empty(arr.shape, dtype=np.float64)
    out[..., 0] = (nx / norm * 0.5 + 0.5) * 255.0
    out[..., 1] = (ny / norm * 0.5 + 0.5) * 255.0
    out[..., 2] = (nz / norm * 0.5 + 0.5) * 255.0
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


# =============================================================================
# FOLIAGE ATLAS  (4x4 distinct cluster tiles, drawn needle silhouettes)
# =============================================================================

ATLAS_GRID = 4
ATLAS_RES = 1024
TILE_RES = ATLAS_RES // ATLAS_GRID      # 256
SS = 4                                   # supersample factor


def _draw_cluster_tile(swatch_flat, rng, light_mul, warm):
    C = TILE_RES * SS
    img = Image.new("RGBA", (C, C), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cen = np.array([C * 0.5, C * 0.5])
    P = len(swatch_flat)

    n_sprig = int(rng.integers(3, 6))      # denser, fuller clusters
    for _ in range(n_sprig):
        ang = rng.uniform(0.0, 2.0 * np.pi)
        dvec = np.array([np.cos(ang), np.sin(ang)])
        perp = np.array([-dvec[1], dvec[0]])
        L = C * rng.uniform(0.66, 0.90)
        start = cen - dvec * (L * 0.5) + perp * rng.uniform(-0.10, 0.10) * C
        steps = int(rng.integers(30, 46))
        base_nl = C * rng.uniform(0.18, 0.27)
        halfw = C * 0.011
        for i in range(steps):
            t = (i + 0.5) / steps
            p = start + dvec * (L * t)
            taper = 0.45 + 0.62 * np.sin(np.pi * t)
            for side in (-1.0, 1.0):
                jit = rng.uniform(-0.30, 0.30)
                ndir = perp * side + dvec * (0.32 + jit)
                ndir = ndir / (np.linalg.norm(ndir) + 1e-9)
                nl = base_nl * taper * float(rng.lognormal(0.0, 0.26))
                tip = p + ndir * nl
                col = swatch_flat[int(rng.integers(P))].astype(np.float64)
                col = col * light_mul + warm + rng.normal(0.0, 6.0, size=3)
                col = tuple(int(c) for c in np.clip(col, 0, 255))
                b1 = p + perp * halfw
                b2 = p - perp * halfw
                d.polygon([tuple(b1), tuple(b2), tuple(tip)],
                          fill=(col[0], col[1], col[2], 255))
    return img.resize((TILE_RES, TILE_RES), Image.LANCZOS)


def _make_foliage_atlas(swatch_flat, rng):
    atlas = Image.new("RGBA", (ATLAS_RES, ATLAS_RES), (0, 0, 0, 0))
    for row in range(ATLAS_GRID):
        for col in range(ATLAS_GRID):
            t = row / (ATLAS_GRID - 1)
            # brighter overall so the canopy reads frosted, not muddy
            light_mul = (1.0 - t) * 1.28 + t * 0.86
            warm = np.array([(1.0 - t) * 10.0 - t * 6.0,
                             (1.0 - t) * 4.0 - t * 2.0,
                             -(1.0 - t) * 6.0 + t * 10.0])
            tile = _draw_cluster_tile(swatch_flat, rng, light_mul, warm)
            atlas.paste(tile, (col * TILE_RES, row * TILE_RES))
    a = np.asarray(atlas)[..., 3]
    a = np.where(a > 128, 255, 0).astype(np.uint8)
    soft = np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.6)))
    out = np.asarray(atlas).copy()
    out[..., 3] = soft
    return Image.fromarray(out, "RGBA")


# =============================================================================
# BARK ALBEDO
# =============================================================================

def _make_bark_albedo(color, rng, res=512):
    h = w = res
    base = np.clip(color, 0, 255).astype(np.float64)
    streak = _value_noise(h, w, 6, 48, rng)
    blotch = _value_noise(h, w, 16, 10, rng)
    fine = _value_noise(h, w, 80, 80, rng)
    lum = 0.85 + 0.30 * streak + 0.18 * (blotch - 0.5) + 0.10 * (fine - 0.5)
    lum = np.clip(lum, 0.65, 1.35)
    arr = base[None, None, :] * lum[..., None]
    redshift = np.clip((streak - 0.6) * 3.0, 0.0, 1.0)[..., None]
    arr = arr + redshift * np.array([12.0, 3.0, -3.0])[None, None, :]
    arr = np.clip(arr, 0, 255)
    tiled = _mirror_tile(arr, res)
    return Image.fromarray(tiled.astype(np.uint8), "RGB")


# =============================================================================
# UVs + VERTEX TINTS
# =============================================================================

_ROT_UV = {
    0: np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
    1: np.array([[0, 1], [0, 0], [1, 0], [1, 1]], dtype=float),
    2: np.array([[1, 1], [0, 1], [0, 0], [1, 0]], dtype=float),
    3: np.array([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=float),
}


def _canopy_uv_and_tint(verts, rng):
    n_cards = len(verts) // 4
    uv = np.zeros((len(verts), 2), dtype=np.float64)
    cell = 1.0 / ATLAS_GRID
    margin = 0.03
    tcol = rng.integers(0, ATLAS_GRID, size=n_cards)
    trow = rng.integers(0, ATLAS_GRID, size=n_cards)
    rot = rng.integers(0, 4, size=n_cards)
    for k in range(n_cards):
        loc = _ROT_UV[int(rot[k])]
        u = (tcol[k] + margin + loc[:, 0] * (1.0 - 2.0 * margin)) * cell
        v = (trow[k] + margin + loc[:, 1] * (1.0 - 2.0 * margin)) * cell
        uv[4 * k:4 * k + 4, 0] = u
        uv[4 * k:4 * k + 4, 1] = v

    y = verts[:, 1]
    y0 = BARE_TRUNK_FRAC * TREE_HEIGHT
    crown_h = TREE_HEIGHT * (1.0 - BARE_TRUNK_FRAC) - APEX_BARE_FRAC * TREE_HEIGHT
    u_h = np.clip((y - y0) / crown_h, 0.0, 1.0)
    rad = np.hypot(verts[:, 0], verts[:, 2])
    rmax = CROWN_BASE_RADIUS * np.power(np.maximum(1.0 - u_h, 0.0), CONE_POWER)
    radfrac = np.clip(rad / (rmax + 1e-4), 0.0, 1.0)
    card_jit = np.repeat(rng.normal(0.0, 0.045, size=n_cards), 4)
    # brighter, higher floor -> frosted look, outer/top brightest
    b = np.clip(0.82 + 0.24 * u_h + 0.12 * radfrac + card_jit, 0.7, 1.18)
    cool = np.array([248.0, 252.0, 250.0])
    tint = np.empty((len(verts), 4), dtype=np.uint8)
    tint[:, 0] = np.clip(cool[0] * b, 0, 255)
    tint[:, 1] = np.clip(cool[1] * b, 0, 255)
    tint[:, 2] = np.clip(cool[2] * np.minimum(1.05, b + 0.05), 0, 255)
    tint[:, 3] = 255
    return uv, tint


def _trunk_uv_and_tint(verts, rng):
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    u = (np.arctan2(z, x) / (2.0 * np.pi)) + 0.5
    v = (y / TREE_HEIGHT) * 3.0
    uv = np.stack([u, v], axis=1)

    ynorm = np.clip(y / TREE_HEIGHT, 0.0, 1.0)
    ao = 0.78 + 0.22 * np.clip(ynorm / 0.1, 0.0, 1.0)
    jit = rng.normal(0.0, 0.04, size=len(verts))
    b = np.clip(ao + jit, 0.6, 1.1)
    warm = np.array([242.0, 232.0, 222.0])
    tint = np.empty((len(verts), 4), dtype=np.uint8)
    tint[:, 0] = np.clip(warm[0] * b, 0, 255)
    tint[:, 1] = np.clip(warm[1] * b, 0, 255)
    tint[:, 2] = np.clip(warm[2] * b, 0, 255)
    tint[:, 3] = 255
    return uv, tint


# =============================================================================
# ASSEMBLE TEXTURED SCENE
# =============================================================================

def build_textured_scene(image_path, seed, density):
    rng = np.random.default_rng(seed)
    img = _load_image(image_path)

    swatch_flat, _ = _sample_foliage(img, rng)
    bark_color = _sample_bark(img)

    atlas = _make_foliage_atlas(swatch_flat, rng)
    bark_albedo = _make_bark_albedo(bark_color, rng, res=512)
    bark_normal = _normal_from_albedo(np.asarray(bark_albedo, dtype=np.float64))

    geo_scene = build_mesh(seed, density)
    trunk = geo_scene.geometry["trunk"]
    canopy = geo_scene.geometry["canopy"]

    canopy_mat = PBRMaterial(
        name="canopy",
        baseColorTexture=atlas,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0,
        roughnessFactor=0.8,
        alphaMode="MASK",
        alphaCutoff=0.45,
        doubleSided=True,
    )
    c_uv, c_tint = _canopy_uv_and_tint(canopy.vertices, rng)
    canopy.visual = TextureVisuals(uv=c_uv, material=canopy_mat)
    canopy.visual.vertex_attributes["color"] = c_tint

    trunk_mat = PBRMaterial(
        name="trunk",
        baseColorTexture=bark_albedo,
        normalTexture=bark_normal,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0,
        roughnessFactor=0.9,
        doubleSided=False,
    )
    t_uv, t_tint = _trunk_uv_and_tint(trunk.vertices, rng)
    trunk.visual = TextureVisuals(uv=t_uv, material=trunk_mat)
    trunk.visual.vertex_attributes["color"] = t_tint

    scene = trimesh.Scene()
    scene.add_geometry(trunk, geom_name="trunk")
    scene.add_geometry(canopy, geom_name="canopy")
    return scene


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Procedural spire-conifer -> textured GLB")
    parser.add_argument("--image", required=True, help="reference photo path")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--density", choices=["high", "med", "low"],
                        default="high")
    parser.add_argument("--output", required=True, help="output .glb path")
    args = parser.parse_args()

    try:
        scene = build_textured_scene(args.image, args.seed, args.density)
        glb = scene.export(file_type="glb")
        with open(args.output, "wb") as f:
            f.write(glb)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())