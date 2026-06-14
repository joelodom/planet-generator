#!/usr/bin/env python3
"""
Procedural low spreading groundcover juniper: geometry + photo-derived,
tileable materials, UV'd by surface type, exported as a textured GLB.

Surfaces (semantic geometry names):  "trunk", "branches", "canopy".
  - canopy : clumped leaf cards mapped onto a 4x4 foliage atlas (MASK alpha)
  - trunk / branches : tapered cylinders with cylindrical bark UVs + normals

CLI:
  python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb

Only numpy, trimesh, PIL and the stdlib are used. Deterministic given --seed.
"""

import argparse
import sys

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial
from PIL import Image, ImageDraw, ImageFilter


# ===========================================================================
# GEOMETRY  (build_mesh: +Y up, base at y=0, meters)
# ===========================================================================

# Proportions retuned to the photo: narrower + a touch taller mound so the
# front aspect lands near ~1.7 (was 2.0), and a denser, rounder cushion.
CROWN_WIDTH = 1.10              # meters: overall spread in X (the chosen size)
DEPTH_OVER_WIDTH = 0.88         # nearly as deep as wide -> rounded plan
HEIGHT_OVER_WIDTH = 0.58        # mounded cushion (front w/h ~1.7)
CROWN_DEPTH = CROWN_WIDTH * DEPTH_OVER_WIDTH
HEIGHT = CROWN_WIDTH * HEIGHT_OVER_WIDTH

RX = CROWN_WIDTH * 0.5          # horizontal envelope semi-axis (X)
RZ = CROWN_DEPTH * 0.5          # horizontal envelope semi-axis (Z)

ENV_CX = 0.06 * CROWN_WIDTH     # crown peak nudged off-center (+X), per image
ENV_CZ = 0.0
THETA_MIN = np.deg2rad(6.0)     # near the top of the mound
THETA_MAX = np.deg2rad(112.0)   # rim, just below equator -> tidy spreading skirt
RIM_Y = 0.10 * HEIGHT           # rim drapes close to the ground
# Solve top = ENV_CY + RY  and  rim = ENV_CY + RY*cos(THETA_MAX):
RY = (HEIGHT - RIM_Y) / (1.0 - np.cos(THETA_MAX))
ENV_CY = HEIGHT - RY

CARD_ASPECT = 1.4               # leaf sprigs read a bit longer than wide

# Density presets. Far more clumps + interior fill than before so the canopy
# reads as a SOLID mound rather than a see-through shell. cards stay <=~3500.
_PRESETS = {
    "high": dict(trunk_sides=14, branch_sides=8, n_limbs=6,
                 n_surf=24, n_int=16, cards_per=(70, 106), base_half=0.070),
    "med":  dict(trunk_sides=10, branch_sides=6, n_limbs=5,
                 n_surf=14, n_int=8,  cards_per=(45, 72),  base_half=0.082),
    "low":  dict(trunk_sides=6,  branch_sides=5, n_limbs=4,
                 n_surf=9,  n_int=4,  cards_per=(26, 42),  base_half=0.098),
}


def _norm_rows(v):
    """Row-wise normalize an (N,3) array, guarding against zero length."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return v / n


def _perp_basis(axis):
    """Two unit vectors spanning the plane perpendicular to `axis`."""
    a = axis / (np.linalg.norm(axis) + 1e-12)
    ref = np.array([0.0, 1.0, 0.0]) if abs(a[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, ref)
    u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(a, u)
    return u, v


def _segment(p0, p1, r0, r1, sides):
    """A capped tapered cylinder from p0 (radius r0) to p1 (radius r1)."""
    u, v = _perp_basis(p1 - p0)
    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    circ = np.cos(ang)[:, None] * u[None, :] + np.sin(ang)[:, None] * v[None, :]
    ring0 = p0 + r0 * circ
    ring1 = p1 + r1 * circ
    verts = np.vstack([ring0, ring1, p0[None, :], p1[None, :]])
    c0, c1 = 2 * sides, 2 * sides + 1
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces.append([i, sides + i, sides + j])
        faces.append([i, sides + j, j])
        faces.append([c0, j, i])                 # bottom cap
        faces.append([c1, sides + i, sides + j])  # top cap
    return verts, np.array(faces, dtype=np.int64)


def _merge(parts):
    """Merge a list of (verts, faces) into one (verts, faces) with offsets."""
    V, F, off = [], [], 0
    for v, f in parts:
        V.append(v)
        F.append(f + off)
        off += len(v)
    return np.vstack(V), np.vstack(F)


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    if density not in _PRESETS:
        density = "high"
    p = _PRESETS[density]
    rng = np.random.default_rng(seed)

    env_center = np.array([ENV_CX, ENV_CY, ENV_CZ])

    # --- Foliage envelope: flattened ellipsoid + a few low-frequency lobes ---
    n_lobes = int(rng.integers(3, 6))
    lobe_freq = rng.integers(2, 5, n_lobes).astype(float)   # azimuthal waviness
    lobe_phase = rng.uniform(0.0, 2.0 * np.pi, n_lobes)
    lobe_amp = rng.uniform(0.03, 0.07, n_lobes)             # gentle -> rounder rim
    s = lobe_amp.sum()
    if s > 0.20:                                            # keep silhouette clean
        lobe_amp *= 0.20 / s

    def envelope_point(theta, phi):
        st, ct = np.sin(theta), np.cos(theta)
        ls = 1.0 + st * np.sum(lobe_amp * np.cos(lobe_freq * phi + lobe_phase))
        x = ENV_CX + RX * ls * st * np.cos(phi)
        z = ENV_CZ + RZ * ls * st * np.sin(phi)
        y = ENV_CY + RY * ct
        return np.array([x, y, z])

    def clamp_inside(pts):
        """Pull points back onto/just-inside the ellipsoid shell (no spikes)."""
        d = pts - env_center
        q = np.sqrt((d[:, 0] / RX) ** 2 + (d[:, 1] / RY) ** 2 + (d[:, 2] / RZ) ** 2)
        fac = np.minimum(1.0, 0.99 / np.maximum(q, 1e-9))
        return env_center + d * fac[:, None]

    # --- Clump centers: dense shell coverage + a filled interior + core -----
    clump_radius = 0.14 * CROWN_WIDTH
    clump_centers = []
    cos_lo, cos_hi = np.cos(THETA_MAX), np.cos(THETA_MIN)

    for _ in range(p["n_surf"]):
        theta = np.arccos(rng.uniform(cos_lo, cos_hi))       # area-weighted skirt
        phi = rng.uniform(0.0, 2.0 * np.pi)
        shell = envelope_point(theta, phi)
        clump_centers.append((env_center + 0.86 * (shell - env_center), 0.6))
    for _ in range(p["n_int"]):                              # fill the volume
        theta = np.arccos(rng.uniform(np.cos(THETA_MAX), cos_hi))
        phi = rng.uniform(0.0, 2.0 * np.pi)
        shell = envelope_point(theta, phi)
        rf = rng.uniform(0.0, 0.78)
        clump_centers.append((env_center + rf * (shell - env_center), 0.7))
    clump_centers.append((env_center.copy(), 0.8))           # central core

    # --- Spawn leaf-card centers inside each clump --------------------------
    centers_list, half_list = [], []
    lo, hi = p["cards_per"]
    for center, ysquash in clump_centers:
        k = int(rng.integers(lo, hi))
        off = rng.normal(0.0, 0.40 * clump_radius, size=(k, 3))
        off[:, 1] *= ysquash                                 # flatter clumps
        d = np.linalg.norm(off, axis=1, keepdims=True)
        scale = np.minimum(1.0, clump_radius / np.maximum(d, 1e-9))
        cc = center[None, :] + off * scale
        cc = clamp_inside(cc)                                # tidy rounded mound
        centers_list.append(cc)
        hx = p["base_half"] * np.exp(rng.normal(0.0, 0.22, size=k))
        hx = np.clip(hx, 0.6 * p["base_half"], 1.5 * p["base_half"])
        half_list.append(hx)

    centers = np.vstack(centers_list)
    hx = np.concatenate(half_list)[:, None]
    hy = hx / CARD_ASPECT
    M = len(centers)

    # --- Orient each card tangent to the shell, normal outward (+jitter) -----
    radial = centers - env_center
    r = _norm_rows(radial)
    ref = np.tile(np.array([0.0, 1.0, 0.0]), (M, 1))
    ref[np.abs(r[:, 1]) > 0.9] = np.array([1.0, 0.0, 0.0])
    a = _norm_rows(np.cross(r, ref))
    b = np.cross(r, a)
    ja = rng.uniform(-0.38, 0.38, (M, 1))                    # ~+/-20deg normal jitter
    jb = rng.uniform(-0.38, 0.38, (M, 1))
    n = _norm_rows(r + ja * a + jb * b)

    ref2 = np.tile(np.array([0.0, 1.0, 0.0]), (M, 1))
    ref2[np.abs(n[:, 1]) > 0.9] = np.array([1.0, 0.0, 0.0])
    a2 = _norm_rows(np.cross(n, ref2))
    b2 = np.cross(n, a2)
    psi = rng.uniform(0.0, 2.0 * np.pi, (M, 1))              # random in-plane spin
    t1 = np.cos(psi) * a2 + np.sin(psi) * b2
    t2 = -np.sin(psi) * a2 + np.cos(psi) * b2

    v0 = centers + (-t1 * hx - t2 * hy)
    v1 = centers + (t1 * hx - t2 * hy)
    v2 = centers + (t1 * hx + t2 * hy)
    v3 = centers + (-t1 * hx + t2 * hy)
    cverts = np.stack([v0, v1, v2, v3], axis=1).reshape(-1, 3)
    idx = np.arange(M) * 4
    cfaces = np.concatenate(
        [np.stack([idx, idx + 1, idx + 2], axis=1),
         np.stack([idx, idx + 2, idx + 3], axis=1)], axis=0).astype(np.int64)
    canopy = trimesh.Trimesh(vertices=cverts, faces=cfaces, process=False)

    # --- Wood: short flared trunk + low, BURIED radiating limbs -------------
    r_trunk = 0.038
    trunk_base = np.array([ENV_CX * 0.25, 0.0, ENV_CZ])
    trunk_top = trunk_base + np.array([0.0, 0.12 * HEIGHT, 0.0])
    trunk_parts = [_segment(trunk_base, trunk_top, r_trunk * 1.45, r_trunk,
                            p["trunk_sides"])]
    trunk_mesh = trimesh.Trimesh(*_merge(trunk_parts), process=True)

    branch_parts = []
    bs = p["branch_sides"]
    for i in range(p["n_limbs"]):
        theta = np.deg2rad(rng.uniform(92.0, 112.0))         # low + outward
        phi = (2.0 * np.pi * i / p["n_limbs"]) + rng.uniform(-0.4, 0.4)
        # keep limb tips well inside the foliage so wood stays hidden
        tip = env_center + 0.58 * (envelope_point(theta, phi) - env_center)
        mid = 0.5 * (trunk_top + tip) + np.array([0.0, 0.05 * HEIGHT, 0.0])
        r0, rm, r1 = r_trunk * 0.80, r_trunk * 0.50, r_trunk * 0.26
        branch_parts.append(_segment(trunk_top, mid, r0, rm, bs))
        branch_parts.append(_segment(mid, tip, rm, r1, bs))
        r_child = r1 / np.sqrt(2.0)                           # r_p^2 = sum r_c^2
        for _ in range(2):
            ct = np.deg2rad(np.rad2deg(theta) - rng.uniform(6.0, 20.0))
            cp = phi + rng.uniform(-0.4, 0.4)
            ctip = env_center + 0.72 * (envelope_point(ct, cp) - env_center)
            branch_parts.append(_segment(tip, ctip, r_child, r_child * 0.30, bs))
    branch_mesh = trimesh.Trimesh(*_merge(branch_parts), process=True)

    # --- Recenter in X/Z, drop lowest point to y=0 --------------------------
    meshes = [trunk_mesh, branch_mesh, canopy]
    allv = np.vstack([m.vertices for m in meshes])
    cx = 0.5 * (allv[:, 0].min() + allv[:, 0].max())
    cz = 0.5 * (allv[:, 2].min() + allv[:, 2].max())
    ymin = allv[:, 1].min()
    trans = np.array([-cx, -ymin, -cz])
    for m in meshes:
        m.apply_translation(trans)

    canopy.fix_normals()

    scene = trimesh.Scene()
    scene.add_geometry(trunk_mesh, geom_name="trunk")
    scene.add_geometry(branch_mesh, geom_name="branches")
    scene.add_geometry(canopy, geom_name="canopy")
    return scene


# ===========================================================================
# PHOTO SAMPLING  (sample real colors from WELL INSIDE the silhouette)
# ===========================================================================

# Foliage boxes biased toward the UPPER, sunlit, frosted blue-green (the pale
# glaucous tone that defines this shrub) plus a couple of mid-canopy patches.
# Wood boxes sit on the warm reddish base peeking out low-center. None touch
# the grey backdrop.
_FOLIAGE_BOXES = [
    (0.30, 0.34, 0.42, 0.45), (0.45, 0.32, 0.58, 0.43),
    (0.58, 0.36, 0.70, 0.47), (0.36, 0.44, 0.48, 0.54),
    (0.52, 0.44, 0.64, 0.54), (0.24, 0.46, 0.34, 0.56),
    (0.66, 0.46, 0.76, 0.56), (0.42, 0.38, 0.52, 0.48),
]
_WOOD_BOXES = [
    (0.40, 0.70, 0.50, 0.80), (0.50, 0.72, 0.60, 0.82),
    (0.45, 0.66, 0.55, 0.74), (0.55, 0.70, 0.63, 0.78),
]


def _load_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64)


def _bg_color(arr):
    """Robust background colour from the four image corners."""
    h, w = arr.shape[:2]
    s = max(4, min(h, w) // 20)
    corners = np.vstack([
        arr[:s, :s].reshape(-1, 3), arr[:s, -s:].reshape(-1, 3),
        arr[-s:, :s].reshape(-1, 3), arr[-s:, -s:].reshape(-1, 3),
    ])
    return np.median(corners, axis=0)


def _box_patches(arr, box, grid=4):
    """Median colour of each grid cell inside a normalized box."""
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = box
    xa, xb = int(x0 * w), int(x1 * w)
    ya, yb = int(y0 * h), int(y1 * h)
    xb = max(xb, xa + grid)
    yb = max(yb, ya + grid)
    region = arr[ya:yb, xa:xb]
    out = []
    rh, rw = region.shape[:2]
    for i in range(grid):
        for j in range(grid):
            cy0, cy1 = i * rh // grid, (i + 1) * rh // grid
            cx0, cx1 = j * rw // grid, (j + 1) * rw // grid
            cell = region[cy0:max(cy1, cy0 + 1), cx0:max(cx1, cx0 + 1)]
            if cell.size:
                out.append(np.median(cell.reshape(-1, 3), axis=0))
    return out


def _sample_colors(arr, boxes, bg, warm_bias=False):
    """Collect interior patches, drop background-coloured ones, return median."""
    cand = []
    for box in boxes:
        cand.extend(_box_patches(arr, box))
    cand = np.array(cand)
    keep = np.linalg.norm(cand - bg[None, :], axis=1) > 16.0     # discard backdrop
    cand = cand[keep] if keep.any() else cand
    if warm_bias and len(cand) > 2:
        warmth = cand[:, 0] - cand[:, 2]                          # reddish-brown
        cand = cand[warmth >= np.median(warmth)]
    return np.median(cand, axis=0)


def _glaucous(foliage_rgb):
    """Anchor on the sampled hue but push to the pale, frosted silvery sage
    that defines this juniper: desaturate, nudge cooler/bluer, lift value."""
    lum = float(foliage_rgb @ np.array([0.299, 0.587, 0.114]))
    desat = foliage_rgb + (np.array([lum, lum, lum]) - foliage_rgb) * 0.45
    cool = desat + np.array([-5.0, 2.0, 12.0])                   # toward blue-green
    return np.clip(cool * 1.12 + 16.0, 0, 255)                   # frosted lift


# ===========================================================================
# TEXTURE SYNTHESIS
# ===========================================================================

def _delight(albedo, clamp=(0.6, 1.6)):
    """Divide by a wrap-safe heavily-blurred luminance; clamp the gain."""
    h, w = albedo.shape[:2]
    lum = albedo @ np.array([0.2126, 0.7152, 0.0722])
    tiled = np.tile(lum, (3, 3)).astype(np.uint8)
    blur = Image.fromarray(tiled).filter(ImageFilter.GaussianBlur(max(h, w) / 8.0))
    blur = np.asarray(blur, dtype=np.float64)[h:2 * h, w:2 * w] + 1e-3
    gain = np.clip(lum.mean() / blur, clamp[0], clamp[1])
    return np.clip(albedo * gain[..., None], 0, 255)


def _lerp(c0, c1, t):
    return tuple(int(round(c0[k] + (c1[k] - c0[k]) * t)) for k in range(3))


def _adjust(base, brightness, warmth):
    """Tint a foliage colour: warmth>1 -> warmer; <1 -> cooler bluer."""
    r = base[0] * brightness * warmth
    g = base[1] * brightness
    b = base[2] * brightness * (0.6 + 0.4 / warmth)
    return np.clip([r, g, b], 0, 255)


def _draw_sprig(draw, base, ang_deg, length, c_base, c_tip, rng):
    """A feathery juniper sprig: a stem of paired pointed needles."""
    ang = np.deg2rad(ang_deg)
    d = np.array([np.cos(ang), np.sin(ang)])
    steps = int(rng.integers(6, 11))
    for s in range(steps):
        t = s / steps
        pos = base + d * (length * t)
        nl = length * 0.30 * (1.0 - 0.45 * t)
        col = _lerp(c_base, c_tip, t)
        for side in (1.0, -1.0):
            na = ang + side * np.deg2rad(rng.uniform(28, 46))
            nd = np.array([np.cos(na), np.sin(na)])
            tip = pos + nd * nl
            perp = np.array([-nd[1], nd[0]]) * (nl * 0.16 + 1.0)
            draw.polygon([tuple(pos + perp), tuple(pos - perp), tuple(tip)],
                         fill=col + (255,))
    tip = base + d * length
    perp = np.array([-d[1], d[0]]) * (length * 0.05 + 1.0)
    draw.polygon([tuple(base + d * length * 0.7 + perp),
                  tuple(base + d * length * 0.7 - perp), tuple(tip)],
                 fill=c_tip + (255,))


def make_foliage_atlas(foliage_rgb, rng, tile=256, grid=4, ss=4):
    """4x4 atlas of distinct foliage clusters; binary alpha, AA edges only."""
    S = tile * ss
    atlas = Image.new("RGBA", (tile * grid, tile * grid), (0, 0, 0, 0))
    for row in range(grid):
        for col in range(grid):
            # sunlit (top rows) brighter & warmer; shaded (lower) cooler.
            # Overall pale/frosted: high base brightness, blue-lifted tips.
            brightness = float(np.clip(1.18 - 0.10 * row + rng.uniform(-0.05, 0.05),
                                       0.82, 1.32))
            warmth = float(1.05 - 0.045 * row)
            tcol = _adjust(foliage_rgb, brightness, warmth)
            c_base = tuple(int(v) for v in np.clip(tcol * 0.80, 0, 255))
            c_tip = tuple(int(v) for v in np.clip(tcol * 1.16 + np.array([8, 12, 18]),
                                                  0, 255))
            tile_img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
            d = ImageDraw.Draw(tile_img)
            for _ in range(int(rng.integers(2, 4))):           # clusters per tile
                cx = rng.uniform(0.25, 0.75) * S
                cy = rng.uniform(0.55, 0.88) * S               # grow upward
                for _ in range(int(rng.integers(5, 9))):
                    ang = -90.0 + rng.uniform(-58, 58)         # mostly upward
                    length = rng.uniform(0.28, 0.50) * S
                    _draw_sprig(d, np.array([cx, cy]), ang, length,
                                c_base, c_tip, rng)
            tile_img = tile_img.resize((tile, tile), Image.LANCZOS)
            ta = np.asarray(tile_img).copy()
            ta[..., 3] = np.where(ta[..., 3] < 40, 0, ta[..., 3])  # binary alpha
            tile_img = Image.fromarray(ta, "RGBA")
            atlas.paste(tile_img, (col * tile, row * tile))
    return atlas


def make_bark(wood_rgb, rng, size=512):
    """Seamless reddish-brown fibrous bark from integer-frequency periodic noise."""
    xs = np.arange(size) / size
    X, Y = np.meshgrid(xs, xs)
    grain = np.zeros((size, size))
    for fx, amp in [(8, 0.50), (13, 0.30), (21, 0.20), (34, 0.12)]:
        wob = 0.35 * np.sin(2 * np.pi * 2 * Y + rng.uniform(0, 2 * np.pi))
        grain += amp * np.sin(2 * np.pi * fx * X + rng.uniform(0, 2 * np.pi) + wob)
    for f, amp in [(2, 0.25), (3, 0.18)]:
        grain += amp * np.sin(2 * np.pi * f * X + rng.uniform(0, 2 * np.pi)) \
                     * np.sin(2 * np.pi * f * Y + rng.uniform(0, 2 * np.pi))
    grain += 0.10 * np.sin(2 * np.pi * 48 * X + rng.uniform(0, 2 * np.pi)) \
                  * np.sin(2 * np.pi * 40 * Y + rng.uniform(0, 2 * np.pi))
    grain = (grain - grain.min()) / (np.ptp(grain) + 1e-9)
    val = 0.5 + 1.0 * grain                                     # lightest ~3x darkest
    base = np.array(wood_rgb, dtype=np.float64)
    albedo = base[None, None, :] * val[..., None]
    albedo += np.clip(val - 1.0, 0, None)[..., None] * np.array([28.0, 8.0, -6.0])
    albedo = np.clip(albedo, 0, 255)
    albedo = _delight(albedo)
    return Image.fromarray(albedo.astype(np.uint8), "RGB")


def make_normal(albedo_img, strength=2.2):
    """Tangent-space normal map from inverse-luminance height (wrap-seamless)."""
    arr = np.asarray(albedo_img, dtype=np.float64) / 255.0
    lum = arr @ np.array([0.2126, 0.7152, 0.0722])
    height = 1.0 - lum
    gx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) * strength
    gy = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) * strength
    nz = np.ones_like(height)
    nrm = np.stack([-gx, -gy, nz], axis=-1)
    nrm /= (np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9)
    out = ((nrm * 0.5 + 0.5) * 255.0).astype(np.uint8)
    return Image.fromarray(out, "RGB")


# ===========================================================================
# UVs + VERTEX COLOURS + MATERIAL ATTACH
# ===========================================================================

def _cyl_uv(V, u_tiles, v_tiles):
    """Cylindrical UVs about the part's vertical axis (bark wrap)."""
    cx = 0.5 * (V[:, 0].min() + V[:, 0].max())
    cz = 0.5 * (V[:, 2].min() + V[:, 2].max())
    ang = np.arctan2(V[:, 2] - cz, V[:, 0] - cx)
    u = (ang / (2 * np.pi) + 0.5) * u_tiles
    ymin, ymax = V[:, 1].min(), V[:, 1].max()
    v = ((V[:, 1] - ymin) / (ymax - ymin + 1e-9)) * v_tiles
    return np.column_stack([u, v]).astype(np.float64)


def _canopy_uv(n_cards, rng):
    """Map each card quad onto a random atlas tile with a random 90deg turn."""
    base = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    tiles = rng.integers(0, 16, n_cards)
    rots = rng.integers(0, 4, n_cards)
    uv = np.zeros((4 * n_cards, 2), dtype=np.float64)
    for i in range(n_cards):
        c, r = int(tiles[i] % 4), int(tiles[i] // 4)
        u0, v0 = c * 0.25, 1.0 - (r + 1) * 0.25
        corners = np.roll(base, int(rots[i]), axis=0)
        uv[4 * i:4 * i + 4, 0] = u0 + corners[:, 0] * 0.25
        uv[4 * i:4 * i + 4, 1] = v0 + corners[:, 1] * 0.25
    return uv


def _canopy_colors(V, rng):
    """Sun/shade COLOR_0: top & outer brighter, inner & low slightly darker/
    cooler. Floor kept HIGH so the pale frosted tone is preserved."""
    ymax = V[:, 1].max()
    hnorm = np.clip(V[:, 1] / (ymax + 1e-9), 0, 1)
    cx = 0.5 * (V[:, 0].min() + V[:, 0].max())
    cz = 0.5 * (V[:, 2].min() + V[:, 2].max())
    rad = np.sqrt((V[:, 0] - cx) ** 2 + (V[:, 2] - cz) ** 2)
    radn = rad / (rad.max() + 1e-9)
    bright = 0.80 + 0.20 * hnorm + 0.10 * radn
    bright = np.clip(bright + rng.normal(0, 0.045, len(V)), 0.62, 1.18)
    r = 255 * bright * np.clip(0.99 + 0.06 * hnorm, 0, 1.12)
    g = 255 * bright
    b = 255 * bright * (1.06 - 0.08 * hnorm)               # interior/low cooler
    col = np.clip(np.column_stack([r, g, b]), 0, 255).astype(np.uint8)
    return np.column_stack([col, np.full(len(V), 255, np.uint8)])


def _wood_colors(V, rng):
    """Subtle AO: darker near the ground / in crevices, warm undertone."""
    ymax = V[:, 1].max()
    ao = 0.55 + 0.45 * np.clip(V[:, 1] / (ymax + 1e-9), 0, 1)
    ao = np.clip(ao + rng.normal(0, 0.03, len(V)), 0.4, 1.0)
    r = 255 * ao * 1.00
    g = 255 * ao * 0.95
    b = 255 * ao * 0.88
    col = np.clip(np.column_stack([r, g, b]), 0, 255).astype(np.uint8)
    return np.column_stack([col, np.full(len(V), 255, np.uint8)])


def _attach(mesh, uv, material, colors):
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    mesh.visual.vertex_attributes["color"] = colors


def texture_scene(scene, image_path, seed):
    """Derive materials from the photo and attach UVs/materials per surface."""
    rng = np.random.default_rng(seed + 12345)               # deterministic swatches
    arr = _load_rgb(image_path)
    bg = _bg_color(arr)
    foliage_rgb = _glaucous(_sample_colors(arr, _FOLIAGE_BOXES, bg))
    wood_rgb = _sample_colors(arr, _WOOD_BOXES, bg, warm_bias=True)
    if not np.isfinite(wood_rgb).all():                     # safety fallback
        wood_rgb = np.clip(foliage_rgb * np.array([1.3, 0.9, 0.7]), 0, 255)

    # --- textures ---
    atlas = make_foliage_atlas(foliage_rgb, rng).convert("RGBA")
    bark = make_bark(wood_rgb, rng, size=512)
    bark_n = make_normal(bark, strength=2.2)

    # --- foliage / canopy ---
    fol_mat = PBRMaterial(name="foliage", baseColorTexture=atlas,
                          baseColorFactor=[255, 255, 255, 255],
                          metallicFactor=0.0, roughnessFactor=0.85)
    fol_mat.alphaMode = "MASK"
    fol_mat.alphaCutoff = 0.45
    fol_mat.doubleSided = True

    canopy = scene.geometry["canopy"]
    n_cards = len(canopy.vertices) // 4
    _attach(canopy, _canopy_uv(n_cards, rng), fol_mat, _canopy_colors(canopy.vertices, rng))

    # --- wood (trunk + branches share the bark look, own UV scale) ---
    for name, u_t, v_t in (("trunk", 3.0, 2.0), ("branches", 4.0, 5.0)):
        mesh = scene.geometry[name]
        mat = PBRMaterial(name="bark_" + name, baseColorTexture=bark,
                          normalTexture=bark_n,
                          baseColorFactor=[255, 255, 255, 255],
                          metallicFactor=0.0, roughnessFactor=0.9)
        mat.doubleSided = False
        _attach(mesh, _cyl_uv(mesh.vertices, u_t, v_t), mat, _wood_colors(mesh.vertices, rng))

    return scene


# ===========================================================================
# CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="Procedural spreading juniper -> GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    scene = build_mesh(args.seed, args.density)
    scene = texture_scene(scene, args.image, args.seed)

    glb = scene.export(file_type="glb")
    with open(args.output, "wb") as f:
        f.write(glb)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:           # exit non-zero on any failure
        sys.stderr.write("ERROR: {}\n".format(exc))
        sys.exit(1)