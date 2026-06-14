#!/usr/bin/env python3
"""
Procedural buttressed emergent rainforest tree -> textured GLB.

Slender ramrod bole with a smooth fluted buttressed foot, a clean self-pruned
shaft, and a narrow high airy crown of leaf cards.  Derives tileable bark + a
4x4 foliage atlas from a reference photo, applies per-surface UVs and PBR
materials, exports a GLB.

Only numpy + trimesh + Pillow + stdlib.  +Y up, base at y=0, meters.

CLI:
    python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb
"""

import argparse
import sys

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual import TextureVisuals
from PIL import Image, ImageDraw, ImageFilter


# ============================================================================
# GEOMETRY
# ============================================================================
TREE_HEIGHT = 30.0          # m -- plausible emergent rainforest giant

CROWN_BASE_FRAC      = 0.56   # clean bole ends / branching starts (frac of H)
CROWN_TOP_FRAC       = 0.98   # crown reaches near the top
BUTTRESS_HEIGHT_FRAC = 0.13   # fluted buttress climbs ~13% up the trunk

# Crown deliberately NARROW + tall to match the photo's slim emergent crown
# (front-view aspect target ~0.43; cards/lobes add a little beyond this).
CROWN_WIDTH_OVER_HEIGHT = 0.34

TRUNK_BASE_R_FRAC = 0.018   # bole radius just above the buttress flare
TRUNK_TOP_R_FRAC  = 0.011   # bole radius where the crown begins

# Buttress is a CONTINUOUS fluted flare folded into the trunk cross-section.
FLARE_BASE = 0.50           # smooth conical widening at the very base
LOBE_AMP   = 1.30           # extra reach of each rounded buttress ridge

CLUMP_RADIUS_FRAC = 0.11    # clump ellipsoid radius ~11% of crown width
CARD_HALF_FRAC    = 0.040   # leaf-card half-size ~4% of crown width (finer)

_DENSITY = {
    "high": dict(trunk_sides=24, bole_rings=28, branch_sides=6, n_fins=6,
                 n_primary=8, n_child=3, cards_total=3200, n_lobes=5,
                 interior_clumps=4),
    "med":  dict(trunk_sides=16, bole_rings=20, branch_sides=5, n_fins=5,
                 n_primary=6, n_child=3, cards_total=1300, n_lobes=4,
                 interior_clumps=2),
    "low":  dict(trunk_sides=10, bole_rings=12, branch_sides=4, n_fins=5,
                 n_primary=4, n_child=2, cards_total=440, n_lobes=3,
                 interior_clumps=1),
}


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.asarray(v, float)


def _rodrigues(v, axis, ang):
    axis = _norm(axis)
    c, s = np.cos(ang), np.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


def _combine(parts):
    Vs, Fs, off = [], [], 0
    for v, f in parts:
        if len(v) == 0 or len(f) == 0:
            continue
        Vs.append(v)
        Fs.append(np.asarray(f) + off)
        off += len(v)
    if not Vs:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    return np.vstack(Vs), np.vstack(Fs)


def _tube(points, radii, sides, cap_start=False, cap_end=False):
    points = np.asarray(points, float)
    radii = np.asarray(radii, float)
    n = len(points)

    tang = np.zeros((n, 3))
    for i in range(n):
        if i == 0:
            tang[i] = points[1] - points[0]
        elif i == n - 1:
            tang[i] = points[-1] - points[-2]
        else:
            tang[i] = points[i + 1] - points[i - 1]
        tang[i] = _norm(tang[i])

    normals = np.zeros((n, 3))
    ref = np.array([0, 0, 1.0]) if abs(tang[0][2]) < 0.9 else np.array([1.0, 0, 0])
    normals[0] = _norm(np.cross(tang[0], ref))
    for i in range(1, n):
        v = np.cross(tang[i - 1], tang[i])
        s = np.linalg.norm(v)
        if s < 1e-9:
            normals[i] = normals[i - 1]
        else:
            axis = v / s
            ang = np.arctan2(s, np.clip(np.dot(tang[i - 1], tang[i]), -1.0, 1.0))
            normals[i] = _rodrigues(normals[i - 1], axis, ang)
        normals[i] = _norm(normals[i] - np.dot(normals[i], tang[i]) * tang[i])

    a = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    cos, sin = np.cos(a), np.sin(a)

    rings = []
    for i in range(n):
        bn = _norm(np.cross(tang[i], normals[i]))
        ring = points[i] + radii[i] * (np.outer(cos, normals[i]) + np.outer(sin, bn))
        rings.append(ring)
    V = np.vstack(rings)

    F = []
    for i in range(n - 1):
        a0, b0 = i * sides, (i + 1) * sides
        for j in range(sides):
            j2 = (j + 1) % sides
            F.append([a0 + j, a0 + j2, b0 + j2])
            F.append([a0 + j, b0 + j2, b0 + j])
    if cap_start:
        c = len(V)
        V = np.vstack([V, points[0]])
        for j in range(sides):
            F.append([c, (j + 1) % sides, j])
    if cap_end:
        c = len(V)
        V = np.vstack([V, points[-1]])
        base = (n - 1) * sides
        for j in range(sides):
            F.append([c, base + j, base + (j + 1) % sides])
    return V, np.array(F, int)


def _build_trunk(rng, p):
    """Slender bole whose lower cross-section folds into smooth fluted
    buttress ridges that flare to the ground -- one continuous skirt."""
    sides = p["trunk_sides"]
    n_lobe = p["n_fins"]
    bole_top = CROWN_BASE_FRAC * TREE_HEIGHT
    Hb = BUTTRESS_HEIGHT_FRAC * TREE_HEIGHT
    r0 = TRUNK_BASE_R_FRAC * TREE_HEIGHT
    r1 = TRUNK_TOP_R_FRAC * TREE_HEIGHT

    nb = max(6, p["bole_rings"] // 2)            # dense rings in the buttress
    nu = max(4, p["bole_rings"] - nb)
    ys = np.concatenate([np.linspace(0.0, Hb, nb, endpoint=False),
                         np.linspace(Hb, bole_top, nu)])

    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    phase = rng.uniform(0, 2 * np.pi)
    rnoise = 1.0 + rng.normal(0, 0.01, len(ys))

    rings = []
    for i, y in enumerate(ys):
        t = y / bole_top
        base = (r0 + (r1 - r0) * t) * rnoise[i]
        if y < Hb:
            fz = (1.0 - y / Hb) ** 1.6                       # 1 ground -> 0 top
            lobe = (0.5 + 0.5 * np.cos(n_lobe * ang + phase)) ** 2.2
            flare = FLARE_BASE * fz + LOBE_AMP * fz * lobe   # rounded ridges
            r = base * (1.0 + flare)
        else:
            r = np.full(sides, base)
        rings.append(np.stack([r * np.cos(ang), np.full(sides, y),
                               r * np.sin(ang)], 1))
    V = np.vstack(rings)

    F = []
    nr = len(ys)
    for i in range(nr - 1):
        a0, b0 = i * sides, (i + 1) * sides
        for j in range(sides):
            j2 = (j + 1) % sides
            F.append([a0 + j, a0 + j2, b0 + j2])
            F.append([a0 + j, b0 + j2, b0 + j])
    c = len(V)
    V = np.vstack([V, [0.0, 0.0, 0.0]])
    for j in range(sides):
        F.append([c, (j + 1) % sides, j])
    c2 = len(V)
    V = np.vstack([V, [0.0, bole_top, 0.0]])
    base = (nr - 1) * sides
    for j in range(sides):
        F.append([c2, base + j, base + (j + 1) % sides])
    return (V, np.array(F, int)), r0, r1


def _make_lobe_fn(rng, n_lobes):
    freqs = rng.integers(2, 5, n_lobes)
    phases = rng.uniform(0, 2 * np.pi, n_lobes)
    amps = rng.uniform(0.04, 0.10, n_lobes)        # gentler, smoother envelope
    pol = rng.uniform(0, 1, n_lobes)

    def fn(d):
        d = _norm(d)
        phi = np.arctan2(d[2], d[0])
        theta = np.arccos(np.clip(d[1], -1.0, 1.0))
        s = 1.0
        for i in range(n_lobes):
            band = 0.5 + 0.5 * np.sin(theta)
            s += amps[i] * np.cos(freqs[i] * phi + phases[i]) * \
                (pol[i] + (1.0 - pol[i]) * band)
        return max(0.6, s)
    return fn


def _env_point(C, r, lobe_fn, d):
    d = _norm(d)
    return C + lobe_fn(d) * (r * d)


def _clamp_to_env(p, C, r, lobe_fn, limit=0.93):
    q = (p - C) / r
    rad = np.linalg.norm(q)
    if rad < 1e-9:
        return p
    shell = lobe_fn(q / rad)
    if rad > limit * shell:
        q = (q / rad) * limit * shell
    return C + q * r


def _branch_points(rng, A, B, nseg, bow):
    A, B = np.asarray(A, float), np.asarray(B, float)
    d = _norm(B - A)
    L = np.linalg.norm(B - A)
    ref = np.array([0, 1, 0.0]) if abs(d[1]) < 0.9 else np.array([1, 0, 0.0])
    p1 = _norm(np.cross(d, ref))
    p2 = _norm(np.cross(d, p1))
    pts = []
    for k in range(nseg + 1):
        t = k / nseg
        base = A + (B - A) * t
        if 0 < k < nseg:
            amp = bow * L * np.sin(np.pi * t)
            base = base + (p1 * rng.normal(0, 1) + p2 * rng.normal(0, 1)) * amp
        pts.append(base)
    return np.array(pts)


def _build_branches(rng, p, C, r, lobe_fn, bole_r_top):
    sides = p["branch_sides"]
    n_primary = p["n_primary"]
    n_child = p["n_child"]
    parts = []
    clumps = []

    bole_top = CROWN_BASE_FRAC * TREE_HEIGHT
    crown_top = CROWN_TOP_FRAC * TREE_HEIGHT
    origin = np.array([0.0, bole_top, 0.0])

    # central leader continuing the trunk up through the crown
    leader_top = _env_point(C, r, lobe_fn,
                            np.array([rng.uniform(-0.08, 0.08), 1.0,
                                      rng.uniform(-0.08, 0.08)]))
    lp = _branch_points(rng, origin, leader_top, 3, 0.03)
    lr = np.linspace(bole_r_top * 0.7, bole_r_top * 0.22, len(lp))
    parts.append(_tube(lp, lr, sides, cap_start=False, cap_end=True))
    clumps.append((leader_top, _norm(leader_top - C)))

    r_p0 = bole_r_top * 0.42                      # thinner primaries
    for i in range(n_primary):
        az = 2.0 * np.pi * i / n_primary + rng.uniform(-0.35, 0.35)
        el = rng.uniform(0.45, 0.95)              # steeper, more upward
        rad = np.sqrt(max(0.0, 1.0 - el * el))
        d_primary = np.array([rad * np.cos(az), el, rad * np.sin(az)])
        target = _env_point(C, r, lobe_fn, d_primary)

        # staggered emergence along the lower crown axis (not one point)
        hstart = bole_top + rng.uniform(0.0, 0.25) * (crown_top - bole_top)
        start = np.array([0.0, hstart, 0.0])
        node = start + (target - start) * rng.uniform(0.40, 0.55)
        node[1] = max(node[1], start[1] + 0.02 * TREE_HEIGHT)

        sp = _branch_points(rng, start, node, 2, 0.05)
        sr = np.linspace(r_p0, r_p0 * 0.7, len(sp))
        parts.append(_tube(sp, sr, sides, cap_start=False, cap_end=False))

        r_node = r_p0 * 0.7
        r_each = r_node / np.sqrt(n_child)
        for _ in range(n_child):
            d_child = _norm(d_primary + rng.normal(0, 0.22, 3))
            d_child[1] = abs(d_child[1]) * 0.6 + 0.4 * d_child[1]
            tip = _env_point(C, r, lobe_fn, d_child)
            cp = _branch_points(rng, node, tip, 3, 0.06)
            cr = np.linspace(r_each * rng.uniform(0.85, 1.1), r_each * 0.4, len(cp))
            parts.append(_tube(cp, cr, sides, cap_start=False, cap_end=True))
            clumps.append((tip, _norm(tip - C)))

    # filler shell clumps so the crown reads full and hides the limbs
    for _ in range(n_primary):
        d = _norm(rng.normal(0, 1, 3) + np.array([0, 0.5, 0]))
        d[1] = abs(d[1]) * 0.7 + 0.3 * d[1]
        sh = _env_point(C, r, lobe_fn, d)
        clumps.append((sh, _norm(sh - C)))

    for _ in range(p["interior_clumps"]):
        d = _norm(rng.normal(0, 1, 3) + np.array([0, 0.3, 0]))
        shell = _env_point(C, r, lobe_fn, d)
        inner = C + (shell - C) * rng.uniform(0.50, 0.78)
        clumps.append((inner, _norm(shell - C)))

    return _combine(parts), clumps


def _build_canopy(rng, p, C, r, lobe_fn, clumps):
    crown_w = 2.0 * r[0]
    clump_r = CLUMP_RADIUS_FRAC * crown_w
    hs0 = CARD_HALF_FRAC * crown_w
    per = max(1, p["cards_total"] // max(1, len(clumps)))

    V = np.zeros((per * len(clumps) * 4, 3))
    F = np.zeros((per * len(clumps) * 2, 3), int)
    vi = 0
    fi = 0
    for (center, nrm) in clumps:
        for _ in range(per):
            off = rng.normal(0, 1, 3)
            off = _norm(off) * (clump_r * rng.uniform(0.0, 1.0) ** (1.0 / 3.0))
            ctr = _clamp_to_env(center + off, C, r, lobe_fn, limit=0.93)

            n = _norm(nrm + rng.normal(0, 0.42, 3))
            ref = np.array([0, 1, 0.0]) if abs(n[1]) < 0.9 else np.array([1, 0, 0.0])
            a = _norm(np.cross(n, ref))
            b = _norm(np.cross(n, a))
            phi = rng.uniform(0, 2 * np.pi)
            ca, sa = np.cos(phi), np.sin(phi)
            a, b = a * ca + b * sa, -a * sa + b * ca

            hs = hs0 * np.exp(rng.normal(0, 0.3))
            ha = a * hs * rng.uniform(0.8, 1.25)
            hb = b * hs * rng.uniform(0.8, 1.25)

            V[vi + 0] = ctr - ha - hb
            V[vi + 1] = ctr + ha - hb
            V[vi + 2] = ctr + ha + hb
            V[vi + 3] = ctr - ha + hb
            F[fi + 0] = [vi + 0, vi + 1, vi + 2]
            F[fi + 1] = [vi + 0, vi + 2, vi + 3]
            vi += 4
            fi += 2
    return V[:vi], F[:fi]


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    if density not in _DENSITY:
        density = "high"
    p = _DENSITY[density]
    rng = np.random.default_rng(seed)

    crown_bottom = CROWN_BASE_FRAC * TREE_HEIGHT
    crown_top = CROWN_TOP_FRAC * TREE_HEIGHT
    C = np.array([0.0, 0.5 * (crown_bottom + crown_top), 0.0])
    ry = 0.5 * (crown_top - crown_bottom)
    rxz = 0.5 * CROWN_WIDTH_OVER_HEIGHT * TREE_HEIGHT
    r = np.array([rxz, ry, rxz])
    lobe_fn = _make_lobe_fn(rng, p["n_lobes"])

    (tV, tF), r_ground, r_bole_top = _build_trunk(rng, p)
    trunk_mesh = trimesh.Trimesh(vertices=tV, faces=tF, process=True)

    (bV, bF), clumps = _build_branches(rng, p, C, r, lobe_fn, r_bole_top)
    branch_mesh = trimesh.Trimesh(vertices=bV, faces=bF, process=True)

    cV, cF = _build_canopy(rng, p, C, r, lobe_fn, clumps)
    canopy_mesh = trimesh.Trimesh(vertices=cV, faces=cF, process=False)

    miny = min(float(trunk_mesh.vertices[:, 1].min()),
               float(branch_mesh.vertices[:, 1].min()),
               float(canopy_mesh.vertices[:, 1].min()))
    for m in (trunk_mesh, branch_mesh, canopy_mesh):
        m.vertices[:, 1] -= miny

    scene = trimesh.Scene()
    scene.add_geometry(trunk_mesh, geom_name="trunk")
    scene.add_geometry(branch_mesh, geom_name="branches")
    scene.add_geometry(canopy_mesh, geom_name="canopy")
    return scene


# ============================================================================
# TEXTURING
# ============================================================================
BARK_RES = 512
ATLAS_RES = 1024
TILE_G = 4

PALE_BARK_CENTERS = [(fx, fy) for fx in (0.45, 0.48, 0.50, 0.52)
                     for fy in (0.58, 0.63, 0.68, 0.72)]
BROWN_BARK_CENTERS = [(fx, fy) for fx in (0.42, 0.46, 0.50, 0.54)
                      for fy in (0.82, 0.87, 0.91)]
FOLIAGE_CENTERS = [(fx, fy) for fx in (0.30, 0.40, 0.50, 0.60, 0.70)
                   for fy in (0.12, 0.20, 0.28, 0.36)]

PALE_FALLBACK = (0.66, 0.64, 0.58)
BROWN_FALLBACK = (0.46, 0.38, 0.30)
LEAF_FALLBACK = (0.32, 0.45, 0.21)
LEAF_TARGET = np.array([0.33, 0.47, 0.21])     # healthy medium green anchor


def _load_image(path):
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.float64) / 255.0
    return arr, im


def _bg_color(arr):
    H, W, _ = arr.shape
    h = max(2, int(0.04 * min(H, W)))
    pts = [(0.04, 0.04), (0.96, 0.04), (0.04, 0.5), (0.96, 0.5),
           (0.5, 0.04), (0.04, 0.96), (0.96, 0.96)]
    cols = []
    for fx, fy in pts:
        cx, cy = int(fx * W), int(fy * H)
        patch = arr[max(0, cy - h):cy + h, max(0, cx - h):cx + h].reshape(-1, 3)
        if len(patch):
            cols.append(np.median(patch, 0))
    return np.median(np.array(cols), 0)


def _sample(arr, centers, half, bg, want="any", fallback=None):
    H, W, _ = arr.shape
    cols = []
    for fx, fy in centers:
        cx, cy = int(round(fx * W)), int(round(fy * H))
        x0, x1 = max(0, cx - half), min(W, cx + half + 1)
        y0, y1 = max(0, cy - half), min(H, cy + half + 1)
        patch = arr[y0:y1, x0:x1].reshape(-1, 3)
        if patch.shape[0] < 1:
            continue
        med = np.median(patch, axis=0)
        if np.linalg.norm(med - bg) < 0.06:
            continue
        if want == "green" and not (med[1] >= med[0] + 0.01 and
                                    med[1] >= med[2] + 0.02):
            continue
        if want == "warm" and not (med[0] >= med[2] + 0.005):
            continue
        cols.append(med)
    if not cols:
        return np.array(fallback, float)
    return np.clip(np.median(np.array(cols), axis=0), 0.0, 1.0)


def _tile_noise(res, rng, fx_range, fy_range, n_terms=6):
    xs = np.linspace(0.0, 2.0 * np.pi, res, endpoint=False)
    X, Y = np.meshgrid(xs, xs)
    field = np.zeros((res, res))
    for _ in range(n_terms):
        ax = int(rng.integers(fx_range[0], fx_range[1] + 1))
        ay = int(rng.integers(fy_range[0], fy_range[1] + 1))
        px, py = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
        field += np.sin(ax * X + px) * np.sin(ay * Y + py)
    field = (field - field.min()) / (np.ptp(field) + 1e-9)
    return field


def _blur(mask, radius):
    im = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
    im = im.filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(im, float) / 255.0


def _albedo_to_normal(alb, strength=2.5):
    lum = 0.299 * alb[..., 0] + 0.587 * alb[..., 1] + 0.114 * alb[..., 2]
    gx = (np.roll(lum, -1, 1) - np.roll(lum, 1, 1)) * 0.5
    gy = (np.roll(lum, -1, 0) - np.roll(lum, 1, 0)) * 0.5
    nx, ny, nz = -gx * strength, -gy * strength, np.ones_like(lum)
    l = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    n = np.stack([nx / l, ny / l, nz / l], -1)
    return Image.fromarray(((n * 0.5 + 0.5) * 255.0).astype(np.uint8), "RGB")


def _make_bark(rng, pale, brown):
    res = BARK_RES
    mottle = _tile_noise(res, rng, (2, 5), (2, 6), 6)
    streak = _tile_noise(res, rng, (10, 22), (1, 2), 8)
    fine = _tile_noise(res, rng, (20, 40), (2, 4), 6)

    val = 0.86 + 0.26 * mottle
    val *= 0.92 + 0.16 * streak
    val *= 0.96 + 0.08 * fine
    alb = pale[None, None, :] * val[..., None]

    warm = _tile_noise(res, rng, (2, 4), (2, 4), 4)
    wm = np.clip((warm - 0.5) * 1.5, 0, 1)[..., None]
    alb = alb * (1 - 0.45 * wm) + (brown[None, None, :] * val[..., None]) * (0.45 * wm)

    lich = _tile_noise(res, rng, (3, 6), (3, 7), 5)
    lm = _blur((lich > 0.80).astype(float), 2.0)[..., None]
    lichen = np.clip(np.array([0.72, 0.75, 0.64]) * (0.6 + 0.4 * float(pale.mean())),
                     0, 1)
    alb = alb * (1 - 0.55 * lm) + lichen[None, None, :] * (0.55 * lm)

    alb = np.clip(alb, 0.0, 1.0)
    img = Image.fromarray((alb * 255.0).astype(np.uint8), "RGB")
    return img, _albedo_to_normal(alb)


def _leaf_color(base, bright, warm, rng, var):
    c = np.array(base, float) * bright
    c[0] *= 1.0 + warm
    c[2] *= 1.0 - warm
    c = c + rng.normal(0, var, 3)
    c = np.clip(c, 0.0, 1.0)
    return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))


def _leaf_poly(cx, cy, length, width, ang):
    base = [(-0.5, 0.0), (-0.25, 0.42), (0.1, 0.5),
            (0.5, 0.0), (0.1, -0.5), (-0.25, -0.42)]
    ca, sa = np.cos(ang), np.sin(ang)
    pts = []
    for lx, ly in base:
        X, Y = lx * length, ly * width
        pts.append((cx + X * ca - Y * sa, cy + X * sa + Y * ca))
    return pts


def _draw_leaf_tile(rng, base, bright, warm, ts, ss=4):
    cs = ts * ss
    im = Image.new("RGBA", (cs, cs), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx, cy = cs * 0.5, cs * 0.5

    nbody = int(rng.integers(4, 7))
    body = _leaf_color(base, bright * 0.97, warm, rng, 0.025)
    for _ in range(nbody):
        bx = cx + rng.normal(0, cs * 0.13)
        by = cy + rng.normal(0, cs * 0.13)
        rr = cs * rng.uniform(0.18, 0.30)
        d.ellipse([bx - rr, by - rr, bx + rr, by + rr * 1.1], fill=body + (255,))

    nleaf = int(rng.integers(55, 90))
    for _ in range(nleaf):
        lx = cx + rng.normal(0, cs * 0.26)
        ly = cy + rng.normal(0, cs * 0.26)
        ang = rng.uniform(0, 2 * np.pi)
        ll = cs * rng.uniform(0.05, 0.12)
        lw = ll * rng.uniform(0.30, 0.50)
        col = _leaf_color(base, bright * rng.uniform(0.85, 1.12), warm, rng, 0.035)
        d.polygon(_leaf_poly(lx, ly, ll, lw, ang), fill=col + (255,))

    im = im.resize((ts, ts), Image.LANCZOS)
    a = np.asarray(im, float)
    al = a[..., 3] / 255.0
    al = np.clip((al - 0.5) * 1.8 + 0.5, 0.0, 1.0)
    a[..., 3] = al * 255.0
    return Image.fromarray(a.astype(np.uint8), "RGBA")


def _make_leaf_atlas(rng, leaf):
    G, ts = TILE_G, ATLAS_RES // TILE_G
    atlas = Image.new("RGBA", (ATLAS_RES, ATLAS_RES), (0, 0, 0, 0))
    for row in range(G):
        for col in range(G):
            f = row / (G - 1)
            bright = 1.14 - 0.30 * f       # sunlit top rows, lightly shaded base
            warm = 0.05 - 0.10 * f         # only mild warm/cool drift
            tile = _draw_leaf_tile(rng, leaf, bright, warm, ts)
            atlas.paste(tile, (col * ts, row * ts))
    return atlas, G


def _wood_tint(V, yref, rng):
    t = np.clip(V[:, 1] / max(yref, 1e-6), 0.0, 1.0)
    shade = 0.66 + 0.34 * t
    r = shade
    g = shade * (0.93 + 0.07 * t)
    b = shade * (0.82 + 0.18 * t)
    n = rng.normal(0, 0.025, len(V))
    r = np.clip(r + n, 0.30, 1.0)
    g = np.clip(g + n, 0.30, 1.0)
    b = np.clip(b + n, 0.30, 1.0)
    col = np.stack([r, g, b, np.ones_like(r)], 1)
    return (col * 255.0).astype(np.uint8)


def _fix_seam(V, F, uv, col):
    Vl, uvl, cl = V.tolist(), uv.tolist(), col.tolist()
    F = F.copy()
    cache = {}
    for fi in range(len(F)):
        tri = [int(F[fi][0]), int(F[fi][1]), int(F[fi][2])]
        us = [uvl[tri[0]][0], uvl[tri[1]][0], uvl[tri[2]][0]]
        if max(us) - min(us) > 0.5:
            for n in range(3):
                vi = tri[n]
                if uvl[vi][0] < 0.5:
                    if vi in cache:
                        ni = cache[vi]
                    else:
                        ni = len(Vl)
                        Vl.append(Vl[vi][:])
                        uvl.append([uvl[vi][0] + 1.0, uvl[vi][1]])
                        cl.append(cl[vi][:])
                        cache[vi] = ni
                    tri[n] = ni
            F[fi] = tri
    return (np.array(Vl, float), np.array(F, np.int64),
            np.array(uvl, float), np.array(cl, np.uint8))


def _apply_cyl(mesh, material, yref, rng, urep, vtile):
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, np.int64)
    u01 = np.arctan2(V[:, 2], V[:, 0]) / (2.0 * np.pi) + 0.5
    v = V[:, 1] / vtile
    uv = np.stack([u01, v], 1)
    col = _wood_tint(V, yref, rng)

    V2, F2, uv2, col2 = _fix_seam(V, F, uv, col)
    uv2[:, 0] *= urep

    m = trimesh.Trimesh(vertices=V2, faces=F2, process=False)
    m.visual = TextureVisuals(uv=uv2, material=material)
    m.visual.vertex_attributes["color"] = col2
    return m


def _apply_canopy(mesh, material, G, rng):
    V = np.asarray(mesh.vertices, float)
    ncards = len(V) // 4
    uv = np.zeros((len(V), 2))
    col = np.zeros((len(V), 4), np.uint8)
    pad = 2.0 / ATLAS_RES

    ymin, ymax = float(V[:, 1].min()), float(V[:, 1].max())
    rmax = float(np.sqrt(V[:, 0] ** 2 + V[:, 2] ** 2).max()) + 1e-6

    for k in range(ncards):
        ti = int(rng.integers(0, G * G))
        rot = int(rng.integers(0, 4))
        row, cc = ti // G, ti % G
        u0, u1 = cc / G + pad, (cc + 1) / G - pad
        v0, v1 = row / G + pad, (row + 1) / G - pad
        corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
        corners = corners[rot:] + corners[:rot]
        for j in range(4):
            uv[4 * k + j] = corners[j]

        ctr = V[4 * k:4 * k + 4].mean(0)
        tv = (ctr[1] - ymin) / (ymax - ymin + 1e-6)
        ro = np.sqrt(ctr[0] ** 2 + ctr[2] ** 2) / rmax
        bm = np.clip(0.72 + 0.26 * tv + 0.08 * ro, 0.0, 1.0)   # brighter, floor up
        warm = (tv - 0.5) * 0.10                               # gentle sun/shade
        r = np.clip(bm * (1 + warm) + rng.normal(0, 0.025), 0.45, 1.0)
        g = np.clip(bm * (1 + 0.25 * warm) + rng.normal(0, 0.025), 0.45, 1.0)
        b = np.clip(bm * (1 - warm) + rng.normal(0, 0.025), 0.45, 1.0)
        c = (int(r * 255), int(g * 255), int(b * 255), 255)
        for j in range(4):
            col[4 * k + j] = c

    m = trimesh.Trimesh(vertices=V, faces=np.asarray(mesh.faces, np.int64),
                        process=False)
    m.visual = TextureVisuals(uv=uv, material=material)
    m.visual.vertex_attributes["color"] = col
    return m


def texture_scene(scene, image_path, seed):
    arr, _ = _load_image(image_path)
    bg = _bg_color(arr)
    half = max(2, int(0.006 * min(arr.shape[0], arr.shape[1])))
    rng = np.random.default_rng(seed * 2 + 101)

    pale = _sample(arr, PALE_BARK_CENTERS, half, bg, "any", PALE_FALLBACK)
    brown = _sample(arr, BROWN_BARK_CENTERS, half, bg, "warm", BROWN_FALLBACK)
    leaf = _sample(arr, FOLIAGE_CENTERS, half, bg, "green", LEAF_FALLBACK)

    # guarantee a healthy, not-too-dark medium green for the foliage
    leaf = 0.55 * leaf + 0.45 * LEAF_TARGET
    lum = float(leaf.mean())
    if lum < 0.32:
        leaf = leaf * (0.32 / max(lum, 1e-6))
    leaf = np.clip(leaf, 0.0, 1.0)

    bark_img, bark_nrm = _make_bark(rng, pale, brown)
    atlas, G = _make_leaf_atlas(rng, leaf)

    bark_mat = PBRMaterial(name="bark", baseColorTexture=bark_img,
                           normalTexture=bark_nrm, metallicFactor=0.0,
                           roughnessFactor=0.9)
    leaf_mat = PBRMaterial(name="foliage", baseColorTexture=atlas,
                           metallicFactor=0.0, roughnessFactor=0.8,
                           alphaMode="MASK", alphaCutoff=0.45, doubleSided=True)

    bole_top = CROWN_BASE_FRAC * TREE_HEIGHT

    out = trimesh.Scene()
    out.add_geometry(_apply_cyl(scene.geometry["trunk"], bark_mat, bole_top,
                                rng, urep=2.2, vtile=3.0), geom_name="trunk")
    out.add_geometry(_apply_cyl(scene.geometry["branches"], bark_mat, bole_top,
                                rng, urep=1.4, vtile=1.6), geom_name="branches")
    out.add_geometry(_apply_canopy(scene.geometry["canopy"], leaf_mat, G, rng),
                     geom_name="canopy")
    return out


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Procedural buttressed rainforest tree -> textured GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        scene = build_mesh(args.seed, args.density)
        scene = texture_scene(scene, args.image, args.seed)
        data = scene.export(file_type="glb")
        with open(args.output, "wb") as f:
            f.write(data)
    except Exception as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()