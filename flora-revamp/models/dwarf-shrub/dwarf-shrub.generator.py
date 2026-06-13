#!/usr/bin/env python3
"""
Procedural "low desert cushion shrub" -> textured GLB.

A broad, ground-hugging mound of fine wiry twigs (sagebrush / dryland scrub)
densely lined with small clumped sage-green leaf-card tufts. Tileable bark and
a foliage atlas are derived from a reference photo; UVs are assigned per surface
type; a binary .glb is exported.

CLI:
    python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter


# ===========================================================================
# GEOMETRY
# ===========================================================================

# Proportions tuned to the photo: a LOW, wide cushion (content aspect ~2.3).
OVERALL_WIDTH = 1.00                                 # full crown spread, meters
HEIGHT_OVER_WIDTH = 0.50                             # width ~2x height -> aspect ~2.3
OVERALL_HEIGHT = OVERALL_WIDTH * HEIGHT_OVER_WIDTH   # ~0.50 m tall mound

CROWN_HALF_W = OVERALL_WIDTH * 0.5
CLUMP_R = OVERALL_WIDTH * 0.060                      # small tuft radius
CARD_HS = OVERALL_WIDTH * 0.028                      # small leaf-card half-size

BASE_Y = 0.04
BASE_R = 0.022
PRIMARY_R = 0.008                                    # thin wiry stems

V_REPEAT = 0.10                                      # bark UV repeat (m)
ATLAS_N = 4

_UP = np.array([0.0, 1.0, 0.0])


def _density_params(density):
    presets = {
        "high": dict(
            n_primary=16, max_level=2, tube_sides=5,
            n_clumps=130, cards_total=2600,
            children=[3, 2, 2], steps=[6, 4, 3], steplen=[0.09, 0.06, 0.045],
            gravity=[0.5, 0.6, 0.7], out=[0.03, 0.05, 0.06],
            noise=[0.04, 0.06, 0.08], spread=[0.6, 0.8, 0.9],
        ),
        "med": dict(
            n_primary=11, max_level=2, tube_sides=4,
            n_clumps=70, cards_total=1100,
            children=[2, 2, 1], steps=[5, 4, 3], steplen=[0.10, 0.06, 0.045],
            gravity=[0.5, 0.6, 0.7], out=[0.03, 0.05, 0.06],
            noise=[0.04, 0.06, 0.08], spread=[0.6, 0.8, 0.9],
        ),
        "low": dict(
            n_primary=7, max_level=1, tube_sides=3,
            n_clumps=34, cards_total=360,
            children=[2, 1], steps=[5, 3], steplen=[0.11, 0.06],
            gravity=[0.5, 0.6], out=[0.03, 0.05],
            noise=[0.04, 0.06], spread=[0.6, 0.8],
        ),
    }
    return presets.get(density, presets["high"])


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _lobe_factor(az, lobe_params):
    s = 1.0
    for amp, freq, ph in lobe_params:
        s += amp * np.cos(freq * az + ph)
    return max(0.40, s)


def _envelope_top(hr, az, lobe_params):
    maxr = CROWN_HALF_W * _lobe_factor(az, lobe_params)
    f = min(1.0, hr / maxr) if maxr > 0 else 0.0
    # flatter-topped cushion (exponent < 1 keeps the crown broad)
    return OVERALL_HEIGHT * max(0.0, 1.0 - f * f) ** 0.65


def _clamp_to_envelope(p, lobe_params, margin=1.0):
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    az = np.arctan2(z, x)
    hr = np.hypot(x, z)
    maxr = CROWN_HALF_W * _lobe_factor(az, lobe_params) * margin
    if maxr > 0 and hr > maxr:
        s = maxr / hr
        x *= s
        z *= s
        hr = maxr
    topy = _envelope_top(hr, az, lobe_params)
    if y > topy:
        y = topy
    if y < 0.0:
        y = 0.0
    return np.array([x, y, z])


def _leaf_normal(pos, tip_dir, rng):
    outward = _norm(np.array([pos[0], 0.0, pos[2]]))
    if np.linalg.norm(outward) < 1e-6:
        outward = _norm(np.array([tip_dir[0], 0.0, tip_dir[2]]))
    n = outward * 0.5 + _UP * 0.8 + rng.normal(0, 0.15, 3)
    return _norm(n)


def _make_tube(points, radii, sides):
    points = np.asarray(points, dtype=float)
    radii = np.asarray(radii, dtype=float)
    n = len(points)
    if n < 2:
        return None

    tang = np.zeros((n, 3))
    tang[1:-1] = points[2:] - points[:-2]
    tang[0] = points[1] - points[0]
    tang[-1] = points[-1] - points[-2]
    lens = np.linalg.norm(tang, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    tang = tang / lens

    ref = np.array([0.0, 1.0, 0.0])
    if abs(float(tang[0] @ ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    n1 = _norm(np.cross(tang[0], ref))
    frames = [n1]
    for i in range(1, n):
        v = frames[-1] - tang[i] * float(frames[-1] @ tang[i])
        ln = np.linalg.norm(v)
        if ln < 1e-6:
            alt = np.array([1.0, 0.0, 0.0]) if abs(float(tang[i] @ _UP)) > 0.9 else _UP
            v = np.cross(tang[i], alt)
            ln = np.linalg.norm(v)
        frames.append(v / ln)

    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)

    rings = np.empty((n, sides, 3))
    for i in range(n):
        a = frames[i]
        b = _norm(np.cross(tang[i], a))
        rings[i] = points[i] + radii[i] * (np.outer(ca, a) + np.outer(sa, b))
    verts = rings.reshape(-1, 3)

    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    clen = np.concatenate([[0.0], np.cumsum(seg)])
    vco = clen / V_REPEAT
    u_row = np.arange(sides) / float(sides)
    uv = np.empty((n, sides, 2))
    uv[:, :, 0] = u_row[None, :]
    uv[:, :, 1] = vco[:, None]
    uv = uv.reshape(-1, 2)

    vrad = np.repeat(radii[:, None], sides, axis=1).reshape(-1)

    faces = []
    for i in range(n - 1):
        for j in range(sides):
            j2 = (j + 1) % sides
            a = i * sides + j
            b = i * sides + j2
            c = (i + 1) * sides + j2
            d = (i + 1) * sides + j
            faces.append([a, b, c])
            faces.append([a, c, d])

    verts = np.vstack([verts, points[0], points[-1]])
    uv = np.vstack([uv, [0.5, vco[0]], [0.5, vco[-1]]])
    vrad = np.concatenate([vrad, [radii[0], radii[-1]]])
    base_c = n * sides
    tip_c = n * sides + 1
    off = (n - 1) * sides
    for j in range(sides):
        j2 = (j + 1) % sides
        faces.append([base_c, j2, j])
        faces.append([tip_c, off + j, off + j2])

    return verts, np.asarray(faces, dtype=np.int64), uv, vrad


def _grow(start, direction, radius, level, p, paths, anchors, lobe_params, rng):
    steps = p["steps"][level]
    step_len = p["steplen"][level]
    r0 = radius
    r1 = radius * 0.40

    pos = np.array(start, dtype=float)
    direction = _norm(direction)
    pts = [pos.copy()]
    rads = [r0]

    for k in range(steps):
        frac = (k + 1) / steps
        droop = p["gravity"][level] * (0.30 + frac)
        outward = _norm(np.array([pos[0], 0.0, pos[2]]))
        if np.linalg.norm(outward) < 1e-6:
            outward = _norm(np.array([direction[0], 0.0, direction[2]]))
        direction = _norm(
            direction
            + np.array([0.0, -droop, 0.0]) * step_len
            + outward * p["out"][level]
            + rng.normal(0.0, p["noise"][level], 3)
        )
        pos = pos + direction * step_len
        pos = _clamp_to_envelope(pos, lobe_params)
        pts.append(pos.copy())
        rads.append(r0 + (r1 - r0) * frac)

    paths.append((np.array(pts), np.array(rads)))

    # Dense foliage anchors along the UPPER/OUTER portion of every twig so the
    # leaf tufts line the whole framework (interior near base stays bare).
    a_start = max(1, len(pts) // 2) if level == 0 else 1
    for i in range(a_start, len(pts)):
        td = _norm(pts[i] - pts[i - 1])
        anchors.append((pts[i].copy(), _leaf_normal(pts[i], td, rng)))

    if level < p["max_level"]:
        nchild = p["children"][level]
        lo = max(1, len(pts) // 2)
        hi = len(pts)
        if lo >= hi:
            lo = hi - 1
        for _ in range(nchild):
            ti = int(rng.integers(lo, hi))
            cstart = pts[ti]
            pdir = _norm(pts[ti] - pts[ti - 1])
            outward = _norm(np.array([cstart[0], 0.0, cstart[2]]))
            if np.linalg.norm(outward) < 1e-6:
                outward = _norm(np.array([pdir[0], 0.0, pdir[2]]))
            child_dir = _norm(
                pdir
                + _UP * 0.10                      # spread laterally, not upward
                + outward * 0.35
                + rng.normal(0.0, 1.0, 3) * p["spread"][level] * 0.4
            )
            _grow(cstart, child_dir, rads[ti] * rng.uniform(0.6, 0.75),
                  level + 1, p, paths, anchors, lobe_params, rng)


def _build_leaf_cards(anchors, n_clumps, cards_total, lobe_params, rng):
    if len(anchors) == 0:
        return None

    centers = np.array([a[0] for a in anchors])
    hr = np.hypot(centers[:, 0], centers[:, 2])
    # broad coverage, mild bias to upper/outer surfaces
    weights = 0.4 + centers[:, 1] * 1.5 + hr * 0.5
    weights = np.clip(weights, 1e-6, None)
    weights = weights / weights.sum()

    k = min(n_clumps, len(anchors))
    idx = rng.choice(len(anchors), size=k, replace=False, p=weights)
    per = max(6, cards_total // max(1, k))

    base_corner = [(0, 1), (1, 1), (1, 0), (0, 0)]
    inset = 0.012

    verts, faces, uvs, vcols = [], [], [], []
    for ci in idx:
        center, normal = anchors[ci]
        clump_jit = rng.uniform(-0.05, 0.05)
        for _ in range(per):
            d = rng.normal(0.0, 1.0, 3)
            d = d / (np.linalg.norm(d) + 1e-9)
            r = CLUMP_R * rng.uniform(0.0, 1.0) ** (1.0 / 3.0)
            pos = center + d * r * np.array([1.0, 0.7, 1.0])
            pos = _clamp_to_envelope(pos, lobe_params, margin=1.03)

            nz = _norm(normal + rng.normal(0.0, 0.35, 3))
            t = _UP if abs(nz[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
            u = _norm(np.cross(nz, t))
            v = _norm(np.cross(nz, u))

            th = rng.uniform(0.0, np.pi)
            uu = u * np.cos(th) + v * np.sin(th)
            vv = -u * np.sin(th) + v * np.cos(th)

            hs = CARD_HS * float(np.exp(rng.normal(0.0, 0.28)))
            hs2 = hs * rng.uniform(0.7, 1.1)

            p0 = pos - uu * hs - vv * hs2
            p1 = pos + uu * hs - vv * hs2
            p2 = pos + uu * hs + vv * hs2
            p3 = pos - uu * hs + vv * hs2
            b = len(verts)
            verts.extend([p0, p1, p2, p3])
            faces.append([b, b + 1, b + 2])
            faces.append([b, b + 2, b + 3])

            tile = int(rng.integers(0, ATLAS_N * ATLAS_N))
            rot = int(rng.integers(0, 4))
            col, row = tile % ATLAS_N, tile // ATLAS_N
            corners = base_corner[rot:] + base_corner[:rot]
            for (cu, cv) in corners:
                su = (col + inset + cu * (1 - 2 * inset)) / ATLAS_N
                sv = (row + inset + cv * (1 - 2 * inset)) / ATLAS_N
                uvs.append([su, sv])

            hn = np.clip(pos[1] / OVERALL_HEIGHT, 0.0, 1.0)
            on = np.clip(np.hypot(pos[0], pos[2]) / CROWN_HALF_W, 0.0, 1.0)
            bright = np.clip(0.74 + 0.28 * hn + 0.12 * on + clump_jit
                             + rng.normal(0, 0.03), 0.55, 1.20)
            col_rgb = np.array([bright * 0.97, bright * 1.03, bright * 0.92])
            rgba = np.clip(np.append(col_rgb, 1.0), 0.0, 1.0)
            for _c in range(4):
                vcols.append(rgba)

    return (np.array(verts), np.asarray(faces, dtype=np.int64),
            np.asarray(uvs, dtype=np.float64), np.asarray(vcols, dtype=np.float64))


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    p = _density_params(density)

    # low-amplitude lobes keep the envelope tight (controls overall width)
    n_lobes = int(rng.integers(3, 6))
    lobe_params = [
        (rng.uniform(0.05, 0.10), int(rng.integers(1, 4)), rng.uniform(0, 2 * np.pi))
        for _ in range(n_lobes)
    ]

    paths = []
    anchors = []

    base_pts = np.array([[0.0, 0.0, 0.0], [0.0, BASE_Y, 0.0]])
    base_rad = np.array([BASE_R * 1.5, BASE_R * 0.9])
    paths.append((base_pts, base_rad))

    n_primary = p["n_primary"]
    az0 = rng.uniform(0, 2 * np.pi)
    for i in range(n_primary):
        az = az0 + (2 * np.pi * i / n_primary) + rng.normal(0, 0.12)
        start = np.array([
            np.cos(az) * 0.02 * rng.uniform(0.3, 1.0),
            BASE_Y,
            np.sin(az) * 0.02 * rng.uniform(0.3, 1.0),
        ])
        horiz = np.array([np.cos(az), 0.0, np.sin(az)])
        # variable up-bias: some stems rise to fill the crown centre, most
        # spread low and outward -> irregular low mound, not a fountain.
        upb = rng.uniform(0.45, 1.10)
        direction = _norm(horiz * 1.0 + _UP * upb)
        _grow(start, direction, PRIMARY_R * rng.uniform(0.85, 1.15),
              0, p, paths, anchors, lobe_params, rng)

    # --- Wood mesh ----------------------------------------------------------
    wverts, wfaces, wuv, wrad = [], [], [], []
    voff = 0
    for pts, rads in paths:
        tube = _make_tube(pts, rads, p["tube_sides"])
        if tube is None:
            continue
        v, f, uv, vr = tube
        wverts.append(v)
        wfaces.append(f + voff)
        wuv.append(uv)
        wrad.append(vr)
        voff += len(v)
    wverts = np.vstack(wverts)
    wfaces = np.vstack(wfaces)
    wuv = np.vstack(wuv)
    wrad = np.concatenate(wrad)
    branches = trimesh.Trimesh(vertices=wverts, faces=wfaces, process=False)

    ynorm = np.clip(wverts[:, 1] / OVERALL_HEIGHT, 0.0, 1.0)
    rnorm = np.clip(wrad / PRIMARY_R, 0.0, 1.0)
    paleness = np.clip(0.45 * (1.0 - rnorm) + 0.40 * ynorm + 0.15, 0.0, 1.0)
    f = 0.72 + 0.38 * paleness
    bcol = np.stack([
        np.clip(f * (1.0 - 0.04 * paleness), 0.4, 1.2),
        np.clip(f * 1.0, 0.4, 1.2),
        np.clip(f * (0.96 + 0.10 * paleness), 0.4, 1.2),
    ], axis=1)
    branches.metadata["uv"] = wuv
    branches.metadata["vcolor"] = np.clip(
        np.column_stack([bcol, np.ones(len(bcol))]), 0, 1)

    meshes = [("branches", branches)]

    # --- Foliage mesh -------------------------------------------------------
    leaf = _build_leaf_cards(anchors, p["n_clumps"], p["cards_total"], lobe_params, rng)
    if leaf is not None:
        lv, lf, luv, lvc = leaf
        foliage = trimesh.Trimesh(vertices=lv, faces=lf, process=False)
        foliage.fix_normals()
        foliage.metadata["uv"] = luv
        foliage.metadata["vcolor"] = lvc
        meshes.append(("foliage", foliage))

    # --- Normalize: rest on XZ plane (min y = 0), centered in X/Z ----------
    allv = np.vstack([m.vertices for _, m in meshes])
    minc = allv.min(axis=0)
    maxc = allv.max(axis=0)
    offset = np.array([
        -(minc[0] + maxc[0]) * 0.5,
        -minc[1],
        -(minc[2] + maxc[2]) * 0.5,
    ])

    scene = trimesh.Scene()
    for name, m in meshes:
        m.vertices = m.vertices + offset
        scene.add_geometry(m, geom_name=name)

    return scene


# ===========================================================================
# IMAGE ANALYSIS
# ===========================================================================

def _gaussian(arr2d, radius):
    img = Image.fromarray(np.clip(arr2d * 255, 0, 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(img).astype(np.float32) / 255.0


def analyze_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    h, w = arr.shape[:2]

    lum = arr.mean(axis=2)
    blur = _gaussian(lum, max(8, min(h, w) // 14))
    blur = np.clip(blur, 1e-3, None)
    gain = np.clip(lum.mean() / blur, 0.6, 1.6)
    delit = np.clip(arr * gain[..., None], 0.0, 1.0)

    frame = np.concatenate([
        delit[:max(1, h // 20), :, :].reshape(-1, 3),
        delit[-max(1, h // 20):, :, :].reshape(-1, 3),
        delit[:, :max(1, w // 20), :].reshape(-1, 3),
        delit[:, -max(1, w // 20):, :].reshape(-1, 3),
    ], axis=0)
    bg = np.median(frame, axis=0)

    cy0, cy1 = int(h * 0.18), int(h * 0.92)
    cx0, cx1 = int(w * 0.12), int(w * 0.88)
    crop = delit[cy0:cy1, cx0:cx1, :].reshape(-1, 3)

    dist_bg = np.linalg.norm(crop - bg[None, :], axis=1)
    obj = crop[dist_bg > 0.10]
    if len(obj) < 200:
        obj = crop

    greenness = obj[:, 1] - 0.5 * (obj[:, 0] + obj[:, 2])
    fol = obj[greenness > 0.035]
    wood = obj[greenness <= 0.035]

    def _safe(pixels, fallback):
        return pixels if len(pixels) >= 40 else np.asarray([fallback])

    fol = _safe(fol, [0.50, 0.57, 0.43])
    wood = _safe(wood, [0.60, 0.55, 0.47])

    fol_mid = np.median(fol, axis=0)
    fol_lit = np.clip(np.percentile(fol, 72, axis=0) * 1.04 + 0.02, 0, 1)
    fol_shade = np.clip(np.percentile(fol, 28, axis=0) * 0.92, 0, 1)

    wlum = wood @ np.array([0.299, 0.587, 0.114])
    order = np.argsort(wlum)
    dark = wood[order[: max(1, len(wood) // 3)]].mean(axis=0)
    light = wood[order[-max(1, len(wood) // 3):]].mean(axis=0)
    dl = float(dark @ np.array([0.299, 0.587, 0.114])) + 1e-3
    ll = float(light @ np.array([0.299, 0.587, 0.114])) + 1e-3
    if ll / dl < 2.0:
        dark = np.clip(dark * 0.78, 0.04, 1)
        light = np.clip(light * 1.12 + 0.04, 0, 1)

    bark_dark = np.clip(dark * np.array([1.02, 0.98, 0.92]), 0.04, 1)
    bark_light = np.clip(light * np.array([1.03, 1.0, 0.93]) + 0.02, 0, 1)

    return dict(fol_mid=fol_mid, fol_lit=fol_lit, fol_shade=fol_shade,
                bark_dark=bark_dark, bark_light=bark_light)


# ===========================================================================
# TEXTURE SYNTHESIS
# ===========================================================================

def make_bark_textures(bark_dark, bark_light, rng, res=512):
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32) / res
    val = np.zeros((res, res), np.float32)
    for _ in range(5):
        fx = int(rng.integers(3, 10))
        fy = int(rng.integers(0, 3))
        amp = rng.uniform(0.15, 0.45)
        ph = rng.uniform(0, 2 * np.pi)
        val += amp * np.sin(2 * np.pi * (fx * xx + fy * yy) + ph)
    for _ in range(4):
        fx = int(rng.integers(8, 24))
        fy = int(rng.integers(8, 24))
        amp = rng.uniform(0.04, 0.12)
        ph = rng.uniform(0, 2 * np.pi)
        val += amp * np.sin(2 * np.pi * (fx * xx + fy * yy) + ph)

    val -= val.min()
    val /= (np.ptp(val) + 1e-6)
    val = val ** 1.1

    alb = (bark_dark[None, None, :] * (1.0 - val[..., None])
           + bark_light[None, None, :] * val[..., None])
    alb = np.clip(alb + rng.normal(0, 0.015, alb.shape), 0, 1)
    albedo = Image.fromarray((alb * 255).astype(np.uint8), "RGB")

    height = 1.0 - (alb @ np.array([0.299, 0.587, 0.114]))
    gx = np.roll(height, -1, 1) - np.roll(height, 1, 1)
    gy = np.roll(height, -1, 0) - np.roll(height, 1, 0)
    strength = 2.5
    nx, ny, nz = -gx * strength, -gy * strength, np.ones_like(height)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    nrm = np.stack([nx / ln, ny / ln, nz / ln], axis=2)
    normal = Image.fromarray((((nrm * 0.5) + 0.5) * 255).astype(np.uint8), "RGB")
    return albedo, normal


def _ovate(cx, cy, length, width, angle, npts=14):
    ts = np.linspace(0, 2 * np.pi, npts, endpoint=False)
    ex = (length * 0.5) * np.cos(ts)
    ey = (width * 0.5) * np.sin(ts)
    ex = ex * (0.70 + 0.30 * (np.cos(ts) * 0.5 + 0.5))   # gentle ovate taper
    ca, sa = np.cos(angle), np.sin(angle)
    px = cx + ex * ca - ey * sa
    py = cy + ex * sa + ey * ca
    return [(float(a), float(b)) for a, b in zip(px, py)]


def _draw_cluster(size, base_rgb, sun, rng):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx = cy = size * 0.5
    # many small leaves -> fine, dense sprig texture (not big blobs)
    n_leaves = int(rng.integers(18, 28))
    for _ in range(n_leaves):
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(0, size * 0.24)
        lx = cx + np.cos(ang) * dist
        ly = cy + np.sin(ang) * dist
        length = size * rng.uniform(0.09, 0.16)
        width = length * rng.uniform(0.52, 0.74)
        jitter = rng.uniform(0.90, 1.08, 3)
        col = np.clip(base_rgb * jitter, 0, 1)
        rgb = tuple(int(c * 255) for c in col)
        d.polygon(_ovate(lx, ly, length, width, ang + rng.uniform(-0.5, 0.5)),
                  fill=rgb + (255,))
    for _ in range(int(rng.integers(2, 5))):
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(0, size * 0.22)
        bx = cx + np.cos(ang) * dist
        by = cy + np.sin(ang) * dist
        rr = size * rng.uniform(0.015, 0.035)
        col = np.clip(base_rgb * (1.08 + 0.10 * sun), 0, 1)
        rgb = tuple(int(c * 255) for c in col)
        d.ellipse([bx - rr, by - rr, bx + rr, by + rr], fill=rgb + (255,))
    return im


def make_foliage_atlas(pal, rng, res=1024):
    atlas = Image.new("RGBA", (res, res), (0, 0, 0, 0))
    tile = res // ATLAS_N
    ss = 4
    big = tile * ss
    for r in range(ATLAS_N):
        for c in range(ATLAS_N):
            sun = 1.0 - r / float(ATLAS_N - 1)
            base = pal["fol_shade"] * (1 - sun) + pal["fol_lit"] * sun
            # keep a soft grey-sage green; only a slight warm lift in sun
            base = np.clip(base + np.array([0.03, 0.02, -0.015]) * sun, 0, 1)
            tile_img = _draw_cluster(big, base, sun, rng)
            tile_img = tile_img.resize((tile, tile), Image.LANCZOS)
            atlas.paste(tile_img, (c * tile, r * tile))
    return atlas


# ===========================================================================
# MATERIAL / UV ASSIGNMENT + EXPORT
# ===========================================================================

def _to_u8_rgba(float_rgba):
    return np.clip(float_rgba * 255.0 + 0.5, 0, 255).astype(np.uint8)


def assign_materials(scene, pal, rng):
    bark_albedo, bark_normal = make_bark_textures(
        pal["bark_dark"], pal["bark_light"], rng, res=512)
    atlas = make_foliage_atlas(pal, rng, res=1024)

    for name, mesh in scene.geometry.items():
        uv = mesh.metadata.get("uv")
        vcolor = mesh.metadata.get("vcolor")
        if uv is None:
            continue

        if name == "branches":
            mat = trimesh.visual.material.PBRMaterial(
                name="bark",
                baseColorTexture=bark_albedo,
                normalTexture=bark_normal,
                baseColorFactor=[255, 255, 255, 255],
                metallicFactor=0.0,
                roughnessFactor=0.9,
                alphaMode="OPAQUE",
                doubleSided=False,
            )
        else:
            mat = trimesh.visual.material.PBRMaterial(
                name="foliage",
                baseColorTexture=atlas,
                baseColorFactor=[255, 255, 255, 255],
                metallicFactor=0.0,
                roughnessFactor=0.82,
                alphaMode="MASK",
                alphaCutoff=0.45,
                doubleSided=True,
            )

        mesh.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uv), material=mat)
        if vcolor is not None:
            mesh.visual.vertex_attributes["color"] = _to_u8_rgba(np.asarray(vcolor))

    return scene


def main():
    ap = argparse.ArgumentParser(description="Procedural desert cushion shrub -> GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        pal = analyze_image(args.image)
        scene = build_mesh(args.seed, args.density)
        tex_rng = np.random.default_rng(args.seed ^ 0x9E3779B9)
        scene = assign_materials(scene, pal, tex_rng)

        data = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(data)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("ERROR: {}\n".format(exc))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()