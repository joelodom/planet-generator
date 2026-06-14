#!/usr/bin/env python3
"""
Procedural slender wild flowering herb (cranesbill / speedwell-like):
builds geometry, derives tileable materials + leaf/petal alpha-cutout atlases
from a reference photo, applies per-surface UVs and PBR materials, and exports
a textured GLB.

Foliage and flowers are textured CARDS with binary alpha-cutout silhouettes
(alphaMode=MASK, doubleSided); the stem is a solid tube with cylindrical UVs.

CLI:
    python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb
"""

import argparse
import sys

import numpy as np
import trimesh
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial
from PIL import Image, ImageDraw, ImageFilter


# ---------------------------------------------------------------------------
# Measured proportions (read off the reference image)
# ---------------------------------------------------------------------------
PLANT_HEIGHT = 0.42          # m, plausible woodland herb height (base->tip)
HW_RATIO     = 2.4           # height / overall foliage width (slender, ~0.42 aspect)
PLANT_WIDTH  = PLANT_HEIGHT / HW_RATIO

SPIKE_FRAC   = 0.22          # flower spike occupies the TOP ~22% of height
FOLIAGE_TOP  = 0.70          # leaves sit below ~lower two-thirds
FORK_FRAC    = 0.16          # secondary stem forks off this low on the main

STEM_BASE_R  = 0.0035        # m at the very base
STEM_TIP_R   = 0.0010        # m up in the spike
BASAL_FLARE  = 0.55          # +55% radius over the bottom ~6%

STEM_VSCALE  = 0.030         # m of stem per vertical texture repeat
GRID         = 4             # 4x4 atlas for foliage + flowers


# ===========================================================================
# Vector / curve helpers
# ===========================================================================
def _n(v):
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-12)


def _basis_from_z(zdir, up_hint=(0.0, 1.0, 0.0)):
    """Orthonormal basis (columns x,y,z) with z aligned to zdir."""
    z = _n(zdir)
    up = np.asarray(up_hint, dtype=float)
    if abs(float(np.dot(z, _n(up)))) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
    x = _n(np.cross(up, z))
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _frames(points):
    """Parallel-transport frames along a polyline -> tangent T, normals N, B."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    T = np.zeros((n, 3))
    if n >= 3:
        T[1:-1] = pts[2:] - pts[:-2]
    T[0] = pts[1] - pts[0]
    T[-1] = pts[-1] - pts[-2]
    T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)

    N = np.zeros((n, 3))
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(T[0], ref))) > 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    N[0] = _n(ref - np.dot(ref, T[0]) * T[0])
    for i in range(1, n):
        prev_t, cur_t, np_prev = T[i - 1], T[i], N[i - 1]
        axis = np.cross(prev_t, cur_t)
        s = np.linalg.norm(axis)
        c = float(np.clip(np.dot(prev_t, cur_t), -1.0, 1.0))
        if s < 1e-9:
            ni = np_prev
        else:
            axis /= s
            ang = np.arctan2(s, c)
            ni = (np_prev * np.cos(ang)
                  + np.cross(axis, np_prev) * np.sin(ang)
                  + axis * np.dot(axis, np_prev) * (1.0 - np.cos(ang)))
        ni = ni - np.dot(ni, cur_t) * cur_t
        N[i] = _n(ni)
    B = np.cross(T, N)
    return T, N, B


def _tube(points, radii, sides, vscale, cap=False):
    """Swept circular tube + cylindrical UVs. Returns (verts, faces, uv)."""
    points = np.asarray(points, dtype=float)
    radii = np.asarray(radii, dtype=float)
    n = len(points)
    _, N, B = _frames(points)
    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    u_around = np.arange(sides) / float(sides)

    rings, uvs = [], []
    for i in range(n):
        ring = (points[i][None, :]
                + radii[i] * (np.outer(ca, N[i]) + np.outer(sa, B[i])))
        rings.append(ring)
        uvs.append(np.column_stack([u_around, np.full(sides, cum[i] / vscale)]))
    verts = np.vstack(rings)
    uv = np.vstack(uvs)

    faces = []
    for i in range(n - 1):
        for j in range(sides):
            a = i * sides + j
            b = i * sides + (j + 1) % sides
            c = (i + 1) * sides + (j + 1) % sides
            d = (i + 1) * sides + j
            faces.append([a, b, c])
            faces.append([a, c, d])

    if cap:
        nv = len(verts)
        cbot, ctop = nv, nv + 1
        verts = np.vstack([verts, points[0][None, :], points[-1][None, :]])
        uv = np.vstack([uv, [[0.5, cum[0] / vscale]], [[0.5, cum[-1] / vscale]]])
        for j in range(sides):
            faces.append([cbot, (j + 1) % sides, j])
        base = (n - 1) * sides
        for j in range(sides):
            faces.append([ctop, base + j, base + (j + 1) % sides])

    return verts, np.asarray(faces, dtype=np.int64), uv


def _grow_stem(start, base_dir, length, n, bend_vec, wob_amp, rng):
    """A gently curving stem centreline starting at `start`."""
    t = np.linspace(0.0, 1.0, n)
    base_dir = _n(base_dir)
    pts = np.asarray(start, dtype=float)[None, :] + np.outer(t * length, base_dir)
    pts = pts + np.outer(t ** 1.7, np.asarray(bend_vec, dtype=float))
    perp1 = np.cross(base_dir, [0.0, 0.0, 1.0])
    if np.linalg.norm(perp1) < 1e-6:
        perp1 = np.cross(base_dir, [0.0, 1.0, 0.0])
    perp1 = _n(perp1)
    perp2 = _n(np.cross(base_dir, perp1))
    ph = rng.uniform(0.0, 2.0 * np.pi, 2)
    fr = rng.uniform(1.2, 2.3, 2)
    wob = (np.sin(fr[0] * np.pi * t + ph[0])[:, None] * perp1
           + np.sin(fr[1] * np.pi * t + ph[1])[:, None] * perp2)
    pts = pts + wob_amp * t[:, None] * wob
    return pts


# ===========================================================================
# Card organ generators (flat/curved cards; shape comes from alpha-cutout)
# Local frame: x = across, y = card normal, z = along (forward from attach pt)
# ===========================================================================
def _leaf_card(size, rng):
    """A gently curved palmate-leaf CARD (narrow across so the plant reads
    slender). Silhouette is supplied by the atlas alpha."""
    ncols, nrows = 3, 3
    w = size * 0.72                      # half-width across (kept tight)
    h = size * 1.55                      # blade length along forward
    droop = size * 0.30 * rng.uniform(0.6, 1.1)
    us = np.linspace(-w, w, ncols)
    vs = np.linspace(0.0, h, nrows)

    verts, uv = [], []
    for vv in vs:
        for uu in us:
            y = -droop * (vv / h) ** 2 - 0.08 * size * (uu / w) ** 2
            verts.append([uu, y, vv])
            uv.append([(uu + w) / (2.0 * w), vv / h])
    verts = np.asarray(verts, dtype=float)
    uv = np.asarray(uv, dtype=float)

    faces = []
    for r in range(nrows - 1):
        for c in range(ncols - 1):
            a = r * ncols + c
            b = r * ncols + c + 1
            cc = (r + 1) * ncols + c + 1
            d = (r + 1) * ncols + c
            faces.append([a, b, cc])
            faces.append([a, cc, d])
    return verts, np.asarray(faces, dtype=np.int64), uv


def _flower_cards(size, rng):
    """Two small crossed CARDS forming a delicate floret billboard. Base
    (z=0) sits at the pedicel tip. Kept small to avoid chunky fins."""
    hw = size * 0.70
    ln = size * 1.45
    quads = [
        ([-hw, 0.0, 0.0], [hw, 0.0, 0.0], [hw, 0.0, ln], [-hw, 0.0, ln]),
        ([0.0, -hw, 0.0], [0.0, hw, 0.0], [0.0, hw, ln], [0.0, -hw, ln]),
    ]
    verts, faces, uv = [], [], []
    for qi, q in enumerate(quads):
        base = qi * 4
        verts.extend(q)
        uv.extend([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        faces.append([base + 0, base + 1, base + 2])
        faces.append([base + 0, base + 2, base + 3])
    return (np.asarray(verts, dtype=float),
            np.asarray(faces, dtype=np.int64),
            np.asarray(uv, dtype=float))


# ===========================================================================
# Per-vertex COLOR_0 tints (multiply the texture; values stay in [~0.5, 1.0])
# ===========================================================================
def _stem_colors(vy, Hp):
    hf = np.clip(vy / max(Hp, 1e-6), 0.0, 1.0)
    brown = np.array([0.98, 0.80, 0.66])     # base reddish-brown push
    green = np.array([0.90, 0.98, 0.82])     # greener upward
    w = np.clip(hf / 0.28, 0.0, 1.0)[:, None]
    return brown[None, :] * (1.0 - w) + green[None, :] * w


def _leaf_colors(lv, size, leaf_y, Hp, rng):
    rad = np.sqrt(lv[:, 0] ** 2 + lv[:, 2] ** 2) / (size + 1e-9)
    bright = 0.80 + 0.26 * np.clip(rad, 0.0, 1.2)
    hf = np.clip(leaf_y / max(Hp, 1e-6), 0.0, 1.0)
    shade = 0.86 + 0.20 * hf
    jit = 1.0 + rng.uniform(-0.05, 0.05)
    m = np.clip(bright * shade * jit, 0.5, 1.0)
    return np.clip(np.column_stack([m, m * 0.985, m * 0.92]), 0.0, 1.0)


def _flower_colors(fv, size, rng):
    zt = fv[:, 2] / (size * 1.45 + 1e-9)
    shade = 0.90 + 0.10 * np.clip(zt, 0.0, 1.0)
    m = 1.0 + rng.uniform(-0.06, 0.0)
    val = np.clip(shade * m, 0.5, 1.0)
    return np.column_stack([val, val, val])


# ===========================================================================
# Atlas tile mapping
# ===========================================================================
def _to_tile(uv01, tile, grid, flip=False, inset=0.02):
    row = tile // grid
    col = tile % grid
    s = uv01[:, 0].copy()
    t = uv01[:, 1].copy()
    if flip:
        s = 1.0 - s
    s = inset + s * (1.0 - 2.0 * inset)
    t = inset + t * (1.0 - 2.0 * inset)
    return np.column_stack([(col + s) / grid, (row + t) / grid])


# ===========================================================================
# Accumulator
# ===========================================================================
class _Accum:
    def __init__(self):
        self.V, self.F, self.UV, self.C, self.n = [], [], [], [], 0

    def add(self, v, f, uv, c):
        self.V.append(np.asarray(v, dtype=float))
        self.F.append(np.asarray(f, dtype=np.int64) + self.n)
        self.UV.append(np.asarray(uv, dtype=float))
        self.C.append(np.asarray(c, dtype=float))
        self.n += len(v)

    def empty(self):
        return self.n == 0

    def pack(self):
        return (np.vstack(self.V), np.vstack(self.F),
                np.vstack(self.UV), np.vstack(self.C))


# ===========================================================================
# Density presets
# ===========================================================================
def _config(density):
    presets = {
        "high": dict(stem_samples=40, sides_main=10, sides_branch=6,
                     n_main_leaves=9, n_sec_leaves=4, lobes=7,
                     n_flowers=30, thin_sides=5),
        "med":  dict(stem_samples=28, sides_main=8, sides_branch=5,
                     n_main_leaves=6, n_sec_leaves=3, lobes=5,
                     n_flowers=18, thin_sides=4),
        "low":  dict(stem_samples=18, sides_main=5, sides_branch=4,
                     n_main_leaves=4, n_sec_leaves=2, lobes=5,
                     n_flowers=10, thin_sides=3),
    }
    return presets.get(density, presets["high"])


# ===========================================================================
# Geometry entry point
# ===========================================================================
def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    cfg = _config(density)

    stem = _Accum()
    foliage = _Accum()
    flowers = _Accum()

    # -- main stem (kept nearly vertical, gentle lean) -----------------------
    Hp = PLANT_HEIGHT * rng.uniform(0.92, 1.08)
    n = cfg["stem_samples"]
    base_dir = _n([rng.uniform(-0.03, 0.03), 1.0, rng.uniform(-0.03, 0.03)])
    lean_a = rng.uniform(0.0, 2.0 * np.pi)
    lean_amp = Hp * rng.uniform(0.04, 0.08)
    bend_vec = np.array([np.cos(lean_a) * lean_amp, 0.0, np.sin(lean_a) * lean_amp])
    pts_m = _grow_stem([0.0, 0.0, 0.0], base_dir, Hp, n, bend_vec, Hp * 0.010, rng)

    t = np.linspace(0.0, 1.0, n)
    r_m = STEM_TIP_R + (STEM_BASE_R - STEM_TIP_R) * (1.0 - t) ** 1.2
    r_m *= 1.0 + BASAL_FLARE * np.clip((0.06 - t) / 0.06, 0.0, None)
    v, f, uv = _tube(pts_m, r_m, cfg["sides_main"], STEM_VSCALE, cap=True)
    stem.add(v, f, uv, _stem_colors(v[:, 1], Hp))
    _, Nm, Bm = _frames(pts_m)

    def out_dir(N, B, phi):
        return _n(np.cos(phi) * N + np.sin(phi) * B)

    # -- secondary (forked) stem : near-vertical, just a slight splay --------
    fk = int(round(FORK_FRAC * (n - 1)))
    a_sec = lean_a + np.pi + rng.uniform(-0.4, 0.4)
    out_h = out_dir(Nm[fk], Bm[fk], a_sec)
    main_tan = _n(pts_m[fk + 1] - pts_m[fk])
    sec_dir = _n(main_tan + out_h * 0.16)          # mostly upward, tiny outward
    Ls = Hp * rng.uniform(0.42, 0.55)
    ns = max(10, n // 2)
    bend_s = out_h * (Ls * 0.06) + np.array([0.0, Ls * 0.03, 0.0])
    pts_s = _grow_stem(pts_m[fk], sec_dir, Ls, ns, bend_s, Ls * 0.012, rng)
    ts = np.linspace(0.0, 1.0, ns)
    r_s = (r_m[fk] * 0.8) * (1.0 - ts) ** 1.1 + 0.0008
    v, f, uv = _tube(pts_s, r_s, cfg["sides_branch"], STEM_VSCALE, cap=True)
    stem.add(v, f, uv, _stem_colors(v[:, 1], Hp))
    _, Ns, Bs = _frames(pts_s)

    # -- leaves : short petioles, blades held steeply UP-and-out -------------
    def place_leaf(P, out, size):
        pet_len = size * rng.uniform(0.30, 0.55)
        leaf_base = P + out * pet_len
        pp = np.linspace(P, leaf_base, 3)
        pr = np.linspace(0.0011, 0.0008, 3)
        pv, pf, puv = _tube(pp, pr, cfg["thin_sides"], 0.02)
        stem.add(pv, pf, puv, _stem_colors(pv[:, 1], Hp))

        lv, lf, luv01 = _leaf_card(size, rng)
        # steep upward attitude collapses horizontal projection -> slender
        hwt = rng.uniform(0.38, 0.55)
        fwd = _n(out * hwt + np.array([0.0, 0.92, 0.0]))
        R = _basis_from_z(fwd)
        world = leaf_base + lv @ R.T
        tile = int(rng.integers(0, GRID * GRID))
        flip = bool(rng.integers(0, 2))
        uv = _to_tile(luv01, tile, GRID, flip)
        col = _leaf_colors(lv, size, leaf_base[1], Hp, rng)
        foliage.add(world, lf, uv, col)

    phi0 = rng.uniform(0.0, 2.0 * np.pi)
    for i, fr in enumerate(np.linspace(0.12, FOLIAGE_TOP - 0.04, cfg["n_main_leaves"])):
        idx = int(round(fr * (n - 1)))
        phi = phi0 + i * np.pi + rng.uniform(-0.35, 0.35)
        out = out_dir(Nm[idx], Bm[idx], phi)
        size = float(np.interp(fr, [0.1, FOLIAGE_TOP], [0.052, 0.028]))
        place_leaf(pts_m[idx], out, size)

    for i, fr in enumerate(np.linspace(0.28, 0.90, cfg["n_sec_leaves"])):
        idx = int(round(fr * (ns - 1)))
        phi = phi0 + i * np.pi + np.pi * 0.5 + rng.uniform(-0.35, 0.35)
        out = out_dir(Ns[idx], Bs[idx], phi)
        size = float(np.interp(fr, [0.3, 1.0], [0.044, 0.026]))
        place_leaf(pts_s[idx], out, size)

    # -- flower spike : tight, narrow, many small florets --------------------
    GOLDEN = 2.399963229728653
    for i, fr in enumerate(np.linspace(1.0 - SPIKE_FRAC, 0.99, cfg["n_flowers"])):
        idx = int(round(fr * (n - 1)))
        phi = phi0 + i * GOLDEN
        out = out_dir(Nm[idx], Bm[idx], phi)
        pdir = _n(out * 0.45 + np.array([0.0, -0.22, 0.0]))   # gentle nod, hugs stem
        plen = rng.uniform(0.004, 0.009)
        tip = pts_m[idx] + pdir * plen
        pp = np.linspace(pts_m[idx], tip, 3)
        pr = np.linspace(0.0009, 0.0006, 3)
        pv, pf, puv = _tube(pp, pr, cfg["thin_sides"], 0.02)
        stem.add(pv, pf, puv, _stem_colors(pv[:, 1], Hp))

        size = float(np.interp(fr, [1.0 - SPIKE_FRAC, 1.0], [0.013, 0.007]))
        fv, ff, fuv01 = _flower_cards(size, rng)
        R = _basis_from_z(pdir)
        world = tip + fv @ R.T
        white = float(np.clip((fr - (1.0 - SPIKE_FRAC)) / SPIKE_FRAC, 0.0, 1.0))
        row = int(round(white * (GRID - 1)))            # upper flowers -> whiter row
        colt = int(rng.integers(0, GRID))
        uv = _to_tile(fuv01, row * GRID + colt, GRID, flip=False)
        col = _flower_colors(fv, size, rng)
        flowers.add(world, ff, uv, col)

    # -- recentre on XZ, base to y=0, assemble (geometry + UV + COLOR_0) ------
    groups = [("stem", stem), ("foliage", foliage), ("flowers", flowers)]
    allV = np.vstack([g.pack()[0] for _, g in groups if not g.empty()])
    cx = 0.5 * (allV[:, 0].min() + allV[:, 0].max())
    cz = 0.5 * (allV[:, 2].min() + allV[:, 2].max())
    my = allV[:, 1].min()
    offset = np.array([-cx, -my, -cz])

    scene = trimesh.Scene()
    for name, g in groups:
        if g.empty():
            continue
        V, F, UV, C = g.pack()
        mesh = trimesh.Trimesh(vertices=V + offset, faces=F, process=False)
        vis = TextureVisuals(uv=UV)
        rgba = np.concatenate(
            [np.clip(C, 0.0, 1.0) * 255.0, np.full((len(C), 1), 255.0)], axis=1
        ).astype(np.uint8)
        vis.vertex_attributes["color"] = rgba
        mesh.visual = vis
        scene.add_geometry(mesh, geom_name=name)

    return scene


# ===========================================================================
# Photo sampling + de-lighting
# ===========================================================================
def _med(px, fallback):
    if px is None or len(px) < 8:
        return np.asarray(fallback, dtype=float)
    return np.median(px, axis=0).astype(float)


def sample_palette(path):
    """De-light the photo, then robustly sample body colours from INSIDE the
    silhouette (never the neutral background)."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=float)
    H, W, _ = arr.shape

    lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    lpil = Image.fromarray(np.clip(lum, 0, 255).astype(np.uint8), "L")
    blur = np.asarray(lpil.filter(ImageFilter.GaussianBlur(max(H, W) / 12.0)), float)
    gain = np.clip(lum.mean() / (blur + 1e-3), 0.6, 1.6)          # clamped de-light
    delit = np.clip(arr * gain[..., None], 0, 255)

    k = max(8, min(H, W) // 20)
    corners = [delit[:k, :k], delit[:k, -k:], delit[-k:, :k], delit[-k:, -k:]]
    bg = np.median(np.vstack([c.reshape(-1, 3) for c in corners]), axis=0)

    flat = delit.reshape(-1, 3)
    dist = np.linalg.norm(flat - bg, axis=1)
    obj = flat[dist > 22.0]
    if len(obj) < 50:
        obj = flat
    R, G, B = obj[:, 0], obj[:, 1], obj[:, 2]

    gm = (G - np.maximum(R, B) > 6.0) & (G > 40.0)
    greens = obj[gm] if gm.sum() > 50 else obj
    fol_med = _med(greens, [96, 140, 74])
    fol_dark = np.percentile(greens, 22, axis=0)
    fol_light = np.percentile(greens, 82, axis=0)

    score = (R - G) + (B - G) + 0.20 * (R + G + B) / 3.0
    thr = np.percentile(score, 97)
    lil = obj[score >= thr]
    lilac = _med(lil, [188, 178, 216])

    sm = (R - G > -6.0) & (B < G + 15.0) & ((R + G + B) / 3.0 < 175.0)
    sm &= ((R + G + B) / 3.0 > lum.mean() * 0.35)
    stem_px = obj[sm]
    if len(stem_px) > 40:
        stem_med = np.median(stem_px, axis=0).astype(float)
    else:
        stem_med = np.clip(fol_dark * np.array([1.06, 0.90, 0.80]), 0, 255)

    white = np.clip(lilac + (255.0 - lilac) * 0.72, 0, 255)
    return dict(
        fol_dark=np.clip(fol_dark, 0, 255), fol_med=np.clip(fol_med, 0, 255),
        fol_light=np.clip(fol_light, 0, 255), lilac=np.clip(lilac, 0, 255),
        white=white, stem=np.clip(stem_med, 0, 255),
        stem_dark=np.clip(stem_med * 0.70, 0, 255),
    )


# ===========================================================================
# Texture synthesis
# ===========================================================================
def _mirror_seamless(arr):
    """Mirror-fold into a perfectly tileable image (edges wrap by reflection)."""
    h, w = arr.shape[:2]
    qh, qw = h // 2, w // 2
    q = arr[:qh, :qw]
    top = np.concatenate([q, q[:, ::-1]], axis=1)
    full = np.concatenate([top, top[::-1]], axis=0)
    img = Image.fromarray(np.clip(full, 0, 255).astype(np.uint8), "RGB")
    return np.asarray(img.resize((w, h), Image.LANCZOS), dtype=float)


def _upnoise(shape_small, out, rng):
    a = rng.random(shape_small)
    im = Image.fromarray((a * 255).astype(np.uint8)).resize((out, out), Image.LANCZOS)
    return np.asarray(im, dtype=float) / 255.0


def make_stem_texture(res, base, dark, rng):
    base = np.asarray(base, float)
    dark = np.asarray(dark, float)
    streak = _upnoise((res, max(4, res // 16)), res, rng)   # vertical fibres
    fine = _upnoise((res // 4, res // 4), res, rng)
    t = np.clip(streak, 0, 1)[..., None]
    col = dark[None, None, :] * (1.0 - t) + base[None, None, :] * t
    col = col * (0.90 + 0.16 * fine[..., None])
    col = _mirror_seamless(np.clip(col, 0, 255))
    return Image.fromarray(col.astype(np.uint8), "RGB")


def make_normal(rgb_img, strength=1.6):
    arr = np.asarray(rgb_img, float) / 255.0
    h = 1.0 - (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2])
    gy, gx = np.gradient(h)
    nx, ny, nz = -gx * strength, -gy * strength, np.ones_like(h)
    l = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    nmap = np.stack([nx / l * 0.5 + 0.5, ny / l * 0.5 + 0.5, nz / l * 0.5 + 0.5], -1)
    return Image.fromarray((nmap * 255).astype(np.uint8), "RGB")


def _palmate_outline(S, n_lobes, rng):
    """Lobed palmate outline filling the tile; base at bottom-centre.
    Broad lobes (shallow notches) so it reads as a full palmate star."""
    cx, cy = S * 0.5, S * 0.90
    R = S * 0.80
    tips = np.deg2rad(np.linspace(-82.0, 82.0, n_lobes))
    pts = [(cx, cy)]
    for i, a in enumerate(tips):
        if i > 0:
            mid = 0.5 * (tips[i - 1] + a)
            nr = 0.52 * R                       # shallower notch -> broader lobes
            pts.append((cx + np.sin(mid) * nr, cy - np.cos(mid) * nr))
        tr = R * rng.uniform(0.88, 1.02)
        pts.append((cx + np.sin(a) * tr, cy - np.cos(a) * tr))
    pts.append((cx, cy))
    return cx, cy, pts


def _draw_leaf_tile(ts, base, rng):
    """RGBA palmate leaf with binary alpha-cutout silhouette + venation."""
    ss = 4
    S = ts * ss
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    n_lobes = int(rng.integers(5, 8))
    cx, cy, pts = _palmate_outline(S, n_lobes, rng)
    fill = (int(base[0]), int(base[1]), int(base[2]), 255)
    d.polygon(pts, fill=fill)

    vein = tuple(np.clip(base * 1.16, 0, 255).astype(int)) + (255,)
    for a in np.deg2rad(np.linspace(-82, 82, n_lobes)):
        ex = cx + np.sin(a) * S * 0.74
        ey = cy - np.cos(a) * S * 0.76
        d.line([(cx, cy), (ex, ey)], fill=vein, width=max(1, S // 220))
    d.line([(cx, cy), (cx, S * 0.12)], fill=vein, width=max(1, S // 180))

    arr = np.asarray(img, float)
    alpha = arr[..., 3:4].copy()
    nz = _upnoise((max(8, S // 16),) * 2, S, rng)
    arr[..., :3] = np.clip(arr[..., :3] * (0.90 + 0.20 * nz[..., None]), 0, 255)
    arr[..., 3:4] = alpha
    img = Image.fromarray(arr.astype(np.uint8), "RGBA")
    return img.resize((ts, ts), Image.LANCZOS)


def make_foliage_atlas(res, grid, st, rng):
    ts = res // grid
    atlas = Image.new("RGBA", (res, res), (0, 0, 0, 0))
    for r in range(grid):
        for c in range(grid):
            lit = (r + c) / (2.0 * (grid - 1))               # sun/shade gradient
            base = st["fol_dark"] * (1.0 - lit) + st["fol_light"] * lit
            base = np.clip(base + np.array([10, 4, -6]) * lit, 0, 255)
            tile = _draw_leaf_tile(ts, base, rng)
            atlas.paste(tile, (c * ts, r * ts))
    return atlas


def _draw_flower_tile(ts, base, white, lilac, rng):
    """RGBA blossom card: clustered petals (binary alpha) with white throat."""
    ss = 4
    S = ts * ss
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bx, by = S * 0.5, S * 0.90
    fill = (int(base[0]), int(base[1]), int(base[2]), 255)
    pr = S * 0.26
    for a in np.deg2rad(np.linspace(-78, 78, 5)):
        pc = (bx + np.sin(a) * S * 0.42, by - np.cos(a) * S * 0.46)
        d.ellipse([pc[0] - pr, pc[1] - pr, pc[0] + pr, pc[1] + pr], fill=fill)
    d.ellipse([bx - pr * 1.1, by - S * 0.78, bx + pr * 1.1, by - S * 0.20], fill=fill)

    arr = np.asarray(img, float)
    alpha = arr[..., 3:4].copy() / 255.0
    yy = np.repeat(np.linspace(0.0, 1.0, S)[:, None], S, axis=1)[..., None]
    wb = np.clip((yy - 0.45) / 0.55, 0.0, 1.0) * 0.7          # throat whitening
    arr[..., :3] = arr[..., :3] * (1.0 - wb) + white[None, None, :] * wb
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")
    d = ImageDraw.Draw(img)
    vein = tuple(np.clip(lilac * 0.78, 0, 255).astype(int)) + (255,)
    for a in np.deg2rad(np.linspace(-45, 45, 5)):
        ex = bx + np.sin(a) * S * 0.34
        ey = by - np.cos(a) * S * 0.62
        d.line([(bx, by), (ex, ey)], fill=vein, width=max(1, S // 280))
    out = np.asarray(img, float)
    out[..., 3:4] = alpha * 255.0                            # veins clipped to petals
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA").resize(
        (ts, ts), Image.LANCZOS)


def make_flower_atlas(res, grid, st, rng):
    ts = res // grid
    atlas = Image.new("RGBA", (res, res), (0, 0, 0, 0))
    for r in range(grid):
        for c in range(grid):
            paleness = r / (grid - 1)                          # row0 lilac -> white
            base = np.clip(st["lilac"] * (1.0 - paleness) + st["white"] * paleness, 0, 255)
            tile = _draw_flower_tile(ts, base, st["white"], st["lilac"], rng)
            atlas.paste(tile, (c * ts, r * ts))
    return atlas


# ===========================================================================
# Materials
# ===========================================================================
def _material(tex, roughness, alpha_mode="OPAQUE", double_sided=False,
              normal=None, cutoff=0.5):
    kw = dict(
        name="mat",
        baseColorTexture=tex,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicFactor=0.0,
        roughnessFactor=float(roughness),
        alphaMode=alpha_mode,
        doubleSided=bool(double_sided),
    )
    if alpha_mode == "MASK":
        kw["alphaCutoff"] = float(cutoff)
    if normal is not None:
        kw["normalTexture"] = normal
    return PBRMaterial(**kw)


def texture_scene(scene, palette, seed):
    rng = np.random.default_rng(seed + 7919)
    stem_tex = make_stem_texture(512, palette["stem"], palette["stem_dark"], rng)
    stem_nrm = make_normal(stem_tex, strength=1.6)
    fol_atlas = make_foliage_atlas(1024, GRID, palette, rng)     # RGBA cutout
    flo_atlas = make_flower_atlas(1024, GRID, palette, rng)      # RGBA cutout

    mats = {
        "stem":    _material(stem_tex, 0.90, "OPAQUE", double_sided=False, normal=stem_nrm),
        "foliage": _material(fol_atlas, 0.85, "MASK", double_sided=True, cutoff=0.5),
        "flowers": _material(flo_atlas, 0.55, "MASK", double_sided=True, cutoff=0.5),
    }
    for name, geom in scene.geometry.items():
        if name in mats:
            geom.visual.material = mats[name]
    return scene


# ===========================================================================
# CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Procedural wildflower -> textured GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    palette = sample_palette(args.image)
    scene = build_mesh(args.seed, args.density)
    scene = texture_scene(scene, palette, args.seed)

    data = scene.export(file_type="glb")
    with open(args.output, "wb") as fh:
        fh.write(data)
    print("wrote", args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("ERROR: {}\n".format(exc))
        sys.exit(1)