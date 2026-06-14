"""
Standalone procedural generator + texturer for a windswept juniper /
bonsai-styled conifer.  Builds geometry, derives tileable materials from a
reference photo, applies UVs by surface type, and exports a textured GLB.

CLI:
    python juniper.py --image PATH --seed INT --density {high,med,low} \
        --output OUT.glb

Only numpy / trimesh / PIL / stdlib are used.  Deterministic given --seed.
"""

import argparse
import sys

import numpy as np
import trimesh
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial
from PIL import Image, ImageDraw, ImageFilter


# ============================================================================
# GEOMETRY  (from the validated module, lightly edited to emit clean UVs)
# ============================================================================

# ----------------------------------------------------------------------------
# Measured proportions (estimated by eye off the reference, ~10% accuracy).
# ----------------------------------------------------------------------------
TREE_HEIGHT_M          = 0.95   # overall height (small alpine/bonsai tree), m
HEIGHT_OVER_WIDTH      = 1.05   # bounding box is nearly square, slightly tall
TRUNK_FRAC_OF_HEIGHT   = 0.60   # trunk/deadwood dominates the lower ~60%
BASE_MOUND_FRAC        = 0.10   # the soil mound is ~10% of total height
CANOPY_OFFSET_RIGHT    = 0.19   # crown center pushed to +X (cantilever) ~0.19*H
CROWN_W_OVER_CROWN_H   = 1.6    # the foliage pad is broad: wider than tall
TRUNK_BASE_FLARE       = 1.5    # basal root-flare multiplier over bottom ~10%

H = 0.90                        # internal working height (scaled at the end)

# Foliage envelope (a lobed ellipsoid). Broad, low, and only modestly offset
# right so it sits as a stout pad hugging the trunk top (not a ball on a neck).
CANOPY_CX = 0.19 * H
CANOPY_CY = 0.60 * H
CANOPY_CZ = 0.00 * H
CANOPY_AX = 0.295 * H           # half-width  (X) -- broad pad
CANOPY_AY = 0.210 * H           # half-height (Y) -- short
CANOPY_AZ = 0.250 * H           # half-depth  (Z)
CANOPY_CENTER = np.array([CANOPY_CX, CANOPY_CY, CANOPY_CZ])
CROWN_WIDTH = 2.0 * CANOPY_AX

# Texturing / UV constants
_TILE_M_WOOD  = 0.10            # meters of trunk surface per texture tile
_TILE_M_MOUND = 0.13            # meters of ground surface per texture tile
ATLAS         = 4               # 4x4 foliage atlas
_UV_MARGIN    = 1.5 / 1024.0    # atlas tile inset to avoid bleeding


def _density_params(density):
    if density == "low":
        return dict(trunk_sides=8,  trunk_stations=18, branch_sides=5,
                    branch_stations=8,  n_primary=2, n_secondary=0,
                    root_sides=5, root_stations=6, n_roots=0,
                    cards_total=360, n_clumps=7,  mound_subdiv=1)
    if density == "med":
        return dict(trunk_sides=12, trunk_stations=32, branch_sides=6,
                    branch_stations=12, n_primary=3, n_secondary=1,
                    root_sides=6, root_stations=8, n_roots=3,
                    cards_total=1200, n_clumps=12, mound_subdiv=2)
    return dict(trunk_sides=18, trunk_stations=48, branch_sides=8,
                branch_stations=16, n_primary=4, n_secondary=1,
                root_sides=6, root_stations=8, n_roots=5,
                cards_total=2800, n_clumps=18, mound_subdiv=3)


def _catmull_rom(points, n_per_seg):
    """Smooth interpolating polyline through `points` (Nx3)."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts.copy()
    p0 = pts[0] + (pts[0] - pts[1])
    pN = pts[-1] + (pts[-1] - pts[-2])
    ext = np.vstack([p0, pts, pN])
    out = []
    t = np.linspace(0.0, 1.0, n_per_seg, endpoint=False)[:, None]
    for i in range(1, len(ext) - 2):
        P0, P1, P2, P3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        seg = 0.5 * ((2 * P1)
                     + (-P0 + P2) * t
                     + (2 * P0 - 5 * P1 + 4 * P2 - P3) * t ** 2
                     + (-P0 + 3 * P1 - 3 * P2 + P3) * t ** 3)
        out.append(seg)
    out.append(pts[-1][None, :])
    return np.vstack(out)


def _frames(curve):
    """Parallel-transport frames (tangent, normal, binormal) along a polyline."""
    t = np.gradient(curve, axis=0)
    t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    N = np.zeros_like(curve)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(t[0], ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    n0 = ref - t[0] * np.dot(ref, t[0])
    N[0] = n0 / (np.linalg.norm(n0) + 1e-9)
    for i in range(1, len(curve)):
        v, w, n = t[i - 1], t[i], N[i - 1]
        axis = np.cross(v, w)
        s = np.linalg.norm(axis)
        c = np.clip(np.dot(v, w), -1.0, 1.0)
        if s < 1e-8:
            N[i] = n
        else:
            axis /= s
            ang = np.arctan2(s, c)
            N[i] = (n * np.cos(ang)
                    + np.cross(axis, n) * np.sin(ang)
                    + axis * np.dot(axis, n) * (1 - np.cos(ang)))
            N[i] /= (np.linalg.norm(N[i]) + 1e-9)
    B = np.cross(t, N)
    return t, N, B


def _build_tube(curve, radii, sides, twist_total=0.0, flute_amp=0.0, flute_k=0):
    """Swept tube with spiral twist + fluting and clean cylindrical UVs.

    A duplicated seam column (sides+1 columns) keeps the U wrap from smearing.
    U runs around the circumference, V runs along the part's length (meters).
    """
    _, N, B = _frames(curve)
    m = len(curve)
    theta = np.linspace(0.0, 2 * np.pi, sides, endpoint=False)
    cols = sides + 1
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    varc = np.concatenate([[0.0], np.cumsum(seg)])
    mean_r = float(np.mean(radii))
    u_rep = max(1.0, (2 * np.pi * mean_r) / _TILE_M_WOOD)

    verts = np.zeros((m * cols, 3))
    uvs = np.zeros((m * cols, 2))
    for i in range(m):
        tw = twist_total * (i / max(m - 1, 1))
        for j in range(cols):
            jj = j % sides
            ang = theta[jj] + tw
            flute = 1.0 + flute_amp * np.cos(flute_k * ang) if flute_k > 0 else 1.0
            rr = radii[i] * flute
            verts[i * cols + j] = (curve[i]
                                   + np.cos(ang) * rr * N[i]
                                   + np.sin(ang) * rr * B[i])
            uvs[i * cols + j] = [(j / sides) * u_rep, varc[i] / _TILE_M_WOOD]

    faces = []
    for i in range(m - 1):
        for j in range(sides):
            a = i * cols + j
            b = i * cols + j + 1
            c = (i + 1) * cols + j + 1
            d = (i + 1) * cols + j
            faces.append((a, b, c))
            faces.append((a, c, d))

    verts = list(verts)
    uvs = list(uvs)
    c0 = len(verts); verts.append(np.asarray(curve[0])); uvs.append([0.5, 0.5])
    for j in range(sides):
        faces.append((c0, j + 1, j))
    base = (m - 1) * cols
    c1 = len(verts); verts.append(np.asarray(curve[-1])); uvs.append([0.5, 0.5])
    for j in range(sides):
        faces.append((c1, base + j, base + j + 1))

    return (np.asarray(verts, dtype=float),
            np.asarray(faces, dtype=np.int64),
            np.asarray(uvs, dtype=float))


class _Accum:
    """Accumulate (verts, faces, uvs) chunks into one indexed mesh."""
    def __init__(self):
        self.v = []
        self.f = []
        self.uv = []
        self.n = 0

    def add(self, verts, faces, uvs=None):
        verts = np.asarray(verts, dtype=float)
        self.v.append(verts)
        self.f.append(np.asarray(faces, dtype=np.int64) + self.n)
        if uvs is None:
            uvs = np.zeros((len(verts), 2))
        self.uv.append(np.asarray(uvs, dtype=float))
        self.n += len(verts)

    def result(self):
        if not self.v:
            return (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64),
                    np.zeros((0, 2)))
        return np.vstack(self.v), np.vstack(self.f), np.vstack(self.uv)


def _envelope_radius(direction):
    """Radius of the lobed foliage envelope along a direction."""
    d = np.asarray(direction, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-9)
    axes = np.array([CANOPY_AX, CANOPY_AY, CANOPY_AZ])
    base = 1.0 / np.sqrt(np.sum((d / axes) ** 2) + 1e-12)
    az = np.arctan2(d[2], d[0])
    el = np.arctan2(d[1], np.hypot(d[0], d[2]))
    lobe = 1.0 + 0.12 * np.cos(3 * az + 0.5) + 0.08 * np.cos(2 * el + 1.0)
    return base * lobe


def _build_wood(rng, p):
    acc = _Accum()
    branch_tips = []

    # Stout S-curve: a low twist-back then a moderate sweep right.  Kept less
    # extreme than before so the canopy sits over the trunk, not on a long neck.
    base_ctrl = np.array([
        [0.000, 0.000, 0.000],
        [0.020, 0.10 * H, 0.020],
        [-0.030, 0.20 * H, -0.020],
        [0.005, 0.30 * H, 0.020],
        [0.060, 0.40 * H, 0.000],
        [0.100, 0.48 * H, 0.015],
        [0.130, TRUNK_FRAC_OF_HEIGHT * H, 0.000],
    ])
    jit = rng.normal(scale=0.015, size=base_ctrl.shape)
    jit[0] = 0.0
    ctrl = base_ctrl + jit

    n_per_seg = max(2, p["trunk_stations"] // 6)
    trunk_curve = _catmull_rom(ctrl, n_per_seg)
    m = len(trunk_curve)

    # Thick, sculptural trunk -- a large fraction of the silhouette, like the
    # photo's massive bole.  Slow taper keeps it stout up to the canopy.
    R_BASE = 0.100
    R_TOP = 0.050
    s = np.linspace(0.0, 1.0, m)
    radii = R_TOP + (R_BASE - R_TOP) * (1.0 - s) ** 1.15
    flare = np.where(s < 0.10, 1.0 + (TRUNK_BASE_FLARE - 1.0) * (1.0 - s / 0.10), 1.0)
    radii = radii * flare

    tv, tf, tuv = _build_tube(trunk_curve, radii, p["trunk_sides"],
                              twist_total=1.3 * np.pi, flute_amp=0.07, flute_k=7)
    acc.add(tv, tf, tuv)

    def trunk_radius_at(idx):
        return radii[min(idx, m - 1)]

    primary_tips = []
    for k in range(p["n_primary"]):
        frac = 0.72 + 0.27 * (k / max(p["n_primary"] - 1, 1))
        si = int(np.clip(frac, 0.0, 0.999) * (m - 1))
        p0 = trunk_curve[si]
        # aim UP into the pad (positive Y bias) so limbs never poke out sideways
        d = np.array([rng.uniform(-0.1, 0.9),
                      rng.uniform(0.25, 1.0),
                      rng.uniform(-0.7, 0.7)])
        d /= np.linalg.norm(d)
        tip = CANOPY_CENTER + d * _envelope_radius(d) * rng.uniform(0.55, 0.82)
        mid = 0.5 * (p0 + tip) + rng.normal(scale=0.03, size=3)
        bcurve = _catmull_rom(np.array([p0, mid, tip]),
                              max(2, p["branch_stations"] // 2))
        rs = trunk_radius_at(si) * 0.55
        re = rs * 0.40
        br = np.linspace(rs, re, len(bcurve))
        bv, bf, buv = _build_tube(bcurve, br, p["branch_sides"],
                                  twist_total=0.6 * np.pi, flute_amp=0.06, flute_k=5)
        acc.add(bv, bf, buv)
        primary_tips.append((tip, re))
        branch_tips.append(tip)

    for (ptip, pr) in primary_tips:
        for _ in range(p["n_secondary"]):
            d = np.array([rng.uniform(-0.3, 1.0),
                          rng.uniform(0.0, 1.0),
                          rng.uniform(-0.8, 0.8)])
            d /= np.linalg.norm(d)
            tip = CANOPY_CENTER + d * _envelope_radius(d) * rng.uniform(0.6, 0.85)
            mid = 0.5 * (ptip + tip) + rng.normal(scale=0.02, size=3)
            bcurve = _catmull_rom(np.array([ptip, mid, tip]),
                                  max(2, p["branch_stations"] // 2))
            br = np.linspace(pr * 0.75, pr * 0.30, len(bcurve))
            bv, bf, buv = _build_tube(bcurve, br, max(4, p["branch_sides"] - 1),
                                      twist_total=0.4 * np.pi, flute_amp=0.05, flute_k=5)
            acc.add(bv, bf, buv)
            branch_tips.append(tip)

    for k in range(p["n_roots"]):
        ang = 2 * np.pi * k / max(p["n_roots"], 1) + rng.uniform(-0.3, 0.3)
        out = np.array([np.cos(ang), 0.0, np.sin(ang)])
        start = np.array([0.0, 0.06 * H, 0.0]) + out * 0.01
        knee = out * 0.075 + np.array([0.0, 0.035 * H, 0.0])
        end = out * (0.115 + rng.uniform(0.0, 0.03)) + np.array([0.0, 0.015 * H, 0.0])
        end[1] = max(end[1], 0.008)
        rcurve = _catmull_rom(np.array([start, knee, end]),
                              max(2, p["root_stations"] // 2))
        rr = np.linspace(R_BASE * 0.40, 0.008, len(rcurve))
        rv, rf, ruv = _build_tube(rcurve, rr, p["root_sides"],
                                  twist_total=0.3 * np.pi, flute_amp=0.07, flute_k=4)
        acc.add(rv, rf, ruv)

    return acc.result(), branch_tips


def _tile_rect(tile):
    """UV corners (ll, lr, ur, ul) of an atlas tile, with inset margin."""
    r = tile // ATLAS
    c = tile % ATLAS
    u0 = c / ATLAS + _UV_MARGIN
    u1 = (c + 1) / ATLAS - _UV_MARGIN
    v0 = r / ATLAS + _UV_MARGIN
    v1 = (r + 1) / ATLAS - _UV_MARGIN
    return [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]


def _build_canopy(rng, p, branch_tips):
    acc = _Accum()

    centers = [np.asarray(t, dtype=float) for t in branch_tips]
    while len(centers) < p["n_clumps"]:
        remaining = p["n_clumps"] - len(centers)
        interior = remaining <= max(2, p["n_clumps"] // 5)
        d = np.array([rng.uniform(-0.25, 1.0),
                      rng.uniform(-0.35, 1.0),
                      rng.uniform(-1.0, 1.0)])
        d /= np.linalg.norm(d)
        frac = rng.uniform(0.30, 0.48) if interior else rng.uniform(0.62, 0.95)
        centers.append(CANOPY_CENTER + d * _envelope_radius(d) * frac)
    centers = centers[:p["n_clumps"]]

    clump_radius = 0.13 * CROWN_WIDTH
    base_hs = 0.055 * CROWN_WIDTH
    per_clump = max(6, p["cards_total"] // max(len(centers), 1))
    card_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    for cc in centers:
        for _ in range(per_clump):
            u = rng.normal(size=3)
            u /= (np.linalg.norm(u) + 1e-9)
            rad = clump_radius * rng.random() ** (1.0 / 3.0)
            pos = cc + u * rad
            outd = pos - CANOPY_CENTER
            dist = np.linalg.norm(outd) + 1e-9
            er = _envelope_radius(outd)
            if dist > 1.03 * er:
                pos = CANOPY_CENTER + (outd / dist) * (1.03 * er)
                outd = pos - CANOPY_CENTER
                dist = np.linalg.norm(outd) + 1e-9
            nrm = outd / dist
            ref = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(nrm, ref)) > 0.9:
                ref = np.array([1.0, 0.0, 0.0])
            t1 = np.cross(ref, nrm); t1 /= (np.linalg.norm(t1) + 1e-9)
            t2 = np.cross(nrm, t1)
            phi = rng.uniform(0, 2 * np.pi)
            a = np.cos(phi) * t1 + np.sin(phi) * t2
            b = -np.sin(phi) * t1 + np.cos(phi) * t2
            jit = np.deg2rad(rng.uniform(-18, 18))
            a = a + np.tan(jit) * nrm * 0.5
            b = b + np.tan(jit) * nrm * 0.5
            a /= (np.linalg.norm(a) + 1e-9)
            b /= (np.linalg.norm(b) + 1e-9)
            hs = base_hs * float(np.exp(rng.normal(0.0, 0.30)))
            quad = np.array([
                pos - a * hs - b * hs,
                pos + a * hs - b * hs,
                pos + a * hs + b * hs,
                pos - a * hs + b * hs,
            ])
            # map this card onto a random atlas tile with a random 90deg turn
            corners = _tile_rect(int(rng.integers(0, ATLAS * ATLAS)))
            rot = int(rng.integers(0, 4))
            corners = corners[rot:] + corners[:rot]
            acc.add(quad, card_faces, np.array(corners, dtype=float))

    return acc.result()


def _build_mound(rng, p):
    ico = trimesh.creation.icosphere(subdivisions=p["mound_subdiv"], radius=1.0)
    v = ico.vertices.copy()
    d0 = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    MOUND_R = 0.16
    MOUND_H = BASE_MOUND_FRAC * H * 0.75
    v[:, 0] *= MOUND_R
    v[:, 2] *= MOUND_R
    v[:, 1] *= MOUND_H

    ph = rng.uniform(0, 2 * np.pi, size=4)
    az = np.arctan2(d0[:, 2], d0[:, 0])
    el = np.arctan2(d0[:, 1], np.hypot(d0[:, 0], d0[:, 2]))
    lump = (1.0 + 0.16 * np.sin(2 * az + ph[0]) + 0.10 * np.sin(3 * az + ph[1])
            + 0.08 * np.sin(2 * el + ph[2]) + 0.05 * np.sin(5 * az + ph[3]))
    v[:, 0] *= lump
    v[:, 2] *= lump
    v[:, 1] *= (0.6 + 0.4 * lump)
    v[:, 1] = np.maximum(v[:, 1], 0.0)
    return v, ico.faces.copy()


def _triplanar_uv(verts, normals, tile):
    """Bake a triplanar projection into a single per-vertex UV set."""
    an = np.abs(normals)
    uv = np.zeros((len(verts), 2))
    xd = (an[:, 0] >= an[:, 1]) & (an[:, 0] >= an[:, 2])
    yd = (an[:, 1] > an[:, 0]) & (an[:, 1] >= an[:, 2])
    zd = ~(xd | yd)
    uv[xd] = verts[xd][:, [2, 1]]
    uv[yd] = verts[yd][:, [0, 2]]
    uv[zd] = verts[zd][:, [0, 1]]
    return uv / tile


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    """Build the windswept-juniper geometry; UVs are stashed in mesh metadata."""
    rng = np.random.default_rng(seed)
    if density not in ("high", "med", "low"):
        density = "high"
    p = _density_params(density)

    (wv, wf, wuv), branch_tips = _build_wood(rng, p)
    cv, cf, cuv = _build_canopy(rng, p, branch_tips)
    mv, mf = _build_mound(rng, p)

    all_y = np.concatenate([wv[:, 1], cv[:, 1], mv[:, 1]])
    min_y = float(all_y.min())
    max_y = float(all_y.max())
    scale = TREE_HEIGHT_M / max((max_y - min_y), 1e-6)

    def finalize(v):
        out = v.copy() * scale
        out[:, 1] -= min_y * scale
        return out

    wv, cv, mv = finalize(wv), finalize(cv), finalize(mv)

    scene = trimesh.Scene()

    wood = trimesh.Trimesh(vertices=wv, faces=wf, process=False)
    wood.fix_normals()
    wood.metadata["uv"] = wuv
    scene.add_geometry(wood, geom_name="trunk")

    canopy = trimesh.Trimesh(vertices=cv, faces=cf, process=False)
    canopy.metadata["uv"] = cuv
    scene.add_geometry(canopy, geom_name="canopy")

    mound = trimesh.Trimesh(vertices=mv, faces=mf, process=False)
    mound.fix_normals()
    mound.metadata["uv"] = _triplanar_uv(mound.vertices, mound.vertex_normals,
                                         _TILE_M_MOUND)
    scene.add_geometry(mound, geom_name="mound")

    return scene


# ============================================================================
# PHOTO SAMPLING
# ============================================================================
def load_image(path):
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.float64) / 255.0


def estimate_bg(img):
    """Median color of the four image corners (assumed background)."""
    h, w, _ = img.shape
    s = max(4, min(h, w) // 12)
    corners = [img[:s, :s], img[:s, -s:], img[-s:, :s], img[-s:, -s:]]
    flat = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
    return np.median(flat, axis=0)


def delight(crop):
    """Divide out a heavily blurred luminance; clamp the gain to [0.6, 1.6]."""
    h, w, _ = crop.shape
    if h < 4 or w < 4:
        return crop
    lum = 0.299 * crop[..., 0] + 0.587 * crop[..., 1] + 0.114 * crop[..., 2]
    limg = Image.fromarray((np.clip(lum, 0, 1) * 255).astype(np.uint8))
    rad = max(2, min(h, w) // 3)
    blur = np.asarray(limg.filter(ImageFilter.GaussianBlur(rad)), dtype=float) / 255.0
    mean = float(np.mean(blur)) + 1e-6
    gain = np.clip(mean / (blur + 1e-6), 0.6, 1.6)
    return np.clip(crop * gain[..., None], 0, 1)


def sample_region(img, box, bg, rng, n=48):
    """Median of small de-lit patches inside `box`, rejecting background hits."""
    h, w, _ = img.shape
    x0, y0, x1, y1 = box
    xs0, xs1 = int(x0 * w), int(x1 * w)
    ys0, ys1 = int(y0 * h), int(y1 * h)
    crop = img[ys0:ys1, xs0:xs1]
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    crop = delight(crop)
    ch, cw, _ = crop.shape
    cols = []
    for _ in range(n):
        px = int(rng.integers(0, max(1, cw - 3)))
        py = int(rng.integers(0, max(1, ch - 3)))
        c = np.median(crop[py:py + 3, px:px + 3].reshape(-1, 3), axis=0)
        if np.linalg.norm(c - bg) > 0.09:        # discard background-like patches
            cols.append(c)
    if len(cols) < 3:
        return None
    return np.median(np.array(cols), axis=0)


def get_color(img, box, bg, rng, default):
    c = sample_region(img, box, bg, rng)
    return np.array(default, dtype=float) if c is None else c


# ============================================================================
# TEXTURE SYNTHESIS  (tileable; colored from the photo palette)
# ============================================================================
def fft_noise(size, rng, beta=2.4, aniso=(1.0, 1.0)):
    """Tileable fractal noise via a 1/f^beta spectrum (periodic by FFT)."""
    fx = np.fft.fftfreq(size)[None, :]
    fy = np.fft.fftfreq(size)[:, None]
    f = np.sqrt((fx * aniso[0]) ** 2 + (fy * aniso[1]) ** 2)
    f[0, 0] = 1.0
    amp = f ** (-beta / 2.0)
    amp[0, 0] = 0.0
    ph = rng.uniform(0, 2 * np.pi, (size, size))
    spec = amp * (np.cos(ph) + 1j * np.sin(ph))
    img = np.fft.ifft2(spec).real
    return (img - img.min()) / (np.ptp(img) + 1e-9)


def height_to_normal(h, strength=2.0):
    """Tangent-space normal map from a tileable height field (np.roll = tile)."""
    gx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.5
    gy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 0.5
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(h)
    l = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    rgb = np.stack([nx / l, ny / l, nz / l], axis=-1) * 0.5 + 0.5
    return Image.fromarray((rgb * 255).astype(np.uint8), "RGB")


def make_wood(deadwood, bark, rng, S=512):
    """Sinewy bone-grey deadwood with ropey grey-brown bark strips spiralling."""
    Y, X = np.mgrid[0:S, 0:S] / float(S)
    grain = fft_noise(S, rng, beta=2.2, aniso=(1.0, 5.0))   # grain along length
    fine = fft_noise(S, rng, beta=2.0, aniso=(1.0, 2.5))
    # Only thin ropey strips of bark twist over an otherwise pale deadwood body.
    spiral = 0.5 + 0.5 * np.sin(2 * np.pi * (3.0 * (X + 0.55 * Y)) + 2.2 * grain)
    bark_mask = np.clip((spiral - 0.62) / 0.14, 0, 1) * (0.55 + 0.45 * fine)
    bark_mask = np.clip(bark_mask * 1.15, 0, 1)

    alb = (deadwood[None, None, :] * (1 - bark_mask)[..., None]
           + bark[None, None, :] * bark_mask[..., None])
    alb *= (0.90 + 0.18 * grain)[..., None]      # gentle grain value variation
    alb *= (0.94 + 0.10 * fine)[..., None]        # fine speckle
    alb = np.clip(alb, 0, 1)
    albimg = Image.fromarray((alb * 255).astype(np.uint8), "RGB")

    height = 0.55 * grain + 0.45 * bark_mask
    return albimg, height_to_normal(height, strength=1.6)


def make_mound(soil, stone, moss, rng, S=512):
    """Soil + stone mottling with dark cracks/veins and patches of moss."""
    mott = fft_noise(S, rng, beta=2.6)
    fine = fft_noise(S, rng, beta=2.0)
    ridg = fft_noise(S, rng, beta=2.3)
    cracks = 1.0 - np.abs(2.0 * ridg - 1.0)
    crack_mask = np.clip((cracks - 0.82) / 0.10, 0, 1)
    stone_mask = np.clip((mott - 0.50) / 0.25, 0, 1)
    mossn = fft_noise(S, rng, beta=3.0)
    moss_mask = np.clip((mossn - 0.62) / 0.12, 0, 1) * 0.7

    alb = (soil[None, None, :] * (1 - stone_mask)[..., None]
           + stone[None, None, :] * stone_mask[..., None])
    alb = alb * (1 - moss_mask)[..., None] + moss[None, None, :] * moss_mask[..., None]
    alb *= (1.0 - 0.55 * crack_mask)[..., None]   # darken cracks
    alb *= (0.85 + 0.30 * fine)[..., None]
    alb = np.clip(alb, 0, 1)
    albimg = Image.fromarray((alb * 255).astype(np.uint8), "RGB")

    height = 0.5 * mott + 0.5 * (1.0 - crack_mask)
    return albimg, height_to_normal(height, strength=2.0)


def _draw_tile(ts, base, rng):
    """One supersampled foliage-cluster tile (RGBA): prickly needle silhouette."""
    im = Image.new("RGBA", (ts, ts), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx = cy = ts / 2.0
    R = ts * 0.40

    def colj(f=0.12):
        c = np.clip(base * (1.0 + rng.uniform(-f, f, 3)), 0, 1)
        return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 255)

    # central overlapping clumps (keeps most texels opaque -> binary alpha)
    for _ in range(5):
        bx = cx + rng.uniform(-0.15, 0.15) * ts
        by = cy + rng.uniform(-0.15, 0.15) * ts
        rr = R * rng.uniform(0.5, 0.8)
        pts = []
        for i in range(14):
            a = 2 * np.pi * i / 14
            rad = rr * (0.7 + 0.3 * rng.random())
            pts.append((bx + np.cos(a) * rad, by + np.sin(a) * rad))
        d.polygon(pts, fill=colj(0.10))

    # radiating needle sprigs -> ragged, tufted edge
    for _ in range(160):
        a = rng.uniform(0, 2 * np.pi)
        r0 = rng.uniform(0.15, 0.5) * R
        r1 = r0 + rng.uniform(0.3, 0.6) * R
        x0 = cx + np.cos(a) * r0
        y0 = cy + np.sin(a) * r0
        x1 = cx + np.cos(a) * r1
        y1 = cy + np.sin(a) * r1
        w = ts * rng.uniform(0.006, 0.014)
        px = -np.sin(a) * w
        py = np.cos(a) * w
        d.polygon([(x0 + px, y0 + py), (x0 - px, y0 - py), (x1, y1)],
                  fill=colj(0.16))
    return im


def make_atlas(green, rng, S=1024):
    """4x4 atlas of distinct clusters; top rows sunlit/warm, bottom shaded/cool."""
    tile = S // ATLAS
    ss = 4
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    for r in range(ATLAS):
        for c in range(ATLAS):
            shade = r / (ATLAS - 1)
            bright = 1.12 - 0.42 * shade
            warm = np.array([0.03, 0.01, -0.02]) * (1.0 - shade)
            base = np.clip(green * bright + warm, 0, 1)
            tim = _draw_tile(tile * ss, base, rng)
            tim = tim.resize((tile, tile), Image.LANCZOS)
            out.paste(tim, (c * tile, r * tile), tim)
    return out


# ============================================================================
# PER-VERTEX TINTS (COLOR_0) -- multiply the texture in glTF
# ============================================================================
def _to_color(col):
    out = np.empty((len(col), 4), dtype=np.uint8)
    out[:, :3] = np.clip(col * 255.0, 0, 255).astype(np.uint8)
    out[:, 3] = 255
    return out


def canopy_tint(mesh, rng):
    """Outer/top brighter & warmer; inner/lower darker; per-clump variation."""
    P = mesh.vertices
    center = P.mean(axis=0)
    y = P[:, 1]
    topf = (y - y.min()) / (np.ptp(y) + 1e-9)
    rel = P - center
    dist = np.linalg.norm(rel, axis=1)
    outer = dist / (dist.max() + 1e-9)
    b = 0.62 + 0.28 * topf + 0.18 * outer
    ncard = max(1, len(P) // 4)
    jit = np.repeat(rng.uniform(0.85, 1.08, ncard), 4)[:len(P)]
    b = np.clip(b * jit, 0.30, 1.10)
    col = np.stack([b * 0.95, b * 1.03, b * 0.90], axis=1)   # cool/green bias
    return _to_color(np.clip(col, 0, 1))


def trunk_tint(mesh):
    """Mild ambient-occlusion darkening near the ground; keeps the pale wood
    bright overall so the deadwood never collapses to a dark silhouette."""
    P = mesh.vertices
    ao = np.clip(P[:, 1] / 0.30, 0, 1)
    b = 0.80 + 0.18 * ao
    col = np.stack([b * 1.02, b * 1.00, b * 0.96], axis=1)
    return _to_color(np.clip(col, 0, 1))


def mound_tint(mesh):
    """Up-facing patches lifted (moss/light); shaded sides darkened."""
    N = mesh.vertex_normals
    up = np.clip(N[:, 1], 0, 1)
    b = 0.70 + 0.25 * up
    col = np.stack([b * 0.95, b * 1.02, b * 0.90], axis=1)
    return _to_color(np.clip(col, 0, 1))


# ============================================================================
# MATERIAL ASSIGNMENT
# ============================================================================
def apply_materials(scene, img, seed):
    rng = np.random.default_rng(seed + 12345)
    bg = estimate_bg(img)

    # Sampling boxes (normalized, y-down) placed well inside the silhouette.
    deadwood = get_color(img, (0.28, 0.40, 0.46, 0.62), bg, rng, (0.78, 0.76, 0.70))
    bark     = get_color(img, (0.33, 0.58, 0.49, 0.74), bg, rng, (0.40, 0.32, 0.24))
    green    = get_color(img, (0.46, 0.14, 0.74, 0.40), bg, rng, (0.26, 0.40, 0.20))
    soil     = get_color(img, (0.32, 0.82, 0.64, 0.92), bg, rng, (0.34, 0.27, 0.18))
    moss     = get_color(img, (0.30, 0.80, 0.46, 0.90), bg, rng, (0.28, 0.38, 0.18))

    def _relight(c, target):
        """Keep the sampled hue but lift/normalize value to a target luminance."""
        lum = float(np.mean(c)) + 1e-6
        return np.clip(c * (target / lum), 0, 1)

    # The trunk is sun-bleached DEADWOOD: force it pale (bone/silver-grey) even
    # if the sampled patch sat in shadow.  Bark stays a mid grey-brown.
    deadwood = _relight(deadwood, 0.80)
    deadwood = np.clip(deadwood * 0.5 + np.array([0.40, 0.39, 0.37]), 0, 1)  # bony
    bark = _relight(bark, 0.42)
    # Saturated mid-to-deep green for the needle pad (sampled tone reads olive).
    green = np.clip(np.array([green[0] * 0.80, green[1] * 1.18, green[2] * 0.78]), 0, 1)
    green = np.clip(0.6 * green + 0.4 * np.array([0.20, 0.42, 0.16]), 0, 1)
    stone = np.clip(0.5 * soil + np.array([0.34, 0.33, 0.31]), 0, 1)

    wood_alb, wood_nrm = make_wood(deadwood, bark, rng, S=512)
    mound_alb, mound_nrm = make_mound(soil, stone, moss, rng, S=512)
    atlas = make_atlas(green, rng, S=1024)

    wood_mat = PBRMaterial(name="bark_deadwood",
                           baseColorTexture=wood_alb, normalTexture=wood_nrm,
                           baseColorFactor=[1, 1, 1, 1],
                           metallicFactor=0.0, roughnessFactor=0.9)
    mound_mat = PBRMaterial(name="soil_stone",
                            baseColorTexture=mound_alb, normalTexture=mound_nrm,
                            baseColorFactor=[1, 1, 1, 1],
                            metallicFactor=0.0, roughnessFactor=0.95)
    leaf_mat = PBRMaterial(name="needles",
                           baseColorTexture=atlas,
                           baseColorFactor=[1, 1, 1, 1],
                           metallicFactor=0.0, roughnessFactor=0.8,
                           alphaMode="MASK", alphaCutoff=0.45, doubleSided=True)

    for name, mesh in scene.geometry.items():
        uv = mesh.metadata["uv"]
        if name == "trunk":
            mat, tint = wood_mat, trunk_tint(mesh)
        elif name == "mound":
            mat, tint = mound_mat, mound_tint(mesh)
        else:
            mat, tint = leaf_mat, canopy_tint(mesh, rng)
        vis = TextureVisuals(uv=uv, material=mat)
        vis.vertex_attributes["color"] = tint
        mesh.visual = vis

    return scene


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Procedural windswept juniper -> GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        img = load_image(args.image)
        scene = build_mesh(args.seed, args.density)
        apply_materials(scene, img, args.seed)
        glb = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(glb)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()