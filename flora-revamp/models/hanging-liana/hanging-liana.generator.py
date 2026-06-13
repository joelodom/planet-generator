"""
Procedural woody climbing vine / liana -- geometry + photo-derived materials,
exported as a single textured GLB.

Geometry: a thick, rope-like woody stem that twists into real overlapping
loops (loose corkscrew -> prominent coil knot -> ascent -> hook) with thin
tendrils and small clustered leaf cards.  Materials are derived from the
reference photo: a tileable pale fibrous-bark albedo+normal for the wood and
a 4x4 leaf-cluster atlas for the foliage cards.

Surfaces (semantic geometry keys):
    "branches" : woody fibrous-bark vine + tendrils (cylindrical UVs)
    "leaves"   : flat leaf cards mapped onto an atlas (MASK alpha)

Only numpy + trimesh + PIL + stdlib.  Deterministic in --seed.
+Y up; base at y=0; centred in X/Z; meters.
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial


# ---------------------------------------------------------------------------
# Measured proportions / real-world scale (METERS)
# ---------------------------------------------------------------------------
TOTAL_HEIGHT      = 1.18           # overall vertical extent of the vine (m)
HEIGHT_OVER_WIDTH = 2.4            # measured silhouette ratio (photo ~0.39 w/h)
CROWN_WIDTH       = TOTAL_HEIGHT / HEIGHT_OVER_WIDTH   # ~0.49 m horiz. extent
STEM_RADIUS       = 0.024          # chunky, rope-like main stem radius (m)
STEM_TIP_TAPER    = 0.42           # how much the stem thins toward the tip
BASAL_FLARE       = 0.30           # mild thickening over the bottom ~6%
TENDRIL_RADIUS_F  = 0.50           # child tendril radius vs. local main radius
LEAF_LENGTH       = 0.070          # a fresh leaf is ~7 cm long
LEAF_WIDTH        = 0.058          # rounded/heart-shaped leaf width (wider)

_BARK_TILE_V      = 0.085          # bark texture repeats every ~8.5 cm of length
_ATLAS_GRID       = 4              # 4x4 leaf atlas


# ---------------------------------------------------------------------------
# Small math helpers
# ---------------------------------------------------------------------------
def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / n if n > eps else v


def _rotate(v, axis, angle):
    """Rodrigues rotation of vector v about a unit axis."""
    axis = _normalize(axis)
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


def _fbm1(s, rng, octaves, base_freq, amp, decay=0.5):
    """Smooth 1-D fractal noise: a sum of sines with random phases."""
    out = np.zeros_like(s, dtype=float)
    f, a = float(base_freq), float(amp)
    for _ in range(octaves):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        out += a * np.sin(2.0 * np.pi * f * s + phase)
        f *= 2.0
        a *= decay
    return out


def _resample(pts, n):
    """Resample a polyline to n points evenly spaced by arc length."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cl = np.concatenate([[0.0], np.cumsum(seg)])
    total = cl[-1]
    if total <= 0:
        return np.repeat(pts[:1], n, axis=0)
    target = np.linspace(0.0, total, n)
    out = np.empty((n, 3))
    for k in range(3):
        out[:, k] = np.interp(target, cl, pts[:, k])
    return out


def _parallel_frames(points):
    """Twist-minimising (parallel-transport) frames along a polyline."""
    n = len(points)
    tang = np.zeros((n, 3))
    tang[:-1] = np.diff(points, axis=0)
    tang[-1] = tang[-2]
    nrm = np.linalg.norm(tang, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    tang /= nrm

    normals = np.zeros((n, 3))
    t0 = tang[0]
    seed_ax = np.array([0.0, 0.0, 1.0]) if abs(t0[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    normals[0] = _normalize(np.cross(t0, seed_ax))
    for i in range(1, n):
        v = np.cross(tang[i - 1], tang[i])
        vn = np.linalg.norm(v)
        prev = normals[i - 1]
        if vn < 1e-8:
            cur = prev
        else:
            axis = v / vn
            angle = np.arctan2(vn, np.dot(tang[i - 1], tang[i]))
            cur = _rotate(prev, axis, angle)
        cur = cur - tang[i] * np.dot(cur, tang[i])
        normals[i] = _normalize(cur)
    binorm = np.cross(tang, normals)
    return tang, normals, binorm


# ---------------------------------------------------------------------------
# Tube sweep (woody stem & tendrils) -- returns verts, faces, cylindrical UVs
# ---------------------------------------------------------------------------
def _build_tube(points, radii, sides, cap=True):
    _, N, B = _parallel_frames(points)
    n = len(points)
    cols = sides + 1                       # duplicate seam column for clean UVs
    ang = 2.0 * np.pi * np.arange(cols) / sides
    cos_a, sin_a = np.cos(ang), np.sin(ang)

    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    verts = np.empty((n * cols, 3))
    uv = np.empty((n * cols, 2))
    u_col = np.arange(cols) / sides
    for i in range(n):
        ring = points[i] + radii[i] * (np.outer(cos_a, N[i]) + np.outer(sin_a, B[i]))
        verts[i * cols:(i + 1) * cols] = ring
        uv[i * cols:(i + 1) * cols, 0] = u_col
        uv[i * cols:(i + 1) * cols, 1] = cum[i] / _BARK_TILE_V

    i_idx = np.arange(n - 1)[:, None]
    j_idx = np.arange(sides)[None, :]
    a = i_idx * cols + j_idx
    b = i_idx * cols + (j_idx + 1)
    c = (i_idx + 1) * cols + (j_idx + 1)
    d = (i_idx + 1) * cols + j_idx
    f1 = np.stack([a, b, c], axis=-1).reshape(-1, 3)
    f2 = np.stack([a, c, d], axis=-1).reshape(-1, 3)
    faces = [f1, f2]

    if cap:
        cb = len(verts)
        ct = cb + 1
        verts = np.vstack([verts, points[0], points[-1]])
        uv = np.vstack([uv, [0.5, cum[0] / _BARK_TILE_V], [0.5, cum[-1] / _BARK_TILE_V]])
        j = np.arange(sides)
        jn = j + 1
        bottom = np.stack([np.full(sides, cb), jn, j], axis=-1)
        top_base = (n - 1) * cols
        top = np.stack([np.full(sides, ct), top_base + j, top_base + jn], axis=-1)
        faces += [bottom, top]

    return verts, np.vstack(faces).astype(np.int64), uv


# ---------------------------------------------------------------------------
# Main vine centreline -- bigger swirl radii & fuller turns -> real loops
# ---------------------------------------------------------------------------
def _main_centerline(rng, n_points):
    """A vertical, twisting space-curve with overlapping loops:
    loose corkscrew -> prominent coil knot -> ascent -> hook."""
    raw = np.linspace(0.0, 1.0, 1500)

    # vertical profile: nearly flat across the coil (so it crosses over
    # itself into a knot), slight hook crest near the top.
    sy = np.array([0.0, 0.30, 0.48, 0.60, 0.85, 0.92, 1.00])
    yk = np.array([0.0, 0.30, 0.49, 0.58, 0.88, 1.00, 0.95])
    yk = yk + rng.uniform(-0.02, 0.02, yk.shape)
    yk[0] = 0.0
    y = TOTAL_HEIGHT * np.interp(raw, sy, yk)

    # swirl radius: fat loose corkscrew at the bottom, big coil in the middle.
    sr = np.array([0.00, 0.15, 0.30, 0.40, 0.55, 0.62, 0.75, 0.90, 1.00])
    rk = np.array([0.18, 0.23, 0.15, 0.09, 0.26, 0.23, 0.10, 0.16, 0.22])
    rk = rk * CROWN_WIDTH * (1.0 + rng.uniform(-0.12, 0.12, rk.shape))
    r = np.interp(raw, sr, rk)

    # accumulated turns: ~1.5 turns corkscrew, ~1.3-turn coil knot.
    st = np.array([0.00, 0.30, 0.48, 0.60, 0.85, 1.00])
    tk = np.array([0.00, 1.55, 1.75, 3.05, 3.30, 3.75])
    tk = tk + np.concatenate([[0.0], rng.uniform(-0.06, 0.06, 5)])
    tk = np.maximum.accumulate(tk)
    spin = 1.0 if rng.random() < 0.5 else -1.0
    theta = 2.0 * np.pi * np.interp(raw, st, tk) * spin

    cx = _fbm1(raw, rng, 2, 0.9, 0.07 * CROWN_WIDTH)
    cz = _fbm1(raw, rng, 2, 1.1, 0.07 * CROWN_WIDTH)

    x = cx + r * np.cos(theta)
    z = cz + r * np.sin(theta)
    pts = np.stack([x, y, z], axis=1)

    pts[:, 0] += _fbm1(raw, rng, 3, 1.5, 0.013)
    pts[:, 1] += _fbm1(raw, rng, 3, 1.7, 0.011)
    pts[:, 2] += _fbm1(raw, rng, 3, 1.6, 0.013)

    return _resample(pts, n_points)


def _stem_radii(n, rng):
    """Gnarled, gradually-tapering radius profile for the main stem."""
    u = np.linspace(0.0, 1.0, n)
    taper = 1.0 - STEM_TIP_TAPER * u
    flare = 1.0 + BASAL_FLARE * np.exp(-u / 0.06)
    gnarl = 1.0 + 0.16 * _fbm1(u, rng, 3, 4.0, 1.0)
    return STEM_RADIUS * taper * flare * np.clip(gnarl, 0.5, 1.7)


# ---------------------------------------------------------------------------
# Tendrils (thin branches that trail off the main vine)
# ---------------------------------------------------------------------------
def _build_tendril(start, frame, base_radius, n_pts, length, sides, rng):
    t_axis, n_axis, b_axis = frame
    ang = rng.uniform(0.0, 2.0 * np.pi)
    direction = _normalize(n_axis * np.cos(ang) + b_axis * np.sin(ang)
                           + t_axis * rng.uniform(-0.2, 0.2))
    curl_axis = _normalize(np.cross(direction, np.array([0.0, 1.0, 0.0]))
                           + rng.uniform(-0.2, 0.2, 3))
    curl = rng.uniform(0.05, 0.13) * (1.0 if rng.random() < 0.5 else -1.0)
    droop = rng.uniform(0.10, 0.22)

    step = length / n_pts
    pos = np.array(start, dtype=float)
    pts = [pos.copy()]
    for _ in range(n_pts):
        direction = _rotate(direction, curl_axis, curl)
        direction = _normalize(direction + np.array([0.0, -droop, 0.0]))
        pos = pos + direction * step
        pts.append(pos.copy())
    pts = np.asarray(pts)

    u = np.linspace(0.0, 1.0, len(pts))
    radii = base_radius * (1.0 - 0.8 * u) + 0.0010
    return _build_tube(pts, radii, sides, cap=True), pts[-1], direction


# ---------------------------------------------------------------------------
# Leaves (wider rounded cards clustered at nodes; bases sit on the stem)
# ---------------------------------------------------------------------------
def _make_leaf(base, out_dir, length, width, rng):
    # modest jitter so leaves mostly face outward (fewer edge-on slivers)
    d = _normalize(out_dir + rng.uniform(-0.18, 0.18, 3))
    up = np.array([0.0, 1.0, 0.0])
    w = np.cross(d, up)
    if np.linalg.norm(w) < 1e-6:
        w = np.cross(d, np.array([1.0, 0.0, 0.0]))
    w = _normalize(w)
    # wider quad (base pair + tip pair) so an edge-on card is less needle-like
    half = width * 0.5
    b0 = base - w * half * 0.55
    b1 = base + w * half * 0.55
    mid = base + d * length * 0.55
    t0 = mid - w * half
    t1 = mid + w * half
    tip = base + d * length
    verts = np.array([b0, b1, t1, tip, t0])        # 5-pt rounded leaf
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]])
    return verts, faces


def _leaf_tile_uv(rng):
    """UVs for one rounded leaf (5 verts: b0,b1,t1,tip,t0) onto a random atlas
    tile with a random 0/90/180/270 rotation."""
    col = int(rng.integers(0, _ATLAS_GRID))
    row = int(rng.integers(0, _ATLAS_GRID))
    rot = int(rng.integers(0, 4))
    cu = (col + 0.5) / _ATLAS_GRID
    cv = (row + 0.5) / _ATLAS_GRID
    h = (0.5 / _ATLAS_GRID) * 0.92

    def rot2(px, py):
        # rotate (px,py) in [-1,1] about tile centre by rot*90 deg
        for _ in range(rot):
            px, py = -py, px
        return (cu + px * h, cv + py * h)

    # local leaf-space coords -> tile space
    return np.array([
        rot2(-0.55, -1.0),   # b0
        rot2(0.55, -1.0),    # b1
        rot2(1.0, 0.1),      # t1
        rot2(0.0, 1.0),      # tip
        rot2(-1.0, 0.1),     # t0
    ])


def _spawn_cluster(center, out_dir, count, rng):
    """A small spray of leaves whose bases meet at a stem node."""
    V, F, UV = [], [], []
    off = 0
    for _ in range(count):
        jitter = center + _normalize(out_dir) * rng.uniform(0.0, 0.014) \
            + rng.uniform(-0.012, 0.012, 3)
        ln = LEAF_LENGTH * rng.lognormal(0.0, 0.20)
        wd = LEAF_WIDTH * rng.lognormal(0.0, 0.18)
        v, f = _make_leaf(jitter, out_dir, ln, wd, rng)
        V.append(v)
        F.append(f + off)
        UV.append(_leaf_tile_uv(rng))
        off += len(v)
    return np.vstack(V), np.vstack(F), np.vstack(UV)


# ---------------------------------------------------------------------------
# Per-vertex COLOR_0 tints (multiply the texture in glTF)
# ---------------------------------------------------------------------------
def _wood_colors(V):
    y = V[:, 1]
    hf = (y - y.min()) / (np.ptp(y) + 1e-9)             # darker/lower, lighter/up
    base = 0.88 + 0.16 * hf + 0.05 * np.sin(V[:, 0] * 37.0 + V[:, 2] * 29.0)
    base = np.clip(base, 0.72, 1.12)
    c = np.stack([base * 1.05, base * 1.00, base * 0.93], axis=1)   # warm undertone
    c = np.clip(c, 0.0, 1.0)
    rgba = np.concatenate([c, np.ones((len(c), 1))], axis=1)
    return (rgba * 255).astype(np.uint8)


def _leaf_colors(V):
    y = V[:, 1]
    hf = (y - y.min()) / (np.ptp(y) + 1e-9)             # outer/top brighter
    base = 0.80 + 0.30 * hf + 0.05 * np.sin(V[:, 0] * 41.0 + V[:, 2] * 23.0)
    base = np.clip(base, 0.62, 1.14)
    c = np.stack([base * 0.92, base * 1.05, base * 0.80], axis=1)   # fresh green tint
    c = np.clip(c, 0.0, 1.0)
    rgba = np.concatenate([c, np.ones((len(c), 1))], axis=1)
    return (rgba * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Entry point: geometry (UVs + vertex colours attached, materials added later)
# ---------------------------------------------------------------------------
def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)

    cfg = {
        "high": dict(sides=14, n_main=400, n_tendrils=6, tend_pts=30,
                     clusters_main=5, leaf_lo=10, leaf_hi=18),
        "med":  dict(sides=10, n_main=240, n_tendrils=4, tend_pts=22,
                     clusters_main=4, leaf_lo=7, leaf_hi=12),
        "low":  dict(sides=7,  n_main=140, n_tendrils=2, tend_pts=14,
                     clusters_main=2, leaf_lo=3, leaf_hi=6),
    }.get(density, None)
    if cfg is None:
        raise ValueError("density must be one of 'high', 'med', 'low'")

    sides = cfg["sides"]

    main = _main_centerline(rng, cfg["n_main"])
    main_r = _stem_radii(cfg["n_main"], rng)
    tang, nrm, bnm = _parallel_frames(main)

    wood_V, wood_F, wood_UV = [], [], []
    voff = 0

    def add_wood(v, f, uv):
        nonlocal voff
        wood_V.append(v)
        wood_F.append(f + voff)
        wood_UV.append(uv)
        voff += len(v)

    v, f, uv = _build_tube(main, main_r, sides, cap=True)
    add_wood(v, f, uv)

    leaf_nodes = []   # (point, outward_direction)

    lo, hi = int(0.12 * cfg["n_main"]), int(0.92 * cfg["n_main"])
    if cfg["n_tendrils"] > 0 and hi > lo:
        idxs = np.linspace(lo, hi, cfg["n_tendrils"]).astype(int)
        idxs = idxs + rng.integers(-4, 5, idxs.shape)
        idxs = np.clip(idxs, 1, cfg["n_main"] - 2)
        for k in idxs:
            frame = (tang[k], nrm[k], bnm[k])
            br = max(main_r[k] * TENDRIL_RADIUS_F, 0.005)
            length = rng.uniform(0.14, 0.24)
            (tv, tf, tuv), tip, tip_dir = _build_tendril(
                main[k], frame, br, cfg["tend_pts"], length, max(sides - 3, 5), rng)
            add_wood(tv, tf, tuv)
            leaf_nodes.append((tip, tip_dir))

    mlo, mhi = int(0.18 * cfg["n_main"]), int(0.82 * cfg["n_main"])
    midx = np.linspace(mlo, mhi, cfg["clusters_main"]).astype(int)
    for k in midx:
        out = _normalize(nrm[k] * rng.uniform(-1, 1) + bnm[k] * rng.uniform(-1, 1)
                         + np.array([0.0, 0.5, 0.0]))
        leaf_nodes.append((main[k], out))

    leaf_V, leaf_F, leaf_UV = [], [], []
    loff = 0
    for center, out_dir in leaf_nodes:
        count = int(rng.integers(cfg["leaf_lo"], cfg["leaf_hi"] + 1))
        lv, lf, luv = _spawn_cluster(center, out_dir, count, rng)
        leaf_V.append(lv)
        leaf_F.append(lf + loff)
        leaf_UV.append(luv)
        loff += len(lv)

    wood_V = np.vstack(wood_V)
    wood_F = np.vstack(wood_F)
    wood_UV = np.vstack(wood_UV)
    leaf_V = np.vstack(leaf_V)
    leaf_F = np.vstack(leaf_F)
    leaf_UV = np.vstack(leaf_UV)

    # ground & centre the whole object
    all_pts = np.vstack([wood_V, leaf_V])
    min_y = all_pts[:, 1].min()
    cen_x = 0.5 * (all_pts[:, 0].min() + all_pts[:, 0].max())
    cen_z = 0.5 * (all_pts[:, 2].min() + all_pts[:, 2].max())
    shift = np.array([cen_x, min_y, cen_z])
    wood_V = wood_V - shift
    leaf_V = leaf_V - shift

    wood = trimesh.Trimesh(vertices=wood_V, faces=wood_F, process=False)
    wood.visual = TextureVisuals(uv=wood_UV)
    wood.visual.vertex_attributes["color"] = _wood_colors(wood_V)

    leaves = trimesh.Trimesh(vertices=leaf_V, faces=leaf_F, process=False)
    leaves.visual = TextureVisuals(uv=leaf_UV)
    leaves.visual.vertex_attributes["color"] = _leaf_colors(leaf_V)

    scene = trimesh.Scene()
    scene.add_geometry(wood, geom_name="branches")
    scene.add_geometry(leaves, geom_name="leaves")
    return scene


# ===========================================================================
# Texturing -- derive palettes & swatches from the reference photo
# ===========================================================================
def _lum(c):
    return float(np.dot(np.asarray(c)[:3], [0.299, 0.587, 0.114]))


def sample_palette(image_path):
    """Sample bark & foliage colours from WELL INSIDE the silhouette.

    The vine is a thin shape on a flat grey ground, so we discard low-
    saturation background pixels and split the remaining body pixels by hue
    into brown (wood) and green (foliage), taking robust medians (and value
    extremes) rather than one rectangle.  De-lighting is approximated by
    sampling across the whole body (mixing lit ridges and shaded grooves);
    the wood is then biased toward its pale, sun-bleached ridge tone so it
    does not collapse to a dark silhouette in render."""
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((420, 420), Image.LANCZOS)
    a = np.asarray(img, dtype=float) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1)
    mn = a.min(-1)
    sat = mx - mn

    wood = (r >= g - 0.03) & (g >= b - 0.03) & (r > b + 0.05) & \
           (sat > 0.05) & (mx > 0.12) & (mx < 0.97)
    green = (g > r + 0.025) & (g > b + 0.015) & (sat > 0.05)

    wp = a[wood].reshape(-1, 3)
    gp = a[green].reshape(-1, 3)
    pal = {}

    if len(wp) > 50:
        lum = wp @ np.array([0.299, 0.587, 0.114])
        order = np.argsort(lum)
        q = max(1, len(wp) // 4)
        dark = np.median(wp[order[:q]], axis=0)
        light = np.median(wp[order[-q:]], axis=0)
        mid = np.median(wp, axis=0)
    else:
        dark = np.array([0.24, 0.16, 0.10])
        mid = np.array([0.48, 0.36, 0.24])
        light = np.array([0.74, 0.64, 0.50])

    # lift toward weathered pale tan / grey-beige (photo reads light overall)
    dark = np.clip(dark * 1.10 + 0.03, 0.08, 0.55)
    mid = np.clip(mid * 1.15 + 0.05, 0.20, 0.75)
    light = np.clip(light * 1.25 + 0.10, 0.50, 0.92)
    if _lum(light) < 2.0 * _lum(dark):
        light = np.clip(light * (2.0 * _lum(dark) / (_lum(light) + 1e-6)), 0, 0.94)
    mid = np.clip(mid * np.array([1.05, 0.98, 0.92]), 0, 0.85)  # reddish grooves
    pal["wood_dark"], pal["wood_mid"], pal["wood_light"] = dark, mid, light

    if len(gp) > 30:
        lum = gp @ np.array([0.299, 0.587, 0.114])
        order = np.argsort(lum)
        q = max(1, len(gp) // 3)
        gd = np.median(gp[order[:q]], axis=0)
        gl = np.median(gp[order[-q:]], axis=0)
    else:
        gd = np.array([0.26, 0.38, 0.13])
        gl = np.array([0.58, 0.74, 0.32])
    pal["leaf_dark"] = np.clip(gd, 0.04, 0.8)
    pal["leaf_light"] = np.clip(gl * np.array([1.02, 1.06, 0.92]), 0.12, 0.94)
    return pal


def build_bark(size, pal, rng):
    """Tileable fibrous bark albedo (stringy longitudinal grain, mottled
    pale sun-bleached ridges over darker grooves) + a derived tangent-space
    normal map.  All noise is periodic over [0,1) so the swatch tiles
    seamlessly (no seam blur)."""
    N = size
    lin = np.linspace(0.0, 1.0, N, endpoint=False)
    U, Vv = np.meshgrid(lin, lin)          # U = around, Vv = along the stem

    fib = np.zeros((N, N))                  # stringy longitudinal fibres
    for k in range(7):
        fu = int(rng.integers(10, 46))
        fv = int(rng.integers(0, 3))
        ph = rng.uniform(0, 2 * np.pi)
        fib += (1.0 / (k + 1)) * np.sin(2 * np.pi * (fu * U + fv * Vv) + ph)
    fib = (fib - fib.min()) / (np.ptp(fib) + 1e-9)
    fib = np.abs(2.0 * fib - 1.0)          # creases -> grooves

    mot = np.zeros((N, N))                  # broad sun-bleached patches
    for k in range(4):
        fu = int(rng.integers(1, 5))
        fv = int(rng.integers(1, 6))
        ph = rng.uniform(0, 2 * np.pi)
        mot += (1.0 / (k + 1)) * np.sin(2 * np.pi * (fu * U + fv * Vv) + ph)
    mot = (mot - mot.min()) / (np.ptp(mot) + 1e-9)

    fine = np.zeros((N, N))                 # fine grain
    for k in range(3):
        fu = int(rng.integers(40, 92))
        fv = int(rng.integers(1, 6))
        ph = rng.uniform(0, 2 * np.pi)
        fine += (1.0 / (k + 1)) * np.sin(2 * np.pi * (fu * U + fv * Vv) + ph)
    fine = (fine - fine.min()) / (np.ptp(fine) + 1e-9)

    # bias toward ridges (lighter) so the weathered pale tone dominates
    t = np.clip(0.18 + 0.55 * (1.0 - fib) + 0.27 * mot + 0.12 * fine, 0.0, 1.0)
    dark = np.asarray(pal["wood_dark"])
    mid = np.asarray(pal["wood_mid"])
    light = np.asarray(pal["wood_light"])
    t2 = t[..., None]
    lowc = dark + (mid - dark) * np.clip(t2 * 2.0, 0, 1)
    highc = mid + (light - mid) * np.clip((t2 - 0.5) * 2.0, 0, 1)
    col = np.where(t2 < 0.5, lowc, highc)
    col = col * (0.93 + 0.14 * fine[..., None])     # per-texel value wear
    col = np.clip(col, 0.0, 1.0)
    bark = Image.fromarray((col * 255).astype(np.uint8), "RGB")

    # normal map from height (inverse luminance -> grooves recede), tileable
    h = col @ np.array([0.299, 0.587, 0.114])
    gx = np.roll(h, -1, 1) - np.roll(h, 1, 1)
    gy = np.roll(h, -1, 0) - np.roll(h, 1, 0)
    s = 2.4
    nx, ny, nz = -gx * s, -gy * s, np.ones_like(h)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    nrm = np.stack([nx / ln, ny / ln, nz / ln], axis=-1)
    normal = Image.fromarray(((nrm * 0.5 + 0.5) * 255).astype(np.uint8), "RGB")
    return bark, normal


def _leaf_polygon(cx, cy, length, angle, width_f=0.46):
    """A rounded, heart-ish leaf silhouette (widest mid, soft rounded base,
    gently pointed tip)."""
    ax = np.array([np.cos(angle), np.sin(angle)])
    pp = np.array([-ax[1], ax[0]])
    start = np.array([cx, cy]) - ax * length * 0.5
    ts = np.linspace(0.0, 1.0, 14)
    left, right = [], []
    for tt in ts:
        w = length * width_f * (np.sin(np.pi * tt) ** 0.5)   # rounder than 0.6
        p = start + ax * length * tt
        left.append(tuple(p + pp * w))
        right.append(tuple(p - pp * w))
    return left + right[::-1]


def _draw_leaf_tile(S, pal, rng, sunlit):
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    gl = np.asarray(pal["leaf_light"])
    gd = np.asarray(pal["leaf_dark"])
    n = int(rng.integers(4, 8))
    for _ in range(n):
        cx = S * rng.uniform(0.32, 0.68)
        cy = S * rng.uniform(0.32, 0.68)
        length = S * rng.uniform(0.50, 0.78)
        angle = rng.uniform(0, 2 * np.pi)
        mix = rng.uniform(0.0, 1.0)
        col = gd + (gl - gd) * mix
        if sunlit:
            col = np.clip(col * 1.18 + 0.04, 0, 1)        # sunlit: brighter/warmer
        else:
            col = np.clip(col * 0.82, 0, 1)               # shaded: darker/cooler
        rgbc = tuple(int(c * 255) for c in col)
        poly = _leaf_polygon(cx, cy, length, angle)
        d.polygon(poly, fill=rgbc + (255,))              # binary alpha (opaque fill)
        ax = np.array([np.cos(angle), np.sin(angle)])
        s0 = (cx - ax[0] * length * 0.5, cy - ax[1] * length * 0.5)
        s1 = (cx + ax[0] * length * 0.5, cy + ax[1] * length * 0.5)
        vein = tuple(int(c * 0.62 * 255) for c in col)
        d.line([s0, s1], fill=vein + (255,), width=max(1, S // 200))
    return im


def build_leaf_atlas(pal, rng, size=1024, grid=4, ss=4):
    """4x4 atlas of distinct leaf-cluster tiles; top rows sunlit, drawn at 4x
    supersample then LANCZOS-downscaled for clean (near-binary) edges."""
    atlas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    tile = size // grid
    for row in range(grid):
        for col in range(grid):
            sunlit = row < 2
            big = _draw_leaf_tile(tile * ss, pal, rng, sunlit)
            small = big.resize((tile, tile), Image.LANCZOS)
            atlas.paste(small, (col * tile, row * tile))
    return atlas


def apply_textures(scene, image_path, seed):
    pal = sample_palette(image_path)
    rng = np.random.default_rng(seed + 12345)            # deterministic swatches

    bark_img, bark_normal = build_bark(1024, pal, rng)
    atlas = build_leaf_atlas(pal, rng, size=1024, grid=_ATLAS_GRID)

    bark_mat = PBRMaterial(
        name="bark",
        baseColorTexture=bark_img,
        normalTexture=bark_normal,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.9,
    )
    leaf_mat = PBRMaterial(
        name="foliage",
        baseColorTexture=atlas,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.8,
        alphaMode="MASK",
        alphaCutoff=0.45,
        doubleSided=True,
    )

    scene.geometry["branches"].visual.material = bark_mat
    scene.geometry["leaves"].visual.material = leaf_mat
    return scene


# ===========================================================================
# CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Procedural woody vine -> textured GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        scene = build_mesh(args.seed, args.density)
        scene = apply_textures(scene, args.image, args.seed)
        glb = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(glb)
    except Exception as exc:  # noqa: BLE001
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1
    print("Wrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())