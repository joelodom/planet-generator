"""
Standalone procedural generator + texturer for a mature deciduous broadleaf tree
(the "broadleaf-maple" archetype): a single sturdy trunk supporting a full, dense,
ovoid / bullet-shaped crown that is taller than it is wide.

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

# --- Measured proportions (read off the reference image) -------------------
TREE_HEIGHT          = 14.0            # overall height in METERS
HEIGHT_OVER_WIDTH    = 1.60           # crown clearly taller than wide (~0.63 aspect)
CROWN_WIDTH          = TREE_HEIGHT / HEIGHT_OVER_WIDTH   # ~8.75 m
CROWN_HALF_WIDTH     = CROWN_WIDTH * 0.5

CLEAR_BOLE_FRAC      = 0.14           # only a short clear bole; foliage drapes low
CROWN_BOTTOM_Y       = TREE_HEIGHT * CLEAR_BOLE_FRAC
CROWN_TOP_Y          = TREE_HEIGHT
CROWN_CENTER_Y       = 0.5 * (CROWN_BOTTOM_Y + CROWN_TOP_Y)
CROWN_HALF_HEIGHT    = 0.5 * (CROWN_TOP_Y - CROWN_BOTTOM_Y)

WIDEST_FRAC          = 0.48           # widest just below mid -> bullet narrowing up
TOP_NARROW           = 0.30           # top distinctly narrower than the bottom

FORK_Y               = TREE_HEIGHT * 0.30
TRUNK_BASE_R         = 0.40
TRUNK_FLARE          = 1.45
TRUNK_TOP_R          = 0.30

ENVELOPE_MARGIN      = 1.0            # keep all foliage inside a clean silhouette


def _density_params(density: str) -> dict:
    if density == "high":
        return dict(trunk_sides=14, branch_sides=8,
                    n_clumps=72, cards_lo=34, cards_hi=46,
                    n_primary=6, n_secondary=2)
    if density == "med":
        return dict(trunk_sides=10, branch_sides=6,
                    n_clumps=40, cards_lo=22, cards_hi=32,
                    n_primary=5, n_secondary=1)
    if density == "low":
        return dict(trunk_sides=7, branch_sides=5,
                    n_clumps=20, cards_lo=14, cards_hi=22,
                    n_primary=3, n_secondary=0)
    raise ValueError("density must be one of 'high', 'med', 'low'")


def _make_envelope(rng):
    # gentle low-frequency lobes -- kept small so the silhouette stays clean
    n_lobes = int(rng.integers(3, 6))
    freqs   = rng.integers(2, 5, size=n_lobes)
    phases  = rng.uniform(0.0, 2.0 * np.pi, size=n_lobes)
    amps    = rng.uniform(0.025, 0.06, size=n_lobes)
    vfreqs  = rng.uniform(0.6, 1.4, size=n_lobes)

    def horiz_scale(v):
        t = 2.0 * v - 1.0
        ellipse = np.sqrt(max(0.0, 1.0 - t * t))
        bullet  = 1.0 - TOP_NARROW * t
        bias = 1.0 - 0.08 * abs(v - WIDEST_FRAC) * 2.0
        return max(0.0, ellipse * bullet * bias)

    def lobe_gain(az, v):
        g = 1.0
        for k in range(n_lobes):
            g += amps[k] * np.sin(freqs[k] * az + phases[k] + vfreqs[k] * v * np.pi)
        return g

    def shell_point(az, v):
        y   = CROWN_BOTTOM_Y + v * (CROWN_TOP_Y - CROWN_BOTTOM_Y)
        rad = CROWN_HALF_WIDTH * horiz_scale(v) * lobe_gain(az, v)
        x   = rad * np.cos(az)
        z   = rad * np.sin(az)
        return np.array([x, y, z]), rad

    def outward_normal(p):
        d = np.array([p[0], (p[1] - CROWN_CENTER_Y) *
                      (CROWN_HALF_WIDTH / CROWN_HALF_HEIGHT) * 0.6, p[2]])
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])

    return shell_point, outward_normal


def _make_tube(centers, radii, sides, cap_start=False):
    centers = np.asarray(centers, dtype=float)
    radii   = np.asarray(radii, dtype=float)
    n = len(centers)

    tang = np.zeros_like(centers)
    tang[:-1] = centers[1:] - centers[:-1]
    tang[-1]  = tang[-2] if n > 1 else np.array([0.0, 1.0, 0.0])
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)

    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    rings = np.zeros((n, sides, 3))
    prev_nrm = None
    for i in range(n):
        t = tang[i]
        if prev_nrm is None:
            a = np.array([1.0, 0.0, 0.0]) if abs(t @ np.array([1.0, 0, 0])) < 0.9 \
                else np.array([0.0, 0.0, 1.0])
            nrm = np.cross(t, a)
        else:
            nrm = prev_nrm - t * (t @ prev_nrm)
        nrm /= (np.linalg.norm(nrm) + 1e-9)
        binr = np.cross(t, nrm)
        binr /= (np.linalg.norm(binr) + 1e-9)
        prev_nrm = nrm
        for j, ang in enumerate(angles):
            d = np.cos(ang) * nrm + np.sin(ang) * binr
            rings[i, j] = centers[i] + radii[i] * d

    verts = rings.reshape(-1, 3)
    faces = []
    for i in range(n - 1):
        for j in range(sides):
            j2 = (j + 1) % sides
            a = i * sides + j
            b = i * sides + j2
            c = (i + 1) * sides + j
            d = (i + 1) * sides + j2
            faces.append([a, c, b])
            faces.append([b, c, d])

    if cap_start:
        center_idx = len(verts)
        verts = np.vstack([verts, centers[0]])
        for j in range(sides):
            j2 = (j + 1) % sides
            faces.append([center_idx, j2, j])

    return verts, np.array(faces, dtype=np.int64)


def _build_trunk(rng, sides):
    n_rings = 7
    vs = np.linspace(0.0, 1.0, n_rings)
    centers, radii = [], []
    for v in vs:
        y = v * FORK_Y
        r = TRUNK_BASE_R * (1.0 - v) + TRUNK_TOP_R * v
        flare = 1.0 + (TRUNK_FLARE - 1.0) * max(0.0, 1.0 - y / (TREE_HEIGHT * 0.08)) ** 2
        r *= flare
        sway = 0.05 * TRUNK_BASE_R
        cx = sway * np.sin(v * 2.3 + rng.uniform(0, 6.28))
        cz = sway * np.cos(v * 1.7 + rng.uniform(0, 6.28))
        centers.append([cx, y, cz])
        radii.append(r)
    return _make_tube(centers, radii, sides, cap_start=True)


def _build_branches(rng, shell_point, sides, n_primary, n_secondary):
    """Primary/secondary limbs that stop WELL INSIDE the crown (stay hidden)."""
    all_v, all_f = [], []
    base_offset = 0

    fork = np.array([0.0, FORK_Y, 0.0])
    parent_r = TRUNK_TOP_R
    child_r = parent_r / np.sqrt(max(1, n_primary)) * 1.10

    az0 = rng.uniform(0, 2 * np.pi)
    for k in range(n_primary):
        az = az0 + 2 * np.pi * k / n_primary + rng.uniform(-0.25, 0.25)
        v_tip = rng.uniform(0.45, 0.88)
        tip, _ = shell_point(az, v_tip)
        tip[1] = CROWN_BOTTOM_Y + v_tip * (CROWN_TOP_Y - CROWN_BOTTOM_Y)
        tip *= 0.72                                     # end deep inside the foliage

        n_seg = 4
        ts = np.linspace(0.0, 1.0, n_seg)
        centers, radii = [], []
        start = fork + np.array([0.0, -parent_r * 0.6, 0.0])
        for t in ts:
            mid = (start + tip) * 0.5 + np.array([0, parent_r * 1.5, 0])
            p = (1 - t) ** 2 * start + 2 * (1 - t) * t * mid + t ** 2 * tip
            centers.append(p)
            radii.append(child_r * (1.0 - 0.75 * t) + 0.02)
        v, f = _make_tube(centers, radii, sides)
        all_v.append(v)
        all_f.append(f + base_offset)
        base_offset += len(v)

        for _ in range(n_secondary):
            s_az = az + rng.uniform(-0.6, 0.6)
            s_v = min(0.92, v_tip + rng.uniform(0.03, 0.12))
            s_tip, _ = shell_point(s_az, s_v)
            s_tip[1] = CROWN_BOTTOM_Y + s_v * (CROWN_TOP_Y - CROWN_BOTTOM_Y)
            s_tip *= 0.72
            s_start = centers[int(n_seg * 0.6)]
            s_r = child_r * 0.55
            sc, srad = [], []
            for t in np.linspace(0, 1, 3):
                p = (1 - t) * s_start + t * s_tip
                sc.append(p)
                srad.append(s_r * (1.0 - 0.7 * t) + 0.015)
            v2, f2 = _make_tube(sc, srad, max(4, sides - 1))
            all_v.append(v2)
            all_f.append(f2 + base_offset)
            base_offset += len(v2)

    V = np.vstack(all_v)
    F = np.vstack(all_f)
    return V, F


def _ortho_basis(n):
    a = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, a)
    u /= (np.linalg.norm(u) + 1e-9)
    w = np.cross(n, u)
    w /= (np.linalg.norm(w) + 1e-9)
    return u, w


def _build_canopy(rng, shell_point, outward_normal, params):
    """
    Dense, CONTINUOUS shell of leaf cards.  Clump centres are placed by a
    golden-angle spiral over the whole envelope so neighbours overlap and tile
    the silhouette with NO bald gaps; cards conform to the shell with a little
    inward thickness, and a few deeper interior clumps give the crown depth.
    """
    n_clumps = params["n_clumps"]
    cards_lo, cards_hi = params["cards_lo"], params["cards_hi"]

    # clump size scales so coverage stays continuous at every density
    clump_radius = CROWN_WIDTH * 0.13 * np.sqrt(72.0 / n_clumps)
    card_half    = CROWN_WIDTH * 0.05
    shell_thick  = CROWN_WIDTH * 0.10
    v_span       = CROWN_TOP_Y - CROWN_BOTTOM_Y
    ga = np.pi * (3.0 - np.sqrt(5.0))                   # golden angle

    n_interior = max(1, int(round(n_clumps * 0.12)))

    verts, faces = [], []
    vbase = 0
    for i in range(n_clumps + n_interior):
        interior = i >= n_clumps
        if not interior:
            v_c = float(np.clip((i + 0.5) / n_clumps + rng.uniform(-0.03, 0.03),
                                0.02, 1.0))
            az_c = i * ga + rng.uniform(-0.2, 0.2)
            depth_bias = 0.0
        else:
            v_c = float(rng.uniform(0.25, 0.9))
            az_c = float(rng.uniform(0, 2 * np.pi))
            depth_bias = CROWN_HALF_WIDTH * rng.uniform(0.25, 0.5)

        _, rad_c = shell_point(az_c, v_c)
        ang_spread = clump_radius / max(0.6, rad_c)
        v_spread   = clump_radius / v_span

        n_cards = int(rng.integers(cards_lo, cards_hi + 1))
        for _ in range(n_cards):
            az = az_c + rng.normal(0.0, ang_spread)
            v  = float(np.clip(v_c + rng.normal(0.0, v_spread), 0.01, 1.0))
            p, rmax = shell_point(az, v)
            nrm = outward_normal(p)
            depth = depth_bias + rng.uniform(0.0, shell_thick)
            pos = p - nrm * depth

            jn = nrm + rng.normal(0.0, 0.18, 3)
            jn /= (np.linalg.norm(jn) + 1e-9)
            u, w = _ortho_basis(jn)

            ang = rng.uniform(0, 2 * np.pi)
            hs = card_half * float(np.exp(rng.normal(0.0, 0.28)))
            hu = hs * rng.uniform(0.85, 1.15)
            hw = hs * rng.uniform(0.85, 1.15)
            e1 = np.cos(ang) * u + np.sin(ang) * w
            e2 = -np.sin(ang) * u + np.cos(ang) * w

            p0 = pos - hu * e1 - hw * e2
            p1 = pos + hu * e1 - hw * e2
            p2 = pos + hu * e1 + hw * e2
            p3 = pos - hu * e1 + hw * e2
            verts.extend([p0, p1, p2, p3])
            faces.append([vbase, vbase + 1, vbase + 2])
            faces.append([vbase, vbase + 2, vbase + 3])
            vbase += 4

    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    params = _density_params(density)

    shell_point, outward_normal = _make_envelope(rng)

    tv, tf = _build_trunk(rng, params["trunk_sides"])
    trunk = trimesh.Trimesh(vertices=tv, faces=tf, process=True)
    trunk.fix_normals()

    bv, bf = _build_branches(rng, shell_point, params["branch_sides"],
                             params["n_primary"], params["n_secondary"])
    branches = trimesh.Trimesh(vertices=bv, faces=bf, process=True)
    branches.fix_normals()

    # Canopy keeps its exact 4-verts-per-card layout for clean atlas UVs.
    cv, cf = _build_canopy(rng, shell_point, outward_normal, params)
    canopy = trimesh.Trimesh(vertices=cv, faces=cf, process=False)

    scene = trimesh.Scene()
    scene.add_geometry(trunk,    geom_name="trunk")
    scene.add_geometry(branches, geom_name="branches")
    scene.add_geometry(canopy,   geom_name="canopy")
    return scene


# ===========================================================================
# IMAGE SAMPLING
# ===========================================================================

def _load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def _patch(arr, nx, ny, half_frac):
    H, W = arr.shape[:2]
    cx, cy = int(nx * W), int(ny * H)
    hx = max(2, int(half_frac * W))
    hy = max(2, int(half_frac * H))
    y0, y1 = max(0, cy - hy), min(H, cy + hy)
    x0, x1 = max(0, cx - hx), min(W, cx + hx)
    return arr[y0:y1, x0:x1]


def _delight(patch):
    if patch.size == 0:
        return patch
    lum = patch @ np.array([0.299, 0.587, 0.114])
    h, w = lum.shape
    rad = max(2, min(h, w) // 4)
    lum_img = Image.fromarray((np.clip(lum, 0, 1) * 255).astype(np.uint8))
    blur = np.asarray(lum_img.filter(ImageFilter.GaussianBlur(rad)),
                      dtype=np.float64) / 255.0
    gain = np.clip(lum.mean() / (blur + 1e-3), 0.6, 1.6)
    return np.clip(patch * gain[..., None], 0.0, 1.0)


# Regions chosen by LOOKING at reference.png, WELL INSIDE the silhouette.
_CANOPY_CENTERS = [(0.50, 0.22), (0.40, 0.30), (0.60, 0.30), (0.33, 0.42),
                   (0.66, 0.42), (0.50, 0.40), (0.45, 0.55), (0.57, 0.55),
                   (0.50, 0.62), (0.42, 0.68), (0.58, 0.68), (0.50, 0.50)]
_TRUNK_CENTERS  = [(0.500, 0.880), (0.490, 0.920), (0.510, 0.860),
                   (0.500, 0.950), (0.485, 0.905)]

_FALLBACK_GREEN = np.array([0.34, 0.48, 0.18])
_FALLBACK_BARK  = np.array([0.43, 0.37, 0.31])


def _green_pool(arr, rng):
    pix = []
    for nx, ny in _CANOPY_CENTERS:
        d = _delight(_patch(arr, nx, ny, 0.04))
        if d.size:
            pix.append(d.reshape(-1, 3))
    if not pix:
        return _FALLBACK_GREEN[None, :] * np.ones((64, 1))
    pix = np.concatenate(pix, 0)
    g = (pix[:, 1] > pix[:, 0] * 1.02) & (pix[:, 1] > pix[:, 2] * 1.02)
    pool = pix[g] if g.sum() > 40 else pix
    if len(pool) == 0:
        pool = _FALLBACK_GREEN[None, :] * np.ones((64, 1))
    if len(pool) > 4000:
        idx = rng.choice(len(pool), 4000, replace=False)
        pool = pool[idx]
    return pool


def _bark_base(arr):
    cols = []
    for nx, ny in _TRUNK_CENTERS:
        d = _delight(_patch(arr, nx, ny, 0.012))
        if d.size == 0:
            continue
        med = np.median(d.reshape(-1, 3), axis=0)
        if med[0] >= med[1] * 0.9:
            cols.append(med)
    if not cols:
        return _FALLBACK_BARK.copy()
    base = np.median(np.array(cols), axis=0)
    if base.mean() < 0.18:
        base = base / (base.mean() + 1e-3) * 0.30
    return np.clip(base, 0.05, 0.95)


# ===========================================================================
# TEXTURES
# ===========================================================================

def _mirror_tile(a):
    a = np.concatenate([a, a[:, ::-1]], axis=1)
    a = np.concatenate([a, a[::-1, :]], axis=0)
    return a


def make_bark_albedo(base_rgb, size, rng):
    S = size
    u = np.linspace(0.0, 1.0, S, endpoint=False)[None, :]
    v = np.linspace(0.0, 1.0, S, endpoint=False)[:, None]

    wander = 0.04 * np.sin(2 * np.pi * 3 * v + 1.0) + 0.03 * np.sin(2 * np.pi * 7 * v)
    uu = u + wander

    fur = np.zeros((S, S))
    for f, a, ph in [(6, 1.0, 0.0), (11, 0.6, 1.3), (17, 0.35, 2.1), (23, 0.22, 0.5)]:
        fur += a * np.sin(2 * np.pi * f * uu + ph)
    fur = (fur - fur.min()) / (np.ptp(fur) + 1e-9)
    val = 0.50 + 0.50 * fur

    small = rng.normal(0.0, 1.0, (S // 8, S // 8))
    small = _mirror_tile(small)
    small = (small - small.min()) / (np.ptp(small) + 1e-9)
    mott = np.asarray(
        Image.fromarray((small * 255).astype(np.uint8)).resize((S, S), Image.LANCZOS),
        dtype=np.float64) / 255.0
    val *= (0.85 + 0.30 * mott)
    val = np.clip(val, 0.0, 1.25)

    rgb = np.clip(base_rgb[None, None, :] * val[..., None], 0.0, 1.0)
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def make_normal_map(albedo_img, strength=3.0):
    g = np.asarray(albedo_img.convert("L"), dtype=np.float64) / 255.0
    height = 1.0 - g
    dy, dx = np.gradient(height)
    nx = -dx * strength
    ny = -dy * strength
    nz = np.ones_like(height)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    out = np.stack([nx / norm, ny / norm, nz / norm], axis=-1) * 0.5 + 0.5
    return Image.fromarray((out * 255).astype(np.uint8), mode="RGB")


_MAPLE_HALF = [(0.00, -1.00), (0.13, -0.60), (0.32, -0.68), (0.22, -0.36),
               (0.55, -0.42), (0.36, -0.14), (0.70, -0.02), (0.40, 0.08),
               (0.46, 0.32), (0.20, 0.20), (0.16, 0.50), (0.05, 0.32),
               (0.00, 0.70)]
_MAPLE_FULL = _MAPLE_HALF + [(-x, y) for (x, y) in _MAPLE_HALF[::-1]]


def _leaf_points(cx, cy, s, ang):
    ca, sa = np.cos(ang), np.sin(ang)
    pts = []
    for (x, y) in _MAPLE_FULL:
        xr = x * ca - y * sa
        yr = x * sa + y * ca
        pts.append((cx + xr * s, cy + yr * s))
    return pts


def _leaf_tile(pool, rng, tile_px, ss, row, grid):
    P = tile_px * ss
    img = Image.new("RGBA", (P, P), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # upper rows brighter/warmer (sunlit), lower rows darker/cooler (shaded)
    sun_amt = (grid - 1 - row) / (grid - 1)
    sun_amt = float(np.clip(sun_amt + rng.uniform(-0.15, 0.15), 0.0, 1.0))

    n_leaf = int(rng.integers(16, 30))
    for _ in range(n_leaf):
        cx = rng.uniform(0.12, 0.88) * P
        cy = rng.uniform(0.12, 0.88) * P
        size = rng.uniform(0.16, 0.30) * P
        ang = rng.uniform(0, 2 * np.pi)

        base = pool[int(rng.integers(len(pool)))].astype(np.float64).copy()
        base *= 1.12                               # lift toward the bright photo greens
        base *= (0.78 + 0.28 * sun_amt)            # shade depth (floor kept high)
        base[1] *= (1.0 + 0.06 * sun_amt)          # lime in the sun
        base[2] *= (1.0 - 0.12 * sun_amt)          # warmer (less blue) in the sun
        base *= (1.0 + rng.uniform(-0.08, 0.08))
        col = tuple(int(np.clip(c, 0, 1) * 255) for c in base) + (255,)

        d.polygon(_leaf_points(cx, cy, size, ang), fill=col)

    return img.resize((tile_px, tile_px), Image.LANCZOS)


def make_foliage_atlas(pool, rng, tile_px=256, grid=4, ss=4):
    S = tile_px * grid                              # 1024x1024
    atlas = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    for row in range(grid):
        for col in range(grid):
            tile = _leaf_tile(pool, rng, tile_px, ss, row, grid)
            atlas.paste(tile, (col * tile_px, row * tile_px))
    return atlas


# ===========================================================================
# MATERIALS + UV ASSIGNMENT
# ===========================================================================

def _make_pbr(base_img, roughness, normal_img=None, mask=False):
    PBR = trimesh.visual.material.PBRMaterial
    kw = dict(baseColorTexture=base_img, metallicFactor=0.0,
              roughnessFactor=float(roughness))
    if mask:
        kw.update(alphaMode="MASK", alphaCutoff=0.45, doubleSided=True)
    if normal_img is not None:
        kw["normalTexture"] = normal_img
    try:
        return PBR(**kw)
    except TypeError:
        kw.pop("normalTexture", None)
        mat = PBR(**kw)
        if normal_img is not None:
            try:
                mat.normalTexture = normal_img
            except Exception:
                pass
        return mat


def _apply_cylindrical(mesh, albedo, normal, roughness, u_rep, v_rep):
    V = mesh.vertices
    y0, y1 = float(V[:, 1].min()), float(V[:, 1].max())
    ang = np.arctan2(V[:, 2], V[:, 0])
    u = (ang / (2 * np.pi) + 0.5) * u_rep
    vv = (V[:, 1] - y0) / ((y1 - y0) + 1e-9) * v_rep
    uv = np.column_stack([u, vv])

    mat = _make_pbr(albedo, roughness, normal_img=normal)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=albedo, material=mat)

    r = np.hypot(V[:, 0], V[:, 2]) + 1e-6
    warmth = np.clip(0.5 + 0.5 * V[:, 0] / r, 0.0, 1.0)
    ao = np.clip(0.72 + 0.28 * (V[:, 1] - y0) / ((y1 - y0) + 1e-9), 0.0, 1.0)
    cool = np.array([0.82, 0.85, 0.95])
    warm = np.array([1.02, 0.97, 0.85])
    col = (cool[None, :] + (warm - cool)[None, :] * warmth[:, None]) * ao[:, None]
    col = np.clip(col, 0.0, 1.0)
    rgba = np.concatenate(
        [(col * 255).astype(np.uint8), np.full((len(V), 1), 255, np.uint8)], axis=1)
    mesh.visual.vertex_attributes["color"] = rgba


def _apply_canopy(mesh, atlas, roughness, rng, grid=4):
    V = mesh.vertices
    n_cards = len(V) // 4
    uv = np.zeros((len(V), 2))
    rgba = np.zeros((len(V), 4), dtype=np.uint8)

    inset = 0.5 / (256.0 * grid)
    # near-neutral tints (the atlas carries the green); outer/top bright, inner/low dark
    shade = np.array([0.52, 0.56, 0.46])
    sunc  = np.array([1.00, 1.00, 0.90])

    for i in range(n_cards):
        col = int(rng.integers(0, grid))
        row = int(rng.integers(0, grid))
        u0, u1 = col / grid + inset, (col + 1) / grid - inset
        v0, v1 = row / grid + inset, (row + 1) / grid - inset
        box = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
        rot = int(rng.integers(0, 4))
        box = box[rot:] + box[:rot]

        verts = V[4 * i:4 * i + 4]
        for k in range(4):
            uv[4 * i + k] = box[k]

        centroid = verts.mean(axis=0)
        vfrac = float(np.clip((centroid[1] - CROWN_BOTTOM_Y) /
                              (CROWN_TOP_Y - CROWN_BOTTOM_Y), 0.0, 1.0))
        rfrac = float(np.clip(np.hypot(centroid[0], centroid[2]) / CROWN_HALF_WIDTH,
                              0.0, 1.0))
        sun = float(np.clip(0.30 + 0.50 * vfrac + 0.30 * rfrac, 0.0, 1.0))
        tint = shade + (sunc - shade) * sun
        tint = np.clip(tint * (1.0 + rng.uniform(-0.05, 0.05)), 0.0, 1.0)
        c8 = (tint * 255).astype(np.uint8)
        for k in range(4):
            rgba[4 * i + k, :3] = c8
            rgba[4 * i + k, 3] = 255

    mat = _make_pbr(atlas, roughness, mask=True)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=atlas, material=mat)
    mesh.visual.vertex_attributes["color"] = rgba


# ===========================================================================
# TOP-LEVEL BUILD
# ===========================================================================

def build_textured_scene(image_path, seed, density):
    arr = _load_image(image_path)
    tex_rng = np.random.default_rng(seed * 2 + 101)

    pool = _green_pool(arr, tex_rng)
    bark_base = _bark_base(arr)

    bark_albedo = make_bark_albedo(bark_base, 1024, tex_rng)
    bark_normal = make_normal_map(bark_albedo, strength=3.0)
    atlas = make_foliage_atlas(pool, tex_rng)

    geo = build_mesh(seed, density)
    trunk = geo.geometry["trunk"]
    branches = geo.geometry["branches"]
    canopy = geo.geometry["canopy"]

    _apply_cylindrical(trunk, bark_albedo, bark_normal, roughness=0.9,
                       u_rep=3.0, v_rep=4.0)
    _apply_cylindrical(branches, bark_albedo, bark_normal, roughness=0.9,
                       u_rep=2.0, v_rep=6.0)
    _apply_canopy(canopy, atlas, roughness=0.8, rng=tex_rng)

    out = trimesh.Scene()
    out.add_geometry(trunk,    geom_name="trunk")
    out.add_geometry(branches, geom_name="branches")
    out.add_geometry(canopy,   geom_name="canopy")
    return out


def main():
    ap = argparse.ArgumentParser(description="Procedural textured broadleaf tree -> GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        scene = build_textured_scene(args.image, args.seed, args.density)
        glb = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(glb)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())