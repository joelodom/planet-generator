#!/usr/bin/env python3
"""Procedural coconut palm -> textured GLB.

Builds a tall, slender, gently-leaning coconut palm (flared ringed trunk,
fibrous crown knot, arching pinnate fronds whose ribs carry overlapping
leafy blade-cards), derives tileable materials from a reference photo,
applies per-surface UVs and COLOR_0 sun/shade tints, and exports a binary
.glb.

CLI:
    python coconut_palm.py --image PATH --seed INT \
        --density {high,med,low} --output OUT.glb
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter


# ==========================================================================
#  GEOMETRY  (build_mesh)
# ==========================================================================
# MEASURED PROPORTIONS (read off reference.png, ~10% accuracy)
TREE_HEIGHT       = 12.0     # m, overall height (top of fronds)
HEIGHT_OVER_WIDTH = 1.70     # tall/narrow; photo front aspect ~0.59
CROWN_RADIUS      = 3.05     # m, narrowed so the crown reads as a rounded fountain

TRUNK_TOP_FRAC    = 0.66     # bare trunk ~66% of total height
LEAN_OFFSET       = 1.6      # m horizontal drift of trunk top

R_BASE_SHAFT      = 0.22     # m shaft radius near ground
R_TOP             = 0.145    # m radius under the crown
FLARE_MULT        = 1.55     # base flare factor
FLARE_FRAC        = 0.07     # fraction of trunk that is flared
RING_AMP          = 0.035    # subtle leaf-scar ring relief
RING_FREQ         = 26.0     # number of faint rings up the shaft

KNOT_RADIUS       = 0.42     # m crown knot

# Fronds: emanate upward/outward then arch down (fountain crown)
THETA_TOP_DEG     = 80.0     # near-vertical fronds
THETA_BOT_DEG     = 8.0      # lowest fronds start just above horizontal
DROOP_TOTAL_DEG   = 102.0    # strong tip-ward arch -> rounded drooping dome
FROND_LEN_FRAC    = 0.98
RACHIS_BASE_R     = 0.045
RACHIS_TIP_R      = 0.006

# Leaflet blade-cards: each card is a broad leafy chunk that overlaps its
# neighbours along the rib so the rib reads as a continuous feather.
LEAFLET_LEN_FRAC  = 0.42     # blade out-reach as fraction of frond length
CARD_OVERLAP      = 1.7      # along-rib half-width as multiple of segment len
LEAF_SPREAD       = 0.80     # sideways feathering out of the frond plane
LEAF_FORWARD      = 0.32     # lean toward the rib tip
LEAF_DROOP0       = 0.30     # base droop of blades
LEAF_DROOP1       = 0.55     # extra droop toward the rib tip

_DEG = np.pi / 180.0
_EPS = 1e-9


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > _EPS else v


def _unit_rows(a):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n < _EPS] = 1.0
    return a / n


def _sweep_tube(centerline, radii, nsides, cap=True):
    """Sweep a circular cross-section along a polyline (parallel-transport)."""
    pts = np.asarray(centerline, dtype=np.float64)
    n = pts.shape[0]
    radii = np.asarray(radii, dtype=np.float64)

    tang = np.zeros_like(pts)
    tang[1:-1] = pts[2:] - pts[:-2]
    tang[0] = pts[1] - pts[0]
    tang[-1] = pts[-1] - pts[-2]
    tang = _unit_rows(tang)

    normals = np.zeros_like(pts)
    ref = np.array([0.0, 0.0, 1.0]) if abs(tang[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    normals[0] = _unit(np.cross(tang[0], ref))
    for i in range(1, n):
        prev = normals[i - 1]
        proj = prev - tang[i] * float(np.dot(prev, tang[i]))
        normals[i] = _unit(proj)
    binorm = _unit_rows(np.cross(tang, normals))

    ang = np.linspace(0.0, 2.0 * np.pi, nsides, endpoint=False)
    cosA = np.cos(ang)[None, :, None]
    sinA = np.sin(ang)[None, :, None]
    ring = (pts[:, None, :]
            + radii[:, None, None] * (cosA * normals[:, None, :]
                                      + sinA * binorm[:, None, :]))
    verts = ring.reshape(-1, 3)

    faces = []
    for i in range(n - 1):
        for k in range(nsides):
            a = i * nsides + k
            b = i * nsides + (k + 1) % nsides
            c = (i + 1) * nsides + (k + 1) % nsides
            d = (i + 1) * nsides + k
            faces.append((a, b, c))
            faces.append((a, c, d))

    verts = list(verts)
    if cap:
        bc = len(verts)
        verts.append(pts[0])
        for k in range(nsides):
            faces.append((bc, (k + 1) % nsides, k))
        tc = len(verts)
        verts.append(pts[-1])
        base = (n - 1) * nsides
        for k in range(nsides):
            faces.append((tc, base + k, base + (k + 1) % nsides))

    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _trunk_radius(s):
    r = R_TOP + (R_BASE_SHAFT - R_TOP) * (1.0 - s) ** 1.15
    flare = np.where(
        s < FLARE_FRAC,
        1.0 + (FLARE_MULT - 1.0) * ((FLARE_FRAC - s) / FLARE_FRAC) ** 2,
        1.0,
    )
    rings = 1.0 + RING_AMP * np.sin(s * RING_FREQ * 2.0 * np.pi) * np.clip(s, 0.05, 1.0)
    return r * flare * rings


def _build_trunk(rng, nseg, nsides):
    s = np.linspace(0.0, 1.0, nseg + 1)
    top_y = TREE_HEIGHT * TRUNK_TOP_FRAC

    lean_az = rng.uniform(0.0, 2.0 * np.pi)
    dirxz = np.array([np.cos(lean_az), np.sin(lean_az)])
    perp = np.array([-dirxz[1], dirxz[0]])

    horiz = LEAN_OFFSET * s ** 1.35
    sway = 0.18 * np.sin(s * np.pi) * rng.uniform(-1.0, 1.0)

    center = np.zeros((nseg + 1, 3))
    center[:, 1] = top_y * s
    center[:, 0] = horiz * dirxz[0] + sway * perp[0]
    center[:, 2] = horiz * dirxz[1] + sway * perp[1]

    radii = _trunk_radius(s)
    verts, faces = _sweep_tube(center, radii, nsides, cap=True)
    crown_center = center[-1].copy()
    return verts, faces, crown_center


def _build_frond(rng, crown_center, phi, theta0, frond_len, nseg, rib_sides):
    """One frond: tapered woody rib + overlapping leafy blade-cards.

    Cards have their WIDTH along the rib tangent (overlapping neighbours) and
    their LENGTH out to the side, so a rib reads as a continuous feather.
    """
    radial = np.array([np.cos(phi), 0.0, np.sin(phi)])
    up = np.array([0.0, 1.0, 0.0])
    plane_n = np.array([-np.sin(phi), 0.0, np.cos(phi)])
    droop_total = DROOP_TOTAL_DEG * _DEG

    ds = frond_len / nseg
    pts = [crown_center.copy()]
    p = crown_center.copy()
    for k in range(1, nseg + 1):
        frac = k / nseg
        theta_k = theta0 - droop_total * frac ** 1.4
        step = np.cos(theta_k) * radial + np.sin(theta_k) * up
        p = p + ds * step
        pts.append(p.copy())
    pts = np.asarray(pts)

    fr = np.linspace(0.0, 1.0, nseg + 1)
    rib_r = RACHIS_BASE_R * (1.0 - fr) + RACHIS_TIP_R * fr
    rib_v, rib_f = _sweep_tube(pts, rib_r, rib_sides, cap=True)

    tang = np.zeros_like(pts)
    tang[1:-1] = pts[2:] - pts[:-2]
    tang[0] = pts[1] - pts[0]
    tang[-1] = pts[-1] - pts[-2]
    tang = _unit_rows(tang)

    leaf_v, leaf_f = [], []
    Lmax = LEAFLET_LEN_FRAC * frond_len
    hw = CARD_OVERLAP * ds                     # along-rib half-width (overlap)
    for k in range(1, nseg + 1):
        frac = k / nseg
        env = max(np.sin(np.pi * frac) ** 0.55, 0.28)   # full mid-frond, taper ends
        attach = pts[k]
        T = tang[k]
        droopl = LEAF_DROOP0 + LEAF_DROOP1 * frac
        for sgn in (1.0, -1.0):
            Llen = Lmax * env * float(rng.lognormal(0.0, 0.16))
            wlen = hw * float(rng.uniform(0.85, 1.15))
            ddir = _unit(sgn * LEAF_SPREAD * plane_n + LEAF_FORWARD * T - droopl * up)
            tip = attach + ddir * Llen
            base = len(leaf_v)
            # base edge spans along the rib; tip edge narrows to a soft point
            leaf_v.extend([
                attach - T * wlen,
                attach + T * wlen,
                tip + T * wlen * 0.30,
                tip - T * wlen * 0.30,
            ])
            leaf_f.append((base, base + 1, base + 2))
            leaf_f.append((base, base + 2, base + 3))

    return (rib_v, rib_f,
            np.asarray(leaf_v, dtype=np.float64),
            np.asarray(leaf_f, dtype=np.int64))


def _build_crown_knot(crown_center):
    ico = trimesh.creation.icosphere(subdivisions=1, radius=KNOT_RADIUS)
    v = ico.vertices.copy()
    v[:, 1] *= 0.7
    v += crown_center
    return v, ico.faces.copy()


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)

    if density == "low":
        n_fronds, leaflets_side = 12, 12
        trunk_seg, trunk_sides, rib_sides = 14, 6, 3
    elif density == "med":
        n_fronds, leaflets_side = 20, 18
        trunk_seg, trunk_sides, rib_sides = 28, 10, 4
    else:  # "high"
        n_fronds, leaflets_side = 32, 26
        trunk_seg, trunk_sides, rib_sides = 44, 14, 4

    t_v, t_f, crown_center = _build_trunk(rng, trunk_seg, trunk_sides)
    crown_center = crown_center + np.array([0.0, -0.05, 0.0])

    k_v, k_f = _build_crown_knot(crown_center)

    golden = np.pi * (3.0 - np.sqrt(5.0))
    rib_V, rib_F = [], []
    leaf_V, leaf_F = [], []
    rib_off, leaf_off = 0, 0
    for i in range(n_fronds):
        phi = (i * golden + rng.uniform(-0.18, 0.18)) % (2.0 * np.pi)
        theta0 = (rng.uniform(THETA_BOT_DEG, THETA_TOP_DEG)) * _DEG
        flen = CROWN_RADIUS * FROND_LEN_FRAC * float(rng.uniform(0.85, 1.08))

        rv, rf, lv, lf = _build_frond(rng, crown_center, phi, theta0,
                                      flen, leaflets_side, rib_sides)
        rib_V.append(rv); rib_F.append(rf + rib_off); rib_off += len(rv)
        if len(lv):
            leaf_V.append(lv); leaf_F.append(lf + leaf_off); leaf_off += len(lv)

    rib_V = np.vstack(rib_V); rib_F = np.vstack(rib_F)
    leaf_V = np.vstack(leaf_V); leaf_F = np.vstack(leaf_F)

    trunk = trimesh.Trimesh(vertices=t_v, faces=t_f, process=True)
    crown = trimesh.Trimesh(vertices=k_v, faces=k_f, process=True)
    ribs = trimesh.Trimesh(vertices=rib_V, faces=rib_F, process=True)
    canopy = trimesh.Trimesh(vertices=leaf_V, faces=leaf_F, process=False)

    meshes = {"trunk": trunk, "crown": crown, "frond_ribs": ribs, "canopy": canopy}

    min_y = min(float(m.vertices[:, 1].min()) for m in meshes.values())
    scene = trimesh.Scene()
    for name, m in meshes.items():
        m.apply_translation([0.0, -min_y, 0.0])
        m.fix_normals()
        scene.add_geometry(m, geom_name=name)

    return scene


# ==========================================================================
#  COLOR SAMPLING FROM THE PHOTO
# ==========================================================================
def _load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64)


def _sample_color(arr, centers, frac, fallback):
    """Median color over small patches inside the silhouette; reject background."""
    H, W = arr.shape[:2]
    half = max(2, int(min(H, W) * frac))
    cols = []
    for nx, ny in centers:
        cx, cy = int(nx * W), int(ny * H)
        y0, y1 = max(0, cy - half), min(H, cy + half)
        x0, x1 = max(0, cx - half), min(W, cx + half)
        patch = arr[y0:y1, x0:x1].reshape(-1, 3)
        if patch.size == 0:
            continue
        cols.append(np.median(patch, axis=0))
    if not cols:
        return np.array(fallback, dtype=np.float64)
    cols = np.array(cols)

    keep = []
    for c in cols:
        mx, mn = float(c.max()), float(c.min())
        sat = (mx - mn) / (mx + 1e-6)
        val = mx / 255.0
        keep.append(not (sat < 0.10 and val > 0.55))
    keep = np.array(keep)
    cols = cols[keep] if keep.any() else cols

    med = np.median(cols, axis=0)
    dist = np.linalg.norm(cols - med, axis=1)
    if len(cols) > 2:
        cols = cols[dist <= (np.median(dist) + 40.0)]
    out = np.median(cols, axis=0)
    if not np.all(np.isfinite(out)):
        return np.array(fallback, dtype=np.float64)
    return out


def _clip8(a):
    return np.clip(a, 0, 255).astype(np.uint8)


# ==========================================================================
#  TILEABLE PROCEDURAL DETAIL
# ==========================================================================
def _tileable_noise(rng, h, w, n_comp, fx_range, fy_range):
    """Sum of integer-frequency cosines -> seamless (period = full image)."""
    xv = np.linspace(0.0, 2.0 * np.pi, w, endpoint=False)
    yv = np.linspace(0.0, 2.0 * np.pi, h, endpoint=False)
    X, Y = np.meshgrid(xv, yv)
    acc = np.zeros((h, w))
    for _ in range(n_comp):
        fx = int(rng.integers(fx_range[0], fx_range[1] + 1))
        fy = int(rng.integers(fy_range[0], fy_range[1] + 1))
        if fx == 0 and fy == 0:
            fx = 1
        ph = float(rng.uniform(0.0, 2.0 * np.pi))
        amp = 1.0 / (1.0 + fx + fy)
        acc += amp * np.cos(fx * X + fy * Y + ph)
    acc = (acc - acc.min()) / (np.ptp(acc) + 1e-9)
    return acc


def _delight(rgb, lo=0.6, hi=1.6):
    """Even out baked lighting: divide by heavily blurred luminance, clamp gain."""
    img = Image.fromarray(_clip8(rgb))
    lum = np.asarray(img.convert("L"), dtype=np.float64)
    blur = np.asarray(
        Image.fromarray(_clip8(lum)).filter(
            ImageFilter.GaussianBlur(radius=max(rgb.shape[0], rgb.shape[1]) / 12.0)),
        dtype=np.float64,
    )
    mean = float(np.clip(blur.mean(), 1.0, 255.0))
    gain = np.clip(mean / (blur + 1e-6), lo, hi)[..., None]
    return rgb * gain


def _normal_from_albedo(rgb, strength=2.0):
    """Tangent-space normal map from albedo (height = inverse luminance)."""
    lum = rgb.mean(axis=2) / 255.0
    height = 1.0 - lum
    gy, gx = np.gradient(height)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(height)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    out = np.stack([nx / norm, ny / norm, nz / norm], axis=2)
    return Image.fromarray(_clip8((out * 0.5 + 0.5) * 255.0))


# ==========================================================================
#  TEXTURE BUILDERS
# ==========================================================================
def _make_bark(rng, base, dark, res=768):
    """Fibrous woody bark, greyish-tan/brown, faint horizontal leaf-scar rings."""
    vv = (np.arange(res) / res)[:, None]
    fiber = _tileable_noise(rng, res, res, 26, fx_range=(4, 30), fy_range=(0, 3))
    mottle = _tileable_noise(rng, res, res, 12, fx_range=(1, 4), fy_range=(1, 4))

    nrings = 16
    ringwave = np.sin(2.0 * np.pi * nrings * vv)
    groove = np.clip((ringwave - 0.45) / 0.55, 0.0, 1.0)
    groove = np.broadcast_to(groove, (res, res))

    mix = np.clip(0.40 * groove + 0.28 * (1.0 - fiber) + 0.16 * (1.0 - mottle), 0.0, 0.72)
    col = base[None, None, :] * (1.0 - mix)[..., None] + dark[None, None, :] * mix[..., None]
    col *= (0.88 + 0.22 * fiber)[..., None]
    col = _delight(col)
    return Image.fromarray(_clip8(col))


def _make_knot(rng, base, res=512):
    """Coarse brown fibrous matter at the crown heart."""
    dark = base * 0.55
    fiber = _tileable_noise(rng, res, res, 30, fx_range=(6, 26), fy_range=(6, 26))
    mottle = _tileable_noise(rng, res, res, 10, fx_range=(1, 4), fy_range=(1, 4))
    mix = np.clip(0.45 * (1.0 - fiber) + 0.25 * (1.0 - mottle), 0.0, 0.8)
    col = base[None, None, :] * (1.0 - mix)[..., None] + dark[None, None, :] * mix[..., None]
    col *= (0.85 + 0.25 * fiber)[..., None]
    return Image.fromarray(_clip8(col))


def _make_rib(rng, green, brown, res=512):
    """Woody green-brown frond rib (rachis)."""
    base = 0.5 * green + 0.5 * brown
    dark = base * 0.6
    fiber = _tileable_noise(rng, res, res, 22, fx_range=(2, 6), fy_range=(8, 32))
    mix = np.clip(0.4 * (1.0 - fiber), 0.0, 0.6)
    col = base[None, None, :] * (1.0 - mix)[..., None] + dark[None, None, :] * mix[..., None]
    col *= (0.9 + 0.2 * fiber)[..., None]
    return Image.fromarray(_clip8(col))


def _draw_leaflet(draw, bx, by, ang, length, width, color):
    tx = bx + np.cos(ang) * length
    ty = by + np.sin(ang) * length
    px = -np.sin(ang) * width * 0.5
    py = np.cos(ang) * width * 0.5
    pts = [(bx + px, by + py), (tx, ty), (bx - px, by - py)]
    draw.polygon(pts, fill=color)


def _make_foliage_atlas(rng, green_light, green_mid, green_dark, green_brown,
                        atlas=1024, ss=4):
    """4x4 atlas of distinct, DENSE frond-cluster tiles; binary alpha, AA edges.

    Each tile is a comb of leaflets fanning upward from a base rib, filling
    the tile so the big blade-cards read as solid feathery foliage.
    """
    tile_px = atlas // 4
    out = Image.new("RGBA", (atlas, atlas), (0, 0, 0, 0))
    palette = [green_light, green_mid, green_mid, green_dark, green_brown]

    for row in range(4):
        for col in range(4):
            # sunlit tiles (top rows) brighter/warmer; shaded (bottom) cooler
            sun = 1.30 - 0.42 * (row / 3.0)
            warm = np.array([1.08, 1.02, 0.80]) if row < 2 else np.array([0.90, 0.98, 0.92])

            big = tile_px * ss
            tile = Image.new("RGBA", (big, big), (0, 0, 0, 0))
            d = ImageDraw.Draw(tile)

            # a central rib running up the tile, dense leaflets along both sides
            nribs = int(rng.integers(2, 4))
            for _ in range(nribs):
                ox = float(rng.uniform(0.30, 0.70)) * big
                oy = big * 0.99
                main = -np.pi / 2.0 + float(rng.uniform(-0.30, 0.30))
                rib_len = float(rng.uniform(0.85, 1.0)) * big
                nleaf = int(rng.integers(34, 48))
                for j in range(nleaf):
                    t = j / max(1, nleaf - 1)
                    bx = ox + np.cos(main) * rib_len * t
                    by = oy + np.sin(main) * rib_len * t
                    for sgn in (-1.0, 1.0):
                        a = main + sgn * float(rng.uniform(0.45, 0.95))
                        L = float(rng.uniform(0.16, 0.27)) * big * (0.55 + 0.85 * np.sin(np.pi * t))
                        Wd = L * float(rng.uniform(0.16, 0.26))
                        gi = int(rng.integers(0, len(palette)))
                        base_c = palette[gi] * sun * warm * float(rng.uniform(0.88, 1.12))
                        c = tuple(int(v) for v in np.clip(base_c, 0, 255))
                        _draw_leaflet(d, bx, by, a, L, Wd, (c[0], c[1], c[2], 255))

            tile = tile.resize((tile_px, tile_px), Image.LANCZOS)
            out.paste(tile, (col * tile_px, row * tile_px), tile)

    return out


# ==========================================================================
#  UV + VERTEX-COLOR ASSIGNMENT
# ==========================================================================
def _cyl_uv(verts, v_tiles):
    cx = float(verts[:, 0].mean())
    cz = float(verts[:, 2].mean())
    ang = np.arctan2(verts[:, 2] - cz, verts[:, 0] - cx)
    u = ang / (2.0 * np.pi) + 0.5
    y = verts[:, 1]
    span = float(np.ptp(y)) + 1e-6
    v = (y - y.min()) / span * v_tiles
    return np.column_stack([u, v]).astype(np.float64)


def _sphere_uv(verts):
    c = verts.mean(axis=0)
    d = verts - c
    u = np.arctan2(d[:, 2], d[:, 0]) / (2.0 * np.pi) + 0.5
    span = float(np.ptp(verts[:, 1])) + 1e-6
    v = (verts[:, 1] - verts[:, 1].min()) / span
    return np.column_stack([u, v]).astype(np.float64)


def _card_uv(n_verts, rng, atlas_res=1024):
    """Map each 4-vertex blade-card to a random atlas tile with a random 90deg rotation."""
    ncards = n_verts // 4
    uv = np.zeros((n_verts, 2), dtype=np.float64)
    inset = 1.5 / atlas_res
    for c in range(ncards):
        tile = int(rng.integers(0, 16))
        rot = int(rng.integers(0, 4))
        col, row = tile % 4, tile // 4
        u0, u1 = col / 4.0 + inset, (col + 1) / 4.0 - inset
        v0, v1 = row / 4.0 + inset, (row + 1) / 4.0 - inset
        corners = [(u0, v1), (u1, v1), (u1, v0), (u0, v0)]
        corners = corners[rot:] + corners[:rot]
        for i in range(4):
            uv[4 * c + i] = corners[i]
    return uv


def _trunk_colors(verts, rng):
    y = verts[:, 1]
    yn = (y - y.min()) / (float(np.ptp(y)) + 1e-6)
    f = np.clip(0.72 + 0.28 * yn, 0.72, 1.0)
    f = f * (1.0 + 0.04 * (rng.random(len(y)) - 0.5))
    base = np.clip(f[:, None] * 255.0, 0, 255)
    rgb = np.repeat(base, 3, axis=1)
    a = np.full((len(y), 1), 255.0)
    return _clip8(np.hstack([rgb, a]))


def _rib_colors(verts, rng):
    n = len(verts)
    f = 0.85 + 0.10 * rng.random(n)
    rgb = np.clip(np.column_stack([f * 235, f * 235, f * 220]), 0, 255)
    a = np.full((n, 1), 255.0)
    return _clip8(np.hstack([rgb, a]))


def _canopy_colors(verts, rng):
    """Per-card sun/shade tint: outer/upper brighter & warmer, lower browner.

    Kept near-white (lifted floor) so the bright atlas greens survive.
    """
    ncards = len(verts) // 4
    cx = float(verts[:, 0].mean())
    cz = float(verts[:, 2].mean())
    ymin, ymax = float(verts[:, 1].min()), float(verts[:, 1].max())
    yspan = (ymax - ymin) + 1e-6

    cols = np.zeros((len(verts), 4), dtype=np.float64)
    sun_lit = np.array([1.0, 1.0, 0.90])
    shade = np.array([0.84, 0.94, 0.86])
    brown = np.array([0.88, 0.74, 0.48])
    for c in range(ncards):
        idx = slice(4 * c, 4 * c + 4)
        cen = verts[idx].mean(axis=0)
        yn = (cen[1] - ymin) / yspan
        radial = np.clip(np.hypot(cen[0] - cx, cen[2] - cz) / CROWN_RADIUS, 0.0, 1.0)
        sun = np.clip(0.5 * yn + 0.5 * radial, 0.0, 1.0)
        bright = np.clip(0.78 + 0.22 * sun, 0.74, 1.0) * float(rng.uniform(0.95, 1.0))
        tint = shade + (sun_lit - shade) * sun
        if yn < 0.22:
            mixb = (0.22 - yn) * 1.8
            tint = tint * (1.0 - mixb) + brown * mixb
        rgb = np.clip(bright * tint * 255.0, 0, 255)
        cols[idx, :3] = rgb
        cols[idx, 3] = 255.0
    return _clip8(cols)


def _attach(mesh, uv, material, colors):
    tv = trimesh.visual.TextureVisuals(uv=uv, material=material)
    tv.vertex_attributes["color"] = colors
    mesh.visual = tv


# ==========================================================================
#  TEXTURING + EXPORT
# ==========================================================================
def texture_scene(scene, image_path, seed):
    rng = np.random.default_rng((seed ^ 0x9E3779B9) & 0xFFFFFFFF)
    arr = _load_image(image_path)

    # --- sample palette from WELL INSIDE the silhouette ---
    bark = _sample_color(arr, [(0.34, 0.92), (0.36, 0.85), (0.40, 0.75),
                               (0.43, 0.62), (0.45, 0.52)], 0.012, (150, 135, 110))
    foliage = _sample_color(arr, [(0.38, 0.18), (0.50, 0.22), (0.32, 0.28),
                                  (0.58, 0.24), (0.45, 0.15), (0.42, 0.33),
                                  (0.28, 0.20), (0.55, 0.32), (0.48, 0.28)],
                            0.018, (110, 140, 60))
    knot = _sample_color(arr, [(0.45, 0.40), (0.46, 0.42), (0.44, 0.44)],
                         0.015, (95, 72, 45))

    bark = np.asarray(bark, dtype=np.float64)
    foliage = np.asarray(foliage, dtype=np.float64)
    knot = np.asarray(knot, dtype=np.float64)

    # brighten/warm the green palette toward the photo's lime / yellow-green
    bark_dark = bark * 0.58
    green_mid = np.clip(foliage * 1.18 + np.array([12, 18, 2]), 0, 255)
    green_light = np.clip(foliage * 1.55 + np.array([40, 46, 10]), 0, 255)
    green_dark = np.clip(foliage * 0.82, 0, 255)
    green_brown = np.clip(0.55 * foliage + 0.45 * np.array([140, 105, 55]), 0, 255)

    # --- textures ---
    bark_tex = _make_bark(rng, bark, bark_dark, res=768)
    bark_norm = _normal_from_albedo(np.asarray(bark_tex, dtype=np.float64), strength=2.2)
    knot_tex = _make_knot(rng, knot, res=512)
    rib_tex = _make_rib(rng, green_dark, green_brown, res=512)
    atlas = _make_foliage_atlas(rng, green_light, green_mid, green_dark, green_brown,
                                atlas=1024, ss=4)

    # --- materials ---
    PBR = trimesh.visual.material.PBRMaterial
    bark_mat = PBR(name="trunk", baseColorTexture=bark_tex, normalTexture=bark_norm,
                   metallicFactor=0.0, roughnessFactor=0.9)
    knot_mat = PBR(name="crown", baseColorTexture=knot_tex,
                   metallicFactor=0.0, roughnessFactor=0.9)
    rib_mat = PBR(name="frond_ribs", baseColorTexture=rib_tex,
                  metallicFactor=0.0, roughnessFactor=0.85)
    leaf_mat = PBR(name="canopy", baseColorTexture=atlas,
                   metallicFactor=0.0, roughnessFactor=0.8,
                   alphaMode="MASK", alphaCutoff=0.45, doubleSided=True)

    g = scene.geometry
    trunk = g["trunk"]
    _attach(trunk, _cyl_uv(trunk.vertices, v_tiles=TREE_HEIGHT * TRUNK_TOP_FRAC / 1.6),
            bark_mat, _trunk_colors(trunk.vertices, rng))
    crown = g["crown"]
    _attach(crown, _sphere_uv(crown.vertices), knot_mat,
            _clip8(np.tile([192, 165, 120, 255], (len(crown.vertices), 1)).astype(np.float64)))
    ribs = g["frond_ribs"]
    _attach(ribs, _cyl_uv(ribs.vertices, v_tiles=6.0), rib_mat,
            _rib_colors(ribs.vertices, rng))
    canopy = g["canopy"]
    _attach(canopy, _card_uv(len(canopy.vertices), rng, atlas_res=1024),
            leaf_mat, _canopy_colors(canopy.vertices, rng))

    return scene


def main(argv=None):
    ap = argparse.ArgumentParser(description="Procedural coconut palm -> textured GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args(argv)

    scene = build_mesh(args.seed, args.density)
    scene = texture_scene(scene, args.image, args.seed)

    data = scene.export(file_type="glb")
    with open(args.output, "wb") as f:
        f.write(data)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)