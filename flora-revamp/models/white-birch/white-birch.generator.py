#!/usr/bin/env python3
"""
Silver birch (Betula pendula) -- procedural geometry + photo-derived textures
-> textured GLB.

Deterministic given --seed. Only numpy / trimesh / PIL / stdlib.

CLI:
  python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter


# ==========================================================================
# GEOMETRY (build_mesh)
# ==========================================================================
TREE_HEIGHT          = 16.0   # meters, overall height
HEIGHT_OVER_WIDTH    = 2.6    # whole-silhouette height / max crown width
CROWN_BASE_FRAC      = 0.46   # foliage begins at ~46% of total height
CROWN_WIDEST_FRAC    = 0.72   # crown widest at ~72% of total height
TRUNK_BARE_FRAC      = 0.46   # lower ~half of trunk is essentially bare

CROWN_WIDTH   = TREE_HEIGHT / HEIGHT_OVER_WIDTH
RX = RZ       = CROWN_WIDTH * 0.5
CROWN_BASE_Y  = CROWN_BASE_FRAC * TREE_HEIGHT
CROWN_TOP_Y   = TREE_HEIGHT
RY            = (CROWN_TOP_Y - CROWN_BASE_Y) * 0.5
CROWN_CENTER_Y = CROWN_BASE_Y + RY

TRUNK_BASE_RADIUS = 0.155
TRUNK_TOP_RADIUS  = 0.022
BASAL_FLARE       = 1.45
BASAL_FLARE_FRAC  = 0.06

_PRESETS = {
    "high": dict(trunk_sides=14, trunk_rings=18, branch_count=11,
                 branch_sides=6, branch_rings=9, n_clumps=40, total_cards=4000),
    "med":  dict(trunk_sides=10, trunk_rings=11, branch_count=7,
                 branch_sides=5, branch_rings=6, n_clumps=18, total_cards=1700),
    "low":  dict(trunk_sides=6,  trunk_rings=6,  branch_count=4,
                 branch_sides=4,  branch_rings=4, n_clumps=8,  total_cards=480),
}


def _normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + eps)


def _frames(points):
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    tang = np.zeros((n, 3))
    tang[:-1] = pts[1:] - pts[:-1]
    tang[-1] = tang[-2]
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)

    t0 = tang[0]
    a = np.array([0.0, 0.0, 1.0]) if abs(t0[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    nrm = [_normalize(np.cross(t0, a))]
    for i in range(1, n):
        v = np.cross(tang[i - 1], tang[i])
        s = np.linalg.norm(v)
        c = np.clip(np.dot(tang[i - 1], tang[i]), -1.0, 1.0)
        if s < 1e-9:
            ni = nrm[-1]
        else:
            v /= s
            ang = np.arctan2(s, c)
            p = nrm[-1]
            ni = (p * np.cos(ang) + np.cross(v, p) * np.sin(ang)
                  + v * np.dot(v, p) * (1.0 - np.cos(ang)))
        ni = ni - np.dot(ni, tang[i]) * tang[i]
        nrm.append(_normalize(ni))
    nrm = np.array(nrm)
    binm = np.cross(tang, nrm)
    return tang, nrm, binm


def _tube(points, radii, sides, cap_start=True, cap_end=True, roughness=None, rng=None):
    points = np.asarray(points, dtype=float)
    radii = np.asarray(radii, dtype=float)
    _, nrm, binm = _frames(points)
    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    cos_a, sin_a = np.cos(ang), np.sin(ang)

    rings = []
    for i in range(len(points)):
        r = radii[i]
        if roughness is not None and rng is not None and roughness[i] > 0:
            r = r * (1.0 + rng.normal(0.0, roughness[i], sides))
        circle = cos_a[:, None] * nrm[i] + sin_a[:, None] * binm[i]
        rings.append(points[i] + r[:, None] * circle if np.ndim(r) else points[i] + r * circle)

    verts = np.vstack(rings)
    faces = []
    for i in range(len(points) - 1):
        a0, b0 = i * sides, (i + 1) * sides
        for j in range(sides):
            j2 = (j + 1) % sides
            faces.append([a0 + j, a0 + j2, b0 + j2])
            faces.append([a0 + j, b0 + j2, b0 + j])

    if cap_start:
        ci = len(verts)
        verts = np.vstack([verts, points[0]])
        for j in range(sides):
            faces.append([ci, (j + 1) % sides, j])
    if cap_end:
        ci = len(verts)
        base = (len(points) - 1) * sides
        verts = np.vstack([verts, points[-1]])
        for j in range(sides):
            faces.append([ci, base + j, base + (j + 1) % sides])

    return verts, np.asarray(faces, dtype=np.int64)


def _trunk_radius(frac):
    r = TRUNK_TOP_RADIUS + (TRUNK_BASE_RADIUS - TRUNK_TOP_RADIUS) * (1.0 - frac) ** 1.25
    if frac < BASAL_FLARE_FRAC:
        t = 1.0 - frac / BASAL_FLARE_FRAC
        r *= 1.0 + (BASAL_FLARE - 1.0) * t * t
    return r


def _make_lobes(rng):
    n = rng.integers(3, 7)
    return [(rng.integers(2, 5), rng.uniform(0.07, 0.15), rng.uniform(0.0, 2.0 * np.pi))
            for _ in range(n)]


def _shell_point(direction, lobes):
    d = _normalize(direction)
    dy = np.clip(d[1], -1.0, 1.0)
    prof = max(0.0, 1.0 - dy * dy) ** 0.4
    prof *= (1.0 - 0.28 * max(dy, 0.0))
    theta = np.arctan2(d[2], d[0])
    rmult = 1.0 + sum(a * np.cos(f * theta + p) for f, a, p in lobes)
    return np.array([
        RX * d[0] * prof * rmult,
        CROWN_CENTER_Y + RY * dy * rmult,
        RZ * d[2] * prof * rmult,
    ])


def _catmull(ctrl, n):
    ctrl = np.asarray(ctrl, dtype=float)
    pad = np.vstack([ctrl[0], ctrl, ctrl[-1]])
    segs = len(ctrl) - 1
    out = []
    for t in np.linspace(0.0, segs, n):
        i = min(int(np.floor(t)), segs - 1)
        f = t - i
        p0, p1, p2, p3 = pad[i], pad[i + 1], pad[i + 2], pad[i + 3]
        f2, f3 = f * f, f * f * f
        out.append(0.5 * (2 * p1 + (-p0 + p2) * f
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * f3))
    return np.array(out)


def _build_branches(rng, lobes, params):
    """Slender weeping limbs, kept short and inside the crown so foliage covers them."""
    sides = params["branch_sides"]
    rings = params["branch_rings"]
    count = params["branch_count"]

    all_v, all_f, tips = [], [], []
    voff = 0
    h_fracs = np.sort(rng.uniform(0.52, 0.95, count))
    base_az = rng.uniform(0.0, 2.0 * np.pi)

    for k, hf in enumerate(h_fracs):
        h0 = hf * TREE_HEIGHT
        az = base_az + k * 2.399963 + rng.uniform(-0.3, 0.3)
        ux, uz = np.cos(az), np.sin(az)

        dy = (h0 - CROWN_CENTER_Y) / RY
        prof = max(0.05, 1.0 - dy * dy) ** 0.4 * (1.0 - 0.28 * max(dy, 0.0))
        reach = RX * prof * rng.uniform(0.65, 0.85)   # shorter -> stays within foliage

        p0 = np.array([0.0, h0, 0.0])
        p1 = np.array([0.40 * reach * ux, h0 + 0.12 * reach, 0.40 * reach * uz])
        p2 = np.array([0.78 * reach * ux, h0 + 0.14 * reach, 0.78 * reach * uz])
        p3 = np.array([1.00 * reach * ux, h0 - 0.14 * reach, 1.00 * reach * uz])
        path = _catmull([p0, p1, p2, p3], rings)

        r0 = _trunk_radius(hf) * rng.uniform(0.35, 0.5)   # thinner limbs
        radii = r0 * (np.linspace(1.0, 0.08, rings) ** 1.1)

        v, f = _tube(path, radii, sides, cap_start=False, cap_end=True)
        all_v.append(v)
        all_f.append(f + voff)
        voff += len(v)
        tips.append(path[-1])

    return np.vstack(all_v), np.vstack(all_f), tips


def _build_canopy(rng, lobes, params, branch_tips):
    """A CONTINUOUS airy plume: a central tapering column for vertical
    continuity plus shell-filling clumps for the body. Heavy overlap so the
    crown reads as one lacy mass, not separated blobs."""
    n_clumps = params["n_clumps"]
    total = params["total_cards"]
    cards_per = max(1, total // n_clumps)

    clump_radius = 0.13 * CROWN_WIDTH
    card_half = 0.040 * CROWN_WIDTH
    center = np.array([0.0, CROWN_CENTER_Y, 0.0])
    crown_h = CROWN_TOP_Y - CROWN_BASE_Y

    centers = []
    # 1) central column -> connects base to a tapered leader tip (teardrop point)
    ncol = max(3, int(n_clumps * 0.30))
    for i in range(ncol):
        fy = (i + 0.5) / ncol
        ty = CROWN_BASE_Y + fy * crown_h
        rr = 0.16 * CROWN_WIDTH * (1.0 - 0.7 * fy) * rng.uniform(0.0, 1.0)
        ang = rng.uniform(0.0, 2.0 * np.pi)
        centers.append(np.array([rr * np.cos(ang), ty, rr * np.sin(ang)]))
    # 2) shell-filling clumps -> body of the crown, biased upward
    while len(centers) < n_clumps:
        d = rng.normal(size=3)
        d[1] = d[1] * 0.7 + rng.uniform(-0.3, 0.7)
        shell = _shell_point(d, lobes)
        f = rng.uniform(0.6, 0.9)
        centers.append(center + (shell - center) * f)

    vlist, flist = [], []
    voff = 0
    for c in centers:
        c = np.asarray(c, dtype=float)
        rad_vec = c - center
        radial = _normalize(rad_vec) if np.linalg.norm(rad_vec) > 1e-6 else np.array([0.0, 1.0, 0.0])
        for _ in range(cards_per):
            off = rng.normal(size=3)
            off = off / (np.linalg.norm(off) + 1e-9) * (rng.random() ** 0.5) * clump_radius
            pos = c + off

            nrm = _normalize(radial + rng.normal(0.0, 0.35, 3))
            up = np.array([0.0, 1.0, 0.0])
            u = np.cross(up, nrm)
            if np.linalg.norm(u) < 1e-4:
                u = np.array([1.0, 0.0, 0.0])
            u = _normalize(u)
            v = _normalize(np.cross(nrm, u))
            a = rng.uniform(0.0, 2.0 * np.pi)
            uu = u * np.cos(a) + v * np.sin(a)
            vv = -u * np.sin(a) + v * np.cos(a)

            hs = card_half * float(np.exp(rng.normal(0.0, 0.3)))
            quad = np.array([
                pos - uu * hs - vv * hs,
                pos + uu * hs - vv * hs,
                pos + uu * hs + vv * hs,
                pos - uu * hs + vv * hs,
            ])
            vlist.append(quad)
            flist.append(np.array([[0, 1, 2], [0, 2, 3]]) + voff)
            voff += 4

    return np.vstack(vlist), np.vstack(flist)


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    if density not in _PRESETS:
        density = "high"
    params = _PRESETS[density]
    rng = np.random.default_rng(seed)

    lobes = _make_lobes(rng)

    rings = params["trunk_rings"]
    sides = params["trunk_sides"]
    ys = np.linspace(0.0, TREE_HEIGHT * 0.985, rings)
    sway_dir = rng.uniform(0.0, 2.0 * np.pi)
    sway_amp = TREE_HEIGHT * rng.uniform(0.004, 0.012)
    path, radii, rough = [], [], []
    for y in ys:
        frac = y / TREE_HEIGHT
        s = sway_amp * (frac ** 1.5)
        path.append([s * np.cos(sway_dir), y, s * np.sin(sway_dir)])
        radii.append(_trunk_radius(frac))
        rough.append(0.05 * max(0.0, 1.0 - frac / 0.12))
    trunk_v, trunk_f = _tube(path, radii, sides, cap_start=True, cap_end=True,
                             roughness=rough, rng=rng)
    trunk = trimesh.Trimesh(vertices=trunk_v, faces=trunk_f, process=True)
    trunk.merge_vertices()
    trunk.fix_normals()

    br_v, br_f, tips = _build_branches(rng, lobes, params)
    branches = trimesh.Trimesh(vertices=br_v, faces=br_f, process=True)
    branches.merge_vertices()

    can_v, can_f = _build_canopy(rng, lobes, params, tips)
    canopy = trimesh.Trimesh(vertices=can_v, faces=can_f, process=False)

    scene = trimesh.Scene()
    scene.add_geometry(trunk, geom_name="trunk")
    scene.add_geometry(branches, geom_name="branches")
    scene.add_geometry(canopy, geom_name="canopy")

    min_y = scene.bounds[0][1]
    if abs(min_y) > 1e-6:
        scene.apply_translation([0.0, -min_y, 0.0])

    return scene


# ==========================================================================
# IMAGE / SAMPLING HELPERS
# ==========================================================================
def _to_pil(a):
    a = np.clip(np.asarray(a, dtype=float), 0.0, 1.0)
    return Image.fromarray((a * 255.0 + 0.5).astype(np.uint8))


def _to_arr(img):
    return np.asarray(img.convert("RGB"), dtype=float) / 255.0


def _lum(c):
    c = np.asarray(c, dtype=float)
    return 0.299 * c[..., 0] + 0.587 * c[..., 1] + 0.114 * c[..., 2]


def _grid_patches(arr, box, n=6, frac=0.02):
    H, W, _ = arr.shape
    x0, y0, x1, y1 = box
    ph = max(1, int(frac * min(H, W)))
    out = []
    for yy in np.linspace(y0, y1, n):
        for xx in np.linspace(x0, x1, n):
            cx, cy = int(xx * W), int(yy * H)
            a = arr[max(0, cy - ph):cy + ph + 1, max(0, cx - ph):cx + ph + 1]
            if a.size:
                out.append(a.reshape(-1, 3).mean(axis=0))
    return np.array(out) if out else np.zeros((0, 3))


def _median_or(patches, fallback):
    if len(patches) >= 3:
        return np.median(patches, axis=0)
    return np.asarray(fallback, dtype=float)


# ==========================================================================
# TEXTURE SYNTHESIS
# ==========================================================================
def make_bark(size, light, dark, rng, n_lenticels=900, n_chevrons=16):
    """Chalky-white birch bark: bright field + SPARSE lenticel dashes +
    a few dark chevrons. Kept light so the trunk reads white in render."""
    light = np.asarray(light, dtype=float)
    dark = np.asarray(dark, dtype=float)

    base = np.ones((size, size, 3), dtype=float) * light[None, None, :]
    # subtle low-frequency value variation (creamy, never grey)
    low = rng.normal(0.0, 1.0, (48, 48))
    low = (low - low.min()) / (np.ptp(low) + 1e-6)
    noise = _to_arr(_to_pil(np.stack([low] * 3, -1)).resize((size, size), Image.LANCZOS))[..., 0]
    arr = np.clip(base * (0.96 + 0.06 * noise)[..., None], 0.0, 1.0)

    # horizontal lenticels -- short, light-grey dashes, wrapped to tile
    nl = int(n_lenticels * (size / 1024.0) ** 2)
    wmin, wmax = max(3, int(0.006 * size)), max(6, int(0.028 * size))
    hmax = max(2, int(0.0025 * size))
    for _ in range(nl):
        y = int(rng.integers(0, size))
        x = int(rng.integers(0, size))
        w = int(rng.integers(wmin, wmax))
        h = int(rng.integers(1, hmax + 1))
        xs = (x + np.arange(w)) % size
        ys = (y + np.arange(h)) % size
        alpha = float(rng.uniform(0.18, 0.5))
        col = np.clip(dark * float(rng.uniform(0.9, 1.4)), 0, 1)
        region = arr[np.ix_(ys, xs)]
        arr[np.ix_(ys, xs)] = region * (1.0 - alpha) + col * alpha

    # dark chevrons / fissures (sparse, off the edges for clean tiling)
    img = _to_pil(arr).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    nc = int(n_chevrons * (size / 1024.0))
    for _ in range(nc):
        cx = int(rng.integers(int(0.10 * size), int(0.90 * size)))
        cy = int(rng.integers(int(0.10 * size), int(0.90 * size)))
        w = int(rng.integers(int(0.012 * size), int(0.04 * size)))
        h = int(w * rng.uniform(1.2, 2.4))
        col = np.clip(dark * float(rng.uniform(0.5, 0.95)), 0, 1)
        a = int(255 * rng.uniform(0.5, 0.85))
        fill = (int(col[0] * 255), int(col[1] * 255), int(col[2] * 255), a)
        od.polygon([(cx, cy - h), (cx + w, cy), (cx, cy + h), (cx - w, cy)], fill=fill)
        if rng.random() < 0.5:
            od.polygon([(cx, cy - h), (cx + int(w * 0.4), cy - int(h * 1.5)),
                        (cx - int(w * 0.4), cy - int(h * 1.5))], fill=fill)
    img = Image.alpha_composite(img, overlay).convert("RGB")
    return img


def make_normal(albedo_img, strength=1.1):
    """Gentle tangent-space normal map (height = inverse luminance)."""
    L = np.asarray(albedo_img.convert("L"), dtype=float) / 255.0
    h = 1.0 - L
    gy, gx = np.gradient(h)
    nx, ny, nz = -gx * strength, -gy * strength, np.ones_like(h)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    out = np.stack([nx / ln, ny / ln, nz / ln], axis=-1) * 0.5 + 0.5
    return _to_pil(out)


_LEAF = np.array([(0.0, -0.62), (0.28, -0.12), (0.20, 0.45),
                  (0.0, 0.60), (-0.20, 0.45), (-0.28, -0.12)])


def _leaf_color(green, rng, bright, warm):
    """Constrained to fresh green: value/brightness varies, hue stays green."""
    c = np.asarray(green, dtype=float) * bright
    c[0] *= rng.uniform(0.88, 1.06)   # mild red jitter
    c[1] *= rng.uniform(0.95, 1.10)   # keep green strong
    c[2] *= rng.uniform(0.85, 1.02)
    c[0] += warm * 0.03               # small warm/cool shift only
    c[2] -= warm * 0.03
    # enforce green dominance so nothing reads brown/olive
    c[1] = max(c[1], 1.04 * max(c[0], c[2]))
    c = np.clip(c, 0.0, 1.0)
    return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 255)


def _draw_leaf_tile(S, green, bright, warm, rng):
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    n_leaves = int(rng.integers(55, 95))
    for _ in range(n_leaves):
        cx, cy = rng.uniform(0.08, 0.92, 2) * S
        size = rng.uniform(0.07, 0.14) * S
        ang = rng.uniform(0.0, 2.0 * np.pi)
        ca, sa = np.cos(ang), np.sin(ang)
        rot = np.array([[ca, -sa], [sa, ca]])
        pts = (_LEAF * size) @ rot.T + np.array([cx, cy])
        fill = _leaf_color(green, rng, bright, warm)
        d.polygon([tuple(p) for p in pts], fill=fill)
    return img


def make_leaf_atlas(green, rng, size=1024, grid=4):
    tile = size // grid
    ss = 4
    atlas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    for r in range(grid):
        for c in range(grid):
            t = r / (grid - 1)
            bright = 1.12 - 0.30 * t          # top sunlit, bottom shaded (modest)
            warm = 0.5 - 1.0 * t              # gentle warm->cool
            big = _draw_leaf_tile(tile * ss, green, bright, warm, rng)
            small = big.resize((tile, tile), Image.LANCZOS)
            atlas.paste(small, (c * tile, r * tile), small)
    return atlas


# ==========================================================================
# UV + VERTEX COLOR ASSIGNMENT
# ==========================================================================
def cylindrical_uv(mesh, u_rep, v_tile):
    v = mesh.vertices
    u = (np.arctan2(v[:, 2], v[:, 0]) / (2.0 * np.pi) + 0.5) * u_rep
    vv = v[:, 1] / v_tile
    return np.column_stack([u, vv])


def canopy_uv(mesh, rng, grid=4):
    nverts = len(mesh.vertices)
    ncards = nverts // 4
    uv = np.zeros((nverts, 2))
    base = np.array([[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]])
    for i in range(ncards):
        col = int(rng.integers(0, grid))
        row = int(rng.integers(0, grid))
        rot = int(rng.integers(0, 4))
        b = np.roll(base, rot, axis=0)
        cell = b / grid + np.array([col / grid, row / grid])
        uv[i * 4:i * 4 + 4] = cell
    return uv


def _rgba(col):
    col = np.clip(np.asarray(col, dtype=float), 0, 255)
    out = np.empty((len(col), 4), dtype=np.uint8)
    out[:, :3] = col.astype(np.uint8)
    out[:, 3] = 255
    return out


def trunk_vcolors(mesh, white_bark, rng):
    v = mesh.vertices
    frac = np.clip(v[:, 1] / TREE_HEIGHT, 0, 1)
    w = np.clip((0.08 - frac) / 0.08, 0, 1)[:, None]   # dark collar only at the foot
    white = np.asarray(white_bark, dtype=float) * 255.0
    collar = np.array([95, 80, 66], dtype=float)
    col = white * (1 - w) + collar * w
    col *= rng.uniform(0.95, 1.0, len(v))[:, None]
    return _rgba(col)


def branch_vcolors(mesh, white_bark, rng):
    v = mesh.vertices
    frac = np.clip(v[:, 1] / TREE_HEIGHT, 0, 1)
    # upper limbs near-white like the trunk, slightly greyer lower
    base = np.asarray(white_bark, dtype=float) * 255.0
    col = base * (0.80 + 0.20 * frac)[:, None]
    col *= rng.uniform(0.92, 1.02, len(v))[:, None]
    return _rgba(col)


def canopy_vcolors(mesh, rng):
    v = mesh.vertices
    ymin, ymax = v[:, 1].min(), v[:, 1].max()
    r = np.sqrt(v[:, 0] ** 2 + v[:, 2] ** 2)
    rmax = r.max() + 1e-6
    t = (v[:, 1] - ymin) / (ymax - ymin + 1e-6)
    rn = r / rmax
    bright = 0.72 + 0.28 * t + 0.12 * rn               # brighter floor; top/outer lit
    ncards = len(v) // 4
    jit = np.repeat(rng.uniform(-0.06, 0.06, ncards), 4)
    bright = np.clip(bright + jit, 0.62, 1.12)
    base = np.array([224, 236, 206], dtype=float)       # cool airy green-white
    col = bright[:, None] * base
    return _rgba(col)


# ==========================================================================
# ASSEMBLY
# ==========================================================================
def texture_scene(scene, image_path, seed):
    src = Image.open(image_path).convert("RGB")
    arr = _to_arr(src)
    H, W, _ = arr.shape
    tx_rng = np.random.default_rng((int(seed) * 2654435761) & 0xFFFFFFFF)

    # ---- Sample palette WELL INSIDE the silhouette -------------------------
    trunk_box = (0.49, 0.58, 0.62, 0.88)     # white bole, lower-center
    foliage_box = (0.18, 0.10, 0.66, 0.46)   # green crown mass

    tp = _grid_patches(arr, trunk_box, n=7, frac=0.012)
    light_sample = _median_or(
        tp[(_lum(tp) > 0.55) & ((tp.max(1) - tp.min(1)) < 0.20)] if len(tp) else tp,
        (0.88, 0.87, 0.83))
    dark_sample = _median_or(
        tp[_lum(tp) < 0.42] if len(tp) else tp,
        np.clip(np.asarray(light_sample) * 0.25, 0, 1))

    # FORCE a chalky-white bark tone (guards against background contamination)
    lb = np.asarray(light_sample, dtype=float)
    L = _lum(lb)
    if L < 0.74:
        lb = lb * (0.80 / (L + 1e-6))
    lb = np.clip(lb, 0, 1)
    white_bark = np.clip(0.4 * lb + 0.6 * np.array([0.93, 0.92, 0.88]), 0, 1)
    # dark marks: ensure they stay dark grey-black, but sparse
    dark_bark = np.clip(np.minimum(np.asarray(dark_sample), [0.22, 0.22, 0.22]), 0.05, 0.30)

    fp = _grid_patches(arr, foliage_box, n=8, frac=0.012)
    if len(fp):
        greenness = fp[:, 1] - 0.5 * (fp[:, 0] + fp[:, 2])
        green = _median_or(fp[greenness > 0.025], (0.46, 0.56, 0.33))
    else:
        green = np.array([0.46, 0.56, 0.33])
    # fresh, light spring green
    green = np.asarray(green, dtype=float)
    green = np.clip(green * 1.25 + np.array([0.04, 0.10, 0.03]), 0, 1)
    green[1] = max(green[1], 1.08 * max(green[0], green[2]))
    green = np.clip(green, 0, 1)

    # ---- Build textures ----------------------------------------------------
    trunk_img = make_bark(1024, white_bark, dark_bark, tx_rng,
                          n_lenticels=900, n_chevrons=16)
    trunk_normal = make_normal(trunk_img, strength=1.1)
    branch_img = make_bark(512, white_bark, dark_bark, tx_rng,
                           n_lenticels=350, n_chevrons=6)
    atlas_img = make_leaf_atlas(green, tx_rng, size=1024, grid=4)

    # ---- Materials ---------------------------------------------------------
    trunk_mat = trimesh.visual.material.PBRMaterial(
        name="birch_trunk", baseColorTexture=trunk_img,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0, roughnessFactor=0.9)
    trunk_mat.normalTexture = trunk_normal

    branch_mat = trimesh.visual.material.PBRMaterial(
        name="birch_branch", baseColorTexture=branch_img,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0, roughnessFactor=0.9)

    leaf_mat = trimesh.visual.material.PBRMaterial(
        name="birch_leaf", baseColorTexture=atlas_img,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0, roughnessFactor=0.8)
    leaf_mat.alphaMode = "MASK"
    leaf_mat.alphaCutoff = 0.45
    leaf_mat.doubleSided = True

    # ---- Apply to geometry -------------------------------------------------
    trunk = scene.geometry["trunk"]
    trunk.visual = trimesh.visual.TextureVisuals(
        uv=cylindrical_uv(trunk, u_rep=2.0, v_tile=2.5), material=trunk_mat)
    trunk.visual.vertex_attributes["color"] = trunk_vcolors(trunk, white_bark, tx_rng)

    branches = scene.geometry["branches"]
    branches.visual = trimesh.visual.TextureVisuals(
        uv=cylindrical_uv(branches, u_rep=2.0, v_tile=0.6), material=branch_mat)
    branches.visual.vertex_attributes["color"] = branch_vcolors(branches, white_bark, tx_rng)

    canopy = scene.geometry["canopy"]
    canopy.visual = trimesh.visual.TextureVisuals(
        uv=canopy_uv(canopy, tx_rng, grid=4), material=leaf_mat)
    canopy.visual.vertex_attributes["color"] = canopy_vcolors(canopy, tx_rng)

    return scene


def main():
    ap = argparse.ArgumentParser(description="Procedural textured silver birch -> GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        scene = build_mesh(args.seed, args.density)
        scene = texture_scene(scene, args.image, args.seed)
        scene.export(args.output)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())