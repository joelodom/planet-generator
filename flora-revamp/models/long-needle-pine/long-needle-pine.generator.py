"""
Procedural Scots-pine asset: geometry + photo-derived materials -> textured GLB.

Usage:
    python pine.py --image reference.png --seed 7 --density high --output pine.glb

Only numpy, trimesh, Pillow and the stdlib are used.
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter


# ==========================================================================
# GEOMETRY
# ==========================================================================
TREE_HEIGHT = 18.0            # meters, overall size of a mature Scots pine

# Front aspect target ~0.58 -> crown ~0.50 of height wide (plus lobe bulges).
CROWN_WIDTH_FRAC = 0.50       # canopy diameter / tree height
CROWN_RMAX = 0.5 * CROWN_WIDTH_FRAC * TREE_HEIGHT

CROWN_BASE_FRAC = 0.28        # lowest branches begin around the lower third
CROWN_BASE_Y = CROWN_BASE_FRAC * TREE_HEIGHT
CROWN_HEIGHT = TREE_HEIGHT - CROWN_BASE_Y

PROFILE_A = 0.55
PROFILE_B = 1.15

TRUNK_BASE_RADIUS = 0.22
TRUNK_TOP_RADIUS = 0.05
TRUNK_FLARE = 1.45
TRUNK_FLARE_FRAC = 0.06
TRUNK_TOP_FRAC = 0.96         # leader reaches near apex; covered by foliage
WHORL_U_MAX = 0.92            # highest branch (keep its base on the trunk)

LOBES = ((3, 0.12), (5, 0.07), (2, 0.06))


def _params(density):
    if density == "high":
        return dict(trunk_sides=14, trunk_segs=12, branch_sides=6,
                    n_whorls=15, per_whorl=(6, 8),
                    clump_fracs=(0.45, 0.66, 0.85, 1.0), per_clump=16,
                    clump_radius_frac=0.155, card_frac=0.050,
                    n_interior=44, n_apex=9)
    if density == "med":
        return dict(trunk_sides=10, trunk_segs=8, branch_sides=5,
                    n_whorls=10, per_whorl=(4, 6),
                    clump_fracs=(0.6, 0.9), per_clump=13,
                    clump_radius_frac=0.17, card_frac=0.056,
                    n_interior=18, n_apex=5)
    if density == "low":
        return dict(trunk_sides=7, trunk_segs=5, branch_sides=4,
                    n_whorls=7, per_whorl=(3, 4),
                    clump_fracs=(0.95,), per_clump=16,
                    clump_radius_frac=0.20, card_frac=0.072,
                    n_interior=8, n_apex=3)
    raise ValueError("density must be 'high', 'med' or 'low'")


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([0.0, 1.0, 0.0])


def _perp(v):
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(_normalize(v), a)) > 0.9:
        a = np.array([0.0, 0.0, 1.0])
    return _normalize(np.cross(v, a))


def _rotate(v, axis, angle):
    axis = _normalize(axis)
    c, s = np.cos(angle), np.sin(angle)
    return (v * c + np.cross(axis, v) * s +
            axis * np.dot(axis, v) * (1.0 - c))


def _jitter_dir(n, max_angle, rng):
    n = _normalize(n)
    ax = _rotate(_perp(n), n, rng.uniform(0.0, 2.0 * np.pi))
    return _normalize(_rotate(n, ax, rng.uniform(-max_angle, max_angle)))


def _profile(u):
    u = np.clip(u, 1e-4, 1.0 - 1e-4)
    f = (u ** PROFILE_A) * ((1.0 - u) ** PROFILE_B)
    peak = (PROFILE_A ** PROFILE_A) * (PROFILE_B ** PROFILE_B) / \
           ((PROFILE_A + PROFILE_B) ** (PROFILE_A + PROFILE_B))
    return f / peak


def _lobe(theta):
    s = 1.0
    for k, amp in LOBES:
        s += amp * np.sin(k * theta + k * 0.7)
    return s


def _envelope_radius(u, theta):
    return CROWN_RMAX * _profile(u) * _lobe(theta)


def _envelope_y(u):
    return CROWN_BASE_Y + u * CROWN_HEIGHT


def _make_tube(centers, radii, sides):
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    n = len(centers)

    tangents = np.zeros((n, 3))
    for i in range(n):
        if i == 0:
            tangents[i] = centers[1] - centers[0]
        elif i == n - 1:
            tangents[i] = centers[-1] - centers[-2]
        else:
            tangents[i] = centers[i + 1] - centers[i - 1]
        tangents[i] = _normalize(tangents[i])

    normals = np.zeros((n, 3))
    normals[0] = _perp(tangents[0])
    for i in range(1, n):
        t0, t1 = tangents[i - 1], tangents[i]
        axis = np.cross(t0, t1)
        if np.linalg.norm(axis) < 1e-8:
            normals[i] = normals[i - 1]
        else:
            ang = np.arccos(np.clip(np.dot(t0, t1), -1.0, 1.0))
            normals[i] = _normalize(_rotate(normals[i - 1], axis, ang))

    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    verts = []
    for i in range(n):
        t = tangents[i]
        u = normals[i]
        w = _normalize(np.cross(t, u))
        ring = (centers[i] + radii[i] *
                (np.outer(np.cos(ang), u) + np.outer(np.sin(ang), w)))
        verts.append(ring)
    verts = np.vstack(verts)

    faces = []
    for i in range(n - 1):
        a = i * sides
        b = (i + 1) * sides
        for j in range(sides):
            j2 = (j + 1) % sides
            faces.append([a + j, a + j2, b + j2])
            faces.append([a + j, b + j2, b + j])

    bottom_c = len(verts)
    top_c = len(verts) + 1
    verts = np.vstack([verts, centers[0], centers[-1]])
    for j in range(sides):
        j2 = (j + 1) % sides
        faces.append([bottom_c, j2, j])
        top = (n - 1) * sides
        faces.append([top_c, top + j, top + j2])

    return verts, np.asarray(faces, dtype=np.int64)


def _accumulate(meshes):
    vs, fs, off = [], [], 0
    for v, f in meshes:
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    return np.vstack(vs), np.vstack(fs)


def _build_trunk(rng, p):
    segs = p["trunk_segs"]
    ys = np.linspace(0.0, TREE_HEIGHT * TRUNK_TOP_FRAC, segs + 1)

    lean_dir = rng.uniform(0.0, 2.0 * np.pi)
    lean_amt = rng.uniform(0.15, 0.40)
    centers, radii = [], []
    for y in ys:
        u = y / TREE_HEIGHT
        bend = lean_amt * (u ** 1.6)
        sway = 0.06 * np.sin(u * 4.0 + lean_dir)
        x = bend * np.cos(lean_dir) + sway
        z = bend * np.sin(lean_dir) + sway * 0.5
        r = TRUNK_TOP_RADIUS + (TRUNK_BASE_RADIUS - TRUNK_TOP_RADIUS) * \
            ((1.0 - u) ** 1.4)
        if u < TRUNK_FLARE_FRAC:
            f = (TRUNK_FLARE_FRAC - u) / TRUNK_FLARE_FRAC
            r *= 1.0 + (TRUNK_FLARE - 1.0) * (f ** 2)
        centers.append([x, y, z])
        radii.append(r)

    verts, faces = _make_tube(centers, radii, p["trunk_sides"])
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    return m, np.asarray(centers), np.asarray(radii)


def _trunk_point(trunk_centers, y):
    cy = trunk_centers[:, 1]
    return np.array([np.interp(y, cy, trunk_centers[:, 0]),
                     y,
                     np.interp(y, cy, trunk_centers[:, 2])])


def _build_branches(rng, p, trunk_centers):
    """Build branch tubes; return geometry plus per-branch polyline data."""
    n_whorls = p["n_whorls"]
    sides = p["branch_sides"]
    meshes = []
    branch_data = []   # (pts(3,3), reach, u, outward)

    us = np.linspace(0.0, WHORL_U_MAX, n_whorls)
    for u in us:
        y0 = _envelope_y(u)
        base = _trunk_point(trunk_centers, y0)
        n_br = int(rng.integers(p["per_whorl"][0], p["per_whorl"][1] + 1))
        theta0 = rng.uniform(0.0, 2.0 * np.pi)
        for bi in range(n_br):
            theta = theta0 + bi * (2.0 * np.pi / n_br) + rng.uniform(-0.2, 0.2)
            shell_r = _envelope_radius(u, theta)
            reach = shell_r * rng.uniform(0.9, 1.0)
            outward = np.array([np.cos(theta), 0.0, np.sin(theta)])

            # thinner limbs (they should largely hide under foliage)
            br_r = (0.014 + 0.038 * _profile(u)) * rng.uniform(0.8, 1.1)
            br_r = min(br_r, 0.075)

            start = base + outward * 0.04 + \
                np.array([0, rng.uniform(-0.04, 0.04), 0])

            # stronger droop, then a gentle upward sweep at the tip
            droop = rng.uniform(0.30, 0.48)
            mid = base + outward * (reach * 0.58) + \
                np.array([0, -reach * droop, 0])
            tip = base + outward * reach + \
                np.array([0, -reach * droop * 0.45 + reach * 0.10, 0])
            tip[1] = np.clip(tip[1], CROWN_BASE_Y - 0.8, TREE_HEIGHT)

            pts = np.array([start, mid, tip])
            radii = [br_r, br_r * 0.55, br_r * 0.28]
            v, f = _make_tube(pts, radii, sides)
            meshes.append((v, f))
            branch_data.append((pts, reach, u, outward))

    verts, faces = _accumulate(meshes)
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    return m, branch_data


def _branch_point(pts, f):
    """Position at fraction f in [0,1] along the 2-segment branch polyline."""
    if f <= 0.5:
        a, b, t = pts[0], pts[1], f / 0.5
    else:
        a, b, t = pts[1], pts[2], (f - 0.5) / 0.5
    return a + (b - a) * t


def _build_canopy(rng, p, branch_data):
    crown_width = 2.0 * CROWN_RMAX
    fracs = p["clump_fracs"]
    per_clump = p["per_clump"]
    clump_radius = p["clump_radius_frac"] * crown_width
    half_base = p["card_frac"] * crown_width

    # --- gather clump centres -------------------------------------------
    centers, normals = [], []

    # clumps strung ALONG every branch so the limbs disappear into foliage
    for pts, reach, u, outward in branch_data:
        for f in fracs:
            c = _branch_point(pts, f)
            nrm = _normalize(outward + np.array([0.0, 0.35, 0.0]))
            centers.append(c)
            normals.append(nrm)

    # interior fillers to make the crown a continuous mass, not a shell
    for _ in range(p["n_interior"]):
        u = rng.uniform(0.12, 0.92)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        rr = _envelope_radius(u, theta) * rng.uniform(0.30, 0.80)
        y = _envelope_y(u)
        c = np.array([rr * np.cos(theta), y, rr * np.sin(theta)])
        centers.append(c)
        normals.append(_normalize(np.array([c[0], 0.3 * CROWN_RMAX, c[2]])))

    # apex clumps to cap the leader with a softly rounded top
    for _ in range(p["n_apex"]):
        y = _envelope_y(rng.uniform(0.93, 1.0))
        rr = rng.uniform(0.0, 0.18 * crown_width)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        c = np.array([rr * np.cos(theta), y, rr * np.sin(theta)])
        centers.append(c)
        normals.append(_normalize(np.array([c[0], 0.6 * CROWN_RMAX, c[2]])))

    # --- spawn cards inside each clump ----------------------------------
    quads_v, quads_f = [], []
    voff = 0
    for c, nrm in zip(centers, normals):
        for _ in range(per_clump):
            d = rng.normal(0.0, 1.0, 3)
            d *= np.array([1.0, 0.7, 1.0])
            pos = c + _normalize(d) * (clump_radius *
                                       rng.uniform(0.0, 1.0) ** 0.55)

            cn = _jitter_dir(nrm, np.radians(28.0), rng)
            t1 = _perp(cn)
            t1 = _rotate(t1, cn, rng.uniform(0.0, 2.0 * np.pi))
            t2 = _normalize(np.cross(cn, t1))

            hx = half_base * float(np.exp(rng.normal(0.0, 0.32)))
            hy = hx * rng.uniform(0.7, 1.1)

            v0 = pos - t1 * hx - t2 * hy
            v1 = pos + t1 * hx - t2 * hy
            v2 = pos + t1 * hx + t2 * hy
            v3 = pos - t1 * hx + t2 * hy
            quads_v.append([v0, v1, v2, v3])
            quads_f.append([[voff, voff + 1, voff + 2],
                            [voff, voff + 2, voff + 3]])
            voff += 4

    verts = np.vstack([np.asarray(q) for q in quads_v])
    faces = np.vstack([np.asarray(f) for f in quads_f]).astype(np.int64)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    p = _params(density)

    trunk, trunk_centers, _ = _build_trunk(rng, p)
    branches, branch_data = _build_branches(rng, p, trunk_centers)
    canopy = _build_canopy(rng, p, branch_data)

    parts = [trunk, branches, canopy]
    min_y = min(part.vertices[:, 1].min() for part in parts)
    all_v = np.vstack([part.vertices[:, [0, 2]] for part in parts])
    cx, cz = all_v[:, 0].mean(), all_v[:, 1].mean()
    shift = np.array([-cx, -min_y, -cz])
    for part in parts:
        part.apply_translation(shift)

    for part in (trunk, branches):
        part.fix_normals()

    scene = trimesh.Scene()
    scene.add_geometry(trunk, geom_name="trunk")
    scene.add_geometry(branches, geom_name="branches")
    scene.add_geometry(canopy, geom_name="canopy")
    return scene


# ==========================================================================
# TEXTURING  (photo-derived materials)
# ==========================================================================
def _load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def _to_pil(arr, mode="RGB"):
    a = arr
    if a.dtype.kind == "f":
        a = (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(a, mode)


def _value_noise(h, w, gx, gy, rng):
    grid = (rng.random((max(gy, 2), max(gx, 2))) * 255).astype(np.uint8)
    im = Image.fromarray(grid).resize((w, h), Image.BILINEAR)
    return np.asarray(im, dtype=np.float64) / 255.0


def _norm01(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-9)


def _make_tileable_mirror(arr, out_size):
    top = np.concatenate([arr, arr[:, ::-1]], axis=1)
    full = np.concatenate([top, top[::-1, :]], axis=0)
    im = _to_pil(full).resize((out_size, out_size), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64) / 255.0


def _get_crop(arr, cx, cy, hw, hh):
    H, W = arr.shape[:2]
    x0 = int(np.clip((cx - hw) * W, 0, W - 2))
    x1 = int(np.clip((cx + hw) * W, x0 + 1, W))
    y0 = int(np.clip((cy - hh) * H, 0, H - 2))
    y1 = int(np.clip((cy + hh) * H, y0 + 1, H))
    return arr[y0:y1, x0:x1].copy()


def _delight(crop):
    pil = _to_pil(crop)
    lum = np.asarray(pil.convert("L"), dtype=np.float64) / 255.0
    blur = np.asarray(
        pil.convert("L").filter(ImageFilter.GaussianBlur(
            radius=max(4, min(crop.shape[:2]) // 3))),
        dtype=np.float64) / 255.0
    gain = np.clip((lum.mean() + 1e-3) / (blur + 1e-3), 0.6, 1.6)
    return np.clip(crop * gain[..., None], 0.0, 1.0)


def _pixel_pool(arr, centers, half):
    H, W = arr.shape[:2]
    pool = []
    for cx, cy in centers:
        x0 = int(np.clip((cx - half) * W, 0, W - 1))
        x1 = int(np.clip((cx + half) * W, x0 + 1, W))
        y0 = int(np.clip((cy - half) * H, 0, H - 1))
        y1 = int(np.clip((cy + half) * H, y0 + 1, H))
        patch = arr[y0:y1, x0:x1].reshape(-1, 3)
        if len(patch):
            pool.append(patch)
    if not pool:
        return np.zeros((1, 3))
    return np.vstack(pool)


def _tone(pool, pct):
    lum = pool @ np.array([0.299, 0.587, 0.114])
    order = np.argsort(lum)
    n = len(order)
    i = int(np.clip(pct, 0.0, 1.0) * (n - 1))
    win = max(1, n // 20)
    lo, hi = max(0, i - win), min(n, i + win + 1)
    return pool[order[lo:hi]].mean(axis=0)


# ---- bark -----------------------------------------------------------------
def _bark_palette(arr):
    centers = [(0.47, 0.78), (0.49, 0.83), (0.46, 0.88),
               (0.48, 0.93), (0.50, 0.80), (0.45, 0.85)]
    pool = _pixel_pool(arr, centers, 0.012)
    lum = pool @ np.array([0.299, 0.587, 0.114])
    r, b = pool[:, 0], pool[:, 2]
    mask = (r >= b * 0.95) & (lum > 0.06) & (lum < 0.85)
    if mask.sum() > 30:
        pool = pool[mask]
    dark = _tone(pool, 0.20)
    mid = _tone(pool, 0.50)
    light = _tone(pool, 0.82)
    dl = dark @ np.array([0.299, 0.587, 0.114]) + 1e-3
    ll = light @ np.array([0.299, 0.587, 0.114]) + 1e-3
    if ll < dl * 2.0:
        light = np.clip(light * (dl * 2.3 / ll), 0, 1)
    return dark, mid, light


def _build_bark(arr, rng, size=768):
    dark, mid, light = _bark_palette(arr)

    furrow = (_value_noise(size, size, 44, 6, rng) * 0.6 +
              _value_noise(size, size, 90, 13, rng) * 0.4)
    plate = _value_noise(size, size, 7, 11, rng)
    grain = _value_noise(size, size, 230, 230, rng)
    shade = _norm01(0.6 * furrow + 0.25 * plate + 0.15 * grain)

    s = shade[..., None]
    albedo = dark[None, None, :] * (1 - s) + light[None, None, :] * s
    albedo = albedo * 0.7 + mid[None, None, :] * 0.3 * (0.5 + 0.5 * s)

    crop = _delight(_get_crop(arr, 0.47, 0.84, 0.03, 0.13))
    field = np.asarray(_to_pil(crop).resize((size, size), Image.LANCZOS),
                       dtype=np.float64) / 255.0
    fmean = field.reshape(-1, 3).mean(axis=0) + 1e-3
    albedo = albedo * (0.7 + 0.3 * field / fmean[None, None, :])

    albedo = np.clip(albedo, 0.0, 1.0)
    albedo = _make_tileable_mirror(albedo, size)
    return _to_pil(albedo)


def _normal_from_albedo(pil_img, strength=2.2):
    lum = np.asarray(pil_img.convert("L"), dtype=np.float64) / 255.0
    height = 1.0 - lum
    gy, gx = np.gradient(height)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(height)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    nmap = np.stack([nx / ln, ny / ln, nz / ln], axis=-1)
    return _to_pil(nmap * 0.5 + 0.5)


# ---- foliage atlas --------------------------------------------------------
def _foliage_palette(arr):
    centers = [(x, y) for x in (0.32, 0.40, 0.48, 0.56, 0.62)
               for y in (0.14, 0.22, 0.30, 0.40, 0.48)]
    pool = _pixel_pool(arr, centers, 0.018)
    r, g, b = pool[:, 0], pool[:, 1], pool[:, 2]
    lum = pool @ np.array([0.299, 0.587, 0.114])
    mask = (g >= r * 0.97) & (g >= b * 0.97) & (lum > 0.07) & (lum < 0.70)
    if mask.sum() > 40:
        pool = pool[mask]
    dark = _tone(pool, 0.22)
    mid = _tone(pool, 0.50)
    light = _tone(pool, 0.80)
    return dark, mid, light


def _draw_needle_tile(px, base_col, rng):
    ss = 4
    big = px * ss
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = cy = big / 2.0

    n_strokes = 175
    spread = big * 0.33
    for _ in range(n_strokes):
        a0 = rng.uniform(0, 2 * np.pi)
        rr = spread * (rng.uniform(0.0, 1.0) ** 0.5)
        ox = cx + rr * np.cos(a0)
        oy = cy + rr * np.sin(a0)
        ang = a0 + rng.uniform(-0.6, 0.6)
        length = big * rng.uniform(0.07, 0.16)
        ex = ox + length * np.cos(ang)
        ey = oy + length * np.sin(ang)
        # tighter, cooler jitter -> clean muted blue-green rather than olive
        jit = rng.uniform(-0.10, 0.10, 3) * np.array([0.9, 1.0, 0.9])
        col = np.clip(base_col + jit, 0.0, 1.0)
        rgb = tuple(int(c * 255) for c in col)
        w = max(2, int(big * rng.uniform(0.006, 0.012)))
        d.line([(ox, oy), (ex, ey)], fill=rgb + (255,), width=w)

    return img.resize((px, px), Image.LANCZOS)


def _build_foliage_atlas(arr, rng, atlas=1024):
    dark, mid, light = _foliage_palette(arr)
    px = atlas // 4
    out = Image.new("RGBA", (atlas, atlas), (0, 0, 0, 0))
    for row in range(4):
        for col in range(4):
            sun = (3 - row) / 3.0  # top rows sunlit, bottom rows shaded
            base = mid * (1 - sun) + light * sun
            # gentle warm-up when sunlit, slight cool-down in shade
            warm = np.array([0.04, 0.02, -0.03]) * (sun - 0.5) * 2.0
            cool = np.array([-0.02, 0.0, 0.03]) * (0.5 - sun) * 2.0
            base = np.clip(base * (0.84 + 0.24 * sun) + warm + cool, 0, 1)
            base = base * 0.84 + dark * 0.16
            tile = _draw_needle_tile(px, np.clip(base, 0, 1), rng)
            out.paste(tile, (col * px, row * px), tile)
    return out


# ---- UV helpers -----------------------------------------------------------
def _cyl_uv(verts, repeat_u=5.0, world_per_v=2.0):
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    u = (np.arctan2(z, x) / (2.0 * np.pi)) % 1.0
    return np.column_stack([u * repeat_u, (y / world_per_v)])


def _canopy_uv(n_cards, rng, margin=0.5 / 1024.0):
    tiles = rng.integers(0, 16, n_cards)
    rots = rng.integers(0, 4, n_cards)
    base = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    uv = np.zeros((n_cards * 4, 2))
    for k in range(n_cards):
        t = int(tiles[k])
        col, row = t % 4, t // 4
        u0, u1 = col / 4.0 + margin, (col + 1) / 4.0 - margin
        v0, v1 = row / 4.0 + margin, (row + 1) / 4.0 - margin
        r = int(rots[k])
        for i in range(4):
            s, tt = base[(i + r) % 4]
            uv[4 * k + i] = [u0 + s * (u1 - u0), v0 + tt * (v1 - v0)]
    return uv


# ---- per-vertex colours ---------------------------------------------------
def _wood_colors(verts, rng):
    y = verts[:, 1]
    ao = np.minimum(np.clip(0.62 + 0.38 * (y / 3.0), 0.62, 1.0), 1.0)
    jit = 1.0 + rng.normal(0.0, 0.03, len(y))
    base = np.clip(ao * jit, 0.0, 1.0)
    col = np.clip(np.stack([base * 1.02, base * 0.98, base * 0.95], -1), 0, 1)
    rgba = np.concatenate([col, np.ones((len(y), 1))], axis=1)
    return (rgba * 255).astype(np.uint8)


def _canopy_colors(verts, rng):
    n_cards = len(verts) // 4
    centers = verts.reshape(n_cards, 4, 3).mean(axis=1)
    cy = centers[:, 1]
    rad = np.sqrt(centers[:, 0] ** 2 + centers[:, 2] ** 2)
    yfrac = _norm01(cy)
    rfrac = np.clip(rad / (CROWN_RMAX + 1e-6), 0, 1)
    lit = np.clip(0.6 * yfrac + 0.4 * rfrac, 0, 1)
    lit = np.clip(lit + rng.normal(0.0, 0.05, n_cards), 0.0, 1.0)
    bright = 0.58 + 0.42 * lit
    warm = np.array([1.05, 1.0, 0.89])
    cool = np.array([0.90, 1.0, 1.05])
    tone = cool[None, :] * (1 - lit[:, None]) + warm[None, :] * lit[:, None]
    col = np.clip(bright[:, None] * tone, 0.0, 1.0)
    col = np.repeat(col, 4, axis=0)
    rgba = np.concatenate([col, np.ones((len(col), 1))], axis=1)
    return (rgba * 255).astype(np.uint8)


# ---- assemble materials ---------------------------------------------------
def texture_scene(scene, image_path, seed):
    arr = _load_image(image_path)
    rng = np.random.default_rng(seed + 9973)

    bark_img = _build_bark(arr, rng)
    bark_normal = _normal_from_albedo(bark_img)
    atlas_img = _build_foliage_atlas(arr, rng)

    PBR = trimesh.visual.material.PBRMaterial
    TV = trimesh.visual.TextureVisuals

    wood_mat = PBR(name="bark", baseColorTexture=bark_img,
                   normalTexture=bark_normal,
                   metallicFactor=0.0, roughnessFactor=0.9,
                   baseColorFactor=[255, 255, 255, 255])
    branch_mat = PBR(name="branch_bark", baseColorTexture=bark_img,
                     normalTexture=bark_normal,
                     metallicFactor=0.0, roughnessFactor=0.9,
                     baseColorFactor=[255, 255, 255, 255])
    leaf_mat = PBR(name="needles", baseColorTexture=atlas_img,
                   metallicFactor=0.0, roughnessFactor=0.8,
                   alphaMode="MASK", alphaCutoff=0.45, doubleSided=True,
                   baseColorFactor=[255, 255, 255, 255])

    trunk = scene.geometry["trunk"]
    branches = scene.geometry["branches"]
    canopy = scene.geometry["canopy"]

    trunk.visual = TV(uv=_cyl_uv(trunk.vertices, 5.0, 2.0), material=wood_mat)
    trunk.visual.vertex_attributes["color"] = _wood_colors(trunk.vertices, rng)

    branches.visual = TV(uv=_cyl_uv(branches.vertices, 6.0, 1.2),
                         material=branch_mat)
    branches.visual.vertex_attributes["color"] = \
        _wood_colors(branches.vertices, rng)

    n_cards = len(canopy.vertices) // 4
    canopy.visual = TV(uv=_canopy_uv(n_cards, rng), material=leaf_mat)
    canopy.visual.vertex_attributes["color"] = \
        _canopy_colors(canopy.vertices, rng)

    return scene


# ==========================================================================
# CLI
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="Procedural pine -> textured GLB")
    ap.add_argument("--image", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    try:
        scene = build_mesh(args.seed, args.density)
        scene = texture_scene(scene, args.image, args.seed)
        scene.export(args.output)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())