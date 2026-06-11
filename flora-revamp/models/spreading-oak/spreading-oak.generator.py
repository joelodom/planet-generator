"""
Standalone generator + texturer for a mature, open-grown white-oak shade tree.

Builds procedural geometry (stout flared trunk + heavy scaffold limbs strictly
contained inside a lobed dome of dense, overlapping leaf CARDS), derives tileable
bark + a foliage leaf-card ATLAS from a reference photo, applies surface-
appropriate UVs and PBR materials, bakes sun/shade vertex colors, and exports a
textured binary GLB.

CLI:
    python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb

Only numpy + trimesh + PIL + stdlib.  +Y up, base at y=0, meters.
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial


# ===========================================================================
# GEOMETRY  (build_mesh; +Y up, base at y=0, meters)
# ===========================================================================
# --- Measured proportions (read by eye off reference.png) + real-world size ---
TREE_HEIGHT = 15.0                 # meters, total height (a big specimen oak)
CROWN_WIDTH = 16.0                 # meters, crown spread  -> H/W ~= 0.94 (photo ~1.07)
HEIGHT_OVER_WIDTH = TREE_HEIGHT / CROWN_WIDTH

BRANCH_START_FRAC = 0.22           # crotch height as fraction of total height
TRUNK_RADIUS = 0.85                # stout trunk: ~1.7 m diameter
BASAL_FLARE = 1.45                 # base root-collar swell vs. trunk radius
FLARE_FRAC = 0.07                  # flare confined to bottom ~7% of trunk

# Foliage envelope: oblate (squashed) lobed ellipsoid -- the crown DOME.
CROWN_CENTER_Y = 8.7               # vertical center of the crown mass (m)
CROWN_RX = 8.0                     # half-width in X (crown radius)
CROWN_RZ = 8.0                     # half-width in Z
CROWN_RY = 6.3                     # half-height -> top ~15.0 m, bottom ~2.4 m
WOOD_INSET = 0.86                  # branches kept inside this fraction of the shell


def _normalize(v, eps=1e-9):
    n = np.linalg.norm(v)
    return v / n if n > eps else v


def _frame(direction):
    """Return two unit vectors orthogonal to `direction`."""
    d = _normalize(np.asarray(direction, dtype=float))
    ref = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = _normalize(np.cross(d, ref))
    v = np.cross(d, u)
    return u, v


def _clamp_into_crown(p):
    """Pull a point inside the (axis-aligned) crown ellipsoid so no wood pokes
    out past the foliage.  Limits horizontal reach per height + caps the top;
    the bottom is left free so the low trunk/crotch is unaffected."""
    p = np.asarray(p, float).copy()
    t = (p[1] - CROWN_CENTER_Y) / CROWN_RY
    if t > 0.96:
        p[1] = CROWN_CENTER_Y + 0.96 * CROWN_RY
        t = 0.96
    hw = CROWN_RX * np.sqrt(max(0.0, 1.0 - min(t, 0.999) ** 2)) * WOOD_INSET
    r = np.hypot(p[0], p[2])
    if r > hw and r > 1e-6:
        s = hw / r
        p[0] *= s
        p[2] *= s
    return p


def _make_lobes(rng, n_lobes=5):
    """A few random low-frequency bulge axes/amplitudes for the crown shell."""
    axes = rng.normal(size=(n_lobes, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    amps = rng.uniform(0.05, 0.11, size=n_lobes)   # gentle -> rounder dome
    return axes, amps


def _lobe_factor(unit_dir, axes, amps):
    f = 1.0 + np.sum(amps * (axes @ unit_dir))
    return float(np.clip(f, 0.86, 1.14))


def _envelope_point(unit_dir, axes, amps, inset=1.0):
    f = _lobe_factor(unit_dir, axes, amps) * inset
    center = np.array([0.0, CROWN_CENTER_Y, 0.0])
    radii = np.array([CROWN_RX, CROWN_RY, CROWN_RZ])
    return center + f * radii * unit_dir


def _frustum(p0, p1, r0, r1, sides):
    """Capped tapered cylinder between p0 (radius r0) and p1 (radius r1)."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    axis = p1 - p0
    if np.linalg.norm(axis) < 1e-6:
        return None
    u, v = _frame(axis)
    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    circ = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v
    ring0 = p0 + r0 * circ
    ring1 = p1 + r1 * circ
    verts = np.vstack([ring0, ring1, p0[None, :], p1[None, :]])
    c0, c1 = 2 * sides, 2 * sides + 1
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        a, b = i, j
        c, d = sides + i, sides + j
        faces.append([a, b, d])
        faces.append([a, d, c])
        faces.append([c0, b, a])
        faces.append([c1, c, d])
    return verts, np.array(faces, dtype=np.int64)


def _grow_branch(start, direction, length, r0, r1, depth, max_depth,
                 sides, rng, segs, tips):
    n_seg = max(2, 4 - depth)
    cur = np.asarray(start, float)
    d = _normalize(direction)
    for i in range(n_seg):
        t1 = (i + 1) / n_seg
        seg_r1 = r0 + (r1 - r0) * t1
        jitter = rng.normal(scale=0.16 + 0.05 * depth, size=3)
        d = _normalize(d + jitter * np.array([1.0, 0.5, 1.0])
                       + np.array([0.0, 0.06, 0.0]))
        nxt = _clamp_into_crown(cur + d * (length / n_seg))
        segs.append((cur.copy(), nxt.copy(), seg_r1 if i else r0, seg_r1))
        cur = nxt
    tips.append(cur.copy())

    if depth >= max_depth or r1 < 0.05:
        return
    n_child = int(rng.integers(2, 4))
    child_r = r1 / np.sqrt(n_child) * rng.uniform(0.9, 1.02, size=n_child)
    for k in range(n_child):
        spread = rng.uniform(0.45, 0.85)
        side = _normalize(rng.normal(size=3) * np.array([1.0, 0.4, 1.0]))
        cdir = _normalize(d * (1.0 - spread) + side * spread
                          + np.array([0.0, 0.12, 0.0]))
        clen = length * rng.uniform(0.62, 0.82)
        _grow_branch(cur, cdir, clen, child_r[k] * 0.95, child_r[k] * 0.5,
                     depth + 1, max_depth, max(6, sides - 2),
                     rng, segs, tips)


def _build_wood(rng, max_depth, sides):
    """Stout flared trunk + low-splitting heavy scaffold limbs -> one mesh."""
    segs = []
    tips = []
    crotch_y = TREE_HEIGHT * BRANCH_START_FRAC

    flare_top = crotch_y * FLARE_FRAC
    segs.append((np.array([0.0, 0.0, 0.0]),
                 np.array([0.0, flare_top, 0.0]),
                 TRUNK_RADIUS * BASAL_FLARE, TRUNK_RADIUS * 1.08))
    n_trunk = 3
    prev = np.array([0.0, flare_top, 0.0])
    for i in range(n_trunk):
        t1 = (i + 1) / n_trunk
        y = flare_top + (crotch_y - flare_top) * t1
        lean = rng.normal(scale=0.05, size=2)
        nxt = np.array([lean[0], y, lean[1]])
        r_top = TRUNK_RADIUS * (1.08 - 0.18 * t1)
        segs.append((prev, nxt, TRUNK_RADIUS * (1.08 - 0.18 * (i / n_trunk)),
                     r_top))
        prev = nxt

    n_primary = int(rng.integers(4, 6))
    base_r = TRUNK_RADIUS * (1.08 - 0.18)
    limb_r = base_r / np.sqrt(n_primary) * 1.25
    az0 = rng.uniform(0, 2 * np.pi)
    for i in range(n_primary):
        az = az0 + 2 * np.pi * i / n_primary + rng.uniform(-0.3, 0.3)
        elev = rng.uniform(0.62, 1.0)                      # lean well up into dome
        direction = np.array([np.cos(az) * (1.0 - elev * 0.5),
                              elev,
                              np.sin(az) * (1.0 - elev * 0.5)])
        length = rng.uniform(4.2, 5.6)                     # shorter -> stays inside
        _grow_branch(prev, direction, length, limb_r * rng.uniform(0.95, 1.05),
                     limb_r * 0.45, 1, max_depth, sides, rng, segs, tips)

    all_v, all_f, offset = [], [], 0
    for (p0, p1, ra, rb) in segs:
        out = _frustum(p0, p1, max(ra, 0.02), max(rb, 0.02), sides)
        if out is None:
            continue
        v, f = out
        all_v.append(v)
        all_f.append(f + offset)
        offset += len(v)
    verts = np.vstack(all_v)
    faces = np.vstack(all_f)
    wood = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    return wood, tips


def _build_canopy(rng, n_clumps, n_cards, axes, amps, branch_tips):
    """Dense, overlapping, shell-hugging leaf cards filling the whole dome
    (sides + lower flanks included) so the silhouette reads as a solid crown."""
    crown_center = np.array([0.0, CROWN_CENTER_Y, 0.0])
    clump_w = (CROWN_RX + CROWN_RZ) * 0.5

    centers = []
    n_inter = max(2, int(round(n_clumps * 0.30)))
    n_low = max(2, n_clumps // 6)
    n_shell = max(1, n_clumps - n_inter - n_low)

    # Shell clumps: vertical compressed so coverage spreads over sides + top.
    for _ in range(n_shell):
        u = rng.normal(size=3)
        u[1] *= 0.72
        u = _normalize(u)
        centers.append(_envelope_point(u, axes, amps, inset=rng.uniform(0.85, 0.97)))
    # Lower-flank clumps: drape the crown down toward the ground.
    for _ in range(n_low):
        ang = rng.uniform(0.0, 2.0 * np.pi)
        u = _normalize(np.array([np.cos(ang), rng.uniform(-0.55, -0.12), np.sin(ang)]))
        centers.append(_envelope_point(u, axes, amps, inset=rng.uniform(0.86, 0.97)))
    # Interior clumps: fill holes so the dome reads solid, not see-through.
    for _ in range(n_inter):
        u = _normalize(rng.normal(size=3))
        centers.append(_envelope_point(u, axes, amps, inset=rng.uniform(0.40, 0.70)))

    if branch_tips:
        tips = [t for t in branch_tips if t[1] > CROWN_CENTER_Y - CROWN_RY]
        rng.shuffle(tips)
        for i in range(min(len(tips), n_shell // 2)):
            centers[i] = 0.5 * centers[i] + 0.5 * tips[i]
    centers = np.array(centers)

    cards_per = max(8, n_cards // len(centers))
    half_base = 0.048 * CROWN_WIDTH          # ~0.77 m cards, generous overlap
    clump_rad = 0.15 * clump_w               # wide clumps -> they merge

    verts = np.empty((n_cards * 4, 3), dtype=np.float64)
    faces = np.empty((n_cards * 2, 3), dtype=np.int64)
    quad = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    written = 0

    for c in centers:
        if written >= n_cards:
            break
        for _ in range(cards_per):
            if written >= n_cards:
                break
            off = _normalize(rng.normal(size=3)) * (clump_rad * rng.uniform(0.0, 1.0) ** 0.5)
            pos = c + off
            nrm = _normalize(pos - crown_center)
            nrm = _normalize(nrm + rng.normal(scale=0.45, size=3))
            u, v = _frame(nrm)
            s = half_base * float(np.exp(rng.normal(0.0, 0.30)))
            s = min(s, half_base * 2.2)
            a = rng.uniform(0, 2 * np.pi)
            ca, sa = np.cos(a), np.sin(a)
            ax = ca * u + sa * v
            ay = -sa * u + ca * v
            base = written * 4
            for k in range(4):
                verts[base + k] = pos + s * (quad[k, 0] * ax + quad[k, 1] * ay)
            faces[written * 2] = [base, base + 1, base + 2]
            faces[written * 2 + 1] = [base, base + 2, base + 3]
            written += 1

    verts = verts[: written * 4]
    faces = faces[: written * 2]
    canopy = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    canopy.fix_normals()
    return canopy


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)

    if density == "high":
        trunk_sides, max_depth, n_clumps, n_cards = 14, 4, 28, 3400
    elif density == "med":
        trunk_sides, max_depth, n_clumps, n_cards = 10, 3, 18, 1400
    elif density == "low":
        trunk_sides, max_depth, n_clumps, n_cards = 6, 2, 12, 460
    else:
        raise ValueError("density must be 'high', 'med' or 'low'")

    axes, amps = _make_lobes(rng, n_lobes=5)

    wood, tips = _build_wood(rng, max_depth=max_depth, sides=trunk_sides)
    canopy = _build_canopy(rng, n_clumps=n_clumps, n_cards=n_cards,
                           axes=axes, amps=amps, branch_tips=tips)

    combined_min_y = min(wood.bounds[0][1], canopy.bounds[0][1])
    shift = np.array([0.0, -combined_min_y, 0.0])
    wood.apply_translation(shift)
    canopy.apply_translation(shift)

    scene = trimesh.Scene()
    scene.add_geometry(wood, geom_name="trunk")
    scene.add_geometry(canopy, geom_name="canopy")
    return scene


# ===========================================================================
# PHOTO SAMPLING  -- pull body colors from WELL INSIDE the silhouette
# ===========================================================================
def _load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def _patch_median(img, fx, fy, patch):
    H, W = img.shape[:2]
    x = int(np.clip(fx * W, 0, W - patch))
    y = int(np.clip(fy * H, 0, H - patch))
    crop = img[y:y + patch, x:x + patch].reshape(-1, 3)
    return np.median(crop, axis=0)


def _sample_color(img, region, rng, filt, fallback, n=48, patch=7):
    """Median of accepted small patches within a fractional region box."""
    x0, x1, y0, y1 = region
    cols = []
    for _ in range(n):
        fx = rng.uniform(x0, x1)
        fy = rng.uniform(y0, y1)
        c = _patch_median(img, fx, fy, patch)
        if filt(c):
            cols.append(c)
    if len(cols) < 3:
        return np.array(fallback, dtype=np.float64)
    return np.median(np.array(cols), axis=0)


def _brighten_to(color, target_lum, lo, hi):
    """Scale a color so its luminance hits a healthy target (clamped gain)."""
    lum = float(color @ np.array([0.299, 0.587, 0.114]))
    scale = np.clip(target_lum / max(lum, 1e-3), lo, hi)
    return np.clip(color * scale, 0.0, 0.96)


def _is_foliage(c):
    return (c[1] >= c[0] - 0.01) and (c[1] > c[2] + 0.01) and (c.max() - c.min() > 0.04)


def _is_bark(c):
    return not ((c[1] > c[0] + 0.03) and (c[1] > c[2] + 0.03))


# ===========================================================================
# TEXTURE SYNTHESIS HELPERS
# ===========================================================================
def _tiled_noise(res, rng, n_waves=18, max_f=6):
    """Seamlessly tileable smooth scalar noise via integer-frequency sinusoids."""
    u = np.linspace(0.0, 1.0, res, endpoint=False)
    U, V = np.meshgrid(u, u)
    acc = np.zeros((res, res))
    for _ in range(n_waves):
        fx = int(rng.integers(1, max_f + 1))
        fy = int(rng.integers(0, max_f + 1))
        ph = rng.uniform(0.0, 2.0 * np.pi)
        amp = 1.0 / (1.0 + fx + fy)
        acc += amp * np.sin(2.0 * np.pi * (fx * U + fy * V) + ph)
    acc -= acc.min()
    span = acc.max() - acc.min()
    if span > 1e-9:
        acc /= span
    return acc


def _delight(rgb01, res):
    """Divide out a heavily blurred luminance, gain clamped to [0.6, 1.6]."""
    lum = rgb01 @ np.array([0.299, 0.587, 0.114])
    limg = Image.fromarray(np.clip(lum * 255, 0, 255).astype(np.uint8))
    blurred = np.asarray(limg.filter(ImageFilter.GaussianBlur(res * 0.08)),
                         dtype=np.float64) / 255.0
    target = float(lum.mean())
    gain = np.clip(target / (blurred + 1e-3), 0.6, 1.6)
    return np.clip(rgb01 * gain[..., None], 0.0, 1.0)


def _normal_from_albedo(rgb01, strength=2.6):
    """Tileable tangent-space normal map from albedo luminance (height=lum)."""
    h = rgb01 @ np.array([0.299, 0.587, 0.114])
    gx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.5
    gy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 0.5
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(h)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    out = np.stack([nx / ln, ny / ln, nz / ln], axis=-1) * 0.5 + 0.5
    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8), "RGB")


def make_bark_textures(bark_color, rng, res=512):
    """Furrowed, ridged grey-brown bark: vertical fissures + grain. Tileable."""
    u = np.linspace(0.0, 1.0, res, endpoint=False)
    U, V = np.meshgrid(u, u)
    warp = _tiled_noise(res, rng, n_waves=10, max_f=4)
    grain = _tiled_noise(res, rng, n_waves=22, max_f=6)

    n_ridges = 8.0
    ridge = 0.5 + 0.5 * np.cos(2.0 * np.pi * (U * n_ridges + 0.18 * warp))
    ridge = ridge ** 1.4
    plates = 0.5 + 0.5 * np.cos(2.0 * np.pi * (V * 5.0 + 0.5 * grain))
    # keep a healthy value range (lightest ~2.5x darkest) so it never reads flat.
    value = 0.50 + 0.42 * ridge * (0.70 + 0.30 * plates) + 0.08 * grain
    value = np.clip(value, 0.30, 1.18)

    warm = np.array([1.08, 1.0, 0.88])
    albedo = np.clip(bark_color[None, None, :] * warm[None, None, :]
                     * value[..., None], 0.0, 1.0)
    albedo = _delight(albedo, res)
    albedo_img = Image.fromarray(np.clip(albedo * 255, 0, 255).astype(np.uint8), "RGB")
    normal_img = _normal_from_albedo(albedo, strength=2.6)
    return albedo_img, normal_img


def _leaf_polygon(cx, cy, scale, angle):
    """A small lobed oak-ish leaf as a polygon point list (image px)."""
    n = 11
    ts = np.linspace(0.0, 1.0, n)
    w = np.sin(np.pi * ts) * (0.55 + 0.18 * np.sin(ts * np.pi * 4.0))
    w *= 0.45
    midrib = ts - 0.5
    left = np.stack([-w, midrib], axis=1)
    right = np.stack([w[::-1], midrib[::-1]], axis=1)
    pts = np.vstack([left, right])
    ca, sa = np.cos(angle), np.sin(angle)
    rot = np.array([[ca, -sa], [sa, ca]])
    pts = (rot @ (pts * scale).T).T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return [(float(p[0]), float(p[1])) for p in pts]


def make_foliage_atlas(leaf_color, rng, tile=256, grid=4, ss=4):
    """4x4 atlas of distinct leaf-cluster tiles (sunlit warmer, shaded cooler)."""
    res = tile * grid
    atlas = Image.new("RGBA", (res, res), (0, 0, 0, 0))
    base = np.clip(leaf_color, 0.06, 0.95)

    for gy in range(grid):
        for gx in range(grid):
            big = tile * ss
            tile_img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
            draw = ImageDraw.Draw(tile_img)
            shade = (gx + gy) / (2 * (grid - 1))
            sun = 1.0 - shade
            tint = base * (0.74 + 0.50 * sun)                      # brighter range
            tint = tint * np.array([0.95 + 0.14 * sun, 1.0, 0.80 + 0.08 * sun])
            n_leaves = int(rng.integers(48, 72))
            cx0, cy0 = big * 0.5, big * 0.5
            cluster_r = big * 0.42
            for _ in range(n_leaves):
                rr = cluster_r * (rng.uniform(0.0, 1.0) ** 0.5)
                aa = rng.uniform(0.0, 2.0 * np.pi)
                cx = cx0 + rr * np.cos(aa)
                cy = cy0 + rr * np.sin(aa)
                sc = big * rng.uniform(0.16, 0.30)
                ang = rng.uniform(0.0, 2.0 * np.pi)
                jit = rng.uniform(0.82, 1.18) * float(np.exp(rng.normal(0, 0.10)))
                col = np.clip(tint * jit, 0.0, 1.0)
                rgb = tuple(int(c * 255) for c in col)
                poly = _leaf_polygon(cx, cy, sc, ang)
                draw.polygon(poly, fill=rgb + (255,))
            tile_small = tile_img.resize((tile, tile), Image.LANCZOS)
            atlas.paste(tile_small, (gx * tile, gy * tile), tile_small)

    arr = np.asarray(atlas).copy()
    a = arr[..., 3].astype(np.float64)
    a = np.where(a > 170, 255, np.where(a < 70, 0, a)).astype(np.uint8)
    arr[..., 3] = a
    return Image.fromarray(arr, "RGBA"), grid


# ===========================================================================
# UV + MATERIAL ASSIGNMENT
# ===========================================================================
def texture_wood(mesh, albedo_img, normal_img, rng):
    v = mesh.vertices
    ang = np.arctan2(v[:, 2], v[:, 0])
    u = (ang / (2.0 * np.pi) + 0.5) * 3.0
    y = v[:, 1]
    vv = y / 3.0
    uv = np.column_stack([u, vv])

    ymax = float(y.max()) + 1e-6
    g = np.clip(0.70 + 0.30 * (y / ymax), 0.62, 1.02)      # brighter, mild AO
    g = np.clip(g + rng.normal(0.0, 0.03, size=g.shape), 0.58, 1.05)
    cols = np.stack([g * 1.0, g * 0.97, g * 0.9], axis=1)
    vcol = np.clip(cols * 255, 0, 255).astype(np.uint8)
    vcol = np.column_stack([vcol, np.full(len(vcol), 255, dtype=np.uint8)])

    mat = PBRMaterial(
        name="bark",
        baseColorTexture=albedo_img,
        normalTexture=normal_img,
        metallicFactor=0.0,
        roughnessFactor=0.9,
        doubleSided=False,
    )
    mesh.visual = TextureVisuals(uv=uv, material=mat, image=albedo_img)
    mesh.visual.vertex_attributes["color"] = vcol


def texture_canopy(mesh, atlas_img, grid, rng):
    v = mesh.vertices
    n_cards = len(v) // 4
    inv = 1.0 / grid
    pad = 0.02 * inv

    base_sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    uv = np.zeros((len(v), 2), dtype=np.float64)
    vcol = np.zeros((len(v), 4), dtype=np.uint8)

    centers = v.reshape(-1, 4, 3).mean(axis=1)
    ymin = float(v[:, 1].min())
    ymax = float(v[:, 1].max())
    yspan = (ymax - ymin) + 1e-6

    for i in range(n_cards):
        t = int(rng.integers(0, grid * grid))
        rot = int(rng.integers(0, 4))
        col = t % grid
        row = t // grid
        u0 = col * inv
        v0 = row * inv
        rotated = base_sq[rot:] + base_sq[:rot]
        for k in range(4):
            cu, cv = rotated[k]
            uu = u0 + pad + cu * (inv - 2 * pad)
            vvv = v0 + pad + cv * (inv - 2 * pad)
            uv[i * 4 + k] = (uu, vvv)

        c = centers[i]
        hy = (c[1] - ymin) / yspan
        outward = np.clip(np.sqrt(c[0] * c[0] + c[2] * c[2]) / CROWN_RX, 0.0, 1.0)
        f = 0.62 + 0.32 * hy + 0.20 * outward + rng.normal(0.0, 0.04)
        f = float(np.clip(f, 0.58, 1.15))                  # brighter floor
        rgb = np.clip(np.array([f * 1.0, f * 1.03, f * 0.82]) * 255, 0, 255).astype(np.uint8)
        for k in range(4):
            vcol[i * 4 + k] = (rgb[0], rgb[1], rgb[2], 255)

    mat = PBRMaterial(
        name="foliage",
        baseColorTexture=atlas_img,
        metallicFactor=0.0,
        roughnessFactor=0.8,
        alphaMode="MASK",
        alphaCutoff=0.4,
        doubleSided=True,
    )
    mesh.visual = TextureVisuals(uv=uv, material=mat, image=atlas_img)
    mesh.visual.vertex_attributes["color"] = vcol


# ===========================================================================
# TOP-LEVEL ASSEMBLY
# ===========================================================================
def build_textured_scene(image_path, seed, density):
    scene = build_mesh(seed, density)
    img = _load_image(image_path)
    rng = np.random.default_rng(seed + 977)

    bark_color = _sample_color(
        img, region=(0.46, 0.57, 0.66, 0.90), rng=rng,
        filt=_is_bark, fallback=(0.42, 0.37, 0.31))
    leaf_color = _sample_color(
        img, region=(0.22, 0.78, 0.12, 0.50), rng=rng,
        filt=_is_foliage, fallback=(0.30, 0.40, 0.21))

    # Lift dark samples to healthy mid-tones so nothing collapses to a silhouette.
    bark_color = _brighten_to(bark_color, target_lum=0.40, lo=0.85, hi=2.4)
    leaf_color = _brighten_to(leaf_color, target_lum=0.42, lo=0.90, hi=2.6)

    bark_albedo, bark_normal = make_bark_textures(bark_color, rng, res=512)
    atlas_img, grid = make_foliage_atlas(leaf_color, rng, tile=256, grid=4, ss=4)

    texture_wood(scene.geometry["trunk"], bark_albedo, bark_normal, rng)
    texture_canopy(scene.geometry["canopy"], atlas_img, grid, rng)
    return scene


def main():
    parser = argparse.ArgumentParser(description="Procedural textured oak -> GLB")
    parser.add_argument("--image", required=True, help="reference photo path")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--density", choices=["high", "med", "low"], default="high")
    parser.add_argument("--output", required=True, help="output .glb path")
    args = parser.parse_args()

    try:
        scene = build_textured_scene(args.image, args.seed, args.density)
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