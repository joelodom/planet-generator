#!/usr/bin/env python3
"""
Standalone procedural generator + texturer for a dense, dome-shaped
multi-stemmed ornamental shrub (archetype: leafy-deciduous-shrub).

Pipeline:
  1. build_mesh(seed, density)         -> geometry (two semantic surfaces)
  2. sample a palette FROM the photo   -> greens (sunlit/mid/shade) + bark browns
  3. synthesize tileable materials     -> leaf-cluster ATLAS (4x4) + bark albedo/normal
  4. attach UVs + PBR materials        -> cylindrical UVs on wood, atlas UVs on cards
  5. export a binary .glb

Only numpy + trimesh + PIL + stdlib. +Y up, base at y=0, meters.

CLI:
  python thisscript --image PATH --seed INT --density {high,med,low} --output OUT.glb
"""

import argparse
import sys
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial


# ===========================================================================
# GEOMETRY
# ===========================================================================

# ---------------------------------------------------------------------------
# MEASURED PROPORTIONS (read off the reference image, ~10% accuracy)
# ---------------------------------------------------------------------------
OVERALL_HEIGHT = 1.30            # meters, a plausible mature garden shrub
# Photo content aspect (width/height) ~0.96 -> slightly TALLER than wide once
# the leaf cards spill past the envelope, so the bare envelope is a touch
# narrower than tall.
HEIGHT_OVER_WIDTH = 1.08
CROWN_WIDTH = OVERALL_HEIGHT / HEIGHT_OVER_WIDTH   # ~1.20 m

# Foliage skirts down to ~10% of the height; only a small thicket of stems
# shows at the very bottom.
FOLIAGE_BOTTOM_FRAC = 0.10
# Widest point sits near the vertical middle of the canopy.

# Foliage envelope (lobed ellipsoid) ----------------------------------------
_FOL_BOTTOM_Y = FOLIAGE_BOTTOM_FRAC * OVERALL_HEIGHT      # ~0.13 m
ENV_AY = (OVERALL_HEIGHT - _FOL_BOTTOM_Y) * 0.5           # vertical half-extent
ENV_CENTER_Y = _FOL_BOTTOM_Y + ENV_AY                     # ~0.72 m
ENV_AX = CROWN_WIDTH * 0.5                                # horizontal half-extents
ENV_AZ = CROWN_WIDTH * 0.5
ENV_CENTER = np.array([0.0, ENV_CENTER_Y, 0.0])
ENV_AXES = np.array([ENV_AX, ENV_AY, ENV_AZ])
LOBE_AMP = 0.06                  # gentle radial bulges (keep dome rounded, no lobes)

# Wood ----------------------------------------------------------------------
STEM_BASE_RADIUS = 0.013         # m, slender stems
BASAL_FLARE = 1.45               # widen the very bottom of each stem
BASAL_FLARE_FRAC = 0.08          # over the bottom 8% of the stem

# Density presets: counts chosen BEFORE building (generate at target density).
_PRESETS = {
    "high": dict(cards=2500, clumps=34, stems=9, child_per_stem=3,
                 sides=10, path_pts=7),
    "med":  dict(cards=1000, clumps=22, stems=6, child_per_stem=2,
                 sides=8,  path_pts=6),
    "low":  dict(cards=320,  clumps=12, stems=4, child_per_stem=1,
                 sides=6,  path_pts=5),
}


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _rot(v, axis, ang):
    """Rodrigues rotation of vector v about unit axis by angle ang."""
    c, s = np.cos(ang), np.sin(ang)
    return (v * c
            + np.cross(axis, v) * s
            + axis * np.dot(axis, v) * (1.0 - c))


def _lobe_factor(direction, lobes):
    """Low-frequency radial bulge as a function of azimuth (gentle)."""
    az = np.arctan2(direction[2], direction[0])
    f = 1.0
    for freq, amp, phase in lobes:
        f += amp * np.cos(freq * az + phase)
    return f


def _env_point(direction, lobes, scale=1.0):
    """Absolute point on (or inside) the foliage envelope along a unit dir."""
    d = _normalize(direction)
    f = _lobe_factor(d, lobes)
    return ENV_CENTER + scale * f * (ENV_AXES * d)


def _env_outward(p):
    """Outward (ellipsoid-gradient) normal at a point near the envelope."""
    return _normalize((p - ENV_CENTER) / (ENV_AXES ** 2))


def _rand_dir(rng, y_lo=-0.15, y_hi=1.0):
    """Random unit direction biased to the upper dome (avoid the bare bottom)."""
    y = rng.uniform(y_lo, y_hi)
    r = np.sqrt(max(0.0, 1.0 - y * y))
    phi = rng.uniform(0.0, 2.0 * np.pi)
    return np.array([r * np.cos(phi), y, r * np.sin(phi)])


def _tube(points, radii, sides):
    points = np.asarray(points, dtype=float)
    radii = np.asarray(radii, dtype=float)
    n = len(points)

    tang = np.zeros((n, 3))
    tang[:-1] = points[1:] - points[:-1]
    tang[-1] = tang[-2]
    tlen = np.linalg.norm(tang, axis=1, keepdims=True)
    tlen[tlen < 1e-12] = 1.0
    tang = tang / tlen

    normals = np.zeros((n, 3))
    a = tang[0]
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    normals[0] = _normalize(np.cross(a, ref))
    for i in range(1, n):
        v0, v1 = tang[i - 1], tang[i]
        axis = np.cross(v0, v1)
        s = np.linalg.norm(axis)
        if s < 1e-9:
            normals[i] = normals[i - 1]
        else:
            ang = np.arctan2(s, np.dot(v0, v1))
            normals[i] = _rot(normals[i - 1], axis / s, ang)
        normals[i] = _normalize(normals[i] - np.dot(normals[i], tang[i]) * tang[i])
    binorm = np.cross(tang, normals)

    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    verts = np.empty((n * sides, 3))
    for i in range(n):
        ring = (points[i]
                + radii[i] * (np.outer(cos_a, normals[i]) + np.outer(sin_a, binorm[i])))
        verts[i * sides:(i + 1) * sides] = ring

    faces = []
    for i in range(n - 1):
        b0, b1 = i * sides, (i + 1) * sides
        for j in range(sides):
            k = (j + 1) % sides
            faces.append([b0 + j, b0 + k, b1 + k])
            faces.append([b0 + j, b1 + k, b1 + j])
    return verts, np.asarray(faces, dtype=np.int64)


def _bezier(p0, p1, p2, n):
    s = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * p1 + s ** 2 * p2


def _build_wood(rng, cfg, lobes):
    vlist, flist, voff = [], [], 0
    tips = []   # branch-tip anchors for foliage clumps

    n_stems = cfg["stems"]
    base_az = rng.uniform(0.0, 2.0 * np.pi)

    for si in range(n_stems):
        az = base_az + 2.0 * np.pi * si / n_stems + rng.uniform(-0.25, 0.25)
        base = np.array([0.03 * np.cos(az), 0.0, 0.03 * np.sin(az)]) \
            + rng.normal(scale=0.012, size=3) * np.array([1, 0, 1])
        base[1] = 0.0

        # tip lands just inside the upper envelope; keep stems mostly VERTICAL
        # (only a little outward) so they cluster tightly and stay hidden.
        tdir = _rand_dir(rng, y_lo=0.3, y_hi=1.0)
        tdir[0] += 0.45 * np.cos(az)
        tdir[2] += 0.45 * np.sin(az)
        tip = _env_point(tdir, lobes, scale=rng.uniform(0.80, 0.94))

        ctrl = 0.5 * (base + tip)
        ctrl[0] += 0.08 * CROWN_WIDTH * np.cos(az)     # gentle arch, little splay
        ctrl[2] += 0.08 * CROWN_WIDTH * np.sin(az)
        ctrl[1] += 0.14 * OVERALL_HEIGHT

        pts = _bezier(base, ctrl, tip, cfg["path_pts"])
        s = np.linspace(0.0, 1.0, cfg["path_pts"])
        radii = STEM_BASE_RADIUS * (1.0 - 0.82 * s) + 0.0022
        radii[s < BASAL_FLARE_FRAC] *= BASAL_FLARE

        v, f = _tube(pts, radii, cfg["sides"])
        vlist.append(v); flist.append(f + voff); voff += len(v)
        tips.append(tip)

        for _ in range(cfg["child_per_stem"]):
            t = rng.uniform(0.45, 0.85)
            i = int(t * (cfg["path_pts"] - 1))
            cbase = pts[i]
            r_parent = radii[i]
            cdir = _rand_dir(rng, y_lo=0.1, y_hi=1.0)
            ctip = _env_point(cdir, lobes, scale=rng.uniform(0.85, 0.97))
            cctrl = 0.5 * (cbase + ctip) + np.array([0, 0.06 * OVERALL_HEIGHT, 0])
            cpts = _bezier(cbase, cctrl, ctip, max(3, cfg["path_pts"] - 2))
            cs = np.linspace(0.0, 1.0, len(cpts))
            r0 = r_parent * 0.62
            cradii = r0 * (1.0 - 0.7 * cs) + 0.0016
            cv, cf = _tube(cpts, cradii, max(5, cfg["sides"] - 2))
            vlist.append(cv); flist.append(cf + voff); voff += len(cv)
            tips.append(ctip)

    verts = np.vstack(vlist)
    faces = np.vstack(flist)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.merge_vertices()
    mesh.fix_normals()
    return mesh, tips


def _build_canopy(rng, cfg, lobes, tips):
    n_clumps = cfg["clumps"]

    # EVEN golden-angle coverage from apex down to the skirt -> one continuous
    # rounded dome (no notch, filled top, foliage descending the sides).
    centers = []
    ga = np.pi * (3.0 - np.sqrt(5.0))
    for k in range(n_clumps):
        yk = 1.0 - (k + 0.5) / n_clumps * 1.7      # +1 apex -> ~-0.7 skirt
        yk = float(np.clip(yk, -0.72, 0.99))
        rk = np.sqrt(max(0.0, 1.0 - yk * yk))
        phi = k * ga + rng.uniform(-0.18, 0.18)
        d = np.array([rk * np.cos(phi), yk, rk * np.sin(phi)])
        centers.append(_env_point(d, lobes, scale=rng.uniform(0.84, 0.98)))
    # a few interior clumps for visual weight / opacity
    for _ in range(max(2, n_clumps // 6)):
        d = _rand_dir(rng, y_lo=-0.2, y_hi=0.9)
        centers.append(_env_point(d, lobes, scale=rng.uniform(0.45, 0.65)))
    centers = np.array(centers)

    clump_radius = 0.12 * CROWN_WIDTH
    half_base = 0.045 * CROWN_WIDTH
    per_clump = max(1, cfg["cards"] // len(centers))

    verts = np.empty((cfg["cards"] * 4, 3))
    faces = np.empty((cfg["cards"] * 2, 3), dtype=np.int64)
    nc = 0
    for c in centers:
        for _ in range(per_clump):
            if nc >= cfg["cards"]:
                break
            p = c + rng.normal(scale=clump_radius * 0.55, size=3)
            d = _normalize(p - ENV_CENTER)
            shell = _env_point(d, lobes, scale=1.0)
            if np.linalg.norm(p - ENV_CENTER) > np.linalg.norm(shell - ENV_CENTER):
                p = ENV_CENTER + 0.97 * (shell - ENV_CENTER)

            n = _env_outward(p)
            jt = _normalize(np.cross(n, _rand_dir(rng)))
            n = _normalize(n + 0.42 * (rng.uniform(-1, 1) * jt
                                       + rng.uniform(-1, 1) * np.cross(n, jt)))
            t1 = _normalize(np.cross(n, [0, 1, 0]) if abs(n[1]) < 0.95
                            else np.cross(n, [1, 0, 0]))
            t2 = np.cross(n, t1)

            a = rng.uniform(0, np.pi)
            u = np.cos(a) * t1 + np.sin(a) * t2
            w = -np.sin(a) * t1 + np.cos(a) * t2
            hs = half_base * float(np.exp(rng.normal(0.0, 0.30)))
            hw, hh = hs, hs * 1.2

            b = nc * 4
            verts[b + 0] = p - hw * u - hh * w
            verts[b + 1] = p + hw * u - hh * w
            verts[b + 2] = p + hw * u + hh * w
            verts[b + 3] = p - hw * u + hh * w
            faces[nc * 2 + 0] = [b + 0, b + 1, b + 2]
            faces[nc * 2 + 1] = [b + 0, b + 2, b + 3]
            nc += 1

    verts = verts[:nc * 4]
    faces = faces[:nc * 2]
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return mesh


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    if density not in _PRESETS:
        density = "high"
    cfg = _PRESETS[density]
    rng = np.random.default_rng(seed)

    n_lobes = int(rng.integers(3, 6))
    lobes = [(int(rng.integers(2, 5)),
              LOBE_AMP * rng.uniform(0.5, 1.0),
              rng.uniform(0.0, 2.0 * np.pi)) for _ in range(n_lobes)]

    wood, tips = _build_wood(rng, cfg, lobes)
    canopy = _build_canopy(rng, cfg, lobes, tips)

    min_y = min(wood.bounds[0, 1], canopy.bounds[0, 1])
    if abs(min_y) > 1e-9:
        wood.apply_translation([0, -min_y, 0])
        canopy.apply_translation([0, -min_y, 0])

    scene = trimesh.Scene()
    scene.add_geometry(wood, geom_name="branches")
    scene.add_geometry(canopy, geom_name="canopy")
    return scene


# ===========================================================================
# PALETTE  (sample colors FROM the photo, inside the silhouette)
# ===========================================================================

def _load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32)


def _sample_region(arr, x0, x1, y0, y1, grid, patch, keep):
    """Median color over a grid of small interior patches, keeping only
    patches whose color passes `keep` (discards background hits)."""
    H, W, _ = arr.shape
    half = patch // 2
    xs = np.linspace(x0, x1, grid)
    ys = np.linspace(y0, y1, grid)
    cols = []
    for fy in ys:
        for fx in xs:
            cx = int(np.clip(fx * W, half, W - half - 1))
            cy = int(np.clip(fy * H, half, H - half - 1))
            patch_px = arr[cy - half:cy + half + 1, cx - half:cx + half + 1]
            c = np.median(patch_px.reshape(-1, 3), axis=0)
            if keep(c):
                cols.append(c)
    if not cols:
        return None
    return np.median(np.array(cols), axis=0)


def _is_green(c):
    r, g, b = c
    return (g >= r - 4) and (g >= b) and (g - min(r, b) > 8) and (g > 30)


def _is_bark(c):
    r, g, b = c
    return (r >= g - 6) and (g >= b - 4) and (r - b > 6) and (40 < r < 210) \
        and (max(c) - min(c) > 6)


def _lum(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def sample_palette(arr, seed):
    """Return greens (light/mid/dark) + bark (mid/light/dark), all from photo."""
    mid = _sample_region(arr, 0.30, 0.70, 0.30, 0.68, 6, 9, _is_green)
    light = _sample_region(arr, 0.30, 0.70, 0.12, 0.32, 6, 9, _is_green)
    dark = _sample_region(arr, 0.32, 0.68, 0.55, 0.78, 6, 9, _is_green)

    if mid is None:
        mid = np.array([78.0, 120.0, 56.0])     # safety fallback (fresh mid-green)
    if light is None or _lum(light) <= _lum(mid):
        light = np.clip(mid * 1.32 + np.array([10, 12, 4]), 0, 255)
    if dark is None or _lum(dark) >= _lum(mid):
        dark = np.clip(mid * 0.70, 0, 255)
    dark = np.maximum(dark, mid * 0.62)         # don't let interior collapse to black

    bark = _sample_region(arr, 0.40, 0.60, 0.80, 0.97, 5, 7, _is_bark)
    if bark is None:
        bark = _sample_region(arr, 0.35, 0.65, 0.72, 0.97, 7, 7, _is_bark)
    if bark is None:
        bark = np.array([122.0, 92.0, 64.0])     # muted earth-tone fallback
    bark_light = np.clip(bark * 1.45 + np.array([14, 10, 6]), 0, 255)
    bark_dark = np.clip(bark * 0.55, 0, 255)

    return dict(
        leaf_light=np.asarray(light, np.float32),
        leaf_mid=np.asarray(mid, np.float32),
        leaf_dark=np.asarray(dark, np.float32),
        bark=np.asarray(bark, np.float32),
        bark_light=np.asarray(bark_light, np.float32),
        bark_dark=np.asarray(bark_dark, np.float32),
    )


# ===========================================================================
# MATERIALS
# ===========================================================================

# --- leaf-cluster atlas ----------------------------------------------------

def _leaf_polygon(cx, cy, length, width, angle, rng):
    """An ovate, lightly-serrated leaf silhouette as a list of (x,y) points."""
    n = 11
    ts = np.linspace(0.04, 1.0, n)
    left, right = [], []
    serr = rng.uniform(7, 11)
    for t in ts:
        hw = width * 0.5 * (np.sin(np.pi * t) ** 0.7) * (1.0 + 0.08 * np.sin(serr * np.pi * t))
        pos = (t - 0.5) * length
        left.append((-hw, pos))
        right.append((hw, pos))
    pts = left + right[::-1]
    ca, sa = np.cos(angle), np.sin(angle)
    out = []
    for x, y in pts:
        out.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    return out


def _pick_leaf_color(pal, bias, warm, rng):
    """Blend the photo greens toward light/dark per `bias`, plus warm/cool shift."""
    if bias > 0:
        base = pal["leaf_mid"] * (1 - bias) + pal["leaf_light"] * bias
    else:
        base = pal["leaf_mid"] * (1 + bias) + pal["leaf_dark"] * (-bias)
    base = base * (1.0 + rng.uniform(-0.10, 0.12, 3))
    base = base + np.array([warm * 14.0, warm * 8.0, -warm * 8.0])
    return tuple(int(v) for v in np.clip(base, 0, 255))


def build_leaf_atlas(pal, seed, grid=4, tile=256, ss=4):
    """4x4 atlas of distinct leaf-cluster tiles; binary alpha, AA edges only.
    Tiles are densely packed so the canopy reads opaque (little interior shows)."""
    rng = np.random.default_rng(seed ^ 0xA71A5)
    atlas = Image.new("RGBA", (grid * tile, grid * tile), (0, 0, 0, 0))
    big = tile * ss
    margin = int(0.07 * big)

    for grow in range(grid):
        for gcol in range(grid):
            shade = grow / (grid - 1)                # rows: sunlit -> shaded
            bright = 1.30 - 0.30 * shade
            warm = 0.7 - 1.2 * shade
            tcol = Image.new("RGBA", (big, big), (0, 0, 0, 0))
            d = ImageDraw.Draw(tcol)

            n_leaves = int(rng.integers(36, 56))     # dense -> opaque cluster
            for _ in range(n_leaves):
                cx = rng.uniform(margin, big - margin)
                cy = rng.uniform(margin, big - margin)
                length = rng.uniform(0.38, 0.56) * big
                width = length * rng.uniform(0.60, 0.82)
                ang = rng.uniform(0, 2 * np.pi)
                bias = rng.uniform(-0.45, 1.0)       # lean fresher/brighter
                lcol = _pick_leaf_color(pal, bias, warm, rng)
                lcol = tuple(int(np.clip(c * bright, 0, 255)) for c in lcol)
                poly = _leaf_polygon(cx, cy, length, width, ang, rng)
                d.polygon(poly, fill=lcol + (255,))
                # midrib vein, slightly darker
                vein = tuple(int(c * 0.72) for c in lcol)
                d.line([(cx + (length * 0.45) * np.sin(ang),
                         cy - (length * 0.45) * np.cos(ang)),
                        (cx - (length * 0.45) * np.sin(ang),
                         cy + (length * 0.45) * np.cos(ang))],
                       fill=vein + (255,), width=max(2, ss))

            tcol = tcol.resize((tile, tile), Image.LANCZOS)
            atlas.paste(tcol, (gcol * tile, grow * tile), tcol)

    return atlas


# --- bark albedo + normal (tileable, sums-of-sines so no seam) --------------

def build_bark(pal, seed, size=512):
    rng = np.random.default_rng(seed ^ 0xBA7C)
    x = np.linspace(0.0, 1.0, size, endpoint=False)
    y = np.linspace(0.0, 1.0, size, endpoint=False)
    X, Y = np.meshgrid(x, y)

    grain = np.zeros((size, size), np.float32)
    for _ in range(7):
        f = int(rng.integers(8, 28))
        ph = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.05, 0.16)
        grain += amp * np.sin(2 * np.pi * f * X + ph + 0.6 * np.sin(2 * np.pi * 2 * Y))
    for _ in range(3):
        f = int(rng.integers(2, 6))
        ph = rng.uniform(0, 2 * np.pi)
        grain += rng.uniform(0.04, 0.09) * np.sin(2 * np.pi * f * Y + ph)
    grain += 0.05 * rng.standard_normal((size, size))

    grain = grain / (np.max(np.abs(grain)) + 1e-6)        # -> [-1, 1]
    value = 1.0 + 0.30 * grain                            # lightest ~2x darkest

    t = np.clip((value - 0.7) / 0.6, 0.0, 1.0)[..., None]
    base = pal["bark_dark"][None, None, :] * (1 - t) + pal["bark_light"][None, None, :] * t
    warm = 1.0 + 0.06 * np.sin(2 * np.pi * 3 * Y)[..., None]
    base = base * warm
    albedo = np.clip(base, 0, 255).astype(np.uint8)
    bark_img = Image.fromarray(albedo, "RGB")

    lum = (0.2126 * albedo[..., 0] + 0.7152 * albedo[..., 1]
           + 0.0722 * albedo[..., 2]) / 255.0
    height = 1.0 - lum
    gx = (np.roll(height, -1, 1) - np.roll(height, 1, 1))
    gy = (np.roll(height, -1, 0) - np.roll(height, 1, 0))
    strength = 2.5
    nx, ny, nz = -gx * strength, -gy * strength, np.ones_like(height)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    normal = np.stack([nx / ln, ny / ln, nz / ln], -1)
    normal_img = Image.fromarray(((normal * 0.5 + 0.5) * 255).astype(np.uint8), "RGB")

    return bark_img, normal_img


# ===========================================================================
# UVs + material attachment
# ===========================================================================

def texture_branches(mesh, bark_img, normal_img):
    """Cylindrical UVs around the +Y axis; bark albedo + derived normal."""
    v = mesh.vertices
    ang = np.arctan2(v[:, 2], v[:, 0])
    u = (ang / (2.0 * np.pi)) % 1.0 * 3.0          # ~3 wraps of bark around
    vv = v[:, 1] / 0.22                            # repeat ~every 0.22 m
    uv = np.column_stack([u, vv]).astype(np.float32)

    mat = PBRMaterial(
        name="bark",
        baseColorTexture=bark_img,
        normalTexture=normal_img,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.9,
    )
    mat.doubleSided = False

    ao = np.clip(0.55 + 0.45 * (v[:, 1] / 0.40), 0.55, 1.0)
    tint = np.empty((len(v), 4), np.uint8)
    tint[:, 0] = np.clip(236 * ao, 0, 255)
    tint[:, 1] = np.clip(230 * ao, 0, 255)
    tint[:, 2] = np.clip(220 * ao, 0, 255)
    tint[:, 3] = 255

    vis = TextureVisuals(uv=uv, material=mat)
    vis.vertex_attributes["color"] = tint
    mesh.visual = vis
    return mesh


def texture_canopy(mesh, atlas_img, seed, grid=4):
    """Map each leaf card (quad of 4 verts) onto a random atlas tile, with
    a random 0/90/180/270 rotation; bright fresh-green sun/shade vertex tint."""
    v = mesh.vertices
    n_cards = len(v) // 4
    rng = np.random.default_rng(seed ^ 0x0CA0)

    uv = np.zeros((len(v), 2), np.float32)
    tint = np.empty((len(v), 4), np.uint8)
    inset = 0.004

    span = OVERALL_HEIGHT - _FOL_BOTTOM_Y
    top_col = np.array([250.0, 250.0, 224.0])      # sunlit (bright, warm)
    bot_col = np.array([150.0, 176.0, 128.0])      # shaded/inner (still fresh green)

    for i in range(n_cards):
        b = i * 4
        tile = int(rng.integers(0, grid * grid))
        gcol, grow = tile % grid, tile // grid
        rot = int(rng.integers(0, 4))

        u0, u1 = gcol / grid + inset, (gcol + 1) / grid - inset
        v0, v1 = grow / grid + inset, (grow + 1) / grid - inset
        corners = [(u0, v1), (u1, v1), (u1, v0), (u0, v0)]   # card order BL,BR,TR,TL
        corners = corners[rot:] + corners[:rot]
        for k in range(4):
            uv[b + k] = corners[k]

        cy = float(np.mean(v[b:b + 4, 1]))
        h = np.clip((cy - _FOL_BOTTOM_Y) / span, 0.0, 1.0)
        jit = 1.0 + rng.uniform(-0.06, 0.06)
        col_rgb = np.clip((bot_col + (top_col - bot_col) * h) * jit, 0, 255)
        tint[b:b + 4, 0] = col_rgb[0]
        tint[b:b + 4, 1] = col_rgb[1]
        tint[b:b + 4, 2] = col_rgb[2]
        tint[b:b + 4, 3] = 255

    mat = PBRMaterial(
        name="foliage",
        baseColorTexture=atlas_img,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.8,
    )
    mat.alphaMode = "MASK"
    mat.alphaCutoff = 0.45
    mat.doubleSided = True

    vis = TextureVisuals(uv=uv, material=mat)
    vis.vertex_attributes["color"] = tint
    mesh.visual = vis
    return mesh


# ===========================================================================
# MAIN
# ===========================================================================

def build_textured_scene(image_path, seed, density):
    arr = _load_image(image_path)
    pal = sample_palette(arr, seed)
    atlas = build_leaf_atlas(pal, seed)
    bark_img, normal_img = build_bark(pal, seed)

    scene = build_mesh(seed, density)
    texture_branches(scene.geometry["branches"], bark_img, normal_img)
    texture_canopy(scene.geometry["canopy"], atlas, seed)
    return scene


def main():
    ap = argparse.ArgumentParser(description="Procedural textured shrub -> GLB")
    ap.add_argument("--image", required=True, help="source reference image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        scene = build_textured_scene(args.image, args.seed, args.density)
        data = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(data)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("ERROR: {}\n".format(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())