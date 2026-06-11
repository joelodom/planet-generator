#!/usr/bin/env python3
"""
Procedural umbrella-thorn acacia: geometry + photo-derived tileable materials + GLB export.

Pipeline:
  1. build_mesh(seed, density) -> trimesh.Scene  (umbrella acacia, +Y up, base at y=0, meters)
  2. Sample bark + foliage palettes FROM the reference photo (small patches, well inside the
     silhouette; reject background grey).
  3. Synthesize tileable bark albedo + tangent-space normal from those colors (vertical fissures).
  4. Build a 4x4 foliage atlas of distinct leaf-cluster tiles (sunlit warm / shaded cool) drawn
     with PIL leaflet polygons, supersampled then LANCZOS-downscaled, binary alpha.
  5. UV the wood cylindrically and the leaf cards per-atlas-tile; attach PBR materials; carry
     per-vertex COLOR_0 sun/shade tints; export a binary GLB.

CLI:
    python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb
"""

import argparse
import math
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw


# =========================================================================== #
#                               GEOMETRY MODULE                               #
# =========================================================================== #
TREE_HEIGHT = 12.0          # overall height in meters (plausible mature acacia)

FORK_FRAC          = 0.56   # trunk clear bole height / total height
CROWN_CENTER_Y_FRAC = 0.80  # vertical center of the foliage plate / total height
CROWN_RX_FRAC      = 0.50   # crown half-width (X) / total height  -> width ~= 1.0*H
CROWN_RZ_FRAC      = 0.48   # crown half-depth (Z) / total height
CROWN_RY_FRAC      = 0.17   # crown half-thickness (Y) / total height -> flat plate

# Slender, tapered trunk with a moderate basal flare and a gentle lean.
R_TRUNK_BASE = 0.33         # trunk radius at ground (before flare multiply)
R_FORK       = 0.23         # trunk radius where it forks
FLARE_MULT   = 1.45         # basal flare multiplier (x1.45 over bottom ~7%)
FLARE_FRAC   = 0.07         # fraction of bole height affected by the flare
LEAN_X       = 0.50         # horizontal lean of the trunk top (meters)
LEAN_Z       = -0.30

# Branch tips stop SHORT of the crown shell so leaf clumps (placed AT the shell)
# fully bury them -- no bare twigs poking past the foliage.
TIP_REACH_FRAC = 0.80

# Leaf-card sizing as fractions of crown width (= 2*CROWN_RX).
CLUMP_R_FRAC   = 0.11       # clump radius (~11% of crown width) -> full coverage
CARD_HALF_FRAC = 0.050      # card half-size (~5% of crown width), log-normal jitter
CARD_NORMAL_JITTER_DEG = 25.0


_DENSITY = {
    "high": dict(clumps_outer=26, clumps_inner=16, cards_per_clump=(90, 140),
                 n_main=5, sides=10, trunk_rings=10, lobes=5),
    "med":  dict(clumps_outer=16, clumps_inner=8,  cards_per_clump=(42, 66),
                 n_main=4, sides=7,  trunk_rings=7,  lobes=4),
    "low":  dict(clumps_outer=9,  clumps_inner=3,  cards_per_clump=(20, 32),
                 n_main=3, sides=5,  trunk_rings=5,  lobes=3),
}


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([0.0, 1.0, 0.0])


def _rot(v, k, ang):
    """Rodrigues rotation of vector v about unit axis k by angle `ang`."""
    c, s = np.cos(ang), np.sin(ang)
    return v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1.0 - c)


def _merge(parts):
    """Merge a list of (verts, faces) into one (verts, faces) pair."""
    V, F, off = [], [], 0
    for v, f in parts:
        V.append(v)
        F.append(f + off)
        off += len(v)
    return np.vstack(V), np.vstack(F)


def _build_tube(points, radii, sides):
    points = np.asarray(points, dtype=float)
    radii = np.asarray(radii, dtype=float)
    n = len(points)

    tang = np.zeros((n, 3))
    for i in range(n):
        if i == 0:
            d = points[1] - points[0]
        elif i == n - 1:
            d = points[-1] - points[-2]
        else:
            d = points[i + 1] - points[i - 1]
        tang[i] = _norm(d)

    t0 = tang[0]
    ref = np.array([0.0, 1.0, 0.0]) if abs(t0[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = _norm(ref - np.dot(ref, t0) * t0)
    frames_u = [u]
    for i in range(1, n):
        tp, tc = tang[i - 1], tang[i]
        ax = np.cross(tp, tc)
        s = np.linalg.norm(ax)
        up = frames_u[-1]
        if s > 1e-8:
            ax = ax / s
            ang = np.arctan2(s, np.dot(tp, tc))
            up = _rot(up, ax, ang)
        up = _norm(up - np.dot(up, tc) * tc)
        frames_u.append(up)

    ang = 2.0 * np.pi * np.arange(sides) / sides
    cosA, sinA = np.cos(ang), np.sin(ang)

    rings = []
    for i in range(n):
        uu = frames_u[i]
        vv = np.cross(tang[i], uu)
        ring = points[i] + radii[i] * (np.outer(cosA, uu) + np.outer(sinA, vv))
        rings.append(ring)
    verts = np.vstack(rings)

    faces = []
    for i in range(n - 1):
        a, b = i * sides, (i + 1) * sides
        for s in range(sides):
            s1 = (s + 1) % sides
            faces.append([a + s, a + s1, b + s1])
            faces.append([a + s, b + s1, b + s])

    cb = len(verts)
    verts = np.vstack([verts, points[0]])
    for s in range(sides):
        s1 = (s + 1) % sides
        faces.append([cb, s1, s])
    ctop = len(verts)
    verts = np.vstack([verts, points[-1]])
    base = (n - 1) * sides
    for s in range(sides):
        s1 = (s + 1) % sides
        faces.append([ctop, base + s, base + s1])

    return verts, np.asarray(faces, dtype=np.int64)


def _lobe_factor(az, lobe_terms):
    f = 1.0
    for amp, freq, ph in lobe_terms:
        f += amp * np.cos(freq * az + ph)
    return f


def _ellip_q(rel, semi):
    return np.sqrt(np.sum((rel / semi) ** 2))


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    if density not in _DENSITY:
        density = "high"
    cfg = _DENSITY[density]
    rng = np.random.default_rng(seed)

    H = TREE_HEIGHT
    RX = CROWN_RX_FRAC * H
    RZ = CROWN_RZ_FRAC * H
    RY = CROWN_RY_FRAC * H
    semi = np.array([RX, RY, RZ])
    crown_width = 2.0 * RX

    fork_y = FORK_FRAC * H
    fork = np.array([LEAN_X, fork_y, LEAN_Z])
    canopy_center = np.array([LEAN_X * 1.3, CROWN_CENTER_Y_FRAC * H, LEAN_Z * 1.3])

    # Gentle low-frequency bulges (kept modest -> rounder, cleaner silhouette).
    lobe_terms = []
    freq_pool = [2, 3, 4, 5, 6]
    for _ in range(cfg["lobes"]):
        amp = rng.uniform(0.04, 0.09)
        freq = int(rng.choice(freq_pool))
        ph = rng.uniform(0.0, 2.0 * np.pi)
        lobe_terms.append((amp, freq, ph))

    # 1) Shell directions -> branch tips (short of shell) AND outer clump centers (on shell).
    n_tips = cfg["clumps_outer"]
    az0 = rng.uniform(0.0, 2.0 * np.pi)
    tips = []            # branch endpoints (inside the canopy)
    tip_az = []
    outer_clumps = []    # (center on shell, outward normal) -> bury the tips
    for i in range(n_tips):
        az = az0 + 2.0 * np.pi * i / n_tips + rng.uniform(-0.4, 0.4) * (2.0 * np.pi / n_tips)
        # Elevation floored ABOVE horizontal so no sideways/downward bare spears.
        el = rng.uniform(0.05, 0.55) * (np.pi / 2.0)
        d = np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])
        lobe = _lobe_factor(az, lobe_terms)
        scale = rng.uniform(0.90, 1.0)
        shell_off = scale * lobe * (semi * d)
        tip_pt = canopy_center + TIP_REACH_FRAC * shell_off
        clump_pt = canopy_center + shell_off
        tips.append(tip_pt)
        tip_az.append(az % (2.0 * np.pi))
        outer_clumps.append((clump_pt, _norm(shell_off)))
    tips = np.array(tips)
    tip_az = np.array(tip_az)

    # 2) Candelabra skeleton: slim trunk -> n_main main branches -> sub-branch per tip.
    wood_parts_trunk = []
    wood_parts_branch = []
    sides = cfg["sides"]

    tr_rings = cfg["trunk_rings"]
    tpts, trad = [], []
    for k in range(tr_rings):
        t = k / (tr_rings - 1)
        r = R_FORK + (R_TRUNK_BASE - R_FORK) * (1.0 - t)
        if t < FLARE_FRAC:
            r *= 1.0 + (FLARE_MULT - 1.0) * (1.0 - t / FLARE_FRAC)
        x = LEAN_X * (t ** 1.3)
        z = LEAN_Z * (t ** 1.3)
        tpts.append([x, t * fork_y, z])
        trad.append(r)
    wood_parts_trunk.append(_build_tube(tpts, trad, sides))

    n_main = cfg["n_main"]
    order = np.argsort(tip_az)
    groups = []
    chunk = int(np.ceil(n_tips / n_main))
    for g in range(n_main):
        idx = order[g * chunk:(g + 1) * chunk]
        if len(idx):
            groups.append(idx)

    r_main = R_FORK / np.sqrt(max(len(groups), 1))
    for grp in groups:
        gtips = tips[grp]
        gmean = gtips.mean(axis=0)
        jct = canopy_center + 0.5 * (gmean - canopy_center)
        jct[1] -= 0.55 * RY
        mid = fork + 0.5 * (jct - fork)
        mid[1] += 0.8
        main_pts = [fork, mid, jct]
        main_rad = [r_main, r_main * 0.70, r_main * 0.46]
        wood_parts_branch.append(_build_tube(main_pts, main_rad, sides))

        r_end_main = r_main * 0.46
        gsz = len(grp)
        r_sub = r_end_main / np.sqrt(gsz)
        for ti in grp:
            tip = tips[ti]
            sub_pts = [jct, tip]
            sub_rad = [r_sub, 0.018]
            wood_parts_branch.append(_build_tube(sub_pts, sub_rad, sides))

    trunk_V, trunk_F = _merge(wood_parts_trunk)
    branch_V, branch_F = _merge(wood_parts_branch)

    # 3) Clump centers = outer (on shell, burying tips) + interior fill across the plate.
    clumps = list(outer_clumps)
    for _ in range(cfg["clumps_inner"]):
        az = rng.uniform(0.0, 2.0 * np.pi)
        rr = np.sqrt(rng.uniform(0.0, 1.0)) * 0.95   # uniform disk fill
        c = canopy_center + np.array([RX * rr * np.cos(az),
                                      rng.uniform(-0.7, 0.5) * RY,
                                      RZ * rr * np.sin(az)])
        nrm = _norm(np.array([0.35 * np.cos(az), 1.0, 0.35 * np.sin(az)]))
        clumps.append((c, nrm))

    # 4) Leaf cards: clumped flat quads, tangent to the shell, normals outward.
    clump_r = CLUMP_R_FRAC * crown_width
    card_half = CARD_HALF_FRAC * crown_width
    jit = np.deg2rad(CARD_NORMAL_JITTER_DEG)
    up = np.array([0.0, 1.0, 0.0])

    cV, cF = [], []
    vbase = 0
    for center, nrm in clumps:
        n_cards = int(rng.integers(cfg["cards_per_clump"][0],
                                   cfg["cards_per_clump"][1] + 1))
        for _ in range(n_cards):
            off = rng.normal(size=3)
            off = _norm(off) * clump_r * (rng.uniform(0.0, 1.0) ** (1.0 / 3.0))
            off[1] *= 0.6
            cc = center + off
            # Keep cards on/inside the envelope shell -> clean rounded plate edge.
            rel = cc - canopy_center
            q = _ellip_q(rel, semi)
            if q > 1.0:
                cc = canopy_center + rel / q

            rv = rng.normal(size=3)
            rv = rv - np.dot(rv, nrm) * nrm
            rv = _norm(rv)
            theta = rng.uniform(0.0, jit)
            cn = _norm(nrm + np.tan(theta) * rv)

            a = np.cross(cn, up)
            if np.linalg.norm(a) < 1e-6:
                a = np.cross(cn, np.array([1.0, 0.0, 0.0]))
            a = _norm(a)
            b = _norm(np.cross(cn, a))

            hu = card_half * float(rng.lognormal(0.0, 0.3))
            hv = hu * rng.uniform(0.7, 1.0)

            p0 = cc + hu * a + hv * b
            p1 = cc - hu * a + hv * b
            p2 = cc - hu * a - hv * b
            p3 = cc + hu * a - hv * b
            cV.extend([p0, p1, p2, p3])
            cF.append([vbase, vbase + 1, vbase + 2])
            cF.append([vbase, vbase + 2, vbase + 3])
            vbase += 4

    canopy_V = np.asarray(cV, dtype=float)
    canopy_F = np.asarray(cF, dtype=np.int64)

    # 5) Build meshes + per-vertex sun/shade colors (become COLOR_0 tints later).
    def _wood_colors(V):
        t = np.clip(V[:, 1] / H, 0.0, 1.0)
        low = np.array([74.0, 60.0, 50.0])
        high = np.array([122.0, 106.0, 88.0])
        col = low[None, :] + (high - low)[None, :] * t[:, None]
        col += rng.uniform(-6.0, 6.0, size=col.shape)
        rgba = np.empty((len(V), 4), dtype=np.uint8)
        rgba[:, :3] = np.clip(col, 0, 255).astype(np.uint8)
        rgba[:, 3] = 255
        return rgba

    def _canopy_colors(V):
        y = V[:, 1]
        t = np.clip((y - (canopy_center[1] - RY)) / (2.0 * RY), 0.0, 1.0)
        dark = np.array([58.0, 82.0, 40.0])
        light = np.array([150.0, 182.0, 92.0])
        col = dark[None, :] + (light - dark)[None, :] * t[:, None]
        col += rng.uniform(-10.0, 10.0, size=col.shape)
        rgba = np.empty((len(V), 4), dtype=np.uint8)
        rgba[:, :3] = np.clip(col, 0, 255).astype(np.uint8)
        rgba[:, 3] = 255
        return rgba

    trunk_mesh = trimesh.Trimesh(vertices=trunk_V, faces=trunk_F, process=True)
    trunk_mesh.visual.vertex_colors = _wood_colors(trunk_mesh.vertices)

    branch_mesh = trimesh.Trimesh(vertices=branch_V, faces=branch_F, process=True)
    branch_mesh.visual.vertex_colors = _wood_colors(branch_mesh.vertices)

    canopy_mesh = trimesh.Trimesh(vertices=canopy_V, faces=canopy_F, process=False)
    canopy_mesh.visual.vertex_colors = _canopy_colors(canopy_mesh.vertices)

    scene = trimesh.Scene()
    scene.add_geometry(trunk_mesh, geom_name="trunk")
    scene.add_geometry(branch_mesh, geom_name="branches")
    scene.add_geometry(canopy_mesh, geom_name="canopy")

    min_y = scene.bounds[0][1]
    if np.isfinite(min_y) and abs(min_y) > 1e-9:
        scene.apply_translation([0.0, -min_y, 0.0])

    return scene


# =========================================================================== #
#                          PHOTO PALETTE SAMPLING                             #
# =========================================================================== #
_BARK_DARK_FB = np.array([58.0, 47.0, 39.0])
_BARK_LIGHT_FB = np.array([138.0, 120.0, 99.0])
_FOL_MID_FB = np.array([96.0, 128.0, 66.0])
_FOL_LIGHT_FB = np.array([156.0, 184.0, 96.0])
_FOL_DARK_FB = np.array([56.0, 82.0, 42.0])


def _load_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64)


def _bg_color(img):
    """Median of the four image corners -> the background to reject."""
    h, w, _ = img.shape
    s = max(4, w // 25)
    corners = [img[:s, :s], img[:s, -s:], img[-s:, :s], img[-s:, -s:]]
    c = np.concatenate([p.reshape(-1, 3) for p in corners], axis=0)
    return np.median(c, axis=0)


def _sample_patches(img, box, n, psize, rng):
    """Mean color of n small patches placed inside a fractional box (x0,x1,y0,y1)."""
    h, w, _ = img.shape
    x0, x1, y0, y1 = box
    xa, xb = int(x0 * w), max(int(x1 * w) - psize, int(x0 * w) + 1)
    ya, yb = int(y0 * h), max(int(y1 * h) - psize, int(y0 * h) + 1)
    out = []
    for _ in range(n):
        px = int(rng.integers(xa, max(xa + 1, xb)))
        py = int(rng.integers(ya, max(ya + 1, yb)))
        patch = img[py:py + psize, px:px + psize].reshape(-1, 3)
        out.append(patch.mean(axis=0))
    return np.asarray(out)


def _dark_light(samples, dark_q=0.20, light_q=0.85, min_ratio=2.0):
    """Pick a dark and a light representative; keep a visible value range."""
    lum = samples @ np.array([0.299, 0.587, 0.114])
    order = np.argsort(lum)
    dark = samples[order[int(dark_q * (len(order) - 1))]]
    light = samples[order[int(light_q * (len(order) - 1))]]
    ld = float(dark @ np.array([0.299, 0.587, 0.114])) + 1e-3
    ll = float(light @ np.array([0.299, 0.587, 0.114]))
    if ll < ld * min_ratio:
        light = np.clip(dark * min_ratio + 18.0, 0, 255)
    return dark, light


def sample_palettes(img, rng):
    """Sample bark + foliage colors from WELL INSIDE the silhouette, rejecting background."""
    bg = _bg_color(img)
    h, w, _ = img.shape
    psize = max(4, w // 130)

    def dist_bg(c):
        return np.linalg.norm(c - bg)

    fol_box = (0.16, 0.84, 0.04, 0.50)
    fol_raw = _sample_patches(img, fol_box, 90, psize, rng)
    fol_keep = []
    for c in fol_raw:
        r, g, b = c
        if g > r + 3 and g > b + 3 and g > 55 and dist_bg(c) > 22:
            fol_keep.append(c)
    fol_keep = np.asarray(fol_keep) if fol_keep else None

    if fol_keep is not None and len(fol_keep) >= 6:
        fol_mid = np.median(fol_keep, axis=0)
        fol_dark, fol_light = _dark_light(fol_keep, 0.15, 0.85, min_ratio=1.6)
    else:
        fol_mid, fol_light, fol_dark = _FOL_MID_FB, _FOL_LIGHT_FB, _FOL_DARK_FB

    bark_box = (0.40, 0.60, 0.55, 0.96)
    bark_raw = _sample_patches(img, bark_box, 90, psize, rng)
    bark_keep = []
    for c in bark_raw:
        r, g, b = c
        mx, mn = float(max(r, g, b)), float(min(r, g, b))
        sat = (mx - mn) / (mx + 1e-6)
        is_green = g > r + 8
        if (r >= g - 6 and g >= b - 8 and r > 55 and not is_green
                and dist_bg(c) > 26 and sat > 0.05):
            bark_keep.append(c)
    bark_keep = np.asarray(bark_keep) if bark_keep else None

    if bark_keep is not None and len(bark_keep) >= 6:
        bark_dark, bark_light = _dark_light(bark_keep, 0.20, 0.88, min_ratio=2.0)
    else:
        bark_dark, bark_light = _BARK_DARK_FB, _BARK_LIGHT_FB

    return dict(
        bark_dark=np.asarray(bark_dark, float),
        bark_light=np.asarray(bark_light, float),
        fol_mid=np.asarray(fol_mid, float),
        fol_light=np.asarray(fol_light, float),
        fol_dark=np.asarray(fol_dark, float),
    )


# =========================================================================== #
#                       TILEABLE NOISE / TEXTURE UTILS                        #
# =========================================================================== #
def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def _tileable_value_noise(h, w, gx, gy, rng):
    """Wrap-tileable bilinear value noise in [0,1]."""
    g = rng.random((gy, gx))
    ys = np.linspace(0.0, gy, h, endpoint=False)
    xs = np.linspace(0.0, gx, w, endpoint=False)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    fy = _smoothstep(ys - y0)
    fx = _smoothstep(xs - x0)
    y1 = (y0 + 1) % gy
    x1 = (x0 + 1) % gx
    y0 %= gy
    x0 %= gx
    g00 = g[np.ix_(y0, x0)]
    g01 = g[np.ix_(y0, x1)]
    g10 = g[np.ix_(y1, x0)]
    g11 = g[np.ix_(y1, x1)]
    fx2 = fx[None, :]
    fy2 = fy[:, None]
    top = g00 * (1 - fx2) + g01 * fx2
    bot = g10 * (1 - fx2) + g11 * fx2
    return top * (1 - fy2) + bot * fy2


def _delight(rgb):
    """Divide out a heavily-blurred luminance, gain clamped to [0.6, 1.6]."""
    h, w, _ = rgb.shape
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    small = Image.fromarray(np.clip(lum, 0, 255).astype(np.uint8))
    blur = np.asarray(small.resize((16, 16), Image.LANCZOS).resize((w, h), Image.LANCZOS),
                      dtype=np.float64)
    gain = np.clip(float(blur.mean()) / (blur + 1e-3), 0.6, 1.6)
    return np.clip(rgb * gain[:, :, None], 0, 255)


def make_bark_albedo(pal, size, rng):
    """Tileable grey-brown bark: vertical fissures, pale ridges, dark grooves, fine grain."""
    h = w = size
    dark = pal["bark_dark"]
    light = pal["bark_light"]

    base = _tileable_value_noise(h, w, gx=26, gy=6, rng=rng)
    detail = _tileable_value_noise(h, w, gx=70, gy=14, rng=rng)
    ridge = np.clip(0.68 * base + 0.32 * detail, 0.0, 1.0)
    ridge = _smoothstep(ridge)

    grain = _tileable_value_noise(h, w, gx=220, gy=300, rng=rng)
    grain_f = 0.85 + 0.30 * grain

    rgb = (dark[None, None, :] + (light - dark)[None, None, :] * ridge[:, :, None])
    rgb = rgb * grain_f[:, :, None]
    mott = _tileable_value_noise(h, w, gx=5, gy=5, rng=rng)
    rgb *= (0.90 + 0.18 * mott)[:, :, None]
    rgb = np.clip(rgb, 0, 255)

    rgb = _delight(rgb)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def make_normal_from_albedo(albedo_u8, strength=2.6):
    """Tangent-space normal map from albedo luminance (pale ridges = raised). Tileable."""
    a = albedo_u8.astype(np.float64) / 255.0
    height = a @ np.array([0.299, 0.587, 0.114])
    gx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * strength
    gy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * strength
    nz = np.ones_like(height)
    nx, ny = -gx, -gy
    ln = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    nx, ny, nz = nx / ln, ny / ln, nz / ln
    out = np.empty((height.shape[0], height.shape[1], 3), dtype=np.uint8)
    out[:, :, 0] = np.clip((nx * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    out[:, :, 1] = np.clip((ny * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    out[:, :, 2] = np.clip((nz * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    return out


# =========================================================================== #
#                           FOLIAGE ATLAS (4x4)                               #
# =========================================================================== #
def _leaf_polygon(cx, cy, length, width, ang):
    """A small pointed-oval leaflet silhouette as a rotated polygon."""
    shape = [(0.0, 0.0), (0.20, 0.36), (0.50, 0.50), (0.80, 0.36),
             (1.0, 0.0), (0.80, -0.36), (0.50, -0.50), (0.20, -0.36)]
    ca, sa = math.cos(ang), math.sin(ang)
    pts = []
    for lx, ly in shape:
        X = lx * length
        Y = ly * width
        pts.append((cx + X * ca - Y * sa, cy + X * sa + Y * ca))
    return pts


def _jit_color(base, rng):
    f = rng.uniform(0.82, 1.12)
    c = base * f
    c = c + np.array([rng.uniform(-6, 6), rng.uniform(-4, 8), rng.uniform(-6, 4)])
    c = np.clip(c, 0, 255)
    return (int(c[0]), int(c[1]), int(c[2]), 255)


def _draw_cluster(D, base_rgb, rng):
    """Draw one lacy bipinnate leaf cluster (binary alpha) at supersampled size D."""
    img = Image.new("RGBA", (D, D), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    cx = cy = D / 2.0
    R = 0.40 * D
    n_fronds = int(rng.integers(13, 20))
    for _ in range(n_fronds):
        a0 = rng.uniform(0.0, 2.0 * math.pi)
        curve = rng.uniform(-0.5, 0.5)
        flen = R * rng.uniform(0.55, 1.0)
        nseg = int(rng.integers(7, 12))
        for s in range(nseg):
            t = (s + 1) / nseg
            ang = a0 + curve * t
            rx = cx + math.cos(ang) * flen * t
            ry = cy + math.sin(ang) * flen * t
            lsize = (1.0 - 0.6 * t) * R * 0.17
            for side in (-1, 1):
                pang = ang + side * math.pi / 2.0 + rng.uniform(-0.3, 0.3)
                ll = lsize * rng.uniform(0.7, 1.2)
                lw = ll * 0.42
                ox = rx + math.cos(pang) * ll * 0.5
                oy = ry + math.sin(pang) * ll * 0.5
                dr.polygon(_leaf_polygon(ox, oy, ll, lw, pang),
                           fill=_jit_color(base_rgb, rng))
    return img


def make_foliage_atlas(pal, rng):
    """4x4 atlas of distinct clusters: sunlit tiles warm/bright, shaded tiles cool/dark."""
    TILE = 256
    SS = 4
    D = TILE * SS
    atlas = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))

    light = pal["fol_light"]
    mid = pal["fol_mid"]
    dark = pal["fol_dark"]
    sun_base = np.clip(0.55 * light + 0.45 * mid + np.array([12, 8, -8]), 0, 255)
    shade_base = np.clip(0.50 * mid + 0.50 * dark + np.array([-8, -2, 8]), 0, 255)

    for idx in range(16):
        col = idx % 4
        row = idx // 4
        sun = bool(rng.integers(0, 2)) if 4 <= idx < 12 else (idx < 4)
        base = sun_base if sun else shade_base
        base = np.clip(base * rng.uniform(0.92, 1.08), 0, 255)
        tile = _draw_cluster(D, base, rng).resize((TILE, TILE), Image.LANCZOS)
        atlas.paste(tile, (col * TILE, row * TILE))
    return atlas


# =========================================================================== #
#                         UVs + MATERIAL APPLICATION                          #
# =========================================================================== #
_BARK_AROUND = 3.0     # texture repeats around the trunk circumference
_BARK_V_TILE = 0.70    # meters of height per bark tile


def _cyl_uv(V):
    """Cylindrical UVs about the +Y axis for bark (repeats via default REPEAT sampler)."""
    x = V[:, 0]
    y = V[:, 1]
    z = V[:, 2]
    ang = np.arctan2(z, x)
    u = (ang / (2.0 * np.pi)) * _BARK_AROUND
    v = y / _BARK_V_TILE
    return np.column_stack([u, v]).astype(np.float64)


def _canopy_uv(mesh, seed):
    """Map each leaf card (4 sequential verts: p0,p1,p2,p3) onto a random atlas tile."""
    n = len(mesh.vertices)
    n_cards = n // 4
    uv = np.zeros((n, 2), dtype=np.float64)
    rng = np.random.default_rng(seed + 9973)
    m = 0.03
    cell = 0.25

    base = [(1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)]

    def rot90(s, t):
        return (t, 1.0 - s)

    for i in range(n_cards):
        tile = int(rng.integers(0, 16))
        r = int(rng.integers(0, 4))
        col = tile % 4
        row = tile // 4
        for k in range(4):
            s, t = base[k]
            for _ in range(r):
                s, t = rot90(s, t)
            s = m + s * (1.0 - 2.0 * m)
            t = m + t * (1.0 - 2.0 * m)
            uv[4 * i + k, 0] = col * cell + s * cell
            uv[4 * i + k, 1] = row * cell + t * cell
    return uv


def _to_tint(colors_u8, lo):
    """Remap stored vertex colors into a gentle [lo,1] multiplier so the texture stays visible."""
    c = colors_u8[:, :3].astype(np.float64) / 255.0
    f = lo + (1.0 - lo) * c
    out = np.empty_like(colors_u8)
    out[:, :3] = np.clip(f * 255.0, 0, 255).astype(np.uint8)
    out[:, 3] = 255
    return out


def apply_textures(scene, pal, seed):
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial

    rng = np.random.default_rng(seed + 4242)

    bark_albedo = make_bark_albedo(pal, 1024, rng)
    bark_normal = make_normal_from_albedo(bark_albedo, strength=2.6)
    bark_img = Image.fromarray(bark_albedo, mode="RGB")
    bark_nrm_img = Image.fromarray(bark_normal, mode="RGB")

    atlas_img = make_foliage_atlas(pal, rng)

    bark_mat = PBRMaterial(
        name="bark",
        baseColorTexture=bark_img,
        normalTexture=bark_nrm_img,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0,
        roughnessFactor=0.9,
        doubleSided=False,
    )
    foliage_mat = PBRMaterial(
        name="foliage",
        baseColorTexture=atlas_img,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0,
        roughnessFactor=0.8,
        alphaMode="MASK",
        alphaCutoff=0.45,
        doubleSided=True,
    )

    for name, mesh in scene.geometry.items():
        try:
            colors = np.asarray(mesh.visual.vertex_colors).copy()
        except Exception:
            colors = None

        if name in ("trunk", "branches"):
            uv = _cyl_uv(mesh.vertices)
            tint = _to_tint(colors, 0.55) if colors is not None else None
            mesh.visual = TextureVisuals(uv=uv, material=bark_mat)
            if tint is not None:
                mesh.visual.vertex_attributes["color"] = tint.astype(np.uint8)
        elif name == "canopy":
            uv = _canopy_uv(mesh, seed)
            tint = _to_tint(colors, 0.60) if colors is not None else None
            mesh.visual = TextureVisuals(uv=uv, material=foliage_mat)
            if tint is not None:
                mesh.visual.vertex_attributes["color"] = tint.astype(np.uint8)

    return scene


# =========================================================================== #
#                                   CLI                                       #
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser(description="Procedural umbrella acacia -> textured GLB.")
    ap.add_argument("--image", required=True, help="Reference photo path.")
    ap.add_argument("--seed", type=int, default=0, help="Deterministic seed.")
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="Output .glb path.")
    args = ap.parse_args()

    try:
        img = _load_rgb(args.image)
        pal = sample_palettes(img, np.random.default_rng(args.seed + 101))
        scene = build_mesh(args.seed, args.density)
        scene = apply_textures(scene, pal, args.seed)

        glb = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(glb)
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1

    print("Wrote {} ({} bytes)".format(args.output, len(glb)))
    return 0


if __name__ == "__main__":
    sys.exit(main())