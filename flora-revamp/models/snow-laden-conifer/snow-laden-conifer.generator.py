#!/usr/bin/env python3
"""
Procedural snow-covered conifer (spruce/fir "Christmas-tree" archetype):
build geometry, derive tileable materials from a reference photo, UV by
surface type, and export a textured GLB.

Only numpy, trimesh, PIL (Pillow) and the Python stdlib are used.

CLI:
    python thisscript.py --image PATH --seed INT \
        --density {high,med,low} --output OUT.glb
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter


# ===========================================================================
# GEOMETRY  (build_mesh -- carries clean UVs + per-vertex tint in metadata)
# ===========================================================================

# --- MEASURED PROPORTIONS (read off reference.png; front aspect target ~0.52)
HEIGHT_OVER_WIDTH = 1.90          # narrow, slender conical silhouette
TREE_HEIGHT       = 4.5           # meters -- plausible young conifer
CROWN_WIDTH       = TREE_HEIGHT / HEIGHT_OVER_WIDTH   # ~2.37 m base diameter
R_BASE            = CROWN_WIDTH / 2.0                  # ~1.18 m base radius

CONE_POWER        = 0.95          # envelope profile: ~linear cone, faint convex sweep
LOBE_AMP          = 0.05          # low-frequency scalloping of the cone plan

R_TRUNK_BASE      = 0.050         # m, bole radius at ground
R_TRUNK_TOP       = 0.008         # m, bole radius near apex
TRUNK_FLARE       = 1.45          # x multiplier of base radius at the very bottom
FLARE_FRAC        = 0.07          # flare lives in the bottom 7% of height
TRUNK_TOP_FRAC    = 0.97          # trunk reaches 97% of total height

DROOP_FRAC        = 0.16          # quadratic sag of a limb as a fraction of its length
HALF_FRAC         = 0.062         # leaf-card half-size ~6.2% of crown width
ENV_CLAMP         = 0.92          # cards kept inside 92% of the envelope -> clean cone

WOOD_U_TILES      = 1.0           # one wrap of bark around a limb
WOOD_V_PER_M      = 3.0           # bark repeats ~3x per meter of length
SNOW_TILE_PER_M   = 1.0           # snow texel scale (world XZ planar)

DENSITY = {
    "high": dict(n_tiers=20, br_bottom=11, br_top=5, br_sides=5, br_segs=3,
                 trunk_sides=14, trunk_segs=9, cards_per_clump=42,
                 clump_mid=True, n_interior=20,
                 snow_rings=2, snow_segs=8, snow_frac=0.90),
    "med":  dict(n_tiers=14, br_bottom=7, br_top=4, br_sides=4, br_segs=2,
                 trunk_sides=9, trunk_segs=6, cards_per_clump=24,
                 clump_mid=True, n_interior=10,
                 snow_rings=2, snow_segs=6, snow_frac=0.80),
    "low":  dict(n_tiers=8, br_bottom=4, br_top=2, br_sides=4, br_segs=1,
                 trunk_sides=6, trunk_segs=3, cards_per_clump=18,
                 clump_mid=False, n_interior=0,
                 snow_rings=1, snow_segs=5, snow_frac=0.60),
}


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _frame(tangent):
    t = _norm(np.asarray(tangent, float))
    up = np.array([0.0, 1.0, 0.0])
    ref = np.array([1.0, 0.0, 0.0]) if abs(t @ up) > 0.98 else up
    right = _norm(np.cross(t, ref))
    n = _norm(np.cross(right, t))
    return right, n


def _tube(path, radii, sides):
    """Tapered tube along a polyline. uv[:,0]=angle in [0,1), uv[:,1]=length(m)."""
    path = np.asarray(path, float)
    radii = np.asarray(radii, float)
    m = len(path)
    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    verts = np.empty((m * sides, 3), float)
    uv = np.empty((m * sides, 2), float)
    for i in range(m):
        t = path[i + 1] - path[i] if i < m - 1 else path[i] - path[i - 1]
        right, n = _frame(t)
        verts[i * sides:(i + 1) * sides] = (
            path[i] + radii[i] * (np.outer(ca, right) + np.outer(sa, n))
        )
        uv[i * sides:(i + 1) * sides, 0] = ang / (2.0 * np.pi)
        uv[i * sides:(i + 1) * sides, 1] = cum[i]
    faces = []
    for i in range(m - 1):
        a0, a1 = i * sides, (i + 1) * sides
        for j in range(sides):
            jn = (j + 1) % sides
            faces.append([a0 + j, a1 + j, a0 + jn])
            faces.append([a0 + jn, a1 + j, a1 + jn])
    return verts, np.asarray(faces, np.int64), uv


def _unit_dome(rings, segments):
    verts = [[0.0, 1.0, 0.0]]
    for i in range(1, rings + 1):
        lat = (i / rings) * (np.pi / 2.0)
        y, r = np.cos(lat), np.sin(lat)
        for j in range(segments):
            a = 2.0 * np.pi * j / segments
            verts.append([r * np.cos(a), y, r * np.sin(a)])
    faces = []
    for j in range(segments):
        faces.append([0, 1 + j, 1 + (j + 1) % segments])
    for i in range(rings - 1):
        r0, r1 = 1 + i * segments, 1 + (i + 1) * segments
        for j in range(segments):
            jn = (j + 1) % segments
            faces.append([r0 + j, r1 + j, r0 + jn])
            faces.append([r0 + jn, r1 + j, r1 + jn])
    return np.asarray(verts, float), np.asarray(faces, np.int64)


def _trunk_radius(h):
    r = R_TRUNK_TOP + (R_TRUNK_BASE - R_TRUNK_TOP) * (1.0 - h) ** 1.3
    if h < FLARE_FRAC:
        r *= 1.0 + (TRUNK_FLARE - 1.0) * (1.0 - h / FLARE_FRAC)
    return r


def _env_radius(h):
    return R_BASE * max(0.0, 1.0 - h) ** CONE_POWER


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    p = DENSITY.get(density, DENSITY["high"])
    half = HALF_FRAC * CROWN_WIDTH

    lobe_freq = np.array([2.0, 3.0, 5.0])
    lobe_phase = rng.uniform(0.0, 2.0 * np.pi, 3)

    def lobe(theta):
        return 1.0 + LOBE_AMP * float(np.sum(np.sin(lobe_freq * theta + lobe_phase)))

    # ---------------------------------------------------------------- trunk
    ts = np.linspace(0.0, 1.0, p["trunk_segs"] + 1)
    trunk_pts = np.array([[0.0, t * TRUNK_TOP_FRAC * TREE_HEIGHT, 0.0] for t in ts])
    trunk_radii = np.array([_trunk_radius(t) for t in ts])
    tv, tf, t_uv = _tube(trunk_pts, trunk_radii, p["trunk_sides"])

    # ------------------------------------------------- branches + clump specs
    br_v, br_f, br_uv, br_off = [], [], [], 0
    clumps = []   # (center[3], normal[3], clump_radius, n_cards, h)

    h_tiers = np.linspace(0.05, 0.97, p["n_tiers"])
    for h in h_tiers:
        n_br = max(2, int(round(p["br_bottom"] + (p["br_top"] - p["br_bottom"]) * h)))
        rt = _trunk_radius(h)
        base_ang = rng.uniform(0.0, 2.0 * np.pi)
        y0 = h * TREE_HEIGHT
        for b in range(n_br):
            theta = base_ang + 2.0 * np.pi * b / n_br + rng.uniform(-0.15, 0.15)
            dirh = np.array([np.cos(theta), 0.0, np.sin(theta)])
            L = max(0.05, _env_radius(h) * lobe(theta) * rng.uniform(0.88, 1.0))

            ss = np.linspace(0.0, 1.0, p["br_segs"] + 1)
            pts = np.array([[dirh[0] * L * s,
                             max(y0 - DROOP_FRAC * L * s * s, 0.0),
                             dirh[2] * L * s] for s in ss])
            r0 = min(max(0.45 * rt, 0.012), 0.05)
            radii = np.linspace(r0, r0 * 0.25, len(ss))
            v, f, uv = _tube(pts, radii, p["br_sides"])
            br_v.append(v); br_f.append(f + br_off); br_uv.append(uv)
            br_off += len(v)

            cn = _norm(dirh - np.array([0.0, 0.25, 0.0]))
            clump_r = (0.07 + 0.05 * (1.0 - h)) * CROWN_WIDTH
            cpc = int(round(p["cards_per_clump"] * (0.7 + 0.6 * (1.0 - h))))
            clumps.append((pts[-1].copy(), cn, clump_r, max(cpc, 4), h))
            if p["clump_mid"] and len(pts) > 2 and L > 0.5:
                clumps.append((pts[len(pts) // 2].copy(), cn,
                               clump_r * 0.85, max(int(cpc * 0.7), 4), h))

    for _ in range(p["n_interior"]):
        h = rng.uniform(0.2, 0.85)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        rr = rng.uniform(0.2, 0.65) * _env_radius(h)
        c = np.array([np.cos(theta) * rr,
                      h * TREE_HEIGHT - DROOP_FRAC * rr * 0.5,
                      np.sin(theta) * rr])
        cn = _norm(np.array([np.cos(theta), -0.2, np.sin(theta)]))
        clumps.append((c, cn, (0.06 + 0.04 * (1.0 - h)) * CROWN_WIDTH,
                       max(int(p["cards_per_clump"] * 0.55), 4), h))

    clumps.append((np.array([0.0, TREE_HEIGHT * 0.985, 0.0]),
                   np.array([0.0, 1.0, 0.0]), 0.045 * CROWN_WIDTH,
                   max(int(p["cards_per_clump"] * 0.5), 6), 0.99))

    # ----------------------------- canopy cards (dense, envelope-clamped, tinted)
    corners, c_uv, c_col = [], [], []
    for (c, cn, cr, cnt, h) in clumps:
        cn = _norm(cn)
        clump_bright = float(np.clip(0.50 + 0.35 * h + rng.normal(0.0, 0.04),
                                     0.40, 0.92))     # deep green, gentle sun gradient
        warm = 0.07 * (h - 0.5)
        col = np.clip(np.array([clump_bright * (1.0 + 0.16 * warm),
                                clump_bright,
                                clump_bright * (1.0 - 0.30 * warm)]), 0.0, 1.0)
        col255 = [int(col[0] * 255), int(col[1] * 255), int(col[2] * 255), 255]
        for _ in range(cnt):
            pos = c + rng.normal(0.0, 1.0, 3) * cr * np.array([0.8, 0.55, 0.8])
            # hard clamp inside the foliage envelope -> clean conical silhouette
            hp = float(np.clip(pos[1] / TREE_HEIGHT, 0.0, 0.999))
            maxr = _env_radius(hp) * ENV_CLAMP + 0.02
            rad = float(np.hypot(pos[0], pos[2]))
            if rad > maxr and rad > 1e-6:
                pos[0] *= maxr / rad
                pos[2] *= maxr / rad
            nn = cn + rng.normal(0.0, 0.30, 3)        # tangent, mild jitter
            nn[1] -= 0.08
            nn = _norm(nn)
            u = np.cross([0.0, 1.0, 0.0], nn)
            if np.linalg.norm(u) < 1e-6:
                u = np.cross([1.0, 0.0, 0.0], nn)
            u = _norm(u); vv = _norm(np.cross(nn, u))
            hu = half * np.exp(rng.normal(0.0, 0.28))
            hv = half * np.exp(rng.normal(0.0, 0.28))
            corners.append(pos - hu * u - hv * vv)
            corners.append(pos + hu * u - hv * vv)
            corners.append(pos + hu * u + hv * vv)
            corners.append(pos - hu * u + hv * vv)
            tr, tc, rot = int(rng.integers(0, 4)), int(rng.integers(0, 4)), int(rng.integers(0, 4))
            e = 0.004
            u0, u1 = tc / 4.0 + e, (tc + 1) / 4.0 - e
            v0, v1 = tr / 4.0 + e, (tr + 1) / 4.0 - e
            quad = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
            quad = quad[rot:] + quad[:rot]
            c_uv.extend(quad)
            c_col.extend([col255, col255, col255, col255])

    cverts = np.asarray(corners, float).reshape(-1, 3)
    ncards = len(cverts) // 4
    idx = np.arange(ncards) * 4
    cfaces = np.empty((ncards * 2, 3), np.int64)
    cfaces[0::2] = np.stack([idx, idx + 1, idx + 2], axis=1)
    cfaces[1::2] = np.stack([idx, idx + 2, idx + 3], axis=1)
    c_uv = np.asarray(c_uv, float)
    c_col = np.asarray(c_col, np.uint8)

    # ------------------------------------------------------------ snow caps
    unit_v, unit_f = _unit_dome(p["snow_rings"], p["snow_segs"])
    sn_v, sn_f, sn_off = [], [], 0
    n_clumps = len(clumps)
    for i, (c, cn, cr, cnt, h) in enumerate(clumps):
        is_apex = (i == n_clumps - 1)
        w = 0.6 + 0.6 * max(0.0, (h - 0.7) / 0.3) + 0.5 * max(0.0, (0.3 - h) / 0.3)
        if is_apex or rng.random() < min(0.98, p["snow_frac"] * w):
            rr = cr * rng.uniform(0.85, 1.2)
            hh = rr * 0.30                              # flatter, blanket-like pillow
            center = c + np.array([0.0, cr * 0.12, 0.0])  # sits in the foliage
            sn_v.append(unit_v * np.array([rr, hh, rr]) + center)
            sn_f.append(unit_f + sn_off)
            sn_off += len(unit_v)

    # ----------------------------------------------------------------- meshes
    trunk_mesh = trimesh.Trimesh(vertices=tv, faces=tf, process=False)
    branch_mesh = trimesh.Trimesh(vertices=np.vstack(br_v),
                                  faces=np.vstack(br_f), process=False)
    canopy_mesh = trimesh.Trimesh(vertices=cverts, faces=cfaces, process=False)
    snow_mesh = trimesh.Trimesh(vertices=np.vstack(sn_v),
                                faces=np.vstack(sn_f), process=False)

    meshes = [trunk_mesh, branch_mesh, canopy_mesh, snow_mesh]
    allv = np.vstack([m.vertices for m in meshes])
    cx = 0.5 * (allv[:, 0].min() + allv[:, 0].max())
    cz = 0.5 * (allv[:, 2].min() + allv[:, 2].max())
    shift = np.array([-cx, -allv[:, 1].min(), -cz])
    for m in meshes:
        m.apply_translation(shift)

    # ---- scaled UVs + per-vertex tints stashed in metadata (textured later)
    def wood_tint(verts):
        ny = np.clip(verts[:, 1] / (0.45 * TREE_HEIGHT), 0.0, 1.0)   # AO darker low
        base = np.clip(0.45 + 0.40 * ny + rng.normal(0.0, 0.04, len(verts)), 0.3, 0.95)
        out = np.empty((len(verts), 4), np.uint8)
        out[:, :3] = np.clip(base[:, None] * 255, 0, 255).astype(np.uint8)
        out[:, 3] = 255
        return out

    branch_uv = np.vstack(br_uv)
    t_uv = t_uv.copy(); t_uv[:, 0] *= WOOD_U_TILES; t_uv[:, 1] *= WOOD_V_PER_M
    branch_uv[:, 0] *= WOOD_U_TILES; branch_uv[:, 1] *= WOOD_V_PER_M

    sv = snow_mesh.vertices
    sny = (sv[:, 1] - sv[:, 1].min()) / (np.ptp(sv[:, 1]) + 1e-9)
    s_bright = 0.90 + 0.10 * sny                       # bright white snow
    snow_col = np.empty((len(sv), 4), np.uint8)
    snow_col[:, 0] = np.clip((s_bright - 0.015 * (1 - sny)) * 255, 0, 255)
    snow_col[:, 1] = np.clip((s_bright - 0.008 * (1 - sny)) * 255, 0, 255)
    snow_col[:, 2] = np.clip((s_bright + 0.030 * (1 - sny)) * 255, 0, 255)  # faint blue hollow
    snow_col[:, 3] = 255
    snow_uv = np.column_stack([sv[:, 0] * SNOW_TILE_PER_M, sv[:, 2] * SNOW_TILE_PER_M])

    trunk_mesh.metadata.update(dict(kind="wood", uv=t_uv, vcolor=wood_tint(trunk_mesh.vertices)))
    branch_mesh.metadata.update(dict(kind="wood", uv=branch_uv, vcolor=wood_tint(branch_mesh.vertices)))
    canopy_mesh.metadata.update(dict(kind="foliage", uv=c_uv, vcolor=c_col))
    snow_mesh.metadata.update(dict(kind="snow", uv=snow_uv, vcolor=snow_col))

    scene = trimesh.Scene()
    scene.add_geometry(trunk_mesh,  geom_name="trunk")
    scene.add_geometry(branch_mesh, geom_name="branches")
    scene.add_geometry(canopy_mesh, geom_name="canopy")
    scene.add_geometry(snow_mesh,   geom_name="snow")
    return scene


# ===========================================================================
# TEXTURING
# ===========================================================================
FOLIAGE_REGIONS = [(0.35, 0.78), (0.64, 0.78), (0.39, 0.63), (0.62, 0.63),
                   (0.50, 0.72), (0.44, 0.50), (0.57, 0.53), (0.50, 0.42),
                   (0.50, 0.86), (0.43, 0.70), (0.58, 0.70), (0.50, 0.58)]
SNOW_REGIONS    = [(0.50, 0.16), (0.41, 0.42), (0.59, 0.45), (0.50, 0.63),
                   (0.38, 0.77), (0.62, 0.78), (0.50, 0.30), (0.50, 0.50)]
WOOD_REGIONS    = [(0.50, 0.88), (0.46, 0.80), (0.54, 0.80), (0.50, 0.83),
                   (0.47, 0.86), (0.53, 0.86)]


def _luma(c):
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def delight(arr):
    """Remove baked lighting: divide by blurred luminance, clamp gain [0.75,1.25]."""
    H, W = arr.shape[:2]
    lum = _luma(arr)
    limg = Image.fromarray(np.clip(lum * 255, 0, 255).astype(np.uint8))
    blur = np.asarray(limg.filter(ImageFilter.GaussianBlur(min(H, W) / 8.0)),
                      float) / 255.0 + 1e-4
    gain = np.clip(blur.mean() / blur, 0.75, 1.25)[..., None]
    return np.clip(arr * gain, 0.0, 1.0)


def sample_color(arr, regions, pred, fallback):
    H, W = arr.shape[:2]
    half = max(3, int(0.012 * min(H, W)))
    cols = []
    for fx, fy in regions:
        cx, cy = int(fx * W), int(fy * H)
        patch = arr[max(0, cy - half):cy + half + 1,
                    max(0, cx - half):cx + half + 1].reshape(-1, 3)
        if patch.size == 0:
            continue
        c = np.median(patch, axis=0)
        if pred(c):
            cols.append(c)
    if not cols:
        return np.array(fallback, float)
    return np.median(np.array(cols), axis=0)


def tileable_blur(arr01, radius):
    h, w = arr01.shape
    pad = int(radius * 3) + 1
    p = np.pad(arr01, pad, mode="wrap")
    img = Image.fromarray(np.clip(p * 255, 0, 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius))
    out = np.asarray(img, float) / 255.0
    return out[pad:pad + h, pad:pad + w]


def make_foliage_atlas(green, rng, res=1024):
    """4x4 atlas of DENSE, opaque needle-spray cluster tiles (feathered edges)."""
    tile = res // 4
    ss = 4
    T = tile * ss
    g0 = np.array(green, float)
    atlas = Image.new("RGBA", (res, res), (0, 0, 0, 0))
    for r in range(4):
        for c in range(4):
            sun = float(rng.uniform(0.72, 1.02))       # sunlit vs shaded tiles
            warm = sun - 0.85

            def shade(j):
                v = sun * j
                col = np.array([g0[0] * v * (1.0 + 0.16 * warm),
                                g0[1] * v,
                                g0[2] * v * (1.0 - 0.12 * warm)])
                col = np.clip(col * 255, 0, 255).astype(int)
                return (int(col[0]), int(col[1]), int(col[2]), 255)

            tim = Image.new("RGBA", (T, T), (0, 0, 0, 0))
            d = ImageDraw.Draw(tim)
            for _ in range(int(rng.integers(8, 12))):  # many overlapping sprays
                ox = float(rng.uniform(0.15, 0.85)) * T
                oy = float(rng.uniform(0.55, 1.0)) * T
                ma = float(rng.uniform(-0.6, 0.6))
                length = float(rng.uniform(0.6, 0.97)) * T
                tx = ox + np.sin(ma) * length
                ty = oy - np.cos(ma) * length
                # filled opaque spray body (teardrop) -> dense, no see-through
                pp = ma + np.pi / 2.0
                hw = length * 0.17
                d.polygon([(ox + np.sin(pp) * hw, oy - np.cos(pp) * hw),
                           (ox - np.sin(pp) * hw, oy + np.cos(pp) * hw),
                           (tx, ty)], fill=shade(0.85))
                # feathered needles on top
                npair = max(5, int(length / (T * 0.045)))
                for k in range(npair):
                    t = k / npair
                    px = ox + np.sin(ma) * length * t
                    py = oy - np.cos(ma) * length * t
                    nl = (1.0 - t) * T * float(rng.uniform(0.11, 0.20)) + ss * 2
                    spread = float(rng.uniform(0.5, 0.9))
                    for side in (-1.0, 1.0):
                        na = ma + side * spread
                        ex = px + np.sin(na) * nl
                        ey = py - np.cos(na) * nl
                        d.line([px, py, ex, ey],
                               fill=shade(float(rng.uniform(0.7, 1.05))),
                               width=max(2, int(ss * 1.5)))
            tim = tim.resize((tile, tile), Image.LANCZOS)
            atlas.paste(tim, (c * tile, r * tile), tim)
    return atlas


def make_bark(wood, rng, res=512):
    """Dark, desaturated grey-brown bark; vertical grain; value range ~2.2x."""
    hue = np.array(wood, float)
    hue = hue / (hue.max() + 1e-6)
    hue = 0.7 * hue + 0.3 * float(hue.mean())          # desaturate -> not orange
    x = np.linspace(0.0, 1.0, res, endpoint=False)
    grain = np.zeros(res)
    for k, amp in [(6, 0.5), (13, 0.3), (27, 0.2), (53, 0.12)]:
        grain += amp * np.sin(2.0 * np.pi * k * x + float(rng.uniform(0, 2 * np.pi)))
    grain = (grain - grain.min()) / (np.ptp(grain) + 1e-6)
    field = np.tile(grain, (res, 1))
    knots = tileable_blur(rng.random((res, res)), res / 24.0)
    knots = (knots - knots.min()) / (np.ptp(knots) + 1e-6)
    field = 0.7 * field + 0.3 * knots
    fine = tileable_blur(rng.random((res, res)), 1.2)
    field = np.clip(field * 0.92 + 0.08 * fine, 0.0, 1.0)
    value = 0.28 + 0.34 * field
    col = np.clip(value[..., None] * hue[None, None, :], 0.0, 1.0)
    return Image.fromarray((col * 255).astype(np.uint8), "RGB")


def normal_from_albedo(img, strength=2.0):
    g = np.asarray(img.convert("L"), float) / 255.0
    h = 1.0 - g
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
    nx, ny, nz = -dx * strength, -dy * strength, np.ones_like(h)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    nm = np.stack([nx / ln, ny / ln, nz / ln], axis=-1)
    return Image.fromarray(((nm * 0.5 + 0.5) * 255).astype(np.uint8), "RGB")


def make_snow(snow, rng, res=1024):
    """Bright matte snow: granular speckle + faint blue-grey shadow mottle."""
    base = np.clip(np.array(snow, float), 0.85, 1.0)
    mott = tileable_blur(rng.random((res, res)), res / 20.0) - 0.5
    gran = rng.random((res, res)) - 0.5
    val = 1.0 + 0.10 * mott + 0.04 * gran
    col = np.stack([base[i] * val for i in range(3)], axis=-1)
    hollow = np.clip(-mott, 0.0, None)
    col[..., 2] += 0.05 * hollow
    col[..., 0] -= 0.02 * hollow
    col = np.clip(col, 0.0, 1.0)
    return Image.fromarray((col * 255).astype(np.uint8), "RGB")


def build_textured_scene(image_path, seed, density):
    scene = build_mesh(seed, density)
    rng = np.random.default_rng(seed + 1)

    src = Image.open(image_path).convert("RGB")
    arr = np.asarray(src, float) / 255.0
    arr = delight(arr)

    green = sample_color(arr, FOLIAGE_REGIONS,
                         lambda c: (c[1] >= c[0] - 0.03) and (c[1] >= c[2] - 0.02)
                         and (0.06 < _luma(c) < 0.62),
                         fallback=(0.12, 0.26, 0.19))
    green = np.clip(green * 0.85, 0.04, 0.6)            # keep a deep forest green
    snow = sample_color(arr, SNOW_REGIONS,
                        lambda c: (_luma(c) > 0.58) and (c.max() - c.min() < 0.18),
                        fallback=(0.92, 0.94, 0.98))
    wdark = sample_color(arr, WOOD_REGIONS, lambda c: _luma(c) < 0.5,
                         fallback=(0.20, 0.16, 0.12))
    wood = np.clip(np.array([wdark[0] * 1.12 + 0.05,
                             wdark[1] * 1.05 + 0.04,
                             wdark[2] * 0.90 + 0.03]), 0.05, 0.55)

    bark_img = make_bark(wood, rng, res=512)
    bark_n = normal_from_albedo(bark_img, strength=2.0)
    foliage_img = make_foliage_atlas(green, rng, res=1024)
    snow_img = make_snow(snow, rng, res=1024)

    PBR = trimesh.visual.material.PBRMaterial
    mat_wood = PBR(name="wood", baseColorTexture=bark_img, normalTexture=bark_n,
                   baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                   metallicFactor=0.0, roughnessFactor=0.9)
    mat_fol = PBR(name="foliage", baseColorTexture=foliage_img,
                  baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                  metallicFactor=0.0, roughnessFactor=0.8,
                  alphaMode="MASK", alphaCutoff=0.45, doubleSided=True)
    mat_snow = PBR(name="snow", baseColorTexture=snow_img,
                   baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                   metallicFactor=0.0, roughnessFactor=0.85, doubleSided=True)
    mats = {"wood": mat_wood, "foliage": mat_fol, "snow": mat_snow}

    for geom in scene.geometry.values():
        kind = geom.metadata.get("kind", "wood")
        uv = np.asarray(geom.metadata["uv"], dtype=np.float32)
        vcol = np.asarray(geom.metadata["vcolor"], dtype=np.uint8)
        geom.visual = trimesh.visual.TextureVisuals(uv=uv, material=mats[kind])
        geom.visual.vertex_attributes["color"] = vcol
    return scene


# ===========================================================================
# CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Snow-covered conifer -> textured GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        scene = build_textured_scene(args.image, args.seed, args.density)
        data = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(data)
    except Exception as exc:                                # noqa: BLE001
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1
    print("wrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())