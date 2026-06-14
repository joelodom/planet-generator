"""
Standing dead tree (snag) on a snow apron — geometry + photo-derived materials,
exported as a single textured GLB.

Pipeline:
  build_mesh(seed, density)   -> trimesh.Scene (geometry + UVs + COLOR_0 tints)
  apply_materials(scene, ...) -> derives tileable deadwood / snow textures and a
                                 4x4 alpha-cutout tuft atlas from the reference
                                 photo and assigns PBR materials.

Only numpy + trimesh + PIL + stdlib. Deterministic per seed. +Y up, base at y=0,
units in meters.
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageFilter, ImageDraw


# ============================================================================
# Measured proportions (read by eye off reference.png), as named constants.
# ============================================================================
TREE_HEIGHT       = 6.0        # m, plausible real snag height (overall size constant)
HEIGHT_OVER_WIDTH = 2.5        # silhouette is ~2.5x taller than wide (front aspect ~0.4)
CROWN_HALF_WIDTH  = TREE_HEIGHT / HEIGHT_OVER_WIDTH / 2.0   # ~1.2 m half-spread
SPLIT_HEIGHT_FRAC = 0.40       # bole splits into leaders at ~40% of height
TRUNK_BASE_RADIUS = 0.42       # m, THICK chunky bole at the ground (pre-flare)
TRUNK_TOP_RADIUS  = 0.30       # m, still-stout radius where the bole splits
BASAL_FLARE       = 1.7        # broad buttressed root collar
N_ROOT_LOBES      = 6          # buttress/root lobes around the base
SNOW_RADIUS       = CROWN_HALF_WIDTH * 0.78   # snow apron at the foot
SNOW_HEIGHT       = 0.18       # m, soft mound rise at the trunk
BOLE_LENGTH       = SPLIT_HEIGHT_FRAC * TREE_HEIGHT   # ~2.4 m main bole

WOOD_TILE_M       = 0.45       # meters of bark per texture tile (UV scale)
SNOW_REPEAT       = 2.0        # snow texture repeats across the apron
FOLIAGE_GRID      = 4          # 4x4 alpha-cutout atlas


# ============================================================================
# small vector helpers
# ============================================================================
def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _perp(v, rng):
    """A random unit vector perpendicular to v."""
    v = _normalize(v)
    for _ in range(4):
        a = rng.normal(size=3)
        a = a - v * np.dot(a, v)
        n = np.linalg.norm(a)
        if n > 1e-6:
            return a / n
    a = np.cross(v, np.array([1.0, 0.0, 0.0]))
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(v, np.array([0.0, 0.0, 1.0]))
    return _normalize(a)


def _rotate(vec, axis, angle):
    """Rodrigues rotation of vec about axis by angle (radians)."""
    axis = _normalize(axis)
    c, s = np.cos(angle), np.sin(angle)
    return vec * c + np.cross(axis, vec) * s + axis * np.dot(axis, vec) * (1.0 - c)


# ============================================================================
# tube builder: sweep a tapered, furrowed, GNARLED tube along a polyline path.
# Returns (vertices, faces, uv).  Cylindrical UVs: V along the axis, U around.
# ============================================================================
def _tube(pts, rad, sides, furrow, rng, flare=1.0, lobes=0, gnarl=0.0,
          flare_h=0.7, lobe_h=0.8):
    pts = np.asarray(pts, float)
    rad = np.asarray(rad, float)
    n = len(pts)

    # unit tangents along the path
    t = np.gradient(pts, axis=0)
    nl = np.linalg.norm(t, axis=1, keepdims=True)
    nl[nl < 1e-9] = 1.0
    t = t / nl

    # parallel-transport frames (avoid twisting / degenerate up vectors)
    normals = np.zeros((n, 3))
    normals[0] = _perp(t[0], rng)
    for i in range(1, n):
        v = np.cross(t[i - 1], t[i])
        s = np.linalg.norm(v)
        if s < 1e-9:
            normals[i] = normals[i - 1]
        else:
            ang = np.arctan2(s, np.clip(np.dot(t[i - 1], t[i]), -1.0, 1.0))
            normals[i] = _rotate(normals[i - 1], v, ang)
        normals[i] = _normalize(normals[i] - t[i] * np.dot(normals[i], t[i]))
    binormals = np.cross(t, normals)

    # angular profile -> subtle vertical grain (low amplitude, no chevron moire)
    theta = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    k1 = int(rng.integers(5, 9))
    k2 = int(rng.integers(9, 16))
    ph1 = rng.uniform(0.0, 2.0 * np.pi)
    ph2 = rng.uniform(0.0, 2.0 * np.pi)
    fpat = 1.0 + furrow * (0.6 * np.sin(k1 * theta + ph1) +
                           0.4 * np.sin(k2 * theta + ph2))
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    if lobes > 0:
        phl = rng.uniform(0.0, 2.0 * np.pi)
        lobepat = np.clip(np.sin(lobes * theta + phl), 0.0, 1.0) ** 1.5

    # gnarl: smooth along-length swelling + traveling angular knot bulges
    if gnarl > 0.0:
        prof = rng.normal(0.0, 1.0, n)
        if n >= 3:
            sm = prof.copy()
            sm[1:-1] = (prof[:-2] + prof[1:-1] + prof[2:]) / 3.0
            prof = sm
        prof = prof / (np.abs(prof).max() + 1e-6)
        globe = int(rng.integers(2, 4))
        gph = rng.uniform(0.0, 2.0 * np.pi)
        gdrift = rng.uniform(-1.5, 1.5)

    base_y = pts[0, 1]
    rings = np.zeros((n, sides, 3))
    for i in range(n):
        h = max(pts[i, 1] - base_y, 0.0)
        flarefac = 1.0 + (flare - 1.0) * np.exp(-h / flare_h)
        rr = rad[i] * flarefac * fpat
        if lobes > 0:                       # buttress lobes that fade upward
            rr = rr + rad[i] * 0.55 * np.exp(-h / lobe_h) * lobepat
        if gnarl > 0.0:                     # knotty swelling
            tfrac = i / max(n - 1, 1)
            gfac = 1.0 + gnarl * (0.6 * prof[i] +
                                  0.4 * np.sin(globe * theta + gph +
                                               gdrift * tfrac * 2.0 * np.pi))
            rr = rr * gfac
        rr = rr * (1.0 + rng.normal(0.0, 0.03, sides))   # coarse bark wobble
        rings[i] = (pts[i][None, :] +
                    (cos_t[:, None] * normals[i][None, :] +
                     sin_t[:, None] * binormals[i][None, :]) * rr[:, None])

    # duplicate the first column at the seam so U is continuous (no wrap face)
    cols = sides + 1
    rings_ext = np.concatenate([rings, rings[:, 0:1, :]], axis=1)   # (n, cols, 3)
    verts_main = rings_ext.reshape(n * cols, 3)

    # UVs: V = arc length / tile, U = integer wraps around -> seamless
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    vv = cum / WOOD_TILE_M
    mean_r = float(np.mean(rad))
    n_around = max(1, int(round((2.0 * np.pi * mean_r) / WOOD_TILE_M)))
    jcol = np.arange(cols) / float(sides)
    uu = (n_around * jcol)[None, :].repeat(n, 0)
    vrep = vv[:, None].repeat(cols, 1)
    uv_main = np.stack([uu, vrep], axis=-1).reshape(n * cols, 2)

    # blunt caps (hidden at joints; broken-stub look at terminal ends)
    verts = np.vstack([verts_main, pts[0][None, :], pts[-1][None, :]])
    uv = np.vstack([uv_main, [[0.0, vv[0]]], [[0.0, vv[-1]]]])

    faces = []
    for i in range(n - 1):
        for j in range(sides):
            a = i * cols + j
            b = i * cols + j + 1
            c = (i + 1) * cols + j + 1
            d = (i + 1) * cols + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    base = n * cols
    c0, c1 = base, base + 1
    for j in range(sides):                          # start cap
        faces.append([c0, j + 1, j])
    last = (n - 1) * cols
    for j in range(sides):                          # end cap
        faces.append([c1, last + j, last + j + 1])

    return verts, np.asarray(faces, int), uv


# ============================================================================
# snow apron: a soft rounded mound (plano-convex lens), flat on the ground.
# ============================================================================
def _snow(rng, R, H, sectors, rings):
    theta = np.linspace(0.0, 2.0 * np.pi, sectors, endpoint=False)
    rim = R * (1.0 + rng.normal(0.0, 0.07, sectors))   # irregular outer edge
    rr = np.linspace(0.0, 1.0, rings + 1)

    verts = [[0.0, H, 0.0]]                              # 0: apex
    idx = np.zeros((rings + 1, sectors), int)
    for k in range(1, rings + 1):
        for j in range(sectors):
            r = rr[k] * rim[j]
            x = r * np.cos(theta[j])
            z = r * np.sin(theta[j])
            y = H * (1.0 - rr[k] ** 2) ** 1.4
            y *= 1.0 + 0.12 * np.sin(3.0 * theta[j] + 0.7) * (1.0 - rr[k])
            verts.append([x, y, z])
            idx[k, j] = len(verts) - 1
    botc = len(verts)
    verts.append([0.0, 0.0, 0.0])                        # bottom center

    faces = []
    for j in range(sectors):                             # apex fan
        faces.append([0, idx[1, j], idx[1, (j + 1) % sectors]])
    for k in range(1, rings):                            # ring quads
        for j in range(sectors):
            j2 = (j + 1) % sectors
            a, b = idx[k, j], idx[k, j2]
            c, d = idx[k + 1, j2], idx[k + 1, j]
            faces.append([a, b, c])
            faces.append([a, c, d])
    for j in range(sectors):                             # flat bottom disc
        faces.append([botc, idx[rings, (j + 1) % sectors], idx[rings, j]])

    V = np.asarray(verts, float)
    F = np.asarray(faces, int)
    uv = np.stack([V[:, 0], V[:, 2]], axis=1) / (2.0 * R) * SNOW_REPEAT + 0.5
    return V, F, uv


# ============================================================================
# very sparse, tiny alpha-cutout debris/tuft cards (satisfy the MASK material
# requirement without reading as bright sprites).
# ============================================================================
def _foliage_cards(rng, seg_branch, p, up):
    tips = [(pts[-1], _normalize(pts[-1] - pts[-2]), float(rad[-1]))
            for pts, rad in seg_branch if len(pts) >= 2]
    if not tips:
        return None

    n_clumps = min(p["fol_clumps"], len(tips))
    anchors = rng.choice(len(tips), size=n_clumps, replace=False)
    m = 0.04
    base_uv = [(m, m), (1 - m, m), (1 - m, 1 - m), (m, 1 - m)]

    V, F, UV, COL = [], [], [], []
    vc = 0
    for ai in anchors:
        tip, tdir, _r = tips[int(ai)]
        nper = int(rng.integers(p["fol_per"][0], p["fol_per"][1] + 1))
        for _ in range(nper):
            center = tip + _perp(tdir, rng) * rng.normal(0.0, 0.02) + tdir * rng.uniform(0.0, 0.03)
            nrm = _normalize(tdir + _perp(tdir, rng) * rng.uniform(0.3, 1.0)
                             + up * rng.uniform(-0.2, 0.4))
            a = _perp(nrm, rng)
            b = np.cross(nrm, a)
            ang = rng.uniform(0.0, 2.0 * np.pi)
            a2 = a * np.cos(ang) + b * np.sin(ang)
            b2 = -a * np.sin(ang) + b * np.cos(ang)
            hs = min(p["fol_size"] * float(np.exp(rng.normal(0.0, 0.3))),
                     p["fol_size"] * 1.8)
            V.append(center + (-a2 - b2) * hs)
            V.append(center + (a2 - b2) * hs)
            V.append(center + (a2 + b2) * hs)
            V.append(center + (-a2 + b2) * hs)
            F.append([vc, vc + 1, vc + 2])
            F.append([vc, vc + 2, vc + 3])

            ti = int(rng.integers(0, FOLIAGE_GRID))
            tj = int(rng.integers(0, FOLIAGE_GRID))
            rot = int(rng.integers(0, 4))
            corners = base_uv[rot:] + base_uv[:rot]
            for lu, lv in corners:
                UV.append([(tj + lu) / FOLIAGE_GRID, (ti + lv) / FOLIAGE_GRID])

            hfrac = np.clip(center[1] / TREE_HEIGHT, 0.0, 1.0)
            br = np.clip((0.50 + 0.18 * hfrac) * rng.uniform(0.9, 1.05), 0.0, 1.0)
            col = [np.clip(br * 1.02, 0, 1), br, np.clip(br * 0.97, 0, 1), 1.0]
            COL.extend([col, col, col, col])
            vc += 4

    if not V:
        return None
    return (np.asarray(V, float), np.asarray(F, int), np.asarray(UV, float),
            (np.asarray(COL, float) * 255).astype(np.uint8))


# ============================================================================
# density presets
# ============================================================================
def _params(density):
    presets = {
        "high": dict(max_depth=6, n_leaders=3, ring_len=0.14, max_rings=10,
                     sides_max=16, sides_min=6, max_seg=420, stub_prob=0.34,
                     spread_deg=34, min_r=0.008, zig_base=0.16, furrow=0.05,
                     snow_sectors=30, snow_rings=10,
                     fol_clumps=14, fol_per=(1, 2), fol_size=0.030),
        "med":  dict(max_depth=5, n_leaders=3, ring_len=0.22, max_rings=7,
                     sides_max=11, sides_min=5, max_seg=210, stub_prob=0.34,
                     spread_deg=34, min_r=0.014, zig_base=0.16, furrow=0.05,
                     snow_sectors=22, snow_rings=7,
                     fol_clumps=10, fol_per=(1, 2), fol_size=0.032),
        "low":  dict(max_depth=4, n_leaders=2, ring_len=0.34, max_rings=4,
                     sides_max=7, sides_min=4, max_seg=85, stub_prob=0.30,
                     spread_deg=32, min_r=0.024, zig_base=0.15, furrow=0.04,
                     snow_sectors=14, snow_rings=4,
                     fol_clumps=6, fol_per=(1, 1), fol_size=0.034),
    }
    return presets.get(density, presets["high"])


def _sides_for(rmax, p):
    s = np.interp(rmax, [p["min_r"], TRUNK_BASE_RADIUS],
                  [p["sides_min"], p["sides_max"]])
    return int(np.clip(round(s), p["sides_min"], p["sides_max"]))


# ============================================================================
# per-vertex COLOR_0 tints (multiply the albedo in glTF) — kept light so the
# pale silvery wood does not go muddy.
# ============================================================================
def _wood_vertex_colors(V, rng):
    y = V[:, 1]
    x = V[:, 0]
    ymax = max(float(y.max()), 1e-3)
    h = np.clip(y / ymax, 0.0, 1.0)
    bright = 0.84 + 0.16 * h                         # lower only mildly darker
    xm = float(np.abs(x).max()) + 1e-6
    sun = 1.0 + 0.04 * np.clip(x / xm, -1.0, 1.0)
    m = bright * sun * (1.0 + rng.normal(0.0, 0.025, len(y)))
    m = np.clip(m, 0.78, 1.0)
    r = np.clip(m * (1.0 + 0.03 * (h - 0.5)), 0.0, 1.0)
    g = np.clip(m, 0.0, 1.0)
    b = np.clip(m * (1.0 - 0.04 * (h - 0.5)), 0.0, 1.0)
    a = np.ones_like(m)
    return (np.stack([r, g, b, a], 1) * 255.0).astype(np.uint8)


def _snow_vertex_colors(V):
    y = V[:, 1]
    h = np.clip(y / (float(y.max()) + 1e-6), 0.0, 1.0)
    m = 0.88 + 0.12 * h
    r = np.clip(m * 0.99, 0.0, 1.0)
    g = np.clip(m * 0.995, 0.0, 1.0)
    b = np.clip(m, 0.0, 1.0)
    a = np.ones_like(m)
    return (np.stack([r, g, b, a], 1) * 255.0).astype(np.uint8)


def _combine(parts):
    V, F, UV, off = [], [], [], 0
    for v, f, uv in parts:
        V.append(v)
        UV.append(uv)
        F.append(f + off)
        off += len(v)
    return np.vstack(V), np.vstack(F), np.vstack(UV)


def _make_mesh(V, F, UV, vcol):
    m = trimesh.Trimesh(vertices=V, faces=F, process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=UV)
    m.visual.vertex_attributes["color"] = vcol
    return m


# ============================================================================
# entry point — geometry only (materials assigned later)
# ============================================================================
def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    p = _params(density)
    up = np.array([0.0, 1.0, 0.0])

    seg_trunk = []   # (path, radii) -> bole + leaders  (semantic "trunk")
    seg_branch = []  # (path, radii) -> thinner limbs   (semantic "branches")
    counter = [0]

    def grow(start, direction, length, r0, r1, depth, is_trunk):
        # zig-zagging, tapering path for this stem
        n = int(np.clip(round(length / p["ring_len"]) + 1, 2, p["max_rings"]))
        pts = np.zeros((n, 3))
        rad = np.zeros(n)
        d = _normalize(direction)
        pos = np.array(start, float)
        pts[0] = pos
        rad[0] = r0
        zig = p["zig_base"] * (0.6 + 0.8 * depth)        # sharper kinks when deeper
        upbias = max(0.0, 0.5 - 0.11 * depth)            # less upward when deeper
        step = length / (n - 1)
        for i in range(1, n):
            tt = i / (n - 1)
            d = _normalize(d + _perp(d, rng) * zig + up * upbias * 0.3)
            pos = pos + d * step
            pts[i] = pos
            rad[i] = r0 + (r1 - r0) * tt
        (seg_trunk if is_trunk else seg_branch).append((pts, rad))

        if (depth >= p["max_depth"] or r1 < p["min_r"]
                or counter[0] >= p["max_seg"]):
            return

        end = pts[-1]
        enddir = _normalize(pts[-1] - pts[-2])

        if depth == 0:
            nch = p["n_leaders"]
        elif depth < p["max_depth"] - 1:
            nch = int(rng.choice([2, 3, 4], p=[0.40, 0.42, 0.18]))   # denser cage
        else:
            nch = 2

        # child radii preserve cross-section: r_parent^2 ~= sum(r_child^2)
        w = rng.uniform(0.6, 1.0, nch)
        rc = r1 * w / np.sqrt(np.sum(w ** 2))

        for k in range(nch):
            counter[0] += 1
            if counter[0] >= p["max_seg"]:
                break

            if depth == 0:
                # 2-3 stout leaders, splayed in azimuth, near-vertical lean
                az = 2.0 * np.pi * k / nch + rng.uniform(-0.4, 0.4)
                tilt = np.deg2rad(rng.uniform(8.0, 24.0))
                horiz = np.array([np.cos(az), 0.0, np.sin(az)])
                cdir = _normalize(up * np.cos(tilt) + horiz * np.sin(tilt))
                clen = BOLE_LENGTH * rng.uniform(0.60, 0.85)
                cr0 = min(rc[k], r1 * 0.97)
                cr1 = cr0 * rng.uniform(0.62, 0.78)          # leaders stay thick
                grow(end, cdir, clen, cr0, cr1, depth + 1, True)
                continue

            spread = min(np.deg2rad(p["spread_deg"]) * (0.6 + 0.32 * depth),
                         np.deg2rad(80.0))
            cdir = _rotate(enddir, _perp(enddir, rng),
                           spread * rng.uniform(0.7, 1.25))
            cdir = _normalize(cdir + up * max(0.0, 0.4 - 0.09 * depth))
            cr0 = min(rc[k], r1 * 0.95)

            if rng.random() < p["stub_prob"]:
                # snapped-off blunt stub: short, barely tapered, terminal
                clen = length * rng.uniform(0.14, 0.28)
                cr1 = cr0 * rng.uniform(0.60, 0.85)
                grow(end, cdir, clen, cr0, cr1, p["max_depth"], False)
            else:
                clen = length * rng.uniform(0.42, 0.62)
                cr1 = cr0 * rng.uniform(0.35, 0.58)
                grow(end, cdir, clen, cr0, cr1, depth + 1, False)

    # --- the main bole (flared root collar handled at mesh time) ---
    grow(np.array([0.0, 0.0, 0.0]),
         np.array([0.05, 1.0, 0.0]),       # slight lean off vertical
         BOLE_LENGTH, TRUNK_BASE_RADIUS, TRUNK_TOP_RADIUS, 0, True)

    # --- mesh the wood (gnarled, UV-carrying tubes) ---
    trunk_parts = []
    for idx, (pts, rad) in enumerate(seg_trunk):
        sides = _sides_for(float(rad.max()), p)
        if idx == 0:                        # bole: flare + root lobes + heavy gnarl
            trunk_parts.append(_tube(pts, rad, sides, p["furrow"], rng,
                                     flare=BASAL_FLARE, lobes=N_ROOT_LOBES,
                                     gnarl=0.22))
        else:                               # leaders: gnarled, thick
            trunk_parts.append(_tube(pts, rad, sides, p["furrow"], rng, gnarl=0.16))

    branch_parts = []
    for pts, rad in seg_branch:
        sides = _sides_for(float(rad.max()), p)
        branch_parts.append(_tube(pts, rad, sides, p["furrow"], rng, gnarl=0.08))

    Vs, Fs, UVs = _snow(rng, SNOW_RADIUS, SNOW_HEIGHT,
                        p["snow_sectors"], p["snow_rings"])

    fol = _foliage_cards(rng, seg_branch, p, up)

    # --- assemble scene, keyed by semantic surface name ---
    scene = trimesh.Scene()
    Vt, Ft, UVt = _combine(trunk_parts)
    scene.add_geometry(_make_mesh(Vt, Ft, UVt, _wood_vertex_colors(Vt, rng)),
                       geom_name="trunk")
    if branch_parts:
        Vb, Fb, UVb = _combine(branch_parts)
        scene.add_geometry(_make_mesh(Vb, Fb, UVb, _wood_vertex_colors(Vb, rng)),
                           geom_name="branches")
    scene.add_geometry(_make_mesh(Vs, Fs, UVs, _snow_vertex_colors(Vs)),
                       geom_name="snow")
    if fol is not None:
        Vf, Ff, UVf, COLf = fol
        scene.add_geometry(_make_mesh(Vf, Ff, UVf, COLf), geom_name="foliage")

    # rest exactly on the XZ plane (lowest point at y = 0)
    miny = scene.bounds[0][1]
    scene.apply_translation([0.0, -miny, 0.0])
    return scene


# ============================================================================
# ----------------------------- TEXTURING ------------------------------------
# ============================================================================
RES_WOOD = 1024
RES_SNOW = 512
ATLAS_RES = 1024
N_FURROWS = 9          # subtle vertical furrows around one tile
N_CRACKS = 3           # a few deep vertical checks
N_KNOTS = 5

# Sampling regions chosen by LOOKING at reference.png, placed WELL INSIDE the
# trunk / snow silhouettes (never the flat grey background).  Normalized coords.
WOOD_BOXES = [(0.46, 0.72), (0.50, 0.74), (0.53, 0.70), (0.47, 0.66),
              (0.51, 0.62), (0.49, 0.78), (0.45, 0.69), (0.55, 0.67)]
WOOD_CROP = (0.42, 0.58, 0.58, 0.80)   # (x0, y0, x1, y1) inside the lower trunk
SNOW_BOXES = [(0.45, 0.90), (0.52, 0.905), (0.49, 0.92), (0.55, 0.89),
              (0.42, 0.91), (0.58, 0.90)]


def _load_image(path):
    return np.asarray(Image.open(path).convert("RGB"))


def _sample(img, centers, half):
    """Median pixels from small patches; discard patches unlike the body."""
    H, W = img.shape[:2]
    patches, meds = [], []
    for cx, cy in centers:
        x0 = max(0, int((cx - half) * W)); x1 = min(W, int((cx + half) * W))
        y0 = max(0, int((cy - half) * H)); y1 = min(H, int((cy + half) * H))
        if x1 <= x0 or y1 <= y0:
            continue
        pat = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
        patches.append(pat)
        meds.append(np.median(pat, 0))
    if not patches:
        return np.array([[165.0, 162.0, 156.0]])
    meds = np.array(meds)
    body = np.median(meds, 0)
    keep = [pat for pat, md in zip(patches, meds)
            if np.linalg.norm(md - body) < 70.0]
    if not keep:
        keep = patches
    return np.concatenate(keep, 0)


def _palette(px, enforce=True):
    """Dark/mid/light photo colors (0..1) spanning a real value range."""
    px = np.asarray(px, float)
    if len(px) == 0:
        g = np.array([0.6, 0.59, 0.57])
        return g * 0.7, g, g * 1.3
    lum = 0.299 * px[:, 0] + 0.587 * px[:, 1] + 0.114 * px[:, 2]
    p = px[np.argsort(lum)]
    n = len(p)
    dark = p[int(0.15 * n)] / 255.0
    light = p[int(0.88 * n)] / 255.0
    mid = np.median(px, 0) / 255.0
    if enforce:   # keep lightest ~2.5x darkest so the form never collapses
        ld = 0.299 * dark[0] + 0.587 * dark[1] + 0.114 * dark[2] + 1e-3
        ll = 0.299 * light[0] + 0.587 * light[1] + 0.114 * light[2]
        if ll < 2.2 * ld:
            light = np.clip(light * (2.5 * ld / (ll + 1e-3)), 0.0, 1.0)
    return dark, mid, light


def _silver_wood(dark, mid, light):
    """Lift to a pale silvery-grey and pull out the brown cast (match photo)."""
    def lum(c):
        return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    scale = float(np.clip(0.62 / (lum(mid) + 1e-3), 1.0, 2.1))
    out = []
    for c in (dark, mid, light):
        c = np.asarray(c, float) * scale
        c = c * 0.62 + lum(c) * 0.38          # desaturate toward grey
        out.append(np.clip(c, 0.0, 1.0))
    return out[0], out[1], out[2]


def _crop_norm(img, box):
    H, W = img.shape[:2]
    x0 = max(0, min(W - 1, int(box[0] * W)))
    x1 = max(x0 + 1, min(W, int(box[2] * W)))
    y0 = max(0, min(H - 1, int(box[1] * H)))
    y1 = max(y0 + 1, min(H, int(box[3] * H)))
    return img[y0:y1, x0:x1].astype(np.float32) / 255.0


def _delight(arr):
    """Divide out a heavily blurred luminance; clamp the gain to [0.6, 1.6]."""
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    pil = Image.fromarray((np.clip(lum, 0, 1) * 255).astype(np.uint8))
    rad = max(2, min(arr.shape[0], arr.shape[1]) // 6)
    blur = np.asarray(pil.filter(ImageFilter.GaussianBlur(rad))).astype(np.float32) / 255.0
    blur = np.maximum(blur, 1e-3)
    gain = np.clip(float(lum.mean()) / blur, 0.6, 1.6)
    return np.clip(arr * gain[..., None], 0.0, 1.0)


def _mirror_tile(arr, res):
    """Seamless mirror-fold mosaic, downscaled to res x res (0..1)."""
    a = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    top = np.concatenate([a, a[:, ::-1]], axis=1)
    full = np.concatenate([top, top[::-1]], axis=0)
    out = Image.fromarray(full).resize((res, res), Image.LANCZOS)
    return np.asarray(out).astype(np.float32) / 255.0


def _vnoise(rng, res, perx, pery):
    """Tileable bilinear value noise on a perx x pery lattice."""
    lat = rng.random((pery, perx))
    cx = np.arange(res) / res * perx
    cy = np.arange(res) / res * pery
    ix0 = np.floor(cx).astype(int) % perx
    ix1 = (ix0 + 1) % perx
    fx = cx - np.floor(cx)
    fx = fx * fx * (3 - 2 * fx)
    iy0 = np.floor(cy).astype(int) % pery
    iy1 = (iy0 + 1) % pery
    fy = cy - np.floor(cy)
    fy = fy * fy * (3 - 2 * fy)
    lerpx = lat[:, ix0] * (1 - fx[None, :]) + lat[:, ix1] * fx[None, :]
    out = lerpx[iy0, :] * (1 - fy[:, None]) + lerpx[iy1, :] * fy[:, None]
    return out


def _fbm(rng, res, layers):
    out = np.zeros((res, res))
    tot = 0.0
    for px, py, amp in layers:
        out += amp * _vnoise(rng, res, px, py)
        tot += amp
    return out / tot


def _wood_value(rng, res):
    """Grayscale wood field in [0,1]: STRAIGHT subtle vertical grain (no chevron
    moire), soft weathered patches, a few checks and knots."""
    U = np.tile(np.arange(res) / res, (res, 1))          # around-trunk axis
    V = np.tile((np.arange(res) / res)[:, None], (1, res))  # length axis

    wav = _fbm(rng, res, [(4, 4, 0.6), (8, 8, 0.3)])
    phase = U * N_FURROWS + 0.4 * (wav - 0.5)            # SPIRAL removed
    furrow = 0.5 + 0.5 * np.cos(2.0 * np.pi * phase)

    patches = _fbm(rng, res, [(2, 2, 0.6), (4, 4, 0.3), (8, 8, 0.2)])
    fiber = _fbm(rng, res, [(160, 16, 0.6), (300, 40, 0.4)])   # vertical grain
    grain = _fbm(rng, res, [(96, 96, 0.5), (220, 220, 0.5)])

    v = (0.62 * patches + 0.20 * furrow
         + 0.10 * (fiber - 0.5) + 0.06 * (grain - 0.5))

    # deep vertical checks (cracks that catch shadow)
    wander = _fbm(rng, res, [(8, 2, 1.0)])
    cl = U * N_CRACKS + 0.7 * (wander - 0.5)
    cd = np.abs(((cl + 0.5) % 1.0) - 0.5)
    v = v - 0.28 * np.exp(-(cd / 0.02) ** 2)

    # woody knots with concentric rings (wrapped for tileability)
    for _ in range(N_KNOTS):
        uc, vc = rng.random(), rng.random()
        kr = rng.uniform(0.04, 0.08)
        du = np.abs(U - uc); du = np.minimum(du, 1 - du)
        dv = np.abs(V - vc); dv = np.minimum(dv, 1 - dv)
        d = np.sqrt((du * 1.3) ** 2 + dv ** 2)
        ring = 0.5 + 0.5 * np.cos(2.0 * np.pi * d / 0.03)
        v = v + np.exp(-(d / kr) ** 2) * (-0.22 + 0.14 * ring)

    return np.clip(v, 0.0, 1.0)


def _snow_value(rng, res):
    base = _fbm(rng, res, [(3, 3, 0.6), (6, 6, 0.3), (12, 12, 0.2)])
    sparkle = _fbm(rng, res, [(180, 180, 0.5), (360, 360, 0.5)])
    v = 0.86 + 0.12 * (base - 0.5) + 0.05 * (sparkle - 0.5)
    return np.clip(v, 0.0, 1.0)


def _albedo_from_value(v, dark, mid, light, res):
    alb = np.empty((res, res, 3), np.float32)
    for ch in range(3):
        alb[..., ch] = np.interp(v.ravel(), [0.0, 0.5, 1.0],
                                 [dark[ch], mid[ch], light[ch]]).reshape(res, res)
    return alb


def _normal_map(alb, strength):
    """Tangent-space normal map from albedo luminance (tileable via wrap)."""
    lum = 0.299 * alb[..., 0] + 0.587 * alb[..., 1] + 0.114 * alb[..., 2]
    gx = (np.roll(lum, -1, 1) - np.roll(lum, 1, 1)) * 0.5
    gy = (np.roll(lum, -1, 0) - np.roll(lum, 1, 0)) * 0.5
    nx, ny, nz = -gx * strength, -gy * strength, np.ones_like(lum)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    rgb = (np.stack([nx * inv, ny * inv, nz * inv], -1) * 0.5 + 0.5)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def _foliage_atlas(rng, base_col):
    """4x4 atlas of tiny compact tuft silhouettes, binary alpha (subtle)."""
    ts = ATLAS_RES // FOLIAGE_GRID
    ss = 4
    big = ts * ss
    atlas = Image.new("RGBA", (ATLAS_RES, ATLAS_RES), (0, 0, 0, 0))
    base = np.clip(np.asarray(base_col, float), 0.0, 1.0)
    for ti in range(FOLIAGE_GRID):
        for tj in range(FOLIAGE_GRID):
            tile = Image.new("RGBA", (big, big), (0, 0, 0, 0))
            d = ImageDraw.Draw(tile)
            sunlit = ((ti + tj) % 2 == 0)
            col = np.clip(base * 1.12 + 0.03, 0, 1) if sunlit else np.clip(base * 0.82, 0, 1)
            cx = cy = big * 0.5
            cc = tuple(int(np.clip(col[k], 0, 1) * 255) for k in range(3)) + (255,)
            d.ellipse([cx - 0.16 * big, cy - 0.18 * big,
                       cx + 0.16 * big, cy + 0.18 * big], fill=cc)   # compact core
            for _ in range(int(rng.integers(8, 14))):                # short blades
                ang = rng.uniform(0.0, 2.0 * np.pi)
                ln = rng.uniform(0.18, 0.30) * big
                wd = rng.uniform(0.03, 0.06) * big
                ex, ey = cx + np.cos(ang) * ln, cy + np.sin(ang) * ln
                px, py = -np.sin(ang) * wd, np.cos(ang) * wd
                sh = rng.uniform(0.85, 1.1)
                c = tuple(int(np.clip(col[k] * sh, 0, 1) * 255) for k in range(3)) + (255,)
                d.polygon([(cx + px, cy + py), (cx - px, cy - py), (ex, ey)], fill=c)
            tile = tile.resize((ts, ts), Image.LANCZOS)
            atlas.paste(tile, (tj * ts, ti * ts))
    return atlas


def apply_materials(scene, image_path, seed):
    """Derive deadwood + snow + tuft materials from the photo and assign them."""
    rng = np.random.default_rng((int(seed) * 2654435761 + 0x5BD1E995) & 0xFFFFFFFF)
    img = _load_image(image_path)

    # --- deadwood (pale silvery grey) ---
    wdark, wmid, wlight = _palette(_sample(img, WOOD_BOXES, 0.018), enforce=True)
    wdark, wmid, wlight = _silver_wood(wdark, wmid, wlight)
    crop = _delight(_crop_norm(img, WOOD_CROP))
    lf = _mirror_tile(crop, RES_WOOD)
    lf_tint = np.clip(lf / (lf.reshape(-1, 3).mean(0)[None, None, :] + 1e-6),
                      0.90, 1.12)                          # subtle, less mosaic banding
    wv = _wood_value(rng, RES_WOOD)
    walb = np.clip(_albedo_from_value(wv, wdark, wmid, wlight, RES_WOOD) * lf_tint,
                   0.0, 1.0)
    wood_img = Image.fromarray((walb * 255).astype(np.uint8))
    wood_norm = _normal_map(walb, 1.0)                     # gentle, no zebra

    # --- snow ---
    sdark, smid, slight = _palette(_sample(img, SNOW_BOXES, 0.015), enforce=False)
    sv = _snow_value(rng, RES_SNOW)
    salb = np.clip(_albedo_from_value(sv, sdark, smid, slight, RES_SNOW), 0.0, 1.0)
    snow_img = Image.fromarray((salb * 255).astype(np.uint8))

    # --- tufts (pale, desaturated) ---
    fol_base = 0.5 * wlight + 0.5 * wmid
    flum = 0.299 * fol_base[0] + 0.587 * fol_base[1] + 0.114 * fol_base[2]
    fol_base = np.clip(0.55 * fol_base + 0.45 * flum, 0.0, 1.0)
    atlas_img = _foliage_atlas(rng, fol_base)

    PBR = trimesh.visual.material.PBRMaterial
    wood_mat = PBR(name="deadwood", baseColorTexture=wood_img,
                   normalTexture=wood_norm, metallicFactor=0.0,
                   roughnessFactor=0.9, doubleSided=False)
    snow_mat = PBR(name="snowcrust", baseColorTexture=snow_img,
                   metallicFactor=0.0, roughnessFactor=0.88, doubleSided=False)
    fol_mat = PBR(name="tuft", baseColorTexture=atlas_img,
                  metallicFactor=0.0, roughnessFactor=0.8,
                  alphaMode="MASK", alphaCutoff=0.45, doubleSided=True)

    for nm in ("trunk", "branches"):
        if nm in scene.geometry:
            scene.geometry[nm].visual.material = wood_mat
    if "snow" in scene.geometry:
        scene.geometry["snow"].visual.material = snow_mat
    if "foliage" in scene.geometry:
        scene.geometry["foliage"].visual.material = fol_mat
    return scene


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Procedural snag (dead tree) -> GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        scene = build_mesh(args.seed, args.density)
        apply_materials(scene, args.image, args.seed)
        scene.export(args.output)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()