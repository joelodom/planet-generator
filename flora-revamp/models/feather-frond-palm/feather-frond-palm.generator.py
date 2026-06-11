"""
Coconut-palm: procedural geometry + photo-derived materials -> textured GLB.

Pipeline:
  build_mesh(seed, density)  -> trimesh.Scene  (geometry only, +Y up, y=0 base)
  derive tileable materials sampled from the reference photo
  attach UVs by surface type + PBR materials, preserve per-vertex COLOR_0
  export a binary .glb

CLI:
  python thisscript --image PATH --seed INT --density {high,med,low} --output OUT.glb

Only numpy, trimesh, PIL (Pillow) and the stdlib are used.
"""

import argparse
import sys
import numpy as np
import trimesh

from PIL import Image, ImageDraw
from trimesh.visual.material import PBRMaterial
from trimesh.visual import TextureVisuals


# ============================================================================
# GEOMETRY
# ============================================================================
# ----------------------------------------------------------------------------
# Measured proportions (read off the reference image, qualitative, ~10%).
#   - bare trunk occupies roughly the lower ~55-60% of total height
#   - feathery crown is a FULL, ROUNDED dome occupying the top ~45%
#   - the whole silhouette is tall & slender -> width/height ~ 0.60
#   - trunk is very slender relative to its height, with a modest basal flare
# ----------------------------------------------------------------------------
TRUNK_HEIGHT = 6.5     # m, height of the crown hub above the ground (bare stem)
FROND_LEN    = 2.35    # m, nominal frond length -> sets crown radius (kept narrow)
CROWN_RADIUS = FROND_LEN
TARGET_ASPECT = 0.60   # measured photo width/height of the content

TRUNK_BASE_R = 0.16    # m, trunk radius near the ground (before flare)
TRUNK_TOP_R  = 0.10    # m, trunk radius just under the crown (slender)
BASAL_FLARE  = 1.5     # base widens x1.5 over the bottom ~7%
FLARE_FRAC   = 0.07
LEAN_AMP     = 0.40    # m, gentle horizontal sway over the height

UP = np.array([0.0, 1.0, 0.0])


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _rodrigues(v, axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


def _tube(points, radii, sides, cap_start=True, cap_end=True):
    """Sweep a circle of `radii` along `points` with parallel-transport frames."""
    points = np.asarray(points, dtype=float)
    n = len(points)

    tang = np.zeros_like(points)
    tang[1:-1] = points[2:] - points[:-2]
    tang[0] = points[1] - points[0]
    tang[-1] = points[-1] - points[-2]
    tang = np.array([_norm(t) for t in tang])

    ref = np.array([0.0, 0.0, 1.0]) if abs(tang[0][2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    normals = np.zeros_like(points)
    normals[0] = _norm(np.cross(tang[0], ref))
    for i in range(1, n):
        axis = np.cross(tang[i - 1], tang[i])
        s = np.linalg.norm(axis)
        if s < 1e-8:
            normals[i] = normals[i - 1]
        else:
            axis /= s
            ang = np.arctan2(s, np.dot(tang[i - 1], tang[i]))
            normals[i] = _rodrigues(normals[i - 1], axis, ang)
        normals[i] = _norm(normals[i] - np.dot(normals[i], tang[i]) * tang[i])

    binorm = np.cross(tang, normals)

    theta = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ct, st = np.cos(theta), np.sin(theta)

    verts = np.empty((n * sides, 3))
    for i in range(n):
        ring = (np.outer(ct, normals[i]) + np.outer(st, binorm[i])) * radii[i] + points[i]
        verts[i * sides:(i + 1) * sides] = ring

    faces = []
    for i in range(n - 1):
        a = i * sides
        b = (i + 1) * sides
        for j in range(sides):
            j2 = (j + 1) % sides
            faces.append([a + j, a + j2, b + j2])
            faces.append([a + j, b + j2, b + j])

    verts = list(verts)
    if cap_start:
        c = len(verts)
        verts.append(points[0])
        for j in range(sides):
            faces.append([c, (j + 1) % sides, j])
    if cap_end:
        c = len(verts)
        verts.append(points[-1])
        base = (n - 1) * sides
        for j in range(sides):
            faces.append([c, base + j, base + (j + 1) % sides])

    return np.asarray(verts), np.asarray(faces, dtype=np.int64)


def _build_trunk(rng, sides, segs):
    lean_dir = rng.uniform(0.0, 2.0 * np.pi)
    lx, lz = np.cos(lean_dir), np.sin(lean_dir)
    lean_amt = LEAN_AMP * rng.uniform(0.7, 1.2)
    sway = rng.uniform(0.4, 0.9)

    t = np.linspace(0.0, 1.0, segs)
    y = t * TRUNK_HEIGHT
    bend = lean_amt * (np.sin(t * np.pi * 0.5) * 0.8 + np.sin(t * np.pi * sway) * 0.2)
    pts = np.column_stack([lx * bend, y, lz * bend])

    radii = TRUNK_TOP_R + (TRUNK_BASE_R - TRUNK_TOP_R) * (1.0 - t) ** 1.4
    flare = 1.0 + (BASAL_FLARE - 1.0) * np.clip(1.0 - t / FLARE_FRAC, 0.0, 1.0) ** 2
    rings = 1.0 + 0.035 * np.sin(t * TRUNK_HEIGHT * 7.0)   # leaf-scar nodes
    radii = radii * flare * rings

    verts, faces = _tube(pts, radii, sides, cap_start=True, cap_end=True)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # LIGHT tints: gentle ground-contact AO at base, brighter up the stem
    ny = (mesh.vertices[:, 1] / TRUNK_HEIGHT).clip(0, 1)
    base_c = np.array([175, 165, 150])
    top_c = np.array([205, 200, 188])
    col = (base_c[None] * (1 - ny[:, None]) + top_c[None] * ny[:, None])
    col += rng.uniform(-8, 8, col.shape)
    mesh.visual.vertex_colors = np.clip(col, 0, 255).astype(np.uint8)

    return mesh, pts[-1]


def _build_crown(rng, hub, density):
    subdiv = {"high": 2, "med": 1, "low": 0}[density]
    s = trimesh.creation.icosphere(subdivisions=subdiv, radius=1.0)
    v = s.vertices.copy()
    v[:, 0] *= 0.34
    v[:, 2] *= 0.34
    v[:, 1] *= 0.30
    v[:, 1] += 0.10
    v += hub
    mesh = trimesh.Trimesh(vertices=v, faces=s.faces.copy(), process=False)
    col = np.array([168, 172, 138]) + rng.uniform(-10, 10, (len(v), 3))  # light tint
    mesh.visual.vertex_colors = np.clip(col, 0, 255).astype(np.uint8)
    return mesh


def _frond_path(hub, az, pitch0, length, droop, samples):
    fx, fz = np.cos(az), np.sin(az)
    s = np.linspace(0.0, 1.0, samples)
    horiz = length * s * np.cos(pitch0)
    y = hub[1] + length * np.sin(pitch0) * s - droop * length * s ** 2
    x = hub[0] + fx * horiz
    z = hub[2] + fz * horiz
    return np.column_stack([x, y, z])


def _build_fronds(rng, hub, n_fronds, leaflets, rachis_sides, rachis_segs):
    base_az = np.linspace(0.0, 2.0 * np.pi, n_fronds, endpoint=False)
    base_az = base_az + rng.uniform(-0.18, 0.18, n_fronds)
    # rounded, full crown: most fronds reach up/out, only the lowest gently droop
    pitch = np.linspace(1.35, -0.12, n_fronds) + rng.uniform(-0.10, 0.10, n_fronds)
    rng.shuffle(pitch)

    r_verts, r_faces = [], []
    c_verts, c_faces, c_cols = [], [], []
    rv_off = 0
    cv_off = 0

    # LIGHT multiplicative tints (modulate the green atlas, never crush it)
    inner_c = np.array([122, 140, 96])     # shaded interior
    outer_c = np.array([208, 222, 162])    # sunlit tips

    for f in range(n_fronds):
        az = base_az[f]
        p0 = pitch[f]
        length = FROND_LEN * rng.uniform(0.85, 1.12)
        # gentler arch so tips don't hang straight down
        droop = 0.32 + 0.20 * max(0.0, np.cos(p0))
        path = _frond_path(hub, az, p0, length, droop, rachis_segs)

        # brightness by frond height: upper sunlit brighter; lower older darker/yellower
        pn = np.clip((p0 + 0.2) / 1.6, 0.0, 1.0)
        fbright = 0.80 + 0.30 * pn
        fwarm = np.array([1.0, 1.0, 1.0]) + np.array([0.06, 0.02, -0.10]) * (1.0 - pn)

        tt = np.linspace(0.0, 1.0, rachis_segs)
        rad = 0.022 * (1.0 - 0.92 * tt) + 0.003   # thin spine (hidden by leaflets)
        rv, rf = _tube(path, rad, rachis_sides, cap_start=False, cap_end=True)
        r_verts.append(rv)
        r_faces.append(rf + rv_off)
        rv_off += len(rv)

        n_pairs = max(2, leaflets // 2)
        ss = np.linspace(0.04, 0.99, n_pairs)
        for sc in ss:
            i = int(sc * (rachis_segs - 1))
            i = min(i, rachis_segs - 2)
            p = path[i]
            tan = _norm(path[i + 1] - path[i])
            side = _norm(np.cross(tan, UP))
            # long, wide leaflets that overlap and bury the bare rachis spine
            lf = length * 0.42 * (0.55 + 0.45 * np.sin(np.pi * sc ** 0.85))
            lf *= rng.lognormal(0.0, 0.10)
            halfw = lf * 0.075 + 0.010

            shade = sc
            for sign in (1.0, -1.0):
                d = side * sign
                d = _norm(d + tan * 0.30 - UP * 0.20)
                d = _rodrigues(d, tan, rng.uniform(-0.40, 0.40))
                tip = p + d * lf - UP * lf * 0.14
                a = tan * halfw
                v0 = p - a
                v1 = p + a
                v2 = tip + a
                v3 = tip - a
                c_verts.extend([v0, v1, v2, v3])
                c_faces.append([cv_off, cv_off + 1, cv_off + 2])
                c_faces.append([cv_off, cv_off + 2, cv_off + 3])
                cv_off += 4
                col = (inner_c * (1 - shade) + outer_c * shade) * fbright * fwarm
                col = col + rng.uniform(-10, 10, 3)
                c_cols.extend([col, col, col, col])

    rachis = trimesh.Trimesh(vertices=np.vstack(r_verts),
                             faces=np.vstack(r_faces), process=False)
    rachis.visual.vertex_colors = np.tile(
        np.array([150, 160, 112], dtype=np.uint8), (len(rachis.vertices), 1))

    canopy = trimesh.Trimesh(vertices=np.asarray(c_verts),
                             faces=np.asarray(c_faces, dtype=np.int64), process=False)
    canopy.visual.vertex_colors = np.clip(np.asarray(c_cols), 0, 255).astype(np.uint8)

    return rachis, canopy


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    if density not in ("high", "med", "low"):
        density = "high"
    rng = np.random.default_rng(seed)

    params = {
        "high": dict(fronds=28, leaflets=72, t_sides=14, t_segs=26, r_sides=5, r_segs=12),
        "med":  dict(fronds=20, leaflets=48, t_sides=10, t_segs=18, r_sides=4, r_segs=9),
        "low":  dict(fronds=12, leaflets=22, t_sides=6,  t_segs=10, r_sides=3, r_segs=6),
    }[density]

    trunk, hub = _build_trunk(rng, params["t_sides"], params["t_segs"])
    crown = _build_crown(rng, hub, density)
    rachis, canopy = _build_fronds(rng, hub, params["fronds"], params["leaflets"],
                                   params["r_sides"], params["r_segs"])

    for m in (trunk, crown, rachis):
        m.fix_normals()
    _ = canopy.face_normals

    scene = trimesh.Scene()
    scene.add_geometry(trunk, geom_name="trunk")
    scene.add_geometry(crown, geom_name="crown")
    scene.add_geometry(rachis, geom_name="fronds")
    scene.add_geometry(canopy, geom_name="canopy")

    lo = scene.bounds[0]
    scene.apply_translation([-((scene.bounds[0][0] + scene.bounds[1][0]) * 0.5),
                             -lo[1],
                             -((scene.bounds[0][2] + scene.bounds[1][2]) * 0.5)])
    return scene


# ============================================================================
# PHOTO SAMPLING
# ============================================================================
def load_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float64)


def robust_sample(arr, centers, half_frac=0.018):
    """Median color of several small patches, discarding outliers (background)."""
    H, W, _ = arr.shape
    hx = max(2, int(half_frac * W))
    hy = max(2, int(half_frac * H))
    cols = []
    for cx, cy in centers:
        x = int(np.clip(cx, 0, 1) * (W - 1))
        y = int(np.clip(cy, 0, 1) * (H - 1))
        x0, x1 = max(0, x - hx), min(W, x + hx + 1)
        y0, y1 = max(0, y - hy), min(H, y + hy + 1)
        patch = arr[y0:y1, x0:x1].reshape(-1, 3)
        if len(patch):
            cols.append(np.median(patch, axis=0))
    cols = np.asarray(cols)
    med = np.median(cols, axis=0)
    d = np.linalg.norm(cols - med, axis=1)
    keep = cols[d <= (np.median(d) * 2.5 + 28.0)]
    if len(keep) == 0:
        keep = cols
    return np.median(keep, axis=0)


def _clip8(c):
    return np.clip(c, 0, 255).astype(np.uint8)


# ============================================================================
# TILEABLE NOISE (periodic by construction -> no seams, no roll+blur)
# ============================================================================
def _tileable_noise(size, cells_x, cells_y, rng):
    grid = rng.random((cells_y, cells_x))
    ys = np.linspace(0, cells_y, size, endpoint=False)
    xs = np.linspace(0, cells_x, size, endpoint=False)
    y0 = np.floor(ys).astype(int) % cells_y
    x0 = np.floor(xs).astype(int) % cells_x
    y1 = (y0 + 1) % cells_y
    x1 = (x0 + 1) % cells_x
    fy = ys - np.floor(ys)
    fx = xs - np.floor(xs)
    fy = fy * fy * (3 - 2 * fy)
    fx = fx * fx * (3 - 2 * fx)
    g00 = grid[np.ix_(y0, x0)]
    g01 = grid[np.ix_(y0, x1)]
    g10 = grid[np.ix_(y1, x0)]
    g11 = grid[np.ix_(y1, x1)]
    fx2 = fx[None, :]
    fy2 = fy[:, None]
    top = g00 * (1 - fx2) + g01 * fx2
    bot = g10 * (1 - fx2) + g11 * fx2
    return top * (1 - fy2) + bot * fy2


def _fractal(size, cells_x, cells_y, octaves, rng):
    out = np.zeros((size, size))
    amp = 1.0
    tot = 0.0
    for o in range(octaves):
        out += amp * _tileable_noise(size, cells_x * (2 ** o), cells_y * (2 ** o), rng)
        tot += amp
        amp *= 0.5
    return out / tot


def make_bark(size, base_col, rng, n_rings=9, ring_strength=0.22, fiber=0.30):
    """Banded, fibrous bark: vertical fibers + horizontal leaf-scar rings."""
    fibers = _fractal(size, 48, 5, 4, rng)            # vertical streaks
    mottle = _fractal(size, 6, 6, 3, rng)
    yy = np.linspace(0, 1, size, endpoint=False)[:, None]
    rings = 0.5 + 0.5 * np.sin(2 * np.pi * n_rings * yy)
    rings = rings ** 3                                 # sharp scar lines
    bright = 1.0 + fiber * (fibers - 0.5) + 0.10 * (mottle - 0.5)
    bright = bright * (1.0 - ring_strength * rings)
    bright = np.clip(bright, 0.55, 1.45)
    img = base_col[None, None, :] * bright[:, :, None]
    return _clip8(img)


def make_woodgreen(size, base_col, rng):
    fibers = _fractal(size, 40, 6, 4, rng)
    bright = np.clip(1.0 + 0.28 * (fibers - 0.5), 0.6, 1.4)
    img = base_col[None, None, :] * bright[:, :, None]
    return _clip8(img)


def make_crown_tex(size, base_col, rng):
    """Opaque green-brown fibrous hub of bundled leaf bases."""
    fibers = _fractal(size, 30, 18, 4, rng)
    mottle = _fractal(size, 8, 8, 3, rng)
    bright = np.clip(1.0 + 0.32 * (fibers - 0.5) + 0.12 * (mottle - 0.5), 0.6, 1.4)
    img = base_col[None, None, :] * bright[:, :, None]
    return _clip8(img)


def albedo_to_normal(rgb_uint8, strength=2.2):
    a = rgb_uint8.astype(np.float64) / 255.0
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    height = 1.0 - lum
    gy, gx = np.gradient(height)
    n = np.dstack([-gx * strength, -gy * strength, np.ones_like(height)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return _clip8((n * 0.5 + 0.5) * 255.0)


# ============================================================================
# FOLIAGE ATLAS  (4x4 distinct cluster tiles, drawn leaf silhouettes)
# ============================================================================
def _draw_blade(draw, root, ang, length, width, color, curve, segs=7):
    dirx, diry = np.cos(ang), np.sin(ang)
    px, py = -diry, dirx
    left, right = [], []
    for i in range(segs + 1):
        t = i / segs
        off = curve * length * np.sin(np.pi * t)
        cx = root[0] + dirx * length * t + px * off
        cy = root[1] + diry * length * t + py * off
        w = width * (1.0 - t) ** 0.7
        left.append((cx + px * w, cy + py * w))
        right.append((cx - px * w, cy - py * w))
    poly = left + right[::-1]
    draw.polygon([(float(a), float(b)) for a, b in poly], fill=color)


def make_foliage_atlas(atlas_px, base_green, rng):
    tile = atlas_px // 4
    ss = 4
    tss = tile * ss
    atlas = Image.new("RGBA", (atlas_px, atlas_px), (0, 0, 0, 0))

    for row in range(4):
        for col in range(4):
            # sunlit tiles (top rows) brighter/warmer; shaded (bottom) darker/cooler
            f = 1.18 - 0.20 * row
            warm = np.array([14.0, 6.0, -10.0]) * (1.5 - row) / 1.5
            cool = np.array([-6.0, 0.0, 8.0]) * row / 3.0
            tile_base = np.clip(base_green * f + warm + cool, 12, 250)

            timg = Image.new("RGBA", (tss, tss), (0, 0, 0, 0))
            d = ImageDraw.Draw(timg)

            n_clusters = int(rng.integers(2, 4))
            for _ in range(n_clusters):
                rx = rng.uniform(0.25, 0.75) * tss
                ry = rng.uniform(0.80, 1.02) * tss
                spread = rng.uniform(0.7, 1.25)
                center_ang = -np.pi / 2 + rng.uniform(-0.5, 0.5)
                n_blades = int(rng.integers(16, 30))
                for _b in range(n_blades):
                    ang = center_ang + rng.uniform(-1.0, 1.0) * spread
                    length = rng.uniform(0.45, 0.95) * tss
                    width = rng.uniform(0.012, 0.030) * tss
                    curve = rng.uniform(-0.12, 0.12)
                    jit = rng.uniform(-18, 18, 3)
                    tipf = rng.uniform(0.95, 1.18)   # tips a touch brighter
                    # NB: distinct name -- do NOT shadow the `col` loop variable
                    bcol = _clip8(tile_base * tipf + jit)
                    color = (int(bcol[0]), int(bcol[1]), int(bcol[2]), 255)
                    _draw_blade(d, (rx, ry), ang, length, width, color, curve)

            timg = timg.resize((tile, tile), Image.LANCZOS)
            atlas.paste(timg, (col * tile, row * tile), timg)

    return atlas


# ============================================================================
# UV HELPERS + MATERIAL ATTACH
# ============================================================================
def cylindrical_uv(verts, u_tiles=2.0, v_tiles=3.0):
    cx = (verts[:, 0].min() + verts[:, 0].max()) * 0.5
    cz = (verts[:, 2].min() + verts[:, 2].max()) * 0.5
    ang = np.arctan2(verts[:, 2] - cz, verts[:, 0] - cx)
    u = (ang / (2 * np.pi) + 0.5) * u_tiles
    ymin, ymax = verts[:, 1].min(), verts[:, 1].max()
    v = (verts[:, 1] - ymin) / max(1e-6, (ymax - ymin)) * v_tiles
    return np.column_stack([u, v]).astype(np.float64)


def card_atlas_uv(verts, rng, pad=0.004):
    """One atlas tile per card (4 verts -> base v0,v1 ; tip v2,v3), random rot."""
    n = len(verts)
    uv = np.zeros((n, 2))
    n_cards = n // 4
    for k in range(n_cards):
        ti = int(rng.integers(0, 16))
        rot = int(rng.integers(0, 4))
        rc, cc = ti // 4, ti % 4
        u0, u1 = cc / 4 + pad, (cc + 1) / 4 - pad
        # image row 0 is top -> flip to gl v
        vb = 1.0 - (rc / 4) - pad          # tile top  (tip)
        va = 1.0 - ((rc + 1) / 4) + pad    # tile bot  (base)
        corners = [(u0, va), (u1, va), (u1, vb), (u0, vb)]  # v0,v1,v2,v3
        corners = corners[rot:] + corners[:rot]
        for i in range(4):
            uv[4 * k + i] = corners[i]
    return uv


def set_texture(geom, uv, image, roughness, mask=False, normal=None):
    vcols = None
    try:
        vc = np.asarray(geom.visual.vertex_colors)
        if vc.ndim == 2 and len(vc) == len(geom.vertices):
            vcols = vc.copy().astype(np.uint8)
    except Exception:
        vcols = None

    kwargs = dict(baseColorTexture=image, metallicFactor=0.0,
                  roughnessFactor=roughness)
    if normal is not None:
        kwargs["normalTexture"] = normal
    if mask:
        kwargs["alphaMode"] = "MASK"
        kwargs["alphaCutoff"] = 0.45
        kwargs["doubleSided"] = True
    mat = PBRMaterial(**kwargs)

    geom.visual = TextureVisuals(uv=uv.astype(np.float64), material=mat)
    if vcols is not None:
        if vcols.shape[1] == 3:
            vcols = np.concatenate([vcols, np.full((len(vcols), 1), 255, np.uint8)], axis=1)
        geom.visual.vertex_attributes["color"] = vcols


# ============================================================================
# TEXTURE + ASSEMBLE
# ============================================================================
def texture_scene(scene, image_path, seed):
    arr = load_rgb(image_path)
    trng = np.random.default_rng(seed * 2 + 17)

    # --- sample colors WELL INSIDE the silhouette (looked at the photo) ---
    # trunk: lower-centre vertical column
    trunk_col = robust_sample(arr, [
        (0.50, 0.70), (0.50, 0.78), (0.49, 0.85), (0.51, 0.80),
        (0.50, 0.90), (0.485, 0.74),
    ], half_frac=0.012)
    # foliage: dense green crown, mid/inner
    foliage_col = robust_sample(arr, [
        (0.50, 0.30), (0.42, 0.26), (0.58, 0.30), (0.46, 0.40),
        (0.55, 0.22), (0.38, 0.34), (0.62, 0.36), (0.50, 0.44),
    ], half_frac=0.016)
    # guard: if foliage didn't read green, nudge toward a plausible palm green
    if not (foliage_col[1] >= foliage_col[0] and foliage_col[1] >= foliage_col[2]):
        foliage_col = np.array([96.0, 128.0, 64.0])

    crown_col = np.clip(foliage_col * 0.7 + np.array([20, 8, -6]), 20, 200)
    stem_col = np.clip(foliage_col * 0.78 + np.array([18, 10, -4]), 20, 210)

    # --- build textures (bark/others >=512 ; foliage atlas 1024) ---
    bark_img = make_bark(1024, trunk_col, trng)
    bark_pil = Image.fromarray(bark_img, "RGB")
    bark_normal = Image.fromarray(albedo_to_normal(bark_img), "RGB")

    crown_pil = Image.fromarray(make_crown_tex(512, crown_col, trng), "RGB")
    stem_pil = Image.fromarray(make_woodgreen(512, stem_col, trng), "RGB")
    atlas_pil = make_foliage_atlas(1024, foliage_col, trng)

    g = scene.geometry

    # trunk: cylindrical UV, matte woody bark + derived normal
    trunk = g["trunk"]
    set_texture(trunk, cylindrical_uv(trunk.vertices, 2.0, 4.0),
                bark_pil, roughness=0.9, normal=bark_normal)

    # crown hub: opaque green-brown
    crown = g["crown"]
    set_texture(crown, cylindrical_uv(crown.vertices, 2.0, 1.0),
                crown_pil, roughness=0.85)

    # rachis / frond stems: green wood
    fronds = g["fronds"]
    set_texture(fronds, cylindrical_uv(fronds.vertices, 1.0, 4.0),
                stem_pil, roughness=0.85)

    # canopy leaflet cards: foliage atlas, alpha MASK, double-sided
    canopy = g["canopy"]
    set_texture(canopy, card_atlas_uv(canopy.vertices, trng),
                atlas_pil, roughness=0.8, mask=True)

    return scene


def main():
    ap = argparse.ArgumentParser(description="Coconut-palm textured GLB generator")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        scene = build_mesh(args.seed, args.density)
        scene = texture_scene(scene, args.image, args.seed)
        data = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(data)
        tris = sum(len(m.faces) for m in scene.geometry.values())
        print(f"OK  density={args.density} seed={args.seed} "
              f"triangles={tris} -> {args.output}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())