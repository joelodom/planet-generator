"""
Procedural textured boxwood-shrub generator.

Builds a compact, densely-foliaged mounding evergreen shrub as procedural
geometry (a continuous mantle of clumped leaf cards over a small hidden woody
base), derives tileable materials by SAMPLING the reference photo, applies
per-surface UVs and PBR materials, bakes sun/shade vertex colors, and exports
a textured GLB.

Dependencies: numpy, trimesh, Pillow (PIL), stdlib only.
Deterministic for a given --seed.

CLI:
    python thisscript.py --image PATH --seed INT \
        --density {high,med,low} --output OUT.glb
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter


# ============================================================================
# GEOMETRY
# ============================================================================
# Measured proportions (off the reference; photo content aspect ~1.28 W/H).
CROWN_WIDTH_M     = 0.90    # overall foliage diameter in X/Z (meters)
HEIGHT_OVER_WIDTH = 0.78    # -> width/height ~1.28, a low rounded mound
CUSHION_BULGE     = 0.06    # mild extra width low down (pillow shape)

# Lobed-ellipsoid foliage envelope (half-extents). Tuned so the textured
# silhouette lands near the photo's 1.28 aspect and drapes to the ground.
_HALF_W  = 0.38    # shell horizontal half-radius (pre-cushion, pre-cards)
_RV      = 0.34    # shell vertical half-radius
_CY      = 0.35    # shell center height -> foliage drapes to near y=0
_TRUNK_H = 0.10    # short woody stub; only its base peeks below the canopy

_DENSITY = {
    "high": dict(n_clumps=30, cards_lo=80, cards_hi=110,
                 trunk_sides=14, branch_sides=7, n_branches=5, lobes=5),
    "med":  dict(n_clumps=18, cards_lo=45, cards_hi=65,
                 trunk_sides=10, branch_sides=6, n_branches=4, lobes=4),
    "low":  dict(n_clumps=10, cards_lo=24, cards_hi=38,
                 trunk_sides=7,  branch_sides=5, n_branches=3, lobes=3),
}


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _orthonormal(n):
    """Right-handed frame (u, v, n) with u x v = n, given a unit-ish normal n."""
    n = _normalize(np.asarray(n, float))
    up = np.array([0.0, 1.0, 0.0])
    if abs(float(n @ up)) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
    u = _normalize(np.cross(up, n))
    v = np.cross(n, u)
    return u, v, n


def _tube(points, radii, sections):
    """Tapered tube along a polyline using parallel-transport frames. Capped."""
    points = np.asarray(points, float)
    radii = np.asarray(radii, float)
    n = len(points)

    tang = np.zeros((n, 3))
    tang[1:-1] = points[2:] - points[:-2]
    tang[0] = points[1] - points[0]
    tang[-1] = points[-1] - points[-2]
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12)

    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(tang[0] @ ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    nrm = np.zeros((n, 3))
    bin_ = np.zeros((n, 3))
    n0 = _normalize(ref - tang[0] * float(tang[0] @ ref))
    nrm[0] = n0
    bin_[0] = np.cross(tang[0], n0)
    for i in range(1, n):
        v = nrm[i - 1] - tang[i] * float(tang[i] @ nrm[i - 1])
        if np.linalg.norm(v) < 1e-7:
            v = bin_[i - 1] - tang[i] * float(tang[i] @ bin_[i - 1])
        nrm[i] = _normalize(v)
        bin_[i] = np.cross(tang[i], nrm[i])

    ang = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]

    rings = []
    for i in range(n):
        rings.append(points[i] + radii[i] * (ca * nrm[i] + sa * bin_[i]))
    verts = np.vstack(rings)

    faces = []
    for i in range(n - 1):
        a, b = i * sections, (i + 1) * sections
        for j in range(sections):
            j2 = (j + 1) % sections
            faces.append([a + j, a + j2, b + j2])
            faces.append([a + j, b + j2, b + j])

    cstart = len(verts)
    cend = cstart + 1
    verts = np.vstack([verts, points[0][None, :], points[-1][None, :]])
    last = (n - 1) * sections
    for j in range(sections):
        j2 = (j + 1) % sections
        faces.append([cstart, j2, j])
        faces.append([cend, last + j, last + j2])

    return trimesh.Trimesh(vertices=verts, faces=np.asarray(faces, np.int64),
                           process=True)


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    if density not in _DENSITY:
        density = "high"
    cfg = _DENSITY[density]

    # per-instance variety (uniform scale keeps the photo proportions intact)
    inst_scale = float(rng.uniform(0.90, 1.08))
    rxj = float(rng.uniform(0.96, 1.04))
    ryj = float(rng.uniform(0.97, 1.05))
    rzj = float(rng.uniform(0.96, 1.04))
    center = np.array([0.0, _CY, 0.0])
    crown = 2.0 * _HALF_W

    # low-frequency horizontal lobes -> irregular dome, not a clean sphere
    L = cfg["lobes"]
    freqs = rng.integers(2, 6, size=L).astype(float)
    amps = rng.uniform(0.03, 0.08, size=L)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=L)

    def lobe(theta):
        return 1.0 + float(np.sum(amps * np.cos(freqs * theta + phases)))

    def shell_point(d):
        d = _normalize(d)
        theta = np.arctan2(d[2], d[0])
        lf = lobe(theta)
        hs = 1.0 + CUSHION_BULGE * max(0.0, -d[1])      # bulge low -> cushion
        return center + np.array([_HALF_W * hs * lf * rxj * d[0],
                                  _RV * ryj * d[1],
                                  _HALF_W * hs * lf * rzj * d[2]])

    def rand_dir(elev_lo, elev_hi):
        az = rng.uniform(0.0, 2.0 * np.pi)
        dy = rng.uniform(elev_lo, elev_hi)
        r = np.sqrt(max(0.0, 1.0 - dy * dy))
        return np.array([r * np.cos(az), dy, r * np.sin(az)])

    # ---- foliage: a CONTINUOUS mantle of cap-filling clumps ----------------
    cverts = []
    cfaces = []
    state = {"base": 0}
    deg18 = np.deg2rad(18.0)

    def spawn_clump(c_center, d, n_cards, cap_r, card_h):
        u, v, nd = _orthonormal(d)            # nd = outward shell normal
        for _ in range(n_cards):
            # spread cards across a tangent CAP (a surface patch, not a ball)
            a2 = rng.uniform(0.0, 2.0 * np.pi)
            rr = cap_r * np.sqrt(rng.random())
            tang = (np.cos(a2) * u + np.sin(a2) * v) * rr
            noff = nd * rng.uniform(-0.35, 0.55) * cap_r
            cpos = c_center + tang + noff

            # card faces outward (radial), tilted up to ~18 deg
            nrm = _normalize(cpos - center)
            uu, vv, nrm = _orthonormal(nrm)
            t1 = np.tan(rng.uniform(-1.0, 1.0) * deg18)
            t2 = np.tan(rng.uniform(-1.0, 1.0) * deg18)
            nrm = _normalize(nrm + t1 * uu + t2 * vv)
            uu, vv, nrm = _orthonormal(nrm)

            phi = rng.uniform(0.0, 2.0 * np.pi)
            U = uu * np.cos(phi) + vv * np.sin(phi)
            V = -uu * np.sin(phi) + vv * np.cos(phi)
            s = float(np.exp(rng.normal(0.0, 0.25)))
            hu = card_h * s
            hv = hu * float(rng.uniform(1.0, 1.3))      # ovate

            b = state["base"]
            cverts.extend((cpos - U * hu - V * hv, cpos + U * hu - V * hv,
                           cpos + U * hu + V * hv, cpos - U * hu + V * hv))
            cfaces.append([b, b + 1, b + 2])
            cfaces.append([b, b + 2, b + 3])
            state["base"] = b + 4

    clump_r = _HALF_W * 0.42
    card_h = _HALF_W * 0.135
    n_clumps = cfg["n_clumps"]

    # evenly distribute clump centers over the dome (spiral) so the surface is
    # continuous from the soft crown down to the draping lower sides
    golden = np.pi * (3.0 - np.sqrt(5.0))
    dy_top, dy_bot = 0.97, -0.88
    for i in range(n_clumps):
        t = (i + 0.5) / n_clumps
        dy = dy_top - t * (dy_top - dy_bot) + rng.uniform(-0.05, 0.05)
        dy = float(np.clip(dy, -0.93, 0.98))
        az = i * golden + rng.uniform(-0.30, 0.30)
        r = np.sqrt(max(0.0, 1.0 - dy * dy))
        d = np.array([r * np.cos(az), dy, r * np.sin(az)])
        f = float(rng.uniform(0.96, 1.03))               # lumpy, not a clean shell
        c_center = center + f * (shell_point(d) - center)
        n_cards = int(rng.integers(cfg["cards_lo"], cfg["cards_hi"] + 1))
        spawn_clump(c_center, d, n_cards, clump_r, card_h)

    # a few interior fill clumps so the mass reads solid (no see-through edges)
    for _ in range(max(2, n_clumps // 5)):
        d = rand_dir(-0.2, 0.9)
        f = float(rng.uniform(0.45, 0.72))
        c_center = center + f * (shell_point(d) - center)
        n_cards = int(rng.integers(cfg["cards_lo"] // 2, cfg["cards_hi"] // 2 + 1))
        spawn_clump(c_center, d, n_cards, clump_r * 1.25, card_h)

    canopy = trimesh.Trimesh(vertices=np.asarray(cverts, float),
                             faces=np.asarray(cfaces, np.int64),
                             process=False)

    # ---- wood: small flared stub + a few SHORT internal branches -----------
    # Everything stays deep inside the canopy (tips at <= 0.55 of the shell),
    # so no bare twig ever protrudes; only the stub base peeks at the bottom.
    tubes = []
    r_top = 0.028
    trunk_pts = np.array([[0.0, 0.0, 0.0],
                          [rng.uniform(-0.008, 0.008), _TRUNK_H * 0.25,
                           rng.uniform(-0.008, 0.008)],
                          [rng.uniform(-0.015, 0.015), _TRUNK_H,
                           rng.uniform(-0.015, 0.015)]])
    trunk_rad = np.array([r_top * 1.6, r_top * 1.18, r_top])   # basal flare
    tubes.append(_tube(trunk_pts, trunk_rad, cfg["trunk_sides"]))

    trunk_top = trunk_pts[-1]
    for _ in range(cfg["n_branches"]):
        d = rand_dir(0.25, 0.90)
        tip = center + float(rng.uniform(0.42, 0.55)) * (shell_point(d) - center)
        start = trunk_top + (tip - trunk_top) * 0.05
        mid = 0.5 * (start + tip) + _normalize(rng.normal(size=3)) * (crown * 0.02)
        pts = np.array([start, mid, tip])
        rad = np.array([0.013, 0.008, 0.004])
        tubes.append(_tube(pts, rad, cfg["branch_sides"]))

    branches = trimesh.util.concatenate(tubes)

    # ---- place on ground ---------------------------------------------------
    canopy.vertices *= inst_scale
    branches.vertices *= inst_scale

    allv = np.vstack([canopy.vertices, branches.vertices])
    mn = allv.min(axis=0)
    mx = allv.max(axis=0)
    offset = np.array([-(mn[0] + mx[0]) * 0.5, -mn[1], -(mn[2] + mx[2]) * 0.5])
    canopy.vertices += offset
    branches.vertices += offset

    canopy.fix_normals()
    branches.fix_normals()

    scene = trimesh.Scene()
    scene.add_geometry(canopy, geom_name="canopy")
    scene.add_geometry(branches, geom_name="branches")
    return scene


# ============================================================================
# PHOTO SAMPLING  (always sample real pixels; reject grey background)
# ============================================================================
_LUM = np.array([0.299, 0.587, 0.114])


def _load_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32)


def _crop(arr, box):
    """box = (x0, y0, x1, y1) normalized; returns sub-array."""
    h, w = arr.shape[:2]
    x0 = int(np.clip(box[0], 0, 1) * w)
    x1 = int(np.clip(box[2], 0, 1) * w)
    y0 = int(np.clip(box[1], 0, 1) * h)
    y1 = int(np.clip(box[3], 0, 1) * h)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    if x1 - x0 < 2:
        x1 = min(w, x0 + 2)
    if y1 - y0 < 2:
        y1 = min(h, y0 + 2)
    return arr[y0:y1, x0:x1]


def _delight(rgb):
    """Divide out broad lighting; clamp gain to [0.6, 1.6]. Returns float RGB."""
    lum = rgb @ _LUM
    radius = max(8, int(min(rgb.shape[:2]) / 6))
    pil = Image.fromarray(np.clip(lum, 0, 255).astype(np.uint8))
    blur = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius)), float) + 1e-3
    gain = np.clip(float(lum.mean()) / blur, 0.6, 1.6)
    return np.clip(rgb * gain[..., None], 0, 255)


def _green_tones(arr):
    """(shadow, mid, highlight) green RGB sampled from inside the shrub body."""
    crop = _delight(_crop(arr, (0.30, 0.22, 0.70, 0.74)))   # well inside silhouette
    px = crop.reshape(-1, 3)
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    mask = (g > r + 4) & (g > b + 4)                         # grey backdrop rejected
    sel = px[mask] if mask.sum() > 200 else px
    lum = sel @ _LUM
    s = sel[np.argsort(lum)]

    def pct(p):
        return s[min(len(s) - 1, int(p * len(s)))].astype(float).copy()

    shadow, mid, high = pct(0.20), pct(0.52), pct(0.86)
    # keep a healthy, slightly brighter range with a faint lime in the highlight
    shadow = np.clip(shadow * 0.92, 0, 255)
    mid = np.clip(mid * 1.04, 0, 255)
    high = np.clip(high * 1.10 + np.array([3.0, 9.0, 0.0]), 0, 255)
    return shadow, mid, high


def _brown_tone(arr):
    """Median grey-brown of the exposed stems at the lower centre (with fallback)."""
    crop = _delight(_crop(arr, (0.38, 0.80, 0.64, 0.97)))
    px = crop.reshape(-1, 3)
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    mask = (r > g + 5) & (g >= b - 3) & (r > b + 8) & (r < 205)
    sel = px[mask]
    if len(sel) < 30:
        return np.array([96.0, 74.0, 56.0])                 # warm grey-brown fallback
    return np.median(sel, axis=0)


# ============================================================================
# TEXTURE SYNTHESIS
# ============================================================================
def _ellipse_poly(cx, cy, a, b, ang, steps=16):
    t = np.linspace(0.0, 2.0 * np.pi, steps, endpoint=False)
    x = a * np.cos(t)
    y = b * np.sin(t)
    ca, sa = np.cos(ang), np.sin(ang)
    X = cx + x * ca - y * sa
    Y = cy + x * sa + y * ca
    return list(zip(X.tolist(), Y.tolist()))


def _draw_leaf_tile(size, tones, bright, warm, rng):
    """A dense, mostly-opaque cluster tile of ovate leaves on transparent bg."""
    shadow, mid, high = tones
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    warm_vec = np.array([1.0 + warm, 1.0 + 0.3 * warm, 1.0 - warm])

    n_leaves = int(rng.integers(36, 48))
    for _ in range(n_leaves):
        t = float(np.clip(bright - 0.45 + rng.uniform(-0.20, 0.25), 0.0, 1.0))
        col = shadow * (1.0 - t) + high * t
        col = col * warm_vec + rng.uniform(-7.0, 7.0, 3)
        col = np.clip(col, 0, 255).astype(int)

        cx = rng.uniform(0.06, 0.94) * size
        cy = rng.uniform(0.06, 0.94) * size
        a = rng.uniform(0.13, 0.23) * size
        b = a * rng.uniform(0.58, 0.82)        # ovate / spoon-shaped
        ang = rng.uniform(0.0, np.pi)
        d.polygon(_ellipse_poly(cx, cy, a, b, ang),
                  fill=(int(col[0]), int(col[1]), int(col[2]), 255))

        # paler lime highlight catching the rounded tip on some leaves
        if rng.random() < 0.5:
            hl = np.clip(high * warm_vec + 18.0, 0, 255).astype(int)
            hx = cx + np.cos(ang) * a * 0.40
            hy = cy + np.sin(ang) * a * 0.40
            d.polygon(_ellipse_poly(hx, hy, a * 0.34, b * 0.34, ang),
                      fill=(int(hl[0]), int(hl[1]), int(hl[2]), 255))
    return img


def _foliage_atlas(tones, rng, ss=4, tile_px=256):
    """4x4 atlas of distinct cluster tiles; top rows sunlit, bottom rows shaded."""
    atlas = Image.new("RGBA", (tile_px * 4, tile_px * 4), (0, 0, 0, 0))
    for row in range(4):
        for col in range(4):
            bright = 1.15 - 0.30 * (row / 3.0)      # 1.15 (top) .. 0.85 (bottom)
            warm = 0.06 * (1.0 - row / 3.0) - 0.03  # warmer up top, cooler below
            tile = _draw_leaf_tile(tile_px * ss, tones, bright, warm, rng)
            tile = tile.resize((tile_px, tile_px), Image.LANCZOS)
            atlas.paste(tile, (col * tile_px, row * tile_px))
    return atlas


def _bark_texture(brown, rng, size=512):
    """Seamless grey-brown bark swatch with vertical grain and a real value range.

    Built from integer-frequency sinusoids so it tiles with no seam; lightest
    tone is ~2.5x the darkest so the woody form still reads.
    """
    light = np.clip(brown * 1.5, 0, 255)
    dark = np.clip(brown * 0.6, 0, 255)

    coord = np.linspace(0.0, 2.0 * np.pi, size, endpoint=False)
    gx, gy = np.meshgrid(coord, coord)          # periodic both axes -> tileable

    grain = np.zeros((size, size), np.float32)
    for k in range(1, 13):                       # mostly across-width -> vertical streaks
        grain += (rng.uniform(0.4, 1.0) / k) * np.sin(k * gx + rng.uniform(0, 6.28))
    for k in range(1, 5):                         # gentle along-length variation
        grain += (rng.uniform(0.2, 0.5) / k) * np.sin(k * gy + rng.uniform(0, 6.28))

    grain -= grain.min()
    grain /= (grain.max() + 1e-6)
    grain = grain ** 1.2                          # bias toward darker fissures

    rgb = dark[None, None, :] * (1.0 - grain[..., None]) + \
        light[None, None, :] * grain[..., None]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


# ============================================================================
# UV + MATERIAL ASSIGNMENT
# ============================================================================
def _rot_uv(x, y, r):
    """Rotate a unit-square uv by r*90 degrees about (0.5, 0.5)."""
    for _ in range(int(r) % 4):
        x, y = 1.0 - y, x
    return x, y


def _texture_canopy(canopy, atlas, rng):
    V = canopy.vertices
    n_cards = len(V) // 4
    ymin, ymax = float(V[:, 1].min()), float(V[:, 1].max())
    yspan = max(1e-6, ymax - ymin)
    rmax = max(1e-6, float(np.sqrt((V[:, 0] ** 2 + V[:, 2] ** 2)).max()))

    q = 0.25
    unit = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]  # c0,c1,c2,c3

    uv = np.zeros((len(V), 2), np.float64)
    vcol = np.zeros((len(V), 4), np.uint8)

    for i in range(n_cards):
        idx = 4 * i
        cen = V[idx:idx + 4].mean(axis=0)
        hyn = (cen[1] - ymin) / yspan                       # 0 low .. 1 high
        rad = float(np.sqrt(cen[0] ** 2 + cen[2] ** 2)) / rmax

        # atlas tile: brighter (lower row) near the sunlit top
        row = int(np.clip(round((1.0 - hyn) * 3.0 + rng.normal(0.0, 0.55)), 0, 3))
        col = int(rng.integers(0, 4))
        rot = int(rng.integers(0, 4))
        for k in range(4):
            ux, uy = _rot_uv(unit[k][0], unit[k][1], rot)
            uv[idx + k] = ((col + ux) * q, 1.0 - (row + uy) * q)

        # sun/shade vertex tint: top & outer brighter/warmer, inner/lower darker
        bright = 0.74 + 0.40 * (0.55 * hyn + 0.45 * rad)
        bright = float(np.clip(bright + rng.normal(0.0, 0.04), 0.62, 1.18))
        tint = np.clip(np.array([bright * 1.02, bright, bright * 0.96]) * 255.0,
                       0, 255).astype(np.uint8)
        vcol[idx:idx + 4, :3] = tint
        vcol[idx:idx + 4, 3] = 255

    mat = trimesh.visual.material.PBRMaterial(
        name="canopy",
        baseColorTexture=atlas,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0,
        roughnessFactor=0.8,
        alphaMode="MASK",
        alphaCutoff=0.45,
        doubleSided=True,
    )
    canopy.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    canopy.visual.vertex_attributes["color"] = vcol
    return canopy


def _texture_branches(branches, bark):
    V = branches.vertices
    ymin, ymax = float(V[:, 1].min()), float(V[:, 1].max())
    yspan = max(1e-6, ymax - ymin)

    # cylindrical UVs around Y (wood is short and mostly hidden)
    ang = np.arctan2(V[:, 2], V[:, 0])
    u = ((ang / (2.0 * np.pi)) * 3.0) % 1.0
    v = ((V[:, 1] - ymin) / yspan) * 4.0
    uv = np.column_stack([u, v]).astype(np.float64)

    # AO-ish darkening toward the ground
    vcol = np.zeros((len(V), 4), np.uint8)
    hyn = (V[:, 1] - ymin) / yspan
    bright = np.clip(0.70 + 0.30 * hyn, 0.0, 1.0)
    g = np.clip(bright * 255, 0, 255).astype(np.uint8)
    vcol[:, 0] = g
    vcol[:, 1] = g
    vcol[:, 2] = g
    vcol[:, 3] = 255

    mat = trimesh.visual.material.PBRMaterial(
        name="branches",
        baseColorTexture=bark,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0,
        roughnessFactor=0.9,
        doubleSided=False,
    )
    branches.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    branches.visual.vertex_attributes["color"] = vcol
    return branches


# ============================================================================
# DRIVER
# ============================================================================
def build_textured_scene(image_path, seed, density):
    scene = build_mesh(seed, density)
    canopy = scene.geometry["canopy"]
    branches = scene.geometry["branches"]

    arr = _load_rgb(image_path)
    tones = _green_tones(arr)
    brown = _brown_tone(arr)

    tex_rng = np.random.default_rng(seed ^ 0x5EED)
    atlas = _foliage_atlas(tones, tex_rng)
    bark = _bark_texture(brown, tex_rng)

    _texture_canopy(canopy, atlas, tex_rng)
    _texture_branches(branches, bark)
    return scene


def main(argv=None):
    ap = argparse.ArgumentParser(description="Textured boxwood-shrub GLB generator")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args(argv)

    scene = build_textured_scene(args.image, args.seed, args.density)
    glb = scene.export(file_type="glb")
    with open(args.output, "wb") as f:
        f.write(glb)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- CLI must exit non-zero on error
        sys.stderr.write("ERROR: {}\n".format(exc))
        sys.exit(1)