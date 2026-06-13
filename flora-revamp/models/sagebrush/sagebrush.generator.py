"""
Procedural silver-sagebrush shrub: geometry + photo-derived materials + GLB export.

CLI:
  python thisscript --image PATH --seed INT --density {high,med,low} --output OUT.glb

Only numpy / trimesh / PIL / stdlib.
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals


# ===========================================================================
# GEOMETRY  (build_mesh)
# ===========================================================================
OVERALL_HEIGHT = 0.72                        # meters, top of canopy
HEIGHT_OVER_WIDTH = 1.04                      # slightly taller than wide -> aspect ~0.96
CROWN_WIDTH = OVERALL_HEIGHT / HEIGHT_OVER_WIDTH   # ~0.69 m full diameter
FOLIAGE_BOTTOM_FRAC = 0.08                    # canopy reaches low; small wood base shows

_A = CROWN_WIDTH * 0.5
_C = CROWN_WIDTH * 0.5
_B = OVERALL_HEIGHT * (1.0 - FOLIAGE_BOTTOM_FRAC) * 0.5
_CY = OVERALL_HEIGHT * FOLIAGE_BOTTOM_FRAC + _B
_CENTER = np.array([0.0, _CY, 0.0])
_SEMI = np.array([_A, _B, _C])

_CARD_HALF_FRAC = 0.052                        # ~5% of crown width
_CLUMP_RADIUS_FRAC = 0.11                       # ~11% of crown width


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([0.0, 1.0, 0.0])
    return v / n


def _ortho_basis(n):
    n = _normalize(n)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _normalize(np.cross(n, ref))
    v = np.cross(n, u)
    return u, v


def _make_lobes(rng):
    k = rng.integers(3, 7)
    freqs = rng.integers(2, 5, size=k).astype(float)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=k)
    amps = rng.uniform(0.05, 0.11, size=k)        # gentler lobes -> cleaner mound
    return freqs, phases, amps


def _lobe(direction, lobes):
    freqs, phases, amps = lobes
    az = np.arctan2(direction[2], direction[0])
    equator = np.sqrt(max(0.0, 1.0 - direction[1] * direction[1]))
    return 1.0 + equator * float(np.sum(amps * np.cos(freqs * az + phases)))


def _shell_radius(direction, lobes):
    d = _normalize(direction)
    t = 1.0 / np.linalg.norm(d / _SEMI)
    return t * _lobe(d, lobes)


def _clamp_to_envelope(p, lobes, frac=1.0):
    d = p - _CENTER
    r = np.linalg.norm(d)
    if r < 1e-9:
        return p.copy()
    shell = _shell_radius(d, lobes) * frac
    if r > shell:
        return _CENTER + d * (shell / r)
    return p


def _envelope_normal(p):
    g = (p - _CENTER) / (_SEMI * _SEMI)
    return _normalize(g)


def _frustum(p0, p1, r0, r1, sides):
    axis = p1 - p0
    L = np.linalg.norm(axis)
    if L < 1e-7:
        return None
    w = axis / L
    u, v = _ortho_basis(w)
    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    circ = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v
    ring0 = p0 + r0 * circ
    ring1 = p1 + r1 * circ
    verts = np.vstack([ring0, ring1])
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        a, b, c, d = i, j, sides + j, sides + i
        faces.append([a, b, c])
        faces.append([a, c, d])
    return verts, np.array(faces, dtype=np.int64)


def _grow(pos, direction, radius, length, depth, params, lobes, rng,
          seg_verts, seg_faces, tips, voff):
    max_depth = params["max_depth"]
    sides = params["wood_sides"]
    min_r = 0.0018

    if depth > max_depth or radius < min_r:
        tips.append(_clamp_to_envelope(pos, lobes, frac=0.96))
        return

    out = np.array([pos[0], 0.0, pos[2]])
    out = _normalize(out) if np.linalg.norm(out) > 1e-6 else _normalize(
        rng.normal(size=3) * np.array([1.0, 0.0, 1.0]))
    up = np.array([0.0, 1.0, 0.0])
    t = depth / max(1, max_depth)
    bias = (1.0 - t) * 0.6 * up + t * 0.45 * out - t * 0.1 * up
    noise = rng.normal(size=3) * 0.32
    newdir = _normalize(direction + bias + noise)

    end = _clamp_to_envelope(pos + newdir * length, lobes, frac=0.98)

    seg = _frustum(pos, end, radius, radius * 0.78, sides)
    if seg is not None:
        v, f = seg
        seg_verts.append(v)
        seg_faces.append(f + voff[0])
        voff[0] += len(v)

    n_child = 2 if rng.random() < 0.7 and depth < max_depth else 1
    r_child = radius / np.sqrt(n_child) * 0.9
    next_len = length * 0.82
    for _ in range(n_child):
        perturb = _normalize(newdir + rng.normal(size=3) * 0.45 + out * 0.2)
        _grow(end, perturb, r_child, next_len, depth + 1, params, lobes, rng,
              seg_verts, seg_faces, tips, voff)


def _build_wood(params, lobes, rng):
    seg_verts, seg_faces, tips = [], [], []
    voff = [0]
    n_stems = params["stems"]
    root_r = 0.020
    seg_len = OVERALL_HEIGHT * 0.62 / max(1, params["max_depth"])
    for _ in range(n_stems):
        ang = rng.uniform(0.0, 2.0 * np.pi)
        rad = rng.uniform(0.0, 0.035)
        base = np.array([np.cos(ang) * rad, 0.0, np.sin(ang) * rad])
        d0 = _normalize(np.array([np.cos(ang) * 0.25, 1.0, np.sin(ang) * 0.25])
                        + rng.normal(size=3) * 0.15)
        _grow(base, d0, root_r * rng.uniform(0.9, 1.3), seg_len, 0,
              params, lobes, rng, seg_verts, seg_faces, tips, voff)

    verts = np.vstack(seg_verts)
    faces = np.vstack(seg_faces)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.fix_normals()
    base_col = np.array([124, 104, 82], dtype=np.float64)
    jitter = rng.normal(scale=10.0, size=(len(verts), 3))
    cols = np.clip(base_col + jitter, 40, 200).astype(np.uint8)
    rgba = np.column_stack([cols, np.full(len(verts), 255, np.uint8)])
    mesh.visual.vertex_colors = rgba
    return mesh, tips


def _clump_centers(params, lobes, rng, tips):
    n_clumps = params["clumps"]
    centers, normals = [], []

    rng.shuffle(tips)
    for p in tips[: n_clumps // 2]:
        centers.append(p)
        normals.append(_envelope_normal(p))

    # Shell clumps covering the WHOLE dome (top and well down the sides) so the
    # mass reads dense and opaque, not as a hollow upper shell.
    while len(centers) < n_clumps:
        theta = rng.uniform(0.0, 2.0 * np.pi)
        cphi = rng.uniform(-0.8, 1.0)
        sphi = np.sqrt(max(0.0, 1.0 - cphi * cphi))
        direction = np.array([sphi * np.cos(theta), cphi, sphi * np.sin(theta)])
        r = _shell_radius(direction, lobes) * rng.uniform(0.82, 0.98)
        p = _CENTER + r * direction
        centers.append(p)
        normals.append(_envelope_normal(p))

    # Many interior clumps so the crown fills solidly.
    for _ in range(max(3, n_clumps // 3)):
        direction = _normalize(rng.normal(size=3))
        r = _shell_radius(direction, lobes) * rng.uniform(0.3, 0.7)
        p = _CENTER + r * direction
        centers.append(p)
        normals.append(_envelope_normal(p))

    return centers, normals


def _build_canopy(params, lobes, rng, tips):
    centers, normals = _clump_centers(params, lobes, rng, tips)
    cards_per = params["cards_per"]
    clump_r = CROWN_WIDTH * _CLUMP_RADIUS_FRAC
    half = CROWN_WIDTH * _CARD_HALF_FRAC

    all_v, all_f, all_c = [], [], []
    voff = 0
    for center, out_n in zip(centers, normals):
        for _ in range(cards_per):
            off = rng.normal(size=3) * clump_r * 0.8
            off -= out_n * np.dot(off, out_n) * 0.45
            pos = center + off
            pos = _clamp_to_envelope(pos, lobes, frac=1.0)   # no protrusion past shell

            cn = _normalize(out_n + rng.normal(size=3) * 0.22)   # tighter normal jitter
            u, v = _ortho_basis(cn)
            a = rng.uniform(0.0, 2.0 * np.pi)
            ca, sa = np.cos(a), np.sin(a)
            u2 = ca * u + sa * v
            v2 = -sa * u + ca * v
            s = half * float(np.exp(rng.normal(0.0, 0.18)))      # tame size spikes
            hu = s
            hv = s * rng.uniform(0.8, 1.0)

            quad = np.array([
                pos - u2 * hu - v2 * hv,    # 0 -> uv (0,0)
                pos + u2 * hu - v2 * hv,    # 1 -> uv (1,0)
                pos + u2 * hu + v2 * hv,    # 2 -> uv (1,1)
                pos - u2 * hu + v2 * hv,    # 3 -> uv (0,1)
            ])
            all_v.append(quad)
            all_f.append(np.array([[0, 1, 2], [0, 2, 3]]) + voff)
            voff += 4

            # Near-white per-vertex tint (multiplies texture): keep the pale
            # silvery color, add only a gentle top-bright / inner-dim gradient.
            top = np.clip(pos[1] / OVERALL_HEIGHT, 0.0, 1.0)
            val = 0.84 + 0.16 * top
            base_col = np.array([val * 1.0, val * 1.02, val * 0.97]) * 255.0
            base_col = base_col + rng.normal(scale=5.0, size=3)
            col = np.clip(base_col, 150, 255).astype(np.uint8)
            all_c.append(np.tile(np.append(col, 255).astype(np.uint8), (4, 1)))

    verts = np.vstack(all_v)
    faces = np.vstack(all_f)
    cols = np.vstack(all_c)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                           vertex_colors=cols, process=False)
    return mesh


def _params(density):
    presets = {
        # Denser fields of smaller cards -> opaque, feathery mass.
        "high": dict(clumps=40, cards_per=80, stems=6, max_depth=5, wood_sides=6),
        "med":  dict(clumps=26, cards_per=50, stems=5, max_depth=4, wood_sides=5),
        "low":  dict(clumps=14, cards_per=28, stems=4, max_depth=3, wood_sides=4),
    }
    return presets.get(density, presets["high"])


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    params = _params(density)
    lobes = _make_lobes(rng)

    wood, tips = _build_wood(params, lobes, rng)
    if not tips:
        tips = [_CENTER + np.array([0.0, _B * 0.5, 0.0])]
    canopy = _build_canopy(params, lobes, rng, tips)

    min_y = min(wood.vertices[:, 1].min(), canopy.vertices[:, 1].min())
    shift = np.array([0.0, -min_y, 0.0])
    wood.apply_translation(shift)
    canopy.apply_translation(shift)

    scene = trimesh.Scene()
    scene.add_geometry(wood, geom_name="trunk")
    scene.add_geometry(canopy, geom_name="canopy")
    return scene


# ===========================================================================
# TEXTURING
# ===========================================================================
FOLIAGE_SUN_BOX = (0.30, 0.14, 0.70, 0.40)
FOLIAGE_SHADE_BOX = (0.30, 0.52, 0.70, 0.76)
WOOD_BOX = (0.40, 0.82, 0.62, 0.94)


def _load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def _sample_color(img, box, rng, n=60, patch=3):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    cols = []
    for _ in range(n):
        cx = int(rng.uniform(x0, x1) * (W - 1))
        cy = int(rng.uniform(y0, y1) * (H - 1))
        xa, xb = max(0, cx - patch), min(W, cx + patch + 1)
        ya, yb = max(0, cy - patch), min(H, cy + patch + 1)
        cols.append(np.median(img[ya:yb, xa:xb].reshape(-1, 3), axis=0))
    cols = np.array(cols)
    med = np.median(cols, axis=0)
    dist = np.linalg.norm(cols - med, axis=1)
    keep = cols[dist <= (np.median(dist) * 2.0 + 1e-4)]
    return np.clip(np.median(keep, axis=0), 0.0, 1.0)


def _crop(img, box, size=256):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    xa, xb = int(x0 * W), int(x1 * W)
    ya, yb = int(y0 * H), int(y1 * H)
    xb = max(xb, xa + 2)
    yb = max(yb, ya + 2)
    sub = (img[ya:yb, xa:xb] * 255).astype(np.uint8)
    pil = Image.fromarray(sub).resize((size, size), Image.LANCZOS)
    return np.asarray(pil, dtype=np.float64) / 255.0


def _delight(arr):
    h, w = arr.shape[:2]
    lum = arr.mean(axis=2)
    lum_img = Image.fromarray((np.clip(lum, 0, 1) * 255).astype(np.uint8))
    rad = max(4, w // 8)
    blurred = np.asarray(lum_img.filter(ImageFilter.GaussianBlur(rad)),
                         dtype=np.float64) / 255.0
    gain = np.clip(lum.mean() / (blurred + 1e-4), 0.6, 1.6)
    return np.clip(arr * gain[..., None], 0.0, 1.0)


def _mirror_tile(a):
    top = np.concatenate([a, a[:, ::-1]], axis=1)
    full = np.concatenate([top, top[::-1]], axis=0)
    return full


def _to_pil(arr, mode="RGB"):
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode)


def _tone(color, target_lum, desat=0.35):
    """Desaturate slightly toward grey, then lift to a target luminance."""
    g = color.mean()
    color = color * (1.0 - desat) + g * desat
    lum = max(color.mean(), 1e-3)
    return np.clip(color * (target_lum / lum), 0.0, 1.0)


def make_bark_texture(img, wood_color, rng, size=512):
    try:
        crop = _delight(_crop(img, WOOD_BOX, size=size // 2))
        if np.ptp(crop.reshape(-1, 3).mean(0)) < 0.01 and crop.mean() > 0.6:
            crop = np.ones((size // 2, size // 2, 3)) * wood_color
    except Exception:
        crop = np.ones((size // 2, size // 2, 3)) * wood_color

    n = crop.shape[0]
    x = np.arange(n)
    fiber = np.zeros(n)
    for _ in range(14):
        freq = rng.uniform(18.0, 70.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.03, 0.09)
        fiber += amp * np.sin(2.0 * np.pi * freq * x / n + phase)
    col_mult = 1.0 + fiber + rng.normal(scale=0.05, size=n)
    for _ in range(6):
        c = rng.integers(0, n)
        wgt = np.exp(-((x - c) ** 2) / (2 * (rng.uniform(1.5, 3.5) ** 2)))
        col_mult -= 0.5 * wgt
    row_mult = 1.0 + rng.normal(scale=0.05, size=n)
    M = np.clip(np.outer(row_mult, col_mult), 0.45, 1.4)

    bark = crop * M[..., None]
    dark = np.clip(1.0 - M, 0.0, 1.0)[..., None]
    bark = bark + dark * np.array([0.06, 0.02, -0.02])
    bark = np.clip(bark, 0.0, 1.0)

    bark = _mirror_tile(bark)
    return _to_pil(bark).resize((size, size), Image.LANCZOS)


def make_bark_normal(bark_pil, strength=2.5):
    arr = np.asarray(bark_pil, dtype=np.float64) / 255.0
    height = 1.0 - arr.mean(axis=2)
    gy, gx = np.gradient(height)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(height)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    rgb = np.stack([nx / norm, ny / norm, nz / norm], axis=2) * 0.5 + 0.5
    return _to_pil(rgb)


def _leaf_polygon(cx, cy, ang, length, width, curve):
    d = np.array([np.cos(ang), np.sin(ang)])
    p = np.array([-d[1], d[0]])
    ts = np.array([0.0, 0.2, 0.45, 0.7, 1.0])
    wf = np.array([0.18, 0.8, 1.0, 0.7, 0.0]) * width
    left, right = [], []
    for t, w in zip(ts, wf):
        c = np.array([cx, cy]) + d * (length * t) + p * (curve * np.sin(np.pi * t))
        left.append(c + p * w)
        right.append(c - p * w)
    pts = left + right[::-1]
    return [(float(x), float(y)) for x, y in pts]


def _draw_cluster_tile(tile_px, sun_color, shade_color, f, rng, ss=4):
    """A DENSE soft leaf cluster filling the tile center (not a sparse star)."""
    S = tile_px * ss
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    base = shade_color + (sun_color - shade_color) * f
    cx0, cy0 = S * 0.5, S * 0.5
    n_leaves = int(rng.integers(95, 135))
    for _ in range(n_leaves):
        ang = rng.uniform(0.0, 2.0 * np.pi)
        rad = rng.uniform(0.0, S * 0.30)
        bx = cx0 + np.cos(ang) * rad
        by = cy0 + np.sin(ang) * rad
        leaf_ang = ang + rng.uniform(-0.6, 0.6)
        length = rng.uniform(0.14, 0.26) * S       # short -> filled blob, not spikes
        width = rng.uniform(0.020, 0.038) * S
        curve = rng.uniform(-0.05, 0.05) * S
        poly = _leaf_polygon(bx, by, leaf_ang, length, width, curve)
        jit = rng.normal(scale=0.04, size=3)
        col = np.clip(base + jit, 0.0, 1.0)
        rgb = tuple(int(c * 255) for c in col)
        draw.polygon(poly, fill=rgb + (255,))      # opaque -> binary alpha
    return img.resize((tile_px, tile_px), Image.LANCZOS)


def make_leaf_atlas(sun_color, shade_color, rng, tiles=4, tile_px=256):
    atlas = Image.new("RGBA", (tiles * tile_px, tiles * tile_px), (0, 0, 0, 0))
    for ty in range(tiles):
        f = 1.0 - ty / (tiles - 1)                  # top rows sunlit
        sun = np.clip(sun_color * (0.95 + 0.12 * f) + np.array([0.03, 0.03, 0.0]) * f, 0, 1)
        shade = np.clip(shade_color * (0.92 + 0.05 * (1 - f)), 0, 1)
        for tx in range(tiles):
            tile = _draw_cluster_tile(tile_px, sun, shade, f, rng)
            atlas.paste(tile, (tx * tile_px, ty * tile_px))
    return atlas


def _trunk_uv(verts, u_repeat=2.0, cell=0.06):
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]
    ang = np.arctan2(z, x)
    u = (ang / (2.0 * np.pi) + 0.5) * u_repeat
    v = y / cell
    return np.column_stack([u, v]).astype(np.float64)


def _rot_uv(p, k):
    for _ in range(int(k)):
        p = np.column_stack([p[:, 1], 1.0 - p[:, 0]])
    return p


def _canopy_uv(n_verts, rng, tiles=4, inset=0.012):
    n_cards = n_verts // 4
    base = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    uv = np.zeros((n_verts, 2), dtype=np.float64)
    for i in range(n_cards):
        tx = int(rng.integers(0, tiles))
        ty = int(rng.integers(0, tiles))
        rot = int(rng.integers(0, 4))
        local = _rot_uv(base.copy(), rot)
        local = inset + local * (1.0 - 2.0 * inset)
        uv[4 * i:4 * i + 4, 0] = (tx + local[:, 0]) / tiles
        uv[4 * i:4 * i + 4, 1] = (ty + local[:, 1]) / tiles
    return uv


def build_textured_scene(image_path, seed, density):
    scene = build_mesh(seed, density)
    trunk = scene.geometry["trunk"]
    canopy = scene.geometry["canopy"]

    img = _load_image(image_path)
    tex_rng = np.random.default_rng(seed + 12345)
    uv_rng = np.random.default_rng(seed + 999)

    # Sample photo colors, then desaturate + lift to the pale silvery target so
    # the foliage reads whitish-mint instead of dark olive.
    sun_color = _tone(_sample_color(img, FOLIAGE_SUN_BOX, tex_rng), 0.80)
    shade_color = _tone(_sample_color(img, FOLIAGE_SHADE_BOX, tex_rng), 0.64)
    wood_color = _sample_color(img, WOOD_BOX, tex_rng)
    if np.ptp(sun_color) < 0.01:
        sun_color = np.array([0.80, 0.82, 0.76])
    if np.ptp(shade_color) < 0.01:
        shade_color = np.array([0.60, 0.63, 0.57])

    # --- BARK material ---
    bark_pil = make_bark_texture(img, wood_color, tex_rng, size=512)
    bark_normal = make_bark_normal(bark_pil)
    trunk_colors = np.asarray(trunk.visual.vertex_colors).copy()
    trunk_uv = _trunk_uv(trunk.vertices)
    bark_mat = PBRMaterial(
        name="bark",
        baseColorTexture=bark_pil,
        normalTexture=bark_normal,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0,
        roughnessFactor=0.9,
        doubleSided=False,
    )
    trunk.visual = TextureVisuals(uv=trunk_uv, material=bark_mat)
    trunk.visual.vertex_attributes["color"] = trunk_colors.astype(np.uint8)

    # --- FOLIAGE material ---
    atlas = make_leaf_atlas(sun_color, shade_color, tex_rng, tiles=4, tile_px=256)
    canopy_colors = np.asarray(canopy.visual.vertex_colors).copy()
    canopy_uv = _canopy_uv(len(canopy.vertices), uv_rng, tiles=4)
    leaf_mat = PBRMaterial(
        name="foliage",
        baseColorTexture=atlas,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0,
        roughnessFactor=0.85,
        alphaMode="MASK",
        alphaCutoff=0.45,
        doubleSided=True,
    )
    canopy.visual = TextureVisuals(uv=canopy_uv, material=leaf_mat)
    canopy.visual.vertex_attributes["color"] = canopy_colors.astype(np.uint8)

    return scene


def main():
    ap = argparse.ArgumentParser(description="Procedural silver-sagebrush shrub -> GLB")
    ap.add_argument("--image", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    try:
        scene = build_textured_scene(args.image, args.seed, args.density)
        glb = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(glb)
    except Exception as exc:
        sys.stderr.write("ERROR: {}\n".format(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()