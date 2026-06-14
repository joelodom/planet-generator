"""
Cottongrass tussock -- procedural geometry + photo-derived materials -> GLB.

A tufted, grass-like sedge: a dark peaty soil plug shot through with pale
roots, a bushy skirt of fine arching leaf-blades, and dozens of thin wiry
culms each crowned by a soft oval cotton pom-pom.

Pipeline:
  * build_mesh(seed, density)            -> trimesh.Scene (geometry + UV + tint)
  * materials are derived from the SOURCE PHOTO (colours sampled from inside
    the silhouette; tileable detail synthesised in those colours)
  * surfaces: "soil", "roots", "stems", "foliage", "cotton"

CLI:
  python thisscript --image PATH --seed INT --density {high,med,low} --output OUT.glb

Only numpy / trimesh / PIL / stdlib.  +Y up, base at y=0, metres.
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial


# --------------------------------------------------------------------------- #
# Overall proportions (measured by eye off the reference image)
# --------------------------------------------------------------------------- #
OVERALL_HEIGHT = 0.45                       # metres, tallest cotton head
HEIGHT_OVER_WIDTH = 1.25                    # silhouette taller than wide
OVERALL_WIDTH = OVERALL_HEIGHT / HEIGHT_OVER_WIDTH   # ~0.36 m

SOIL_HEIGHT = 0.06 * OVERALL_HEIGHT         # rounded root-ball mound
SOIL_RADIUS = 0.24 * OVERALL_WIDTH          # compact plug (narrowed)
SOIL_TOP = SOIL_HEIGHT * 0.65               # where culms/blades emerge

SKIRT_TOP = 0.42 * OVERALL_HEIGHT           # top of the leaf-blade skirt
SKIRT_SPREAD = 0.34 * OVERALL_WIDTH         # narrower, bushier fountain

STEM_TOP_MAX = 0.90 * OVERALL_HEIGHT        # tallest culm tip
STEM_SPLAY = 0.27 * OVERALL_WIDTH           # near-vertical splay (narrowed)
CLUMP_RADIUS = 0.10 * OVERALL_WIDTH         # radius of the basal emergence ring

HEAD_LONG = 0.085 * OVERALL_WIDTH           # cotton along-stem (long) radius
HEAD_WIDE = 0.070 * OVERALL_WIDTH           # cotton cross (short) radius

# Materials derived from the photo are installed here by main() before
# build_mesh() runs.  If None, build_mesh falls back to flat vertex colours.
_MATERIALS = None

# Fallback body colours (only used if a photo patch hits background).
FALLBACK = {
    "soil": np.array([46, 33, 23], float),
    "roots": np.array([192, 172, 132], float),
    "green": np.array([108, 150, 70], float),
    "cotton": np.array([242, 241, 233], float),
}


# --------------------------------------------------------------------------- #
# Vector helpers
# --------------------------------------------------------------------------- #
def _norm(v):
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n


def _perp(t):
    up = np.array([0.0, 1.0, 0.0]) if abs(t[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    return _norm(np.cross(up, t))


def _rotmat_between(a, b):
    a = _norm(a)
    b = _norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = _perp(a)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def _tangents(pts):
    t = np.zeros_like(pts)
    t[1:-1] = pts[2:] - pts[:-2]
    t[0] = pts[1] - pts[0]
    t[-1] = pts[-1] - pts[-2]
    return np.array([_norm(x) for x in t])


def _smooth(t):
    return t * t * (3.0 - 2.0 * t)


def _tint(fac):
    """fac : (N,3) float multipliers in 0..1  -> uint8 (N,4) COLOR_0 array."""
    c = np.empty((len(fac), 4), np.uint8)
    c[:, :3] = np.clip(fac * 255.0, 0, 255).astype(np.uint8)
    c[:, 3] = 255
    return c


# --------------------------------------------------------------------------- #
# Mesh primitives (each returns verts, faces, uv)
# --------------------------------------------------------------------------- #
def _tube(pts, radii, sides, uv_repeat=0.05):
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    tans = _tangents(pts)

    us = [_perp(tans[0])]
    for i in range(1, n):
        R = _rotmat_between(tans[i - 1], tans[i])
        u = R @ us[-1]
        u = _norm(u - np.dot(u, tans[i]) * tans[i])
        us.append(u)

    seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    vcoord = cum / max(uv_repeat, 1e-6)

    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    verts, uv = [], []
    for i in range(n):
        u = us[i]
        w = np.cross(tans[i], u)
        ring = (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w) * radii[i]
        verts.append(pts[i] + ring)
        for j in range(sides):
            uv.append([j / sides, vcoord[i]])
    verts = np.vstack(verts)

    faces = []
    for i in range(n - 1):
        a = i * sides
        b = (i + 1) * sides
        for j in range(sides):
            k = (j + 1) % sides
            faces.append([a + j, b + j, b + k])
            faces.append([a + j, b + k, a + k])

    c0 = len(verts)
    verts = np.vstack([verts, pts[0], pts[-1]])
    uv.append([0.5, vcoord[0]])
    uv.append([0.5, vcoord[-1]])
    bi, ti = c0, c0 + 1
    base = (n - 1) * sides
    for j in range(sides):
        k = (j + 1) % sides
        faces.append([bi, k, j])
        faces.append([ti, base + j, base + k])
    return verts, np.asarray(faces, dtype=np.int64), np.asarray(uv, float)


def _ribbon(pts, widths, side_dir, uv_rect):
    """Flat double-sided blade card mapped onto an atlas tile."""
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    s = _norm(side_dir)
    half = (np.asarray(widths) * 0.5)[:, None] * s
    left = pts + half
    right = pts - half
    verts = np.empty((2 * n, 3))
    verts[0::2] = left
    verts[1::2] = right

    u0, u1, vb, vt = uv_rect
    uv = np.empty((2 * n, 2))
    f = np.linspace(0.0, 1.0, n)
    vv = vb + (vt - vb) * f
    uv[0::2, 0] = u0
    uv[1::2, 0] = u1
    uv[0::2, 1] = vv
    uv[1::2, 1] = vv

    faces = []
    for i in range(n - 1):
        a, b = 2 * i, 2 * i + 1
        c, d = 2 * (i + 1), 2 * (i + 1) + 1
        faces.append([a, b, d])
        faces.append([a, d, c])
        faces.append([a, d, b])   # back faces (double-sided card)
        faces.append([a, c, d])
    return verts, np.asarray(faces, dtype=np.int64), uv


def _soil_mound(rng, seg, rings):
    R, H = SOIL_RADIUS, SOIL_HEIGHT
    verts = [np.array([0.0, H, 0.0])]
    ang = np.linspace(0.0, 2.0 * np.pi, seg, endpoint=False)
    for ri in range(1, rings + 1):
        t = ri / rings
        r = R * t
        y = H * (1.0 - t * t)
        jit = rng.normal(0.0, 0.06 * H, seg)
        rj = r * (1.0 + rng.normal(0.0, 0.05, seg))
        for j in range(seg):
            verts.append([rj[j] * np.cos(ang[j]),
                          max(0.0, y + (jit[j] if ri < rings else 0.0)),
                          rj[j] * np.sin(ang[j])])
    bottom_c = len(verts)
    verts.append([0.0, 0.0, 0.0])
    verts = np.asarray(verts, dtype=float)

    faces = []
    for j in range(seg):
        k = (j + 1) % seg
        faces.append([0, 1 + j, 1 + k])
    for ri in range(rings - 1):
        a = 1 + ri * seg
        b = 1 + (ri + 1) * seg
        for j in range(seg):
            k = (j + 1) % seg
            faces.append([a + j, b + j, b + k])
            faces.append([a + j, b + k, a + k])
    outer = 1 + (rings - 1) * seg
    for j in range(seg):
        k = (j + 1) % seg
        faces.append([bottom_c, outer + k, outer + j])
    faces = np.asarray(faces, dtype=np.int64)
    # top-down planar UV (dominant axis is +Y for a flat plug)
    uv = np.column_stack([verts[:, 0] / (2 * R) + 0.5,
                          verts[:, 2] / (2 * R) + 0.5])
    return verts, faces, uv


def _cotton_head(rng, tip, axis, subdiv, rng_scale):
    """Soft, smooth, rounded oval pom-pom seated on a culm tip."""
    sph = trimesh.creation.icosphere(subdivisions=subdiv, radius=1.0)
    base = sph.vertices.copy()
    # spherical UV from the unmodified unit-sphere directions
    uv = np.column_stack([
        np.arctan2(base[:, 2], base[:, 0]) / (2 * np.pi) + 0.5,
        np.arccos(np.clip(base[:, 1], -1, 1)) / np.pi,
    ])
    # GENTLE lumpiness only -- keep the silhouette soft and round, not jagged
    noise = 1.0 + rng.normal(0.0, 0.05, len(base))
    v = base * noise[:, None]
    long_r = HEAD_LONG * rng_scale
    wide_r = HEAD_WIDE * rng_scale
    v = v * np.array([wide_r, long_r, wide_r])
    R = _rotmat_between(np.array([0.0, 1.0, 0.0]), axis)
    v = v @ R.T
    v = v + (tip + axis * (long_r * 0.45))
    return v, sph.faces.copy(), uv


# --------------------------------------------------------------------------- #
# Assembly helpers
# --------------------------------------------------------------------------- #
def _concat(chunks):
    vs, fs, uvs, off = [], [], [], 0
    for v, f, uv in chunks:
        vs.append(v)
        fs.append(f + off)
        uvs.append(uv)
        off += len(v)
    return np.vstack(vs), np.vstack(fs), np.vstack(uvs)


def _add(scene, name, v, f, uv, tint):
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mat = _MATERIALS.get(name) if _MATERIALS else None
    if mat is not None:
        mesh.visual = TextureVisuals(uv=uv, material=mat)
        mesh.visual.vertex_attributes["color"] = tint
    else:
        mesh.visual.vertex_colors = tint
    scene.add_geometry(mesh, geom_name=name)


# --------------------------------------------------------------------------- #
# Geometry entry point
# --------------------------------------------------------------------------- #
def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)

    if density == "low":
        n_stems, n_blades, n_roots = 12, 48, 4
        stem_sides, stem_seg, blade_seg = 3, 3, 4
        cotton_subdiv, soil_seg, soil_ring = 1, 16, 3
    elif density == "med":
        n_stems, n_blades, n_roots = 22, 120, 7
        stem_sides, stem_seg, blade_seg = 4, 5, 5
        cotton_subdiv, soil_seg, soil_ring = 2, 28, 4
    else:  # "high"
        n_stems, n_blades, n_roots = 32, 220, 10
        stem_sides, stem_seg, blade_seg = 5, 7, 6
        cotton_subdiv, soil_seg, soil_ring = 3, 40, 6

    scene = trimesh.Scene()

    # ---- soil plug -------------------------------------------------------- #
    sv, sf, suv = _soil_mound(rng, soil_seg, soil_ring)
    sy = sv[:, 1] / max(SOIL_HEIGHT, 1e-6)
    sfac = (0.5 + 0.45 * sy)[:, None] * np.ones(3)          # darker low/crevice
    _add(scene, "soil", sv, sf, suv, _tint(sfac))

    # ---- pale roots ------------------------------------------------------- #
    root_chunks = []
    for _ in range(n_roots):
        ang = rng.uniform(0.0, 2.0 * np.pi)
        rad = rng.uniform(0.4, 1.0) * SOIL_RADIUS
        bx, bz = rad * np.cos(ang), rad * np.sin(ang)
        pts = np.array([
            [bx * 0.5, SOIL_HEIGHT * 0.4, bz * 0.5],
            [bx, SOIL_HEIGHT * 0.15, bz],
            [bx * 1.25 + rng.normal(0, 0.01), 0.001,
             bz * 1.25 + rng.normal(0, 0.01)],
        ])
        r = np.array([0.0016, 0.0011, 0.0006])
        root_chunks.append(_tube(pts, r, max(3, stem_sides - 1), uv_repeat=0.03))
    rv, rf, ruv = _concat(root_chunks)
    _add(scene, "roots", rv, rf, ruv, _tint(np.full((len(rv), 3), 0.85)))

    # ---- dense arching leaf-blade skirt ---------------------------------- #
    blade_chunks, blade_tints = [], []
    for _ in range(n_blades):
        a = rng.uniform(0.0, 2.0 * np.pi)
        br = rng.uniform(0.0, CLUMP_RADIUS)
        bx, bz = br * np.cos(a), br * np.sin(a)
        rad_ang = a + rng.normal(0.0, 0.5)
        radial = np.array([np.cos(rad_ang), 0.0, np.sin(rad_ang)])
        peak_h = rng.uniform(0.55, 1.0) * SKIRT_TOP
        reach = rng.uniform(0.25, 1.0) * SKIRT_SPREAD
        ts = np.linspace(0.0, 1.0, blade_seg)
        pts = []
        for t in ts:
            out = radial * (reach * _smooth(t))
            y = SOIL_TOP + peak_h * np.sin(t * np.pi * 0.78)
            pts.append([bx + out[0], y, bz + out[2]])
        pts = np.asarray(pts)
        # WIDER cards with a solid core -> reads as a blade, no dashed alpha
        w = 0.011 * (1.0 - 0.7 * ts) + 0.0018
        side = _norm(np.cross(np.array([0.0, 1.0, 0.0]), radial))

        idx = int(rng.integers(0, 16))
        col, row = idx % 4, idx // 4
        tw = 0.25
        u0 = col * tw + 0.12 * tw
        u1 = col * tw + 0.88 * tw
        if rng.random() < 0.5:
            u0, u1 = u1, u0
        vb = row * tw + 0.96 * tw          # blade base -> tile bottom
        vt = row * tw + 0.04 * tw          # blade tip  -> tile top
        v, f, uv = _ribbon(pts, w, side, (u0, u1, vb, vt))
        blade_chunks.append((v, f, uv))

        clump = rng.uniform(0.82, 1.05)
        bright = (0.62 + 0.35 * ts) * clump
        bright = np.repeat(bright, 2)[:, None] * np.ones(3)
        blade_tints.append(np.clip(bright, 0, 1))
    bv, bf, buv = _concat(blade_chunks)
    _add(scene, "foliage", bv, bf, buv, _tint(np.vstack(blade_tints)))

    # ---- near-vertical wiry culms + cotton heads ------------------------- #
    stem_chunks, stem_tints = [], []
    cotton_chunks, cotton_tints = [], []
    for _ in range(n_stems):
        a = rng.uniform(0.0, 2.0 * np.pi)
        br = rng.uniform(0.0, 1.0) ** 0.5 * CLUMP_RADIUS
        bx, bz = br * np.cos(a), br * np.sin(a)
        rad_ang = a + rng.normal(0.0, 0.6)
        radial = np.array([np.cos(rad_ang), 0.0, np.sin(rad_ang)])
        reach = rng.uniform(0.0, 1.0) * STEM_SPLAY
        top_y = rng.uniform(0.72, 1.0) * STEM_TOP_MAX
        sway_dir = _norm(np.cross(np.array([0.0, 1.0, 0.0]), radial))
        sway_amp = rng.uniform(0.0, 0.010)
        sway_ph = rng.uniform(0.0, np.pi)

        ts = np.linspace(0.0, 1.0, stem_seg)
        pts = []
        for t in ts:
            out = radial * (reach * _smooth(t))
            sway = sway_dir * (sway_amp * np.sin(t * np.pi + sway_ph))
            y = SOIL_TOP + (top_y - SOIL_TOP) * t
            pts.append([bx + out[0] + sway[0], y, bz + out[2] + sway[2]])
        pts = np.asarray(pts)
        radii = 0.0016 * (1.0 - 0.45 * ts)
        v, f, uv = _tube(pts, radii, stem_sides, uv_repeat=0.05)
        stem_chunks.append((v, f, uv))
        sfac = (0.6 + 0.4 * (v[:, 1] / STEM_TOP_MAX))[:, None] * np.ones(3)
        stem_tints.append(np.clip(sfac, 0, 1))

        axis = _norm(pts[-1] - pts[-2])
        cv, cf, cuv = _cotton_head(rng, pts[-1], axis, cotton_subdiv,
                                   rng.uniform(0.85, 1.15))
        cotton_chunks.append((cv, cf, cuv))
        hb = (0.86 + 0.16 * (top_y / STEM_TOP_MAX)) * rng.uniform(0.92, 1.0)
        cotton_tints.append(np.clip(np.full((len(cv), 3), hb), 0, 1))

    sv, sf, suv = _concat(stem_chunks)
    _add(scene, "stems", sv, sf, suv, _tint(np.vstack(stem_tints)))
    cv, cf, cuv = _concat(cotton_chunks)
    _add(scene, "cotton", cv, cf, cuv, _tint(np.vstack(cotton_tints)))

    return scene


# --------------------------------------------------------------------------- #
# Photo sampling
# --------------------------------------------------------------------------- #
def _load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32)


def _sample(img, box, predicate, fallback):
    """Median colour of in-silhouette pixels passing ``predicate``."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    sub = img[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)].reshape(-1, 3)
    if len(sub) == 0:
        return np.asarray(fallback, float)
    sel = sub[predicate(sub)]
    if len(sel) < 25:
        return np.asarray(fallback, float)
    med = np.median(sel, axis=0)
    d = np.linalg.norm(sel - med, axis=1)
    keep = sel[d < (np.median(d) * 2.5 + 1.0)]
    if len(keep) >= 25:
        med = np.median(keep, axis=0)
    return med


def sample_palette(img):
    lum = lambda s: s @ np.array([0.299, 0.587, 0.114])
    chroma = lambda s: s.max(1) - s.min(1)

    soil = _sample(img, (0.40, 0.80, 0.62, 0.96),
                   lambda s: lum(s) < 100, FALLBACK["soil"])
    roots = _sample(img, (0.38, 0.78, 0.64, 0.97),
                    lambda s: (lum(s) > 105) & (lum(s) < 200) &
                              (s[:, 0] >= s[:, 2]), FALLBACK["roots"])
    green = _sample(img, (0.34, 0.50, 0.66, 0.76),
                    lambda s: (s[:, 1] > s[:, 0] + 4) &
                              (s[:, 1] > s[:, 2] + 4), FALLBACK["green"])
    cotton = _sample(img, (0.22, 0.06, 0.78, 0.42),
                     lambda s: (lum(s) > 155) & (chroma(s) < 48),
                     FALLBACK["cotton"])
    return {"soil": soil, "roots": roots, "green": green, "cotton": cotton}


# --------------------------------------------------------------------------- #
# Tileable texture synthesis (detail in the sampled colours)
# --------------------------------------------------------------------------- #
def _tile_noise(size, rng, octaves=5, base=3):
    x = np.linspace(0.0, 2.0 * np.pi, size, endpoint=False)
    X, Y = np.meshgrid(x, x)
    acc = np.zeros((size, size), float)
    amp_sum = 0.0
    for o in range(octaves):
        f = base * (2 ** o)
        a = 0.55 ** o
        p1, p2, p3 = rng.uniform(0, 2 * np.pi, 3)
        acc += a * (np.sin(f * X + p1) * np.cos(f * Y + p2) +
                    0.6 * np.sin(f * (0.7 * X + 1.3 * Y) + p3))
        amp_sum += a
    acc /= (amp_sum * 1.6)
    acc = (acc - acc.min()) / (np.ptp(acc) + 1e-9)
    return acc


def _delight(rgb):
    """Divide out a heavily blurred luminance; clamp gain to [0.6, 1.6]."""
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    lum = np.asarray(img.convert("L").filter(
        ImageFilter.GaussianBlur(img.size[0] * 0.12)), float) + 1e-3
    gain = np.clip(lum.mean() / lum, 0.6, 1.6)[..., None]
    return np.clip(rgb * gain, 0, 255)


def _soil_texture(size, rng, soil_col, root_col):
    n1 = _tile_noise(size, rng, 5, 3)
    n2 = _tile_noise(size, rng, 4, 9)
    rgb = soil_col[None, None, :] * (0.55 + 0.9 * n1[..., None])
    streak = n2 > 0.80                                   # stringy root flecks
    rgb[streak] = root_col * 0.85
    crack = n1 < 0.16
    rgb[crack] *= 0.55                                   # dark crevices
    rgb = _delight(rgb)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def _fiber_texture(size, rng, col, vertical=True):
    n = _tile_noise(size, rng, 5, 4)
    line = _tile_noise(size, rng, 4, 14)
    if vertical:
        streak = np.repeat(line[:, :1], size, axis=1)
    else:
        streak = np.repeat(line[:1, :], size, axis=0)
    val = 0.7 + 0.45 * n + 0.18 * (streak - 0.5)
    rgb = col[None, None, :] * val[..., None]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def _stem_texture(size, rng, green):
    straw = np.clip(green * 0.55 + np.array([205, 200, 140]) * 0.55, 0, 255)
    rows = np.linspace(0.0, 1.0, size)[:, None]          # 0 bottom..1 top
    grad = green[None, None, :] * (1 - rows[..., None]) + \
        straw[None, None, :] * rows[..., None]
    line = _tile_noise(size, rng, 4, 18)
    streak = np.repeat(line[:1, :], size, axis=0)[..., None]
    rgb = grad * (0.9 + 0.18 * streak)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def _cotton_texture(size, rng, col):
    n1 = _tile_noise(size, rng, 6, 6)
    n2 = _tile_noise(size, rng, 5, 16)
    val = 0.9 + 0.1 * n1 + 0.05 * (n2 - 0.5)
    cream = np.clip(col, 0, 255)
    rgb = cream[None, None, :] * val[..., None]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def _albedo_to_normal(pil_img, strength=2.5):
    g = np.asarray(pil_img.convert("L"), float) / 255.0
    h = 1.0 - g                                          # darker = lower
    gy, gx = np.gradient(h)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(h)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    out = np.stack([nx / ln, ny / ln, nz / ln], axis=-1)
    out = ((out * 0.5 + 0.5) * 255).astype(np.uint8)
    return Image.fromarray(out, "RGB")


# --------------------------------------------------------------------------- #
# Foliage atlas (4x4 distinct blade tiles, full opaque blade cores)
# --------------------------------------------------------------------------- #
def _blade_mask(S4, rng):
    img = Image.new("L", (S4, S4), 0)
    d = ImageDraw.Draw(img)
    cx = S4 * 0.5
    baseW = S4 * rng.uniform(0.30, 0.42)                 # fuller blade
    curve = S4 * rng.uniform(-0.09, 0.09)
    npts = 16
    ys = np.linspace(0.97, 0.05, npts) * S4
    ts = np.linspace(0.0, 1.0, npts)
    # gentle taper + generous minimum width keeps the alpha core solid
    widths = baseW * (1 - ts) ** 0.55 + S4 * 0.022
    centers = cx + curve * np.sin(ts * np.pi * 0.6)
    left = [(centers[i] - widths[i], ys[i]) for i in range(npts)]
    right = [(centers[i] + widths[i], ys[i]) for i in range(npts)]
    d.polygon(left + right[::-1], fill=255)
    return img


def _blade_tile(S, rng, base_col, tip_col, bright):
    S4 = S * 4
    mask = _blade_mask(S4, rng).resize((S, S), Image.LANCZOS)
    a = np.asarray(mask, float) / 255.0
    fb = np.linspace(0.0, 1.0, S)[:, None]               # 0 top..1 bottom
    col = tip_col[None, None, :] * (1 - fb[..., None]) + \
        base_col[None, None, :] * fb[..., None]
    out = np.zeros((S, S, 4), float)
    out[..., :3] = np.clip(col * bright, 0, 255)
    out[..., 3] = a * 255.0
    return out.astype(np.uint8)


def _foliage_atlas(tile, rng, green):
    base_col = np.clip(green, 0, 255)
    tip_col = np.clip(green * 0.5 + np.array([215, 210, 150]) * 0.55, 0, 255)
    atlas = np.zeros((tile * 4, tile * 4, 4), np.uint8)
    for r in range(4):
        for c in range(4):
            sun = (r + c) % 2 == 0
            bright = 1.12 if sun else 0.82
            warm = np.array([1.05, 1.0, 0.9]) if sun else np.array([0.94, 1.0, 1.07])
            bc = np.clip(base_col * warm, 0, 255)
            tc = np.clip(tip_col * warm, 0, 255)
            t = _blade_tile(tile, rng, bc, tc, bright)
            atlas[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = t
    return Image.fromarray(atlas, "RGBA")


# --------------------------------------------------------------------------- #
# Material construction
# --------------------------------------------------------------------------- #
def build_materials(img, seed):
    rng = np.random.default_rng(seed + 12345)            # deterministic swatches
    pal = sample_palette(img)

    soil_img = _soil_texture(512, rng, pal["soil"], pal["roots"])
    root_img = _fiber_texture(512, rng, pal["roots"], vertical=True)
    stem_img = _stem_texture(512, rng, pal["green"])
    cotton_img = _cotton_texture(512, rng, pal["cotton"])
    atlas_img = _foliage_atlas(256, rng, pal["green"])   # 1024x1024

    white = [1.0, 1.0, 1.0, 1.0]
    mats = {
        "soil": PBRMaterial(
            name="soil", baseColorFactor=white, baseColorTexture=soil_img,
            normalTexture=_albedo_to_normal(soil_img, 2.5),
            metallicFactor=0.0, roughnessFactor=0.95),
        "roots": PBRMaterial(
            name="roots", baseColorFactor=white, baseColorTexture=root_img,
            metallicFactor=0.0, roughnessFactor=0.9),
        "stems": PBRMaterial(
            name="stems", baseColorFactor=white, baseColorTexture=stem_img,
            metallicFactor=0.0, roughnessFactor=0.8, doubleSided=True),
        "foliage": PBRMaterial(
            name="foliage", baseColorFactor=white, baseColorTexture=atlas_img,
            metallicFactor=0.0, roughnessFactor=0.8,
            alphaMode="MASK", alphaCutoff=0.45, doubleSided=True),
        "cotton": PBRMaterial(
            name="cotton", baseColorFactor=white, baseColorTexture=cotton_img,
            metallicFactor=0.0, roughnessFactor=0.95, doubleSided=True),
    }
    return mats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    global _MATERIALS
    ap = argparse.ArgumentParser(description="Cottongrass tussock -> textured GLB")
    ap.add_argument("--image", required=True, help="source reference photo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", choices=("high", "med", "low"), default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args(argv)

    img = _load_image(args.image)
    _MATERIALS = build_materials(img, args.seed)

    scene = build_mesh(args.seed, args.density)

    glb = scene.export(file_type="glb")
    with open(args.output, "wb") as fh:
        fh.write(glb)

    tris = sum(len(g.faces) for g in scene.geometry.values())
    print(f"wrote {args.output}  surfaces={list(scene.geometry.keys())}  "
          f"triangles={tris}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)