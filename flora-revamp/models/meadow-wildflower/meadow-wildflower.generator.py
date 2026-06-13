"""Procedural blanketflower (Gaillardia) generator + photo-derived texturing.

Builds a single upright wildflower stem (two open composite blooms, a terminal
bud, alternating lance leaves), derives tileable materials from a reference
photo, applies per-surface UVs, and exports a textured GLB.

CLI:
    python thisscript.py --image PATH --seed INT --density {high,med,low} \
                         --output OUT.glb
"""

import argparse
import sys
import numpy as np
import trimesh
from PIL import Image, ImageFilter


# ============================================================================
# GEOMETRY (build_mesh) -- proportions measured (~10%) off the reference image.
# ============================================================================
TOTAL_HEIGHT = 0.60                 # m, plausible Gaillardia flowering stem
HEIGHT_OVER_WIDTH = 3.6             # tall & narrow silhouette
FLOWER_ZONE_FRAC = 0.25            # heads crowded in the TOP ~25%
BLOOM_DIA_OVER_HEIGHT = 0.11        # one bloom ~11% of height across
LEAF_LEN_OVER_HEIGHT = 0.15         # largest leaves ~15% of height

H = TOTAL_HEIGHT

DISC_RADIUS = 0.0125                # central disc button radius (m) -- enlarged
DOME_HEIGHT = 0.0058                # disc dome rise (m)
PETAL_LEN = 0.021                   # ray floret length (m)
PETAL_WIDTH = 0.0115               # ray floret max width (m) -- broad straps
TOOTH_DEPTH = 0.0022               # subtle 3-toothed ray tip depth (m)


def _counts(density):
    if density == "low":
        return dict(sides=5, stem_seg=16, ped_seg=6,
                    n_leaves=5, leaf_nu=5, leaf_nv=3,
                    n_petals=14, petal_nu=4, petal_nv=4,
                    disc_rings=4, disc_sectors=10, bud_sub=0)
    if density == "med":
        return dict(sides=8, stem_seg=26, ped_seg=8,
                    n_leaves=10, leaf_nu=7, leaf_nv=4,
                    n_petals=20, petal_nu=5, petal_nv=5,
                    disc_rings=6, disc_sectors=14, bud_sub=1)
    return dict(sides=12, stem_seg=40, ped_seg=12,
                n_leaves=16, leaf_nu=9, leaf_nv=5,
                n_petals=26, petal_nu=6, petal_nv=5,
                disc_rings=8, disc_sectors=22, bud_sub=2)


def _norm(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def _perp(a):
    a = _norm(a)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(a @ ref) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    return _norm(ref - a * (a @ ref))


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _grid_faces(nu, nv):
    faces = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a = i * nv + j
            b = i * nv + j + 1
            c = (i + 1) * nv + j
            d = (i + 1) * nv + j + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return np.array(faces, dtype=np.int64)


def _place(local, ex, ey, ez, origin):
    B = np.column_stack([ex, ey, ez])
    return origin + local @ B.T


def _stack(meshes):
    """Concatenate card meshes WITHOUT welding, preserving block layout."""
    vs, fs, off = [], [], 0
    for m in meshes:
        vs.append(m.vertices)
        fs.append(m.faces + off)
        off += len(m.vertices)
    return trimesh.Trimesh(vertices=np.vstack(vs), faces=np.vstack(fs),
                           process=False)


def _tube(points, radii, sides):
    points = np.asarray(points, float)
    radii = np.asarray(radii, float)
    n = len(points)

    t = np.zeros_like(points)
    t[1:-1] = points[2:] - points[:-2]
    t[0] = points[1] - points[0]
    t[-1] = points[-1] - points[-2]
    t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)

    normals = np.zeros_like(points)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(t[0] @ ref) > 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    normals[0] = _norm(ref - t[0] * (t[0] @ ref))
    for i in range(1, n):
        v = normals[i - 1] - t[i] * (t[i] @ normals[i - 1])
        if np.linalg.norm(v) < 1e-9:
            v = ref - t[i] * (t[i] @ ref)
        normals[i] = _norm(v)
    binorm = np.cross(t, normals)

    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    cos, sin = np.cos(ang), np.sin(ang)

    verts = []
    for i in range(n):
        verts.append(points[i] + radii[i] *
                     (np.outer(cos, normals[i]) + np.outer(sin, binorm[i])))
    verts = np.vstack(verts)

    faces = []
    for i in range(n - 1):
        a = i * sides
        b = (i + 1) * sides
        for j in range(sides):
            j2 = (j + 1) % sides
            faces.append([a + j, a + j2, b + j2])
            faces.append([a + j, b + j2, b + j])

    c0 = len(verts)
    verts = np.vstack([verts, points[0][None, :]])
    for j in range(sides):
        j2 = (j + 1) % sides
        faces.append([c0, j2, j])
    c1 = len(verts)
    verts = np.vstack([verts, points[-1][None, :]])
    base = (n - 1) * sides
    for j in range(sides):
        j2 = (j + 1) % sides
        faces.append([c1, base + j, base + j2])

    return trimesh.Trimesh(vertices=verts,
                           faces=np.array(faces, dtype=np.int64), process=True)


def _arc(p0, p1, bow_dir, bow, segs):
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    ts = np.linspace(0.0, 1.0, segs)
    line = p0[None, :] + (p1 - p0)[None, :] * ts[:, None]
    bend = (np.sin(ts * np.pi) * bow)[:, None] * _norm(bow_dir)[None, :]
    return line + bend


def _disc(center, axis, scale, P):
    A = _norm(axis)
    u = _perp(A)
    w = np.cross(A, u)
    R = P["disc_rings"]
    S = P["disc_sectors"]
    rad = DISC_RADIUS * scale
    dome = DOME_HEIGHT * scale

    verts = []
    for i in range(R + 1):
        fr = i / R
        ri = rad * fr
        y = dome * (1.0 - fr * fr)
        for j in range(S):
            ang = 2.0 * np.pi * j / S
            radial = np.cos(ang) * u + np.sin(ang) * w
            verts.append(center + A * y + radial * ri)
    verts = np.array(verts)

    faces = []
    for i in range(R):
        for j in range(S):
            j2 = (j + 1) % S
            a = i * S + j
            b = i * S + j2
            c = (i + 1) * S + j
            d = (i + 1) * S + j2
            faces.append([a, b, d])
            faces.append([a, d, c])
    base_pt = center - A * (dome * 0.25)
    bi = len(verts)
    verts = np.vstack([verts, base_pt[None, :]])
    rim = R * S
    for j in range(S):
        j2 = (j + 1) % S
        faces.append([bi, rim + j2, rim + j])

    return trimesh.Trimesh(vertices=verts,
                           faces=np.array(faces, dtype=np.int64), process=True)


def _petals(center, axis, scale, rng, P):
    A = _norm(axis)
    u = _perp(A)
    w = np.cross(A, u)

    nu, nv = P["petal_nu"], P["petal_nv"]
    npet = P["n_petals"]
    uu = np.linspace(0.0, 1.0, nu)
    vv = np.linspace(-1.0, 1.0, nv)
    U, V = np.meshgrid(uu, vv, indexing="ij")
    tip_mask = _smooth(0.75, 1.0, U)
    faces_template = _grid_faces(nu, nv)

    all_v, all_f, voff = [], [], 0
    rim = DISC_RADIUS * 0.80 * scale
    for k in range(npet):
        ang = 2.0 * np.pi * k / npet + rng.normal(0.0, 0.04)
        plen = PETAL_LEN * scale * (1.0 + rng.normal(0.0, 0.06))
        pwid = PETAL_WIDTH * scale * (1.0 + rng.normal(0.0, 0.06))

        # broad strap: stays wide near the disc, overlaps neighbours, then a
        # gentle taper to a shallow 3-tooth tip.
        wp = 0.55 + 0.45 * np.sin(np.pi * U) ** 0.5
        x = plen * U + TOOTH_DEPTH * scale * np.cos(2.0 * np.pi * V) * tip_mask
        y = 0.5 * pwid * V * wp
        z = -0.10 * plen * (U ** 1.8)          # nearly flat, slight droop
        local = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

        radial = np.cos(ang) * u + np.sin(ang) * w
        tilt = np.deg2rad(rng.uniform(-6.0, 4.0))   # rays near-horizontal
        ex = _norm(radial * np.cos(tilt) + A * np.sin(tilt))
        ez = _norm(-radial * np.sin(tilt) + A * np.cos(tilt))
        ey = np.cross(ez, ex)
        origin = center + A * (DOME_HEIGHT * scale * 0.20) + ex * rim

        all_v.append(_place(local, ex, ey, ez, origin))
        all_f.append(faces_template + voff)
        voff += nu * nv

    return trimesh.Trimesh(vertices=np.vstack(all_v),
                           faces=np.vstack(all_f), process=False)


def _bud(center, axis, scale, P):
    A = _norm(axis)
    u = _perp(A)
    w = np.cross(A, u)
    ico = trimesh.creation.icosphere(subdivisions=P["bud_sub"], radius=1.0)
    v = ico.vertices.copy()
    v[:, 1] *= 1.30                                   # rounder teardrop
    r = np.linalg.norm(v, axis=1)
    dir_ = v / (r[:, None] + 1e-9)
    spike = (v[:, 1] > 0.0) & (np.arange(len(v)) % 2 == 0)
    v[spike] = dir_[spike] * (r[spike] * 1.80)[:, None]   # prominent bracts
    v *= 0.009 * scale                                # enlarged bud
    world = center + v[:, 0:1] * u + v[:, 1:2] * A + v[:, 2:3] * w
    return trimesh.Trimesh(vertices=world, faces=ico.faces.copy(), process=True)


def _leaf(stem_pt, tangent, outward, length, width, rng, P):
    nu, nv = P["leaf_nu"], P["leaf_nv"]
    uu = np.linspace(0.0, 1.0, nu)
    vv = np.linspace(-1.0, 1.0, nv)
    U, V = np.meshgrid(uu, vv, indexing="ij")

    wp = (U + 0.04) ** 0.5 * (1.0 - U) ** 0.9
    wp = wp / (wp.max() + 1e-9)
    wp = wp * (1.0 + 0.12 * np.sin(7.0 * np.pi * U))

    x = length * U
    y = 0.5 * width * V * wp
    z = (-0.18 * length * (U ** 1.6) + 0.05 * width * (V ** 2) * wp)
    local = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    ex = _norm(tangent * 0.6 + outward * 0.85)
    ez = _norm(outward - ex * (outward @ ex))
    ey = np.cross(ez, ex)
    twist = np.deg2rad(rng.uniform(-12.0, 12.0))
    ey = ey * np.cos(twist) + ez * np.sin(twist)
    ez = np.cross(ex, ey)

    origin = stem_pt + outward * 0.003
    return trimesh.Trimesh(vertices=_place(local, ex, ey, ez, origin),
                           faces=_grid_faces(nu, nv), process=False)


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    if density not in ("high", "med", "low"):
        density = "high"
    P = _counts(density)

    branch_y = 0.70 * H
    nseg = P["stem_seg"]
    ts = np.linspace(0.0, 1.0, nseg)
    ys = ts * branch_y
    sway = 0.018 * H * np.sin(ts * 2.2) * ts + rng.normal(0.0, 0.0015, nseg)
    zs = 0.010 * H * np.sin(ts * 1.7) * ts
    stem_pts = np.column_stack([sway, ys, zs])

    r_base, r_top = 0.0045, 0.0016
    radii = r_top + (r_base - r_top) * (1.0 - ts) ** 1.3
    flare = 1.0 + 0.45 * _smooth(0.07, 0.0, ts)
    radii = radii * flare

    stem_meshes = [_tube(stem_pts, radii, P["sides"])]
    branch = stem_pts[-1]

    # Upright, forward-facing heads; gentle pedicels (no crossing/loop).
    blooms = [
        dict(center=np.array([-0.050 * H, 0.90 * H, 0.020 * H]),
             axis=_norm([-0.10, 0.55, 0.83]), scale=1.00,
             bow=np.array([-1.0, 0.2, 0.5]), bowmag=0.025 * H),
        dict(center=np.array([0.085 * H, 0.83 * H, 0.010 * H]),
             axis=_norm([0.18, 0.45, 0.87]), scale=0.82,
             bow=np.array([1.0, 0.1, 0.4]), bowmag=0.025 * H),
    ]
    bud_info = dict(center=np.array([0.0 * H, 0.97 * H, -0.010 * H]),
                    axis=_norm([0.0, 1.0, 0.15]),
                    bow=np.array([0.2, 0.3, 0.2]), bowmag=0.015 * H)

    disc_meshes, petal_meshes = [], []
    for b in blooms:
        attach = b["center"] - b["axis"] * 0.013
        path = _arc(branch, attach, b["bow"], b["bowmag"], P["ped_seg"])
        pr = np.linspace(0.0016, 0.0011, len(path))
        stem_meshes.append(_tube(path, pr, max(5, P["sides"] - 4)))
        disc_meshes.append(_disc(b["center"], b["axis"], b["scale"], P))
        petal_meshes.append(_petals(b["center"], b["axis"], b["scale"], rng, P))

    b = bud_info
    attach = b["center"] - b["axis"] * 0.008
    path = _arc(branch, attach, b["bow"], b["bowmag"], P["ped_seg"])
    pr = np.linspace(0.0016, 0.0010, len(path))
    stem_meshes.append(_tube(path, pr, max(5, P["sides"] - 4)))
    bud_mesh = _bud(b["center"], b["axis"], 1.0, P)

    leaf_meshes = []
    n_leaves = P["n_leaves"]
    golden = np.deg2rad(137.5)
    for i in range(n_leaves):
        f = 0.12 + 0.56 * (i / max(1, n_leaves - 1))
        idx = int(np.clip(f, 0, 1) * (nseg - 1))
        pt = stem_pts[idx]
        if idx < nseg - 1:
            tan = _norm(stem_pts[idx + 1] - stem_pts[idx])
        else:
            tan = _norm(stem_pts[idx] - stem_pts[idx - 1])
        az = i * golden + rng.normal(0.0, 0.15)
        u = _perp(tan)
        w = np.cross(tan, u)
        outward = _norm(np.cos(az) * u + np.sin(az) * w)
        bump = np.sin(np.pi * np.clip((f - 0.05) / 0.7, 0, 1))
        length = LEAF_LEN_OVER_HEIGHT * H * (0.45 + 0.6 * bump)
        length *= (1.0 + rng.normal(0.0, 0.08))
        width = length * 0.26
        leaf_meshes.append(_leaf(pt, tan, outward, length, width, rng, P))

    named = {
        "stem": trimesh.util.concatenate(stem_meshes),
        "leaves": _stack(leaf_meshes),     # keep card block layout (no weld)
        "petals": _stack(petal_meshes),    # keep card block layout (no weld)
        "disc": trimesh.util.concatenate(disc_meshes),
        "bud": bud_mesh,
    }

    mins = np.array([np.inf, np.inf, np.inf])
    maxs = -mins.copy()
    for m in named.values():
        mins = np.minimum(mins, m.vertices.min(axis=0))
        maxs = np.maximum(maxs, m.vertices.max(axis=0))
    shift = np.array([-0.5 * (mins[0] + maxs[0]), -mins[1],
                      -0.5 * (mins[2] + maxs[2])])

    scene = trimesh.Scene()
    for name, m in named.items():
        m.apply_translation(shift)
        if name in ("stem", "disc", "bud"):   # cards left unwelded for UVs
            m.merge_vertices()
        scene.add_geometry(m, geom_name=name)
    return scene


# ============================================================================
# TEXTURING -- colors sampled from the reference photo, detail synthesized
# tileably and tinted to those colors.
# ============================================================================
def load_image(path):
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.float64) / 255.0


def _rgb_to_hsv(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    df = mx - mn
    h = np.zeros_like(mx)
    m = df > 1e-6
    idx = (mx == r) & m
    h[idx] = (60 * ((g[idx] - b[idx]) / df[idx]) + 360) % 360
    idx = (mx == g) & m
    h[idx] = (60 * ((b[idx] - r[idx]) / df[idx]) + 120) % 360
    idx = (mx == b) & m
    h[idx] = (60 * ((r[idx] - g[idx]) / df[idx]) + 240) % 360
    s = np.where(mx > 1e-6, df / (mx + 1e-9), 0.0)
    return h, s, mx


def sample_colors(img):
    """Median colors sampled from inside the flower's silhouette only."""
    Hh, Ww, _ = img.shape
    h, s, v = _rgb_to_hsv(img)

    corners = []
    for yy, xx in [(0.02, 0.02), (0.02, 0.95), (0.95, 0.02), (0.95, 0.95),
                   (0.5, 0.03), (0.5, 0.96)]:
        y0, x0 = int(yy * Hh), int(xx * Ww)
        patch = img[max(0, y0 - 5):y0 + 6, max(0, x0 - 5):x0 + 6]
        corners.append(np.median(patch.reshape(-1, 3), axis=0))
    bg = np.median(np.array(corners), axis=0)

    dist = np.linalg.norm(img - bg[None, None, :], axis=-1)
    notbg = (dist > 0.12) | (s > 0.25)
    rows = np.arange(Hh)[:, None] * np.ones((1, Ww))
    top = rows < (0.45 * Hh)
    green = (h > 70) & (h < 175) & (s > 0.18) & notbg
    flower = notbg & (~green) & top

    def med(mask):
        if int(mask.sum()) < 20:
            return None
        return np.median(img[mask], axis=0)

    yellow = med(flower & (h >= 42) & (h <= 70) & (v > 0.55))
    red = med(flower & ((h < 42) | (h > 330)) & (v > 0.30) & (v <= 0.95) & (s > 0.30))
    disc = med(flower & (v < 0.45) & (s > 0.25) & ((h < 55) | (h > 320)))
    leaf = med(green)
    bud = med(green & (rows < 0.20 * Hh))

    leaf = leaf if leaf is not None else np.array([0.33, 0.46, 0.18])
    yellow = yellow if yellow is not None else np.array([0.95, 0.78, 0.16])
    red = red if red is not None else np.array([0.80, 0.18, 0.06])
    disc = disc if disc is not None else np.array([0.42, 0.20, 0.12])
    bud = bud if bud is not None else leaf * 1.05
    stem = np.clip(0.55 * leaf + 0.45 * np.array([0.78, 0.80, 0.45]), 0, 1)

    return dict(leaf=np.clip(leaf, 0, 1), yellow=np.clip(yellow, 0, 1),
                red=np.clip(red, 0, 1), disc=np.clip(disc, 0, 1),
                bud=np.clip(bud, 0, 1), stem=stem)


def _vnoise(h, w, resh, resw, rng):
    g = (rng.random((resh, resw)) * 255).astype(np.uint8)
    im = Image.fromarray(g, "L").resize((w, h), Image.BILINEAR)
    return np.asarray(im, dtype=np.float64) / 255.0


def fbm(h, w, rng, octaves=4, res0=4):
    out = np.zeros((h, w))
    amp, tot, r = 1.0, 0.0, res0
    for _ in range(octaves):
        out += amp * _vnoise(h, w, r, r, rng)
        tot += amp
        amp *= 0.5
        r *= 2
    return out / tot


def _delight(rgb):
    """Divide out a blurred luminance, gain clamped to [0.6, 1.6]."""
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    im = Image.fromarray((np.clip(lum, 0, 1) * 255).astype(np.uint8), "L")
    radius = max(rgb.shape[0] // 8, 4)
    blur = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=radius)),
                      dtype=np.float64) / 255.0
    gain = np.clip(lum.mean() / (blur + 1e-3), 0.6, 1.6)
    return np.clip(rgb * gain[..., None], 0, 1)


# ---- non-card swatches (>=512) ---------------------------------------------
def make_stem_swatch(cols, rng):
    S = 512
    base = np.array(cols["stem"])
    fib = _vnoise(S, S, 256, 8, rng)          # fine vertical hairs
    detail = fbm(S, S, rng, 4, 8)
    rgb = np.ones((S, S, 3)) * base[None, None, :]
    rgb *= (0.82 + 0.36 * fib)[..., None]
    rgb *= (0.92 + 0.16 * detail)[..., None]
    rgb = _delight(np.clip(rgb, 0, 1))
    return Image.fromarray((rgb * 255).astype(np.uint8), "RGB")


def make_disc_swatch(cols, rng):
    S = 512
    # Lighten the sampled disc toward a legible rusty-brown so the button does
    # not collapse to near-black in render; keep a visible value range.
    base = np.clip(np.array(cols["disc"]) * 1.45 + np.array([0.10, 0.04, 0.0]),
                   0, 1)
    dk = np.clip(base * 0.7, 0, 1)
    ye = np.array(cols["yellow"])
    xx, yy = np.meshgrid(np.linspace(0, 1, S), np.linspace(0, 1, S))
    dots = 0.5 + 0.5 * np.sin(xx * 2 * np.pi * 22) * np.sin(yy * 2 * np.pi * 22)
    fuzz = fbm(S, S, rng, 5, 8)
    rgb = np.ones((S, S, 3)) * base[None, None, :]
    rgb = rgb * (0.78 + 0.30 * fuzz)[..., None] + dk[None, None, :] * (0.22 * (1 - fuzz))[..., None]
    speck = ((dots > 0.85) & (fuzz > 0.5))[..., None]
    rgb = rgb * (1 - speck) + (0.55 * ye + 0.45 * base)[None, None, :] * speck
    rgb = _delight(np.clip(rgb, 0, 1))
    return Image.fromarray((rgb * 255).astype(np.uint8), "RGB")


def make_bud_swatch(cols, rng):
    S = 512
    base = np.array(cols["bud"])
    red = np.array(cols["red"])
    bump = fbm(S, S, rng, 5, 6)
    rgb = np.ones((S, S, 3)) * base[None, None, :]
    rgb *= (0.80 + 0.40 * bump)[..., None]
    tip = (fbm(S, S, rng, 4, 5) > 0.62)[..., None]
    rgb = rgb * (1 - 0.5 * tip) + (0.5 * red + 0.5 * base)[None, None, :] * (0.5 * tip)
    rgb = _delight(np.clip(rgb, 0, 1))
    return Image.fromarray((rgb * 255).astype(np.uint8), "RGB")


# ---- card atlases (1024, full-bleed; geometry supplies the silhouette) -----
def _leaf_tile(col, rng):
    T = 256
    xx, yy = np.meshgrid(np.linspace(0, 1, T), np.linspace(0, 1, T))
    rgb = np.ones((T, T, 3)) * np.array(col)[None, None, :]
    rgb *= (0.92 + 0.12 * yy)[..., None]                      # tip lighter
    vein = np.exp(-((xx - 0.5) ** 2) / (2 * 0.02 ** 2))       # central vein
    rgb *= (1 - 0.30 * vein)[..., None]
    phase = (yy * 12 + np.abs(xx - 0.5) * 6.0)                # lateral veins
    latmask = (np.abs((phase % 1.0) - 0.5) < 0.04)
    rgb *= (1 - 0.07 * latmask)[..., None]
    spk = fbm(T, T, rng, 4, 10)                               # speckle/waxy
    rgb *= (0.90 + 0.20 * spk)[..., None]
    return np.clip(rgb, 0, 1)


def _petal_tile(cols, rng, bright):
    """Banded firewheel ray: deep red at the inner base -> bright golden
    yellow over the outer half (the photo's signature two-tone)."""
    T = 256
    red = np.clip(np.array(cols["red"]), 0, 1)
    yel = np.clip(np.array(cols["yellow"]) * 1.12, 0, 1)
    xx, yy = np.meshgrid(np.linspace(0, 1, T), np.linspace(0, 1, T))
    # yy: 0 = inner base (red), 1 = outer tip (yellow)
    t2 = _smooth(0.16, 0.78, yy)[..., None]
    rgb = (red[None, None, :] * (1 - t2) + yel[None, None, :] * t2) * bright
    streak = (np.abs(((xx * 9) % 1.0) - 0.5) < 0.05)          # faint veins
    rgb *= (1 - 0.06 * streak)[..., None]
    n = fbm(T, T, rng, 4, 12)                                 # velvet noise
    rgb *= (0.95 + 0.10 * n)[..., None]
    return np.clip(rgb, 0, 1)


def make_leaf_atlas(cols, rng):
    T = 256
    atlas = np.zeros((4 * T, 4 * T, 3))
    base = np.array(cols["leaf"])
    for ty in range(4):
        for tx in range(4):
            shade = 0.80 + 0.28 * (1 - ty / 3.0)   # top tiles sunlit/warmer
            warm = 0.05 * (1 - ty / 3.0)
            col = np.clip(base * shade +
                          np.array([warm, warm * 0.2, -warm * 0.5]), 0, 1)
            atlas[ty * T:(ty + 1) * T, tx * T:(tx + 1) * T] = _leaf_tile(col, rng)
    a = np.dstack([np.clip(atlas, 0, 1), np.ones((4 * T, 4 * T))])
    return Image.fromarray((a * 255).astype(np.uint8), "RGBA")


def make_petal_atlas(cols, rng):
    T = 256
    atlas = np.zeros((4 * T, 4 * T, 3))
    for ty in range(4):
        for tx in range(4):
            bright = 0.86 + 0.22 * (1 - ty / 3.0)
            atlas[ty * T:(ty + 1) * T, tx * T:(tx + 1) * T] = \
                _petal_tile(cols, rng, bright)
    a = np.dstack([np.clip(atlas, 0, 1), np.ones((4 * T, 4 * T))])
    return Image.fromarray((a * 255).astype(np.uint8), "RGBA")


# ---- UV + material assignment ----------------------------------------------
def apply_swatch(mesh, tex, rough, proj):
    V = mesh.vertices
    if proj == "cyl":
        cx, cz = V[:, 0].mean(), V[:, 2].mean()
        u = np.arctan2(V[:, 2] - cz, V[:, 0] - cx) / (2 * np.pi) + 0.5
        vv = (V[:, 1] - V[:, 1].min()) / 0.035
    elif proj == "sphere":
        cx, cz = V[:, 0].mean(), V[:, 2].mean()
        u = np.arctan2(V[:, 2] - cz, V[:, 0] - cx) / (2 * np.pi) + 0.5
        vv = (V[:, 1] - V[:, 1].min()) / (np.ptp(V[:, 1]) + 1e-9)
    else:  # planar
        u = (V[:, 0] - V[:, 0].min()) / (np.ptp(V[:, 0]) + 1e-9)
        vv = (V[:, 2] - V[:, 2].min()) / (np.ptp(V[:, 2]) + 1e-9)
    uv = np.column_stack([u, vv])

    yr = np.ptp(V[:, 1]) + 1e-9
    f = 0.80 + 0.20 * np.clip((V[:, 1] - V[:, 1].min()) / yr, 0, 1)  # base AO
    cc = (np.clip(np.column_stack([f, f, f * 0.98]), 0, 1) * 255).astype(np.uint8)
    col = np.column_stack([cc, np.full(len(V), 255, np.uint8)]).astype(np.uint8)

    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=tex, baseColorFactor=[1, 1, 1, 1],
        metallicFactor=0.0, roughnessFactor=rough)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    mesh.visual.vertex_attributes["color"] = col


def apply_cards(mesh, tex, P, kind, rng):
    nu, nv = P[kind + "_nu"], P[kind + "_nv"]
    LV = nu * nv
    V = mesh.vertices
    N = len(V)
    ncards = N // LV
    i_idx = np.repeat(np.arange(nu), nv)
    j_idx = np.tile(np.arange(nv), nu)
    s = i_idx / (nu - 1)               # along length
    t = j_idx / (nv - 1)               # across width

    uv = np.zeros((N, 2))
    col = np.zeros((N, 4), np.uint8)
    ymin = V[:, 1].min()
    yr = np.ptp(V[:, 1]) + 1e-9

    for c in range(ncards):
        blk = slice(c * LV, (c + 1) * LV)
        tile = int(rng.integers(0, 16))
        tx, ty = tile % 4, tile // 4
        flip = (kind == "leaf") and (rng.random() < 0.5)
        sc = (1 - s) if flip else s
        uv[blk, 0] = (tx + 0.02 + 0.96 * t) / 4.0
        uv[blk, 1] = (ty + 0.02 + 0.96 * sc) / 4.0

        Vb = V[blk]
        jit = rng.uniform(0.93, 1.05)
        if kind == "petal":
            # near-neutral so the texture's banding reads; tips a touch brighter
            val = np.clip((0.93 + 0.12 * s) * jit, 0, 1)
            warm = np.clip(np.column_stack([val, val * 0.99, val * 0.96]), 0, 1)
        else:
            base = 0.80 + 0.22 * np.clip((Vb[:, 1] - ymin) / yr, 0, 1)
            val = np.clip(base * jit, 0, 1)
            warm = np.clip(np.column_stack([val, val, val * 0.96]), 0, 1)
        col[blk, :3] = (warm * 255).astype(np.uint8)
        col[blk, 3] = 255

    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=tex, baseColorFactor=[1, 1, 1, 1],
        metallicFactor=0.0, roughnessFactor=(0.55 if kind == "petal" else 0.85),
        alphaMode="MASK", alphaCutoff=0.45, doubleSided=True)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    mesh.visual.vertex_attributes["color"] = col


def texture_scene(scene, density, image_path, seed):
    rng = np.random.default_rng(seed + 101)
    cols = sample_colors(load_image(image_path))
    P = _counts(density)

    leaf_atlas = make_leaf_atlas(cols, rng)
    petal_atlas = make_petal_atlas(cols, rng)
    stem_tex = make_stem_swatch(cols, rng)
    disc_tex = make_disc_swatch(cols, rng)
    bud_tex = make_bud_swatch(cols, rng)

    g = scene.geometry
    apply_swatch(g["stem"], stem_tex, 0.85, "cyl")
    apply_swatch(g["disc"], disc_tex, 0.80, "planar")
    apply_swatch(g["bud"], bud_tex, 0.85, "sphere")
    apply_cards(g["leaves"], leaf_atlas, P, "leaf", rng)
    apply_cards(g["petals"], petal_atlas, P, "petal", rng)
    return scene


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Blanketflower GLB generator")
    ap.add_argument("--image", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    try:
        scene = build_mesh(args.seed, args.density)
        scene = texture_scene(scene, args.density, args.image, args.seed)
        scene.export(args.output)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())