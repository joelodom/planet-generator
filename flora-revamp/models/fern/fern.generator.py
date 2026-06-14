#!/usr/bin/env python3
"""
Procedural shuttlecock / vase-shaped clump FERN -> textured GLB.

One self-contained script that:
  * builds the geometry (build_mesh, +Y up, base at y=0, meters),
  * derives tileable materials by SAMPLING COLORS from a reference photo,
  * draws a 4x4 leaf-card atlas with binary-alpha pinna silhouettes,
  * applies per-surface UVs + COLOR_0 sun/shade tints,
  * exports an embedded-texture .glb.

CLI:
  python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb
"""

import argparse
import math
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw


# ============================================================================
# GEOMETRY MODULE  (surfaces: foliage / stipe / rootball)
# ============================================================================
PLANT_HEIGHT      = 0.85                       # overall height of the rosette
# Photo reads slightly WIDER than tall (full width/height ~1.08); the rosette
# is a broad arching vase, not a tall spike.
PLANT_HALF_WIDTH  = 0.35                        # foliage envelope half-width (m)

ENV_LOBE_AMPS  = (0.07, 0.045, 0.03)           # bulge amplitudes (frac of half-width)
ENV_LOBE_FREQS = (2.0, 3.0, 5.0)               # angular frequencies of the bulges

CROWN_Y       = 0.035                           # y where stipes emerge (inside rootball)
CROWN_RADIUS  = 0.010                           # radius of the converging crown ring
ROOTBALL_R    = 0.050                           # small root knot, mostly hidden by foliage
ROOTBALL_H    = 0.050

STIPE_BASE_R = 0.0040
STIPE_TIP_R  = 0.0009

START_T        = 0.07        # pinnae start low so blades drape over the rootball
END_T          = 0.97        # pinnae stop just short of the pointed tip
PINNA_LEN_FRAC = 0.095       # max pinna reach as a fraction of frond length
FORWARD_SWEEP  = 0.40        # how much pinnae sweep toward the frond tip
NORMAL_JITTER  = 0.18        # +/- out-of-plane wobble of card normals
CARD_WIDTH_K   = 0.95        # card half-width / pinna spacing (overlap -> full blade)

FOLIAGE_BASE = np.array([46, 92, 33],  dtype=float)
FOLIAGE_TIP  = np.array([150, 196, 84], dtype=float)
STIPE_COLOR  = np.array([122, 96, 58, 255], dtype=np.uint8)
ROOT_COLOR   = np.array([74, 52, 33, 255], dtype=np.uint8)


def _nz(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.asarray(v, float)


def _bez(p0, p1, p2, t):
    mt = 1.0 - t
    return mt * mt * p0 + 2.0 * mt * t * p1 + t * t * p2


def _bez_d(p0, p1, p2, t):
    mt = 1.0 - t
    return 2.0 * mt * (p1 - p0) + 2.0 * t * (p2 - p1)


def _concat(parts):
    """Merge a list of (vertices, faces) into one (V, F)."""
    V, F, off = [], [], 0
    for v, f in parts:
        V.append(v)
        F.append(f + off)
        off += len(v)
    return np.vstack(V), np.vstack(F).astype(np.int64)


def _make_tube(points, radii, sides):
    """Sweep a circle of `sides` along `points` with a parallel-transport frame."""
    points = np.asarray(points, float)
    radii = np.asarray(radii, float)
    n = len(points)

    tang = np.zeros((n, 3))
    tang[1:-1] = points[2:] - points[:-2]
    tang[0] = points[1] - points[0]
    tang[-1] = points[-1] - points[-2]
    L = np.linalg.norm(tang, axis=1, keepdims=True)
    L[L == 0] = 1.0
    tang /= L

    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(tang[0] @ ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    normals = np.zeros((n, 3))
    normals[0] = _nz(np.cross(tang[0], ref))
    for i in range(1, n):
        v = np.cross(tang[i - 1], tang[i])
        s = np.linalg.norm(v)
        c = float(tang[i - 1] @ tang[i])
        p = normals[i - 1]
        if s < 1e-9:
            q = p
        else:
            a = v / s
            q = p * c + np.cross(a, p) * s + a * float(a @ p) * (1.0 - c)
        q = q - tang[i] * float(tang[i] @ q)
        normals[i] = _nz(q)
    binorm = np.cross(tang, normals)

    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ca = np.cos(ang)[:, None]
    sa = np.sin(ang)[:, None]
    verts = np.empty((n * sides, 3))
    for i in range(n):
        verts[i * sides:(i + 1) * sides] = (
            points[i] + radii[i] * (ca * normals[i] + sa * binorm[i])
        )

    faces = []
    for i in range(n - 1):
        b0, b1 = i * sides, (i + 1) * sides
        for j in range(sides):
            jn = (j + 1) % sides
            faces.append([b0 + j, b1 + j, b0 + jn])
            faces.append([b0 + jn, b1 + j, b1 + jn])

    ci = len(verts)
    verts = np.vstack([verts, points[0]])
    for j in range(sides):
        jn = (j + 1) % sides
        faces.append([ci, jn, j])

    return verts, np.array(faces, dtype=np.int64)


def _density_params(density):
    table = {
        "high": dict(n_fronds=30, pinnae=42, segs=14, sides=6, rb_sub=2, rootlets=4),
        "med":  dict(n_fronds=20, pinnae=26, segs=10, sides=5, rb_sub=1, rootlets=3),
        "low":  dict(n_fronds=14, pinnae=12, segs=6,  sides=4, rb_sub=1, rootlets=3),
    }
    return table.get(density, table["high"])


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    P = _density_params(density)

    lobe_ph = rng.uniform(0.0, 2.0 * np.pi, len(ENV_LOBE_AMPS))

    def env_radius(az):
        r = 1.0
        for amp, freq, ph in zip(ENV_LOBE_AMPS, ENV_LOBE_FREQS, lobe_ph):
            r += amp * np.cos(freq * az + ph)
        return PLANT_HALF_WIDTH * r

    pk_t = 0.35 / (0.35 + 0.70)
    PEAK = (pk_t ** 0.35) * ((1.0 - pk_t) ** 0.70)

    stipe_parts = []
    fv, ff, fc = [], [], []

    def add_card(inner, outer, vdir, hw, color):
        i = len(fv)
        p0 = inner - vdir * hw
        p1 = outer - vdir * hw
        p2 = outer + vdir * hw
        p3 = inner + vdir * hw
        fv.extend([p0, p1, p2, p3])
        ff.append([i, i + 1, i + 2])
        ff.append([i, i + 2, i + 3])
        fc.append(color)
        fc.append(color)

    n_fronds = P["n_fronds"]
    for k in range(n_fronds):
        az = (k / n_fronds) * 2.0 * np.pi + rng.uniform(-0.16, 0.16)
        cs, sn = np.cos(az), np.sin(az)
        n_p = np.array([-sn, 0.0, cs])

        # Every frond fans OUTWARD (no near-vertical central spike): high
        # minimum tip radius, modest height spread.  Bow the control point
        # outward (not up) so fronds arch from low down into a broad vase.
        u = rng.random()
        tip_y = PLANT_HEIGHT * (0.52 + 0.32 * u)
        tip_r = env_radius(az) * (0.90 - 0.24 * u + 0.10 * rng.random())

        P0 = np.array([CROWN_RADIUS, CROWN_Y])
        P1 = np.array([0.40 * tip_r, 0.50 * tip_y + 0.10 * PLANT_HEIGHT])
        P2 = np.array([tip_r, tip_y])

        ts = np.linspace(0.0, 1.0, P["segs"] + 1)
        pts = np.empty((len(ts), 3))
        for i, t in enumerate(ts):
            s, y = _bez(P0, P1, P2, t)
            pts[i] = [s * cs, y, s * sn]
        radii = STIPE_TIP_R + (STIPE_BASE_R - STIPE_TIP_R) * (1.0 - ts) ** 0.7
        stipe_parts.append(_make_tube(pts, radii, P["sides"]))

        frond_len = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        spacing = frond_len * (END_T - START_T) / P["pinnae"]

        for j in range(P["pinnae"]):
            tt = (j + 0.5) / P["pinnae"]
            t = START_T + tt * (END_T - START_T)
            s, y = _bez(P0, P1, P2, t)
            pt = np.array([s * cs, y, s * sn])
            ds, dy = _bez_d(P0, P1, P2, t)
            T = _nz(np.array([ds * cs, dy, ds * sn]))
            side = np.cross(n_p, T)
            if np.linalg.norm(side) < 1e-6:
                continue
            side = _nz(side)

            scale = (tt ** 0.35) * ((1.0 - tt) ** 0.70) / PEAK
            plen = max(PINNA_LEN_FRAC * frond_len * scale, 0.010)
            # Wider cards (CARD_WIDTH_K ~1) overlap into a continuous lacy blade.
            hw = CARD_WIDTH_K * spacing * (0.55 + 0.45 * scale)

            base_col = FOLIAGE_BASE + (FOLIAGE_TIP - FOLIAGE_BASE) * (tt ** 0.8)
            for sgn in (-1.0, 1.0):
                u_axis = _nz(side * sgn + FORWARD_SWEEP * T)
                normal = _nz(n_p + NORMAL_JITTER * rng.uniform(-1.0, 1.0, 3))
                vdir = np.cross(normal, u_axis)
                if np.linalg.norm(vdir) < 1e-6:
                    vdir = np.cross(T, u_axis)
                vdir = _nz(vdir)
                col = np.clip(base_col + rng.uniform(-12, 12, 3), 0, 255)
                color = np.array([col[0], col[1], col[2], 255], dtype=np.uint8)
                add_card(pt, pt + u_axis * plen, vdir, hw, color)

        s, y = _bez(P0, P1, P2, 0.99)
        tip_pt = np.array([s * cs, y, s * sn])
        ds, dy = _bez_d(P0, P1, P2, 0.99)
        T = _nz(np.array([ds * cs, dy, ds * sn]))
        vdir = _nz(np.cross(n_p, T))
        col = np.clip(FOLIAGE_TIP + rng.uniform(-12, 12, 3), 0, 255)
        color = np.array([col[0], col[1], col[2], 255], dtype=np.uint8)
        add_card(tip_pt, tip_pt + T * (spacing * 1.2), vdir, spacing * 0.40, color)

    foliage_v = np.asarray(fv, dtype=float)
    foliage_f = np.asarray(ff, dtype=np.int64)
    foliage_c = np.asarray(fc, dtype=np.uint8)

    stipe_v, stipe_f = _concat(stipe_parts)

    # ---- small, flat rootball + a few short, downward rootlets -----------
    ico = trimesh.creation.icosphere(subdivisions=P["rb_sub"], radius=1.0)
    rv = ico.vertices.copy()
    rv *= (1.0 + rng.uniform(-0.16, 0.16, len(rv)))[:, None]
    rv[:, 0] *= ROOTBALL_R
    rv[:, 2] *= ROOTBALL_R
    rv[:, 1] = rv[:, 1] * (ROOTBALL_H * 0.5) + ROOTBALL_H * 0.5
    root_parts = [(rv, ico.faces.copy())]

    nr = P["rootlets"]
    for r in range(nr):
        a = (r / nr) * 2.0 * np.pi + rng.uniform(-0.3, 0.3)
        ca, sa = np.cos(a), np.sin(a)
        start = np.array([ROOTBALL_R * 0.40 * ca, ROOTBALL_H * 0.45,
                          ROOTBALL_R * 0.40 * sa])
        end = np.array([ROOTBALL_R * (0.9 + 0.3 * rng.random()) * ca, 0.0,
                        ROOTBALL_R * (0.9 + 0.3 * rng.random()) * sa])
        mid = 0.5 * (start + end) + np.array([0.0, -ROOTBALL_H * 0.28, 0.0])
        rl_pts = np.array([start, mid, end])
        rl_rad = np.array([0.0035, 0.0022, 0.0009])
        root_parts.append(_make_tube(rl_pts, rl_rad, max(4, P["sides"] - 1)))
    root_v, root_f = _concat(root_parts)

    min_y = min(stipe_v[:, 1].min(), foliage_v[:, 1].min(), root_v[:, 1].min())
    stipe_v[:, 1] -= min_y
    foliage_v[:, 1] -= min_y
    root_v[:, 1] -= min_y

    scene = trimesh.Scene()

    foliage = trimesh.Trimesh(vertices=foliage_v, faces=foliage_f, process=False)
    foliage.visual.face_colors = foliage_c
    scene.add_geometry(foliage, geom_name="foliage")

    stipe = trimesh.Trimesh(vertices=stipe_v, faces=stipe_f, process=True)
    stipe.visual.face_colors = STIPE_COLOR
    scene.add_geometry(stipe, geom_name="stipe")

    rootball = trimesh.Trimesh(vertices=root_v, faces=root_f, process=True)
    rootball.visual.face_colors = ROOT_COLOR
    scene.add_geometry(rootball, geom_name="rootball")

    return scene


# ============================================================================
# COLOR SAMPLING  (always read real colors out of the photo body)
# ============================================================================
def load_image_rgb(path):
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=float)


def _patch_medians(arr, x0, x1, y0, y1, n, p):
    """Median color of small p-radius patches over a grid inside the silhouette."""
    H, W, _ = arr.shape
    cols = np.linspace(x0, x1, n)
    rows = np.linspace(y0, y1, n)
    out = []
    for ry in rows:
        cy = int(ry * H)
        for rx in cols:
            cx = int(rx * W)
            xa, xb = max(0, cx - p), min(W, cx + p + 1)
            ya, yb = max(0, cy - p), min(H, cy + p + 1)
            patch = arr[ya:yb, xa:xb].reshape(-1, 3)
            if len(patch):
                out.append(np.median(patch, axis=0))
    return np.array(out) if out else np.zeros((0, 3))


def sample_greens(arr):
    """Foliage greens: green-dominant, non-grey patches; discard background.
    Brightened slightly toward fresh chartreuse to counter render shading."""
    meds = _patch_medians(arr, 0.12, 0.88, 0.06, 0.72, 28, 4)
    keep = []
    for m in meds:
        r, g, b = m
        spread = m.max() - m.min()
        if g > r + 6 and g > b + 1 and g > 45 and spread > 12:
            keep.append(m)
    if len(keep) < 6:
        for m in meds:
            r, g, b = m
            if g >= r and g > b and g > 40 and (m.max() - m.min()) > 8:
                keep.append(m)
    if not keep:
        keep = [np.array([95, 135, 60.]), np.array([70, 110, 45.]),
                np.array([150, 180, 90.])]
    out = []
    for k in keep:
        c = np.clip(k.astype(float) * 1.12 + np.array([6.0, 9.0, 2.0]), 0, 255)
        out.append(c)
    return out


def sample_brown(arr):
    """Rootball brown from the bottom-centre knot; warm, non-grey."""
    meds = _patch_medians(arr, 0.38, 0.62, 0.80, 0.97, 12, 4)
    keep = []
    for m in meds:
        r, g, b = m
        spread = m.max() - m.min()
        if r > b + 6 and r >= g - 6 and 45 < r < 210 and spread > 8:
            keep.append(m)
    if keep:
        return np.median(np.array(keep), axis=0)
    return np.array([95., 68., 45.])


def sample_tan(arr, brown):
    """Straw stipe colour; excludes green foliage. Falls back to lit brown."""
    meds = _patch_medians(arr, 0.40, 0.60, 0.58, 0.80, 12, 3)
    keep = []
    for m in meds:
        r, g, b = m
        spread = m.max() - m.min()
        if r > b + 8 and r >= g - 4 and g > b and r > 70 and (g - r) < 12 and spread > 10:
            keep.append(m)
    if keep:
        return np.median(np.array(keep), axis=0)
    c = brown * 1.45
    c[1] *= 0.98
    return np.clip(c, 0, 235)


# ============================================================================
# TILEABLE PROCEDURAL TEXTURES  (synthesised from the sampled palette)
# ============================================================================
def tileable_noise(W, H, rng, octaves=6, aniso=1.0):
    xs = np.linspace(0.0, 2.0 * np.pi, W, endpoint=False)
    ys = np.linspace(0.0, 2.0 * np.pi, H, endpoint=False)
    X, Y = np.meshgrid(xs, ys)
    acc = np.zeros((H, W))
    amp_sum = 0.0
    for o in range(octaves):
        fx = int(rng.integers(1, 4)) * (o + 1)
        fy = int(rng.integers(1, 4)) * (o + 1)
        amp = 1.0 / (o + 1)
        acc += amp * np.sin(fx * X * aniso + rng.uniform(0, 2 * np.pi)) \
                   * np.sin(fy * Y + rng.uniform(0, 2 * np.pi))
        amp_sum += amp
    acc /= amp_sum
    return (acc - acc.min()) / (np.ptp(acc) + 1e-9)


def build_wood_texture(color, rng, size=512):
    """Straw-wood with vertical grain and a 3x light/dark value range."""
    color = np.asarray(color, float)
    xs = np.linspace(0.0, 2.0 * np.pi, size, endpoint=False)
    ys = np.linspace(0.0, 2.0 * np.pi, size, endpoint=False)
    X, Y = np.meshgrid(xs, ys)
    grain = np.zeros((size, size))
    for o in range(6):
        fx = (o + 1) * int(rng.integers(2, 5))
        amp = 1.0 / (o + 1)
        grain += amp * np.sin(fx * X + rng.uniform(0, 2 * np.pi)
                              + 0.6 * np.sin(2 * Y + rng.uniform(0, 6.0)))
    grain = (grain - grain.min()) / (np.ptp(grain) + 1e-9)
    val = np.clip(0.55 + 0.75 * grain, 0.45, 1.4)
    rgb = np.clip(color[None, None, :] * val[:, :, None], 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, "RGB")
    d = ImageDraw.Draw(img)
    streak = tuple(int(v) for v in np.clip(color * 0.55, 0, 255))
    for _ in range(14):
        x = int(rng.integers(0, size))
        d.line([(x, 0), (x + int(rng.integers(-18, 18)), size)],
               fill=streak, width=int(rng.integers(1, 3)))
    return img


def build_fiber_texture(color, rng, size=512):
    """Fibrous rootball: mottled brown flecked with reddish and dark earthy tones."""
    color = np.asarray(color, float)
    base = tileable_noise(size, size, rng, octaves=6)
    val = 0.5 + 0.75 * base
    rgb = np.clip(color[None, None, :] * val[:, :, None], 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, "RGB")
    d = ImageDraw.Draw(img)
    dark = tuple(int(v) for v in np.clip(color * 0.35, 0, 255))
    red = tuple(int(v) for v in np.clip(color * np.array([1.3, 0.7, 0.6]), 0, 255))
    for _ in range(500):
        x, y = int(rng.integers(0, size)), int(rng.integers(0, size))
        r = int(rng.integers(1, 4))
        d.ellipse([x - r, y - r, x + r, y + r], fill=dark)
    for _ in range(220):
        x, y = int(rng.integers(0, size)), int(rng.integers(0, size))
        r = int(rng.integers(1, 3))
        d.ellipse([x - r, y - r, x + r, y + r], fill=red)
    for _ in range(8):  # wandering rootlet strands
        x, y = rng.uniform(0, size), rng.uniform(0, size)
        pts = []
        for _ in range(12):
            x += rng.uniform(-28, 28)
            y += rng.uniform(8, 28)
            pts.append((x % size, y % size))
        d.line(pts, fill=dark, width=2)
    return img


def albedo_to_normal(img, strength=2.2):
    """Sobel height (inverse luminance) -> tangent-space normal map."""
    a = np.asarray(img.convert("RGB"), float) / 255.0
    lum = a @ np.array([0.299, 0.587, 0.114])
    gx = np.zeros_like(lum)
    gy = np.zeros_like(lum)
    gx[:, 1:-1] = lum[:, 2:] - lum[:, :-2]
    gy[1:-1, :] = lum[2:, :] - lum[:-2, :]
    nx, ny, nz = -gx * strength, -gy * strength, np.ones_like(lum)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    out = np.stack([nx / ln, ny / ln, nz / ln], -1) * 0.5 + 0.5
    return Image.fromarray((out * 255).astype(np.uint8), "RGB")


def _tint(green, mult, warm):
    c = np.asarray(green, float) * mult
    c[0] += warm
    c[1] += warm * 0.4
    c[2] -= warm * 0.4
    c = np.clip(c, 0, 255)
    return (int(c[0]), int(c[1]), int(c[2]))


def build_foliage_atlas(greens, rng, size=1024):
    """4x4 atlas of distinct pinna clusters; binary alpha, sun/shade variation."""
    S = 4
    A = size * S
    tile = A // 4
    img = Image.new("RGBA", (A, A), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ng = len(greens)
    for tj in range(4):
        for ti in range(4):
            ox, oy = ti * tile, tj * tile
            sun = rng.uniform(0.86, 1.18)          # sunlit tiles brighter, none too dark
            warm = rng.uniform(-6.0, 18.0)         # sunlit warmer, shaded cooler
            base_x = ox + tile * (0.5 + rng.uniform(-0.05, 0.05))
            base_y = oy + tile * 0.93
            top_x = ox + tile * (0.5 + rng.uniform(-0.10, 0.10))
            top_y = oy + tile * 0.07
            nseg = 16
            curve = rng.uniform(-0.08, 0.08)
            stem = np.zeros((nseg + 1, 2))
            for s in range(nseg + 1):
                uu = s / nseg
                x = (1 - uu) * base_x + uu * top_x + math.sin(uu * math.pi) * tile * curve
                y = (1 - uu) * base_y + uu * top_y
                stem[s] = [x, y]
            rcol = _tint(greens[int(rng.integers(ng))], sun, warm)
            d.line([tuple(p) for p in stem], fill=rcol + (255,),
                   width=max(2, int(S * 0.9)))
            nleaf = int(rng.integers(15, 21))
            for li in range(nleaf):
                uu = (li + 0.5) / nleaf
                fi = uu * nseg
                i0 = min(int(fi), nseg - 1)
                fr = fi - i0
                sx = stem[i0, 0] * (1 - fr) + stem[i0 + 1, 0] * fr
                sy = stem[i0, 1] * (1 - fr) + stem[i0 + 1, 1] * fr
                tx = stem[i0 + 1, 0] - stem[i0, 0]
                ty = stem[i0 + 1, 1] - stem[i0, 1]
                tl = math.hypot(tx, ty) + 1e-9
                tx, ty = tx / tl, ty / tl
                scale = math.sin(math.pi * min(max(uu, 1e-3), 1 - 1e-3)) ** 0.6
                llen = tile * (0.10 + 0.24 * scale)
                lwid = tile * (0.022 + 0.046 * scale)
                for sgn in (-1.0, 1.0):
                    px, py = -ty * sgn, tx * sgn
                    dx, dy = px * 0.8 + tx * 0.6, py * 0.8 + ty * 0.6
                    dl = math.hypot(dx, dy) + 1e-9
                    dx, dy = dx / dl, dy / dl
                    tipx, tipy = sx + dx * llen, sy + dy * llen
                    m1 = (sx + dx * llen * 0.42 + px * lwid, sy + dy * llen * 0.42 + py * lwid)
                    m2 = (sx + dx * llen * 0.42 - px * lwid, sy + dy * llen * 0.42 - py * lwid)
                    col = _tint(greens[int(rng.integers(ng))], sun, warm)
                    d.polygon([(sx, sy), m1, (tipx, tipy), m2], fill=col + (255,))
                    vcol = _tint(greens[int(rng.integers(ng))], sun * 1.22 + 0.1, warm + 12)
                    d.line([(sx, sy), (tipx, tipy)], fill=vcol + (255,),
                           width=max(1, int(S * 0.7)))
    return img.resize((size, size), Image.LANCZOS)


# ============================================================================
# UVS + VERTEX COLORS
# ============================================================================
def foliage_uv(n_verts, rng, atlas_px=1024):
    """Map each card (4 verts: inner-, outer-, outer+, inner+) onto a random tile."""
    inset = 3.0 / atlas_px
    nq = n_verts // 4
    uv = np.zeros((n_verts, 2), dtype=np.float32)
    for q in range(nq):
        tile = int(rng.integers(0, 16))
        rot = int(rng.integers(0, 4))
        ti, tj = tile % 4, tile // 4
        u0, u1 = ti * 0.25 + inset, (ti + 1) * 0.25 - inset
        v0, v1 = tj * 0.25 + inset, (tj + 1) * 0.25 - inset
        corners = [(u0, v0), (u0, v1), (u1, v1), (u1, v0)]
        corners = corners[rot:] + corners[:rot]
        uv[q * 4:q * 4 + 4] = corners
    return uv


def triplanar_uv(verts, normals, scale):
    """Per-vertex triplanar projection baked to UVs (woody/earthy tiling)."""
    an = np.abs(normals)
    axis = np.argmax(an, axis=1)
    uv = np.zeros((len(verts), 2))
    m0, m1, m2 = axis == 0, axis == 1, axis == 2
    uv[m0, 0], uv[m0, 1] = verts[m0, 2], verts[m0, 1]
    uv[m1, 0], uv[m1, 1] = verts[m1, 0], verts[m1, 2]
    uv[m2, 0], uv[m2, 1] = verts[m2, 0], verts[m2, 1]
    return (uv * scale).astype(np.float32)


def foliage_vertex_colors(verts, rng):
    """COLOR_0 sun/shade: higher + more outward = brighter/warmer.
    Lifted floor so interior blades stay fresh green, not dark olive."""
    y = verts[:, 1]
    ymax = y.max() if y.max() > 0 else 1.0
    rad = np.sqrt(verts[:, 0] ** 2 + verts[:, 2] ** 2)
    rmax = rad.max() + 1e-9
    f = np.clip(0.45 + 0.40 * (y / ymax) + 0.20 * (rad / rmax), 0, 1)
    nq = len(verts) // 4
    if nq:
        f = np.clip(f + rng.uniform(-0.07, 0.07, nq).repeat(4)[:len(verts)], 0, 1)
    shade = np.array([120, 145, 95], float)
    sun = np.array([248, 250, 230], float)
    rgb = shade[None, :] + (sun - shade)[None, :] * f[:, None]
    out = np.empty((len(verts), 4), np.uint8)
    out[:, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[:, 3] = 255
    return out


def wood_vertex_colors(verts):
    """Slightly darker / cooler near the ground (ambient occlusion)."""
    y = verts[:, 1]
    f = 0.55 + 0.45 * np.clip(y / (y.max() + 1e-9), 0, 1)
    warm = np.array([1.0, 0.96, 0.9])
    rgb = np.clip(f[:, None] * warm[None, :] * 255, 0, 255).astype(np.uint8)
    out = np.empty((len(verts), 4), np.uint8)
    out[:, :3] = rgb
    out[:, 3] = 255
    return out


def root_vertex_colors(verts, rng):
    """Crevice darkening flecks on the rootball."""
    f = np.clip(0.7 + rng.uniform(-0.22, 0.18, len(verts)), 0.4, 1.0)
    warm = np.array([1.0, 0.94, 0.86])
    rgb = np.clip(f[:, None] * warm[None, :] * 255, 0, 255).astype(np.uint8)
    out = np.empty((len(verts), 4), np.uint8)
    out[:, :3] = rgb
    out[:, 3] = 255
    return out


# ============================================================================
# APPLY MATERIALS
# ============================================================================
def texture_scene(scene, arr, seed):
    rng = np.random.default_rng(seed + 1234)

    greens = sample_greens(arr)
    brown = sample_brown(arr)
    tan = sample_tan(arr, brown)

    atlas = build_foliage_atlas(greens, rng)
    wood = build_wood_texture(tan, rng)
    wood_n = albedo_to_normal(wood, 2.0)
    fiber = build_fiber_texture(brown, rng)
    fiber_n = albedo_to_normal(fiber, 2.6)

    fol_mat = trimesh.visual.material.PBRMaterial(
        name="foliage", baseColorTexture=atlas,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0, roughnessFactor=0.85,
        alphaMode="MASK", alphaCutoff=0.45, doubleSided=True)
    stipe_mat = trimesh.visual.material.PBRMaterial(
        name="stipe", baseColorTexture=wood, normalTexture=wood_n,
        metallicFactor=0.0, roughnessFactor=0.9, doubleSided=False)
    root_mat = trimesh.visual.material.PBRMaterial(
        name="rootball", baseColorTexture=fiber, normalTexture=fiber_n,
        metallicFactor=0.0, roughnessFactor=0.95, doubleSided=False)

    # --- foliage cards ---
    fol = scene.geometry["foliage"]
    fol_uv = foliage_uv(len(fol.vertices), rng)
    fol.visual = trimesh.visual.TextureVisuals(uv=fol_uv, material=fol_mat)
    fol.visual.vertex_attributes["color"] = foliage_vertex_colors(fol.vertices, rng)

    # --- woody stipes ---
    st = scene.geometry["stipe"]
    st_uv = triplanar_uv(st.vertices, st.vertex_normals, scale=6.0)
    st.visual = trimesh.visual.TextureVisuals(uv=st_uv, material=stipe_mat)
    st.visual.vertex_attributes["color"] = wood_vertex_colors(st.vertices)

    # --- fibrous rootball ---
    rb = scene.geometry["rootball"]
    rb_uv = triplanar_uv(rb.vertices, rb.vertex_normals, scale=9.0)
    rb.visual = trimesh.visual.TextureVisuals(uv=rb_uv, material=root_mat)
    rb.visual.vertex_attributes["color"] = root_vertex_colors(rb.vertices, rng)

    return scene


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Procedural textured fern -> GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    arr = load_image_rgb(args.image)
    scene = build_mesh(args.seed, args.density)
    texture_scene(scene, arr, args.seed)

    glb = scene.export(file_type="glb")
    with open(args.output, "wb") as fh:
        fh.write(glb)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)