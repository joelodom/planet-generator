#!/usr/bin/env python3
"""Blue-agave succulent rosette: procedural geometry + photo-derived tileable
materials (incl. an alpha-cutout foliage-card atlas), exported as a textured GLB.

    python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb
"""
import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw

# ===========================================================================
# GEOMETRY
# ===========================================================================
ROSETTE_DIAMETER  = 1.05                              # m, mature agave spread
ROSETTE_RADIUS    = ROSETTE_DIAMETER * 0.5
HEIGHT_OVER_WIDTH = 0.92                              # nearly as tall as wide (photo ~0.99)
ROSETTE_HEIGHT    = ROSETTE_DIAMETER * HEIGHT_OVER_WIDTH

LEAF_LEN_OUTER    = 0.48
LEAF_LEN_INNER    = 0.70
LEAF_HALFW_OUTER  = 0.105                             # broad fleshy blades
LEAF_HALFW_INNER  = 0.080
THICK_RATIO       = 0.50

ELEV_OUTER_DEG    = 46.0                              # raised: less flat splay
ELEV_INNER_DEG    = 86.0
BEND_OUTER_DEG    = 54.0
BEND_INNER_DEG    = 8.0

CROWN_RADIUS      = 0.080
CROWN_VRAD        = 0.05

GOLDEN_ANGLE      = np.deg2rad(137.507764)
TARGET_ASPECT     = 1.0                               # final width/height target

PROF_U = np.array([-1.00,  0.00,  1.00,  0.55,  0.00, -0.55])
PROF_N = np.array([ 0.15, -0.05,  0.15, -0.55, -1.00, -0.55])
N_CROSS = 6
U_LOOP = 0.5 + 0.5 * PROF_U
V_PER_M = 5.0

CARD_HALF = 0.030 * ROSETTE_DIAMETER                  # small central-tuft leaflets


def _density_params(density):
    if density == "high":
        return dict(n_leaves=72, n_stations=16, teeth=9, crown_sub=2, spine_sides=6,
                    n_clumps=6, cards=10)
    if density == "med":
        return dict(n_leaves=46, n_stations=10, teeth=5, crown_sub=1, spine_sides=5,
                    n_clumps=4, cards=8)
    if density == "low":
        return dict(n_leaves=24, n_stations=6,  teeth=0, crown_sub=0, spine_sides=4,
                    n_clumps=3, cards=5)
    raise ValueError("density must be 'high', 'med' or 'low'")


def _lerp(a, b, t):
    return a + (b - a) * t


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _width_profile(t):
    """Broad lance blade: wide through the body, sharp tip."""
    base, tp, tip = 0.50, 0.22, 0.03
    rise = base + (1.0 - base) * (t / tp)
    fall = tip + (1.0 - tip) * np.clip(1.0 - (t - tp) / (1.0 - tp), 0.0, 1.0) ** 0.7
    return np.where(t < tp, rise, fall)


def _rgba(tone, chan):
    rgb = np.clip(255.0 * tone * np.asarray(chan), 0.0, 255.0)
    return np.array([rgb[0], rgb[1], rgb[2], 255.0])


def _cone(base_c, axis, length, radius, sides):
    d = _unit(np.asarray(axis, float))
    ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    p1 = _unit(np.cross(d, ref))
    p2 = np.cross(d, p1)
    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = base_c + radius * (np.cos(ang)[:, None] * p1 + np.sin(ang)[:, None] * p2)
    verts = np.vstack([ring, base_c[None, :], (base_c + d * length)[None, :]])
    ci, ti = sides, sides + 1
    f = []
    for k in range(sides):
        kn = (k + 1) % sides
        f.append((k, kn, ti))
        f.append((ci, kn, k))
    uv = np.zeros((sides + 2, 2))
    uv[:sides, 0] = np.arange(sides) / max(1, sides)
    uv[:sides, 1] = 0.35
    uv[ci] = (0.5, 0.0)
    uv[ti] = (0.5, 1.0)
    return verts, np.array(f, int), uv


class _Acc:
    def __init__(self):
        self.v, self.f, self.uv, self.c, self.n = [], [], [], [], 0

    def add(self, v, f, uv, c):
        v = np.asarray(v, float)
        f = np.asarray(f, int)
        self.v.append(v)
        self.f.append(f + self.n)
        self.uv.append(np.asarray(uv, float))
        self.c.append(np.asarray(c, float))
        self.n += len(v)

    def arrays(self):
        return (np.vstack(self.v), np.vstack(self.f),
                np.vstack(self.uv), np.vstack(self.c))


def _atlas_tile_uv(tile, k_rot):
    inset = 0.004
    col, row = tile % 4, tile // 4
    u0, v0, span = col * 0.25 + inset, row * 0.25 + inset, 0.25 - 2 * inset
    base = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    rot = base[k_rot:] + base[:k_rot]
    return np.array([(u0 + uu * span, v0 + vv * span) for uu, vv in rot])


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    p = _density_params(density)
    S = p["n_stations"]

    leaf = _Acc()
    spine = _Acc()
    foliage = _Acc()

    t = np.linspace(0.0, 1.0, S)
    wprof = _width_profile(t)
    start = rng.uniform(0.0, 2.0 * np.pi)
    nL = p["n_leaves"]

    for i in range(nL):
        frac = i / (nL - 1) if nL > 1 else 0.0

        azim = start + i * GOLDEN_ANGLE + rng.normal(0.0, np.deg2rad(3.0))
        elev = np.deg2rad(_lerp(ELEV_OUTER_DEG, ELEV_INNER_DEG, frac) + rng.normal(0.0, 2.5))
        bend = np.deg2rad(_lerp(BEND_OUTER_DEG, BEND_INNER_DEG, frac) * (1.0 + rng.normal(0.0, 0.07)))
        L    = _lerp(LEAF_LEN_OUTER, LEAF_LEN_INNER, frac) * (1.0 + rng.normal(0.0, 0.05))
        hw   = _lerp(LEAF_HALFW_OUTER, LEAF_HALFW_INNER, frac) * (1.0 + rng.normal(0.0, 0.05))
        curl = rng.normal(0.0, np.deg2rad(8.0))

        a = azim + curl * (t ** 2)
        phi = elev - bend * t
        T = np.stack([np.cos(phi) * np.cos(a), np.sin(phi), np.cos(phi) * np.sin(a)], axis=1)
        ds = L / (S - 1)
        p0 = np.array([CROWN_RADIUS * 0.45 * np.cos(azim), 0.045,
                       CROWN_RADIUS * 0.45 * np.sin(azim)])
        pts = p0 + np.vstack([np.zeros(3), np.cumsum(T[:-1] * ds, axis=0)])

        U = np.stack([-np.sin(a), np.zeros_like(a), np.cos(a)], axis=1)
        Nn = np.cross(U, T)
        Nn /= np.linalg.norm(Nn, axis=1, keepdims=True)

        W = hw * wprof
        H = W * THICK_RATIO
        ring = (pts[:, None, :]
                + U[:, None, :] * (PROF_U[None, :, None] * W[:, None, None])
                + Nn[:, None, :] * (PROF_N[None, :, None] * H[:, None, None]))

        rv = ring.reshape(-1, 3)
        base_c = pts[0]
        apex = pts[-1] + T[-1] * L * 0.01
        verts = np.vstack([rv, base_c[None, :], apex[None, :]])
        bc_i, ap_i = S * N_CROSS, S * N_CROSS + 1

        f = []
        for s in range(S - 1):
            for k in range(N_CROSS):
                kn = (k + 1) % N_CROSS
                a0 = s * N_CROSS + k
                b0 = s * N_CROSS + kn
                c0 = (s + 1) * N_CROSS + kn
                d0 = (s + 1) * N_CROSS + k
                f.append((a0, b0, c0))
                f.append((a0, c0, d0))
        for k in range(N_CROSS):
            f.append((bc_i, (k + 1) % N_CROSS, k))
        last = (S - 1) * N_CROSS
        for k in range(N_CROSS):
            f.append((ap_i, last + k, last + (k + 1) % N_CROSS))

        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        vcum = np.concatenate([[0.0], np.cumsum(seg)]) * V_PER_M
        uv_ring = np.stack([np.tile(U_LOOP, S), np.repeat(vcum, 6)], axis=1)
        uv = np.vstack([uv_ring, [0.5, 0.0], [0.5, vcum[-1] * 1.02]])

        along = np.concatenate([np.repeat(t, 6), [0.0], [1.0]])
        yv = verts[:, 1]
        base_tone = 0.84 + 0.12 * (1.0 - frac)
        leaf_rand = float(rng.normal(0.0, 0.02))
        tone = base_tone + 0.07 * np.clip(yv / ROSETTE_HEIGHT, 0, 1) + 0.04 * along + leaf_rand
        tone = np.clip(tone, 0.72, 1.0)
        warmth = 0.04 * (1.0 - frac) - 0.02
        chan = np.array([1.0 + warmth, 1.0, 1.0 - 0.6 * warmth])
        rgb = np.clip(255.0 * tone[:, None] * chan[None, :], 0, 255)
        col = np.concatenate([rgb, np.full((len(verts), 1), 255.0)], axis=1)
        leaf.add(verts, np.array(f, int), uv, col)

        # short dark terminal spine (a tip, not a quill)
        sv, sf, suv = _cone(pts[-1], T[-1], L * 0.04, hw * 0.12, p["spine_sides"])
        s_tone = 0.92 + float(rng.normal(0.0, 0.03))
        s_col = np.tile(_rgba(s_tone, [1.03, 1.0, 0.96]), (len(sv), 1))
        spine.add(sv, sf, suv, s_col)

        # small marginal teeth, nearly flush with the edge
        if p["teeth"] > 0:
            sts = np.unique(np.linspace(2, S - 3, p["teeth"]).round().astype(int))
            left, right = ring[:, 0, :], ring[:, 2, :]
            t_uv = np.array([[0.1, 0.1], [0.6, 0.1], [0.1, 0.6], [0.35, 0.9]])
            t_face = np.array([(0, 1, 2), (0, 1, 3), (1, 2, 3), (2, 0, 3)])
            for st in sts:
                for M, sgn in ((left[st], -1.0), (right[st], 1.0)):
                    outw = _unit(sgn * U[st] * 0.9 + T[st] * 0.35 + Nn[st] * 0.1)
                    tb = hw * 0.05
                    tv = np.array([M + T[st] * tb, M - T[st] * tb,
                                   M + Nn[st] * tb, M + outw * hw * 0.18])
                    t_col = np.tile(_rgba(0.88, [1.03, 1.0, 0.95]), (4, 1))
                    spine.add(tv, t_face, t_uv, t_col)

    # central crown
    ico = trimesh.creation.icosphere(subdivisions=p["crown_sub"], radius=CROWN_RADIUS)
    cv = ico.vertices.copy()
    cv[:, 1] *= (CROWN_VRAD / CROWN_RADIUS)
    cv[:, 1] += CROWN_VRAD * 0.6
    c_uv = np.stack([cv[:, 0] * 3.0 + 0.5, cv[:, 2] * 3.0 + 0.5], axis=1)
    c_col = np.tile(_rgba(0.80, [0.98, 1.0, 1.02]), (len(cv), 1))
    leaf.add(cv, ico.faces, c_uv, c_col)

    # ---- foliage CARDS: tiny, tight, upright central tuft (alpha-cutout) ----
    quad_face = np.array([(0, 1, 2), (0, 2, 3)])
    up0 = np.array([0.0, 1.0, 0.0])
    for ci in range(p["n_clumps"]):
        ca = start * 1.3 + ci * GOLDEN_ANGLE
        hf = 0.50 + 0.45 * rng.random()
        cy = hf * ROSETTE_HEIGHT
        cr = ROSETTE_RADIUS * 0.10 * rng.random()
        centre = np.array([cr * np.cos(ca), cy, cr * np.sin(ca)])
        for _ in range(p["cards"]):
            off = rng.normal(0.0, 1.0, 3) * (0.05 * ROSETTE_DIAMETER)
            off[1] *= 0.8
            pos = centre + off
            radial = np.array([np.cos(ca), 0.0, np.sin(ca)])
            normal = _unit(radial * 0.4 + up0 * 1.0 + rng.normal(0.0, 0.15, 3))
            up_card = up0 - normal * float(np.dot(up0, normal))
            up_card = _unit(up_card) if np.linalg.norm(up_card) > 1e-6 else np.array([0.0, 0.0, 1.0])
            right_card = _unit(np.cross(up_card, normal))
            sc = float(np.exp(rng.normal(0.0, 0.2)))
            rh, uh = CARD_HALF * 0.5 * sc, CARD_HALF * 1.6 * sc
            q = np.array([pos - right_card * rh - up_card * uh,
                          pos + right_card * rh - up_card * uh,
                          pos + right_card * rh + up_card * uh,
                          pos - right_card * rh + up_card * uh])
            tile = int(rng.integers(0, 16))
            uv = _atlas_tile_uv(tile, int(rng.integers(0, 4)))
            tone = np.clip(0.86 + 0.14 * hf + float(rng.normal(0.0, 0.03)), 0.7, 1.05)
            ccol = np.tile(_rgba(tone, [1.0 + 0.03 * (1 - hf), 1.0, 1.0 - 0.02]), (4, 1))
            foliage.add(q, quad_face, uv, ccol)

    # ---- assemble ----
    lv, lf, luv, lc = leaf.arrays()
    sv, sf, suv, sc = spine.arrays()
    fv, ff, fuv, fc = foliage.arrays()
    leaves_mesh = trimesh.Trimesh(lv, lf, process=False)
    spine_mesh = trimesh.Trimesh(sv, sf, process=False)
    foliage_mesh = trimesh.Trimesh(fv, ff, process=False)
    for m in (leaves_mesh, spine_mesh):
        try:
            trimesh.repair.fix_normals(m)
        except Exception:
            pass

    meshes = (leaves_mesh, spine_mesh, foliage_mesh)
    allv = np.vstack([m.vertices for m in meshes])
    miny = allv[:, 1].min()
    cx = 0.5 * (allv[:, 0].min() + allv[:, 0].max())
    cz = 0.5 * (allv[:, 2].min() + allv[:, 2].max())
    for m in meshes:
        m.apply_translation([-cx, -miny, -cz])

    # deterministic proportion fix: shrink footprint so width ~= height
    allv = np.vstack([m.vertices for m in meshes])
    width = max(np.ptp(allv[:, 0]), np.ptp(allv[:, 2]))
    height = np.ptp(allv[:, 1])
    if width > 1e-6:
        s = float(np.clip((height / width) * TARGET_ASPECT, 0.72, 1.0))
        if s < 0.999:
            scale = np.diag([s, 1.0, s, 1.0])
            for m in meshes:
                m.apply_transform(scale)

    leaves_mesh.visual = trimesh.visual.TextureVisuals(uv=luv)
    leaves_mesh.visual.vertex_attributes["color"] = lc.astype(np.uint8)
    spine_mesh.visual = trimesh.visual.TextureVisuals(uv=suv)
    spine_mesh.visual.vertex_attributes["color"] = sc.astype(np.uint8)
    foliage_mesh.visual = trimesh.visual.TextureVisuals(uv=fuv)
    foliage_mesh.visual.vertex_attributes["color"] = fc.astype(np.uint8)

    scene = trimesh.Scene()
    scene.add_geometry(leaves_mesh, geom_name="leaves")
    scene.add_geometry(spine_mesh, geom_name="spines")
    scene.add_geometry(foliage_mesh, geom_name="foliage")
    return scene


# ===========================================================================
# TEXTURING
# ===========================================================================
def _periodic_noise(size, rng, n_waves, fmin, fmax):
    xs = np.linspace(0.0, 1.0, size, endpoint=False)
    X, Y = np.meshgrid(xs, xs)
    acc = np.zeros((size, size))
    for _ in range(n_waves):
        fx = int(rng.integers(fmin, fmax + 1))
        fy = int(rng.integers(fmin, fmax + 1))
        sy = 1.0 if rng.random() < 0.5 else -1.0
        ph = rng.uniform(0.0, 2.0 * np.pi)
        acc += (1.0 / (1.0 + fx + fy)) * np.sin(2.0 * np.pi * (fx * X + sy * fy * Y) + ph)
    acc -= acc.min()
    m = acc.max()
    return acc / m if m > 1e-9 else acc


def _stripes(size, rng, n_waves, fmin, fmax):
    xs = np.linspace(0.0, 1.0, size, endpoint=False)
    X, Y = np.meshgrid(xs, xs)
    acc = np.zeros((size, size))
    for _ in range(n_waves):
        fx = int(rng.integers(fmin, fmax + 1))
        ph = rng.uniform(0.0, 2.0 * np.pi)
        wob = rng.uniform(0.0, 0.6)
        acc += (1.0 / (1.0 + fx)) * np.sin(2.0 * np.pi * fx * X + 0.6 * wob * np.sin(2.0 * np.pi * Y) + ph)
    acc -= acc.min()
    m = acc.max()
    return acc / m if m > 1e-9 else acc


def _normal_from_height(h, strength):
    gx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.5
    gy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 0.5
    nx, ny, nz = -gx * strength, -gy * strength, np.ones_like(h)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    out = np.stack([nx * inv * 0.5 + 0.5, ny * inv * 0.5 + 0.5, nz * inv * 0.5 + 0.5], axis=2)
    return Image.fromarray(np.clip(out * 255.0, 0, 255).astype(np.uint8), "RGB")


def _sample_palette(image_path):
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float64)
    H, W = arr.shape[:2]
    centres = [(0.50, 0.52), (0.44, 0.46), (0.56, 0.46), (0.50, 0.42),
               (0.46, 0.60), (0.54, 0.60), (0.50, 0.66), (0.42, 0.52),
               (0.58, 0.52), (0.50, 0.58)]
    half = max(4, int(0.018 * min(H, W)))
    meds = []
    for cx, cy in centres:
        px, py = int(cx * W), int(cy * H)
        x0, x1 = max(0, px - half), min(W, px + half)
        y0, y1 = max(0, py - half), min(H, py + half)
        reg = arr[y0:y1, x0:x1].reshape(-1, 3)
        if len(reg):
            meds.append(np.median(reg, axis=0))
    meds = np.array(meds)
    gmed = np.median(meds, axis=0)
    dist = np.linalg.norm(meds - gmed, axis=1)
    keep = meds[dist <= max(35.0, np.median(dist) * 1.5)]
    if len(keep) < 3:
        keep = meds
    base = np.median(keep, axis=0)
    lum = keep @ np.array([0.299, 0.587, 0.114])
    order = np.argsort(lum)
    dark = keep[order[int(0.20 * (len(keep) - 1))]]
    light = keep[order[int(0.80 * (len(keep) - 1))]]
    light = np.clip(base + (light - base) * 1.3, 0, 255)
    dark = np.clip(base + (dark - base) * 1.3, 0, 255)
    if float(np.linalg.norm(light - dark)) < 14.0:
        light = np.clip(base * 1.12, 0, 255)
        dark = np.clip(base * 0.85, 0, 255)
    return base, light, dark


def _glaucous_lift(col):
    """Lighten + push toward chalky powder-blue (keeps photo-derived hue)."""
    glauc = np.array([198.0, 210.0, 211.0])
    col = col * 0.85 + glauc * 0.15
    return np.clip(col * 1.08, 0, 255)


def _leaf_textures(base, light, dark, seed):
    rng = np.random.default_rng(seed + 101)
    size = 1024
    mott = _periodic_noise(size, rng, 12, 1, 4)
    fine = _periodic_noise(size, rng, 10, 5, 14)
    stri = _stripes(size, rng, 9, 2, 9)
    bloom = _periodic_noise(size, rng, 6, 1, 3)

    t1 = np.clip(0.55 * mott + 0.28 * stri + 0.17 * fine, 0, 1)[..., None]
    col = dark[None, None, :] * (1 - t1) + light[None, None, :] * t1
    bl = (np.clip((bloom - 0.45) / 0.55, 0, 1) * 0.5)[..., None]
    lum = col.mean(axis=2, keepdims=True)
    col = col * (1 - 0.45 * bl) + lum * (0.45 * bl)
    col = col * (1 + 0.10 * bl)
    col = col * (1 - 0.04 * (stri[..., None] - 0.5))
    col = _glaucous_lift(col).astype(np.uint8)
    h = col.mean(axis=2) / 255.0
    return Image.fromarray(col, "RGB"), _normal_from_height(h, 1.8)


def _spine_texture(base, seed):
    rng = np.random.default_rng(seed + 202)
    size = 512
    Lb = float(base @ np.array([0.299, 0.587, 0.114]))
    wood_dark = np.maximum(np.array([Lb * 0.32, Lb * 0.25, Lb * 0.19]), [28.0, 22.0, 16.0])
    wood_light = np.array([Lb * 0.88, Lb * 0.68, Lb * 0.48])
    grain = _stripes(size, rng, 11, 3, 16)
    noise = _periodic_noise(size, rng, 8, 2, 10)
    t = np.clip(0.7 * grain + 0.3 * noise, 0, 1)[..., None]
    col = wood_dark[None, None, :] * (1 - t) + wood_light[None, None, :] * t
    return Image.fromarray(np.clip(col, 0, 255).astype(np.uint8), "RGB")


def _draw_leaflet(N, fill, rng):
    im = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    s = np.linspace(0.0, 1.0, 14)
    hw = 0.26 * (np.sqrt(np.clip(s, 0, 1)) * (1.0 - 0.5 * s)) * (1.0 + 0.25 * np.sin(np.pi * s))
    y = 0.05 + 0.9 * s
    fr, fg, fb = int(fill[0]), int(fill[1]), int(fill[2])
    rpts, lpts = [], []
    for k in range(len(s)):
        bump = (0.014 if k % 2 == 0 else 0.0) + 0.004 * rng.standard_normal()
        w = max(0.0, hw[k] + bump)
        rpts.append(((0.5 + w) * N, y[k] * N))
        lpts.append(((0.5 - w) * N, y[k] * N))
    poly = rpts + [(0.5 * N, 0.985 * N)] + lpts[::-1]
    dr.polygon(poly, fill=(fr, fg, fb, 255))
    dr.line([(0.5 * N, 0.08 * N), (0.5 * N, 0.95 * N)],
            fill=(int(fr * 0.8), int(fg * 0.8), int(fb * 0.74), 255), width=max(1, int(N * 0.02)))
    return np.asarray(im)


def _foliage_atlas(base, light, dark, seed):
    rng = np.random.default_rng(seed + 303)
    tile, ss = 256, 4
    atlas = np.zeros((tile * 4, tile * 4, 4), np.uint8)
    for row in range(4):
        for col in range(4):
            bright = 1.12 - 0.20 * (row / 3.0)
            warm = 0.06 - 0.12 * (col / 3.0)
            if bright >= 1.0:
                cc = base + (light - base) * min(1.0, (bright - 1.0) / 0.12)
            else:
                cc = base + (dark - base) * min(1.0, (1.0 - bright) / 0.30)
            chan = np.array([1.0 + warm, 1.0, 1.0 - 0.5 * warm])
            fill = _glaucous_lift(cc * chan)
            big = _draw_leaflet(tile * ss, fill, rng)
            small = Image.fromarray(big, "RGBA").resize((tile, tile), Image.LANCZOS)
            atlas[row * tile:(row + 1) * tile, col * tile:(col + 1) * tile] = np.asarray(small)
    return Image.fromarray(atlas, "RGBA")


def apply_textures(scene, image_path, seed):
    from trimesh.visual.material import PBRMaterial
    base, light, dark = _sample_palette(image_path)
    leaf_img, leaf_nrm = _leaf_textures(base, light, dark, seed)
    spine_img = _spine_texture(base, seed)
    atlas_img = _foliage_atlas(base, light, dark, seed)

    leaf_mat = PBRMaterial(name="leaves", baseColorTexture=leaf_img, normalTexture=leaf_nrm,
                           metallicFactor=0.0, roughnessFactor=0.78, doubleSided=True)
    spine_mat = PBRMaterial(name="spines", baseColorTexture=spine_img,
                            metallicFactor=0.0, roughnessFactor=0.9, doubleSided=True)
    foliage_mat = PBRMaterial(name="foliage", baseColorTexture=atlas_img,
                              metallicFactor=0.0, roughnessFactor=0.8,
                              alphaMode="MASK", alphaCutoff=0.45, doubleSided=True)
    mats = {"leaves": leaf_mat, "spines": spine_mat, "foliage": foliage_mat}
    for name, mesh in scene.geometry.items():
        mesh.visual.material = mats[name]
    return scene


# ===========================================================================
# CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Procedural textured blue-agave rosette -> GLB")
    ap.add_argument("--image", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    try:
        scene = build_mesh(args.seed, args.density)
        apply_textures(scene, args.image, args.seed)
        glb = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(glb)
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1
    print("wrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())