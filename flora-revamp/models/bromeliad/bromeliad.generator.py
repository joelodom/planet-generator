"""
Bromeliad (Guzmania-type) -- procedural geometry + photo-derived materials,
exported as a single textured GLB.

A compact, mounded rosette of broad, upward-arching strap leaves (deep green
at the base flushing to coral / brick-red at the tips) radiating from a tight
central cup, with a short, broad, flame-coloured bract cluster nested at the
heart.

CLI:
    python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb

Surfaces (scene geometry names): "leaves", "bracts", "throat".
Only numpy / trimesh / PIL / stdlib are used.
"""

import argparse
import sys
import traceback

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter


# ==========================================================================
# GEOMETRY
# ==========================================================================

# --- Measured proportions (read off the reference image) -------------------
# The plant is a COMPACT rosette: front silhouette nearly as tall as wide
# (target width/height ~1.1-1.3), NOT a flat splayed star.
PLANT_WIDTH        = 0.36   # m, full rosette spread (emergent)
HEIGHT_OVER_WIDTH  = 0.80   # compact mound: height close to width
THROAT_TOP_RADIUS  = 0.044  # m
THROAT_HEIGHT      = 0.085  # m

# Foliage envelope: a lobed dome (the gently lobed "star" outline).
ENV_LOBES_5        = 0.08
ENV_LOBES_3        = 0.05
ENV_PROTRUDE       = 0.03


def _grid_faces(n_len, n_across, offset=0):
    faces = []
    for i in range(n_len - 1):
        for j in range(n_across - 1):
            a = offset + i * n_across + j
            b = a + 1
            c = a + n_across
            d = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d])
    return faces


def _curved_card(theta, r0, base_h, length, arch_up_deg, tip_down_deg,
                 max_width, width_base, width_peak_pos, taper_pow,
                 channel, n_len, n_across, sway_deg, twist_deg):
    """Curved, channelled, lance-shaped card (leaf or bract).

    Tangent angle goes from +arch_up_deg (rising) to -tip_down_deg at the tip.
    `length` is the arc length in metres. Returns verts, faces, s, u.
    """
    s = np.linspace(0.0, 1.0, n_len)
    phi = np.deg2rad(arch_up_deg + (-tip_down_deg - arch_up_deg) * s)
    dl = 1.0 / (n_len - 1)
    rho_raw = np.cumsum(np.cos(phi) * dl); rho_raw -= rho_raw[0]
    y_raw = np.cumsum(np.sin(phi) * dl);   y_raw -= y_raw[0]

    rho = r0 + rho_raw * length
    yy = base_h + y_raw * length

    th = theta + np.deg2rad(sway_deg) * s
    er = np.stack([np.cos(th), np.zeros_like(th), np.sin(th)], axis=1)
    centre = er * rho[:, None] + np.stack(
        [np.zeros_like(yy), yy, np.zeros_like(yy)], axis=1)

    T = np.gradient(centre, axis=0)
    T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    eaz = np.stack([-np.sin(th), np.zeros_like(th), np.cos(th)], axis=1)
    Nn = np.cross(T, eaz)
    Nn /= (np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-9)
    tw = np.deg2rad(twist_deg) * s
    B = eaz * np.cos(tw)[:, None] + Nn * np.sin(tw)[:, None]
    Nn2 = -eaz * np.sin(tw)[:, None] + Nn * np.cos(tw)[:, None]

    wp = np.where(
        s <= width_peak_pos,
        width_base + (1.0 - width_base) * (s / max(width_peak_pos, 1e-6)),
        (1.0 - (s - width_peak_pos) / max(1.0 - width_peak_pos, 1e-6)) ** taper_pow,
    )
    wp = np.clip(wp, 0.0, 1.0)
    halfw = 0.5 * max_width * wp

    u = np.linspace(-1.0, 1.0, n_across)
    verts = np.zeros((n_len, n_across, 3))
    for j, uu in enumerate(u):
        side = B * (uu * halfw)[:, None]
        chan = channel * (uu * uu) * max_width
        verts[:, j, :] = centre + side + Nn2 * chan

    verts = verts.reshape(-1, 3)
    s_grid = np.repeat(s[:, None], n_across, axis=1).reshape(-1)
    u_grid = np.repeat(u[None, :], n_len, axis=0).reshape(-1)
    return verts, _grid_faces(n_len, n_across, 0), s_grid, u_grid


def _lathe(profile, seg):
    ang = np.linspace(0.0, 2.0 * np.pi, seg, endpoint=False)
    verts, rows = [], []
    for (r, y) in profile:
        if r <= 1e-9:
            verts.append([0.0, y, 0.0]); rows.append(("apex", len(verts) - 1))
        else:
            row = []
            for a in ang:
                verts.append([r * np.cos(a), y, r * np.sin(a)]); row.append(len(verts) - 1)
            rows.append(("ring", row))
    faces = []
    for k in range(len(rows) - 1):
        t0, v0 = rows[k]; t1, v1 = rows[k + 1]
        if t0 == "apex":
            for i in range(seg):
                faces.append([v0, v1[i], v1[(i + 1) % seg]])
        elif t1 == "apex":
            for i in range(seg):
                faces.append([v0[i], v1, v0[(i + 1) % seg]])
        else:
            for i in range(seg):
                a, b = v0[i], v0[(i + 1) % seg]
                c, d = v1[i], v1[(i + 1) % seg]
                faces.append([a, c, b]); faces.append([b, c, d])
    return np.array(verts, dtype=float), np.array(faces, dtype=np.int64)


def _envelope_factor(theta, ph5, ph3):
    return 1.0 + ENV_LOBES_5 * np.sin(5.0 * theta + ph5) \
               + ENV_LOBES_3 * np.sin(3.0 * theta + ph3)


def _build_leaves(rng, rings):
    Vs, Fs, off = [], [], 0
    s_a, u_a, ring_a, tile_a, shade_a, jit_a = [], [], [], [], [], []
    ph5 = rng.uniform(0, 2 * np.pi)
    ph3 = rng.uniform(0, 2 * np.pi)
    n_rings = len(rings)

    for ri, (count, base_len, up, down, bh, r0, mw, nl, na) in enumerate(rings):
        ring_frac = ri / max(n_rings - 1, 1)          # 0 outer .. 1 inner
        ang0 = rng.uniform(0, 2 * np.pi) + ri * np.pi / max(count, 1)
        for k in range(count):
            theta = ang0 + 2 * np.pi * k / count + rng.uniform(-0.08, 0.08)
            length = base_len * _envelope_factor(theta, ph5, ph3) \
                * float(rng.lognormal(0.0, 0.07)) * (1.0 - ENV_PROTRUDE * 0.5)
            verts, faces, s_g, u_g = _curved_card(
                theta=theta,
                r0=r0 * rng.uniform(0.85, 1.15),
                base_h=bh * rng.uniform(0.85, 1.15),
                length=length,
                arch_up_deg=up + rng.uniform(-5, 5),
                tip_down_deg=down + rng.uniform(-7, 7),
                max_width=mw * float(rng.lognormal(0.0, 0.10)),
                width_base=0.42, width_peak_pos=0.45, taper_pow=1.3,
                channel=0.16, n_len=nl, n_across=na,
                sway_deg=rng.uniform(-8, 8), twist_deg=rng.uniform(-8, 8),
            )
            nv = len(verts)
            tc = int(np.clip(round((1.0 - ring_frac) * 3) + int(rng.integers(-1, 2)), 0, 3))
            tr = int(rng.integers(0, 4))
            tile = tr * 4 + tc
            shade = float(rng.uniform(0.90, 1.10))
            Vs.append(verts); Fs.append(np.array(faces, dtype=np.int64) + off)
            s_a.append(s_g); u_a.append(u_g)
            ring_a.append(np.full(nv, ring_frac))
            tile_a.append(np.full(nv, tile, dtype=np.int64))
            shade_a.append(np.full(nv, shade))
            jit_a.append(rng.random(nv))
            off += nv

    meta = dict(s=np.concatenate(s_a), u=np.concatenate(u_a),
                ring=np.concatenate(ring_a), tile=np.concatenate(tile_a),
                shade=np.concatenate(shade_a), jit=np.concatenate(jit_a))
    return np.concatenate(Vs), np.concatenate(Fs), meta


def _build_bracts(rng, whorls, nl, na):
    Vs, Fs, off = [], [], 0
    s_a, u_a, warm_a, tile_a, shade_a = [], [], [], [], []
    for (count, br, length, up, down, bh, mw, warmth) in whorls:
        ang0 = rng.uniform(0, 2 * np.pi)
        for k in range(count):
            theta = ang0 + 2 * np.pi * k / count + rng.uniform(-0.12, 0.12)
            verts, faces, s_g, u_g = _curved_card(
                theta=theta,
                r0=br * rng.uniform(0.9, 1.1),
                base_h=bh,
                length=length * float(rng.lognormal(0.0, 0.08)),
                arch_up_deg=up + rng.uniform(-5, 5),
                tip_down_deg=down + rng.uniform(-6, 6),
                max_width=mw * float(rng.lognormal(0.0, 0.10)),
                width_base=0.40, width_peak_pos=0.38, taper_pow=1.6,
                channel=0.20, n_len=nl, n_across=na,
                sway_deg=rng.uniform(-6, 6), twist_deg=rng.uniform(-6, 6),
            )
            nv = len(verts)
            tc = int(np.clip(round(warmth * 3) + int(rng.integers(-1, 2)), 0, 3))
            tr = int(rng.integers(0, 4))
            Vs.append(verts); Fs.append(np.array(faces, dtype=np.int64) + off)
            s_a.append(s_g); u_a.append(u_g)
            warm_a.append(np.full(nv, warmth))
            tile_a.append(np.full(nv, tr * 4 + tc, dtype=np.int64))
            shade_a.append(np.full(nv, float(rng.uniform(0.9, 1.1))))
            off += nv
    meta = dict(s=np.concatenate(s_a), u=np.concatenate(u_a),
                warmth=np.concatenate(warm_a), tile=np.concatenate(tile_a),
                shade=np.concatenate(shade_a))
    return np.concatenate(Vs), np.concatenate(Fs), meta


def _build_throat(seg):
    r, h = THROAT_TOP_RADIUS, THROAT_HEIGHT
    profile = [(0.0, 0.012), (0.018, 0.0), (0.026, 0.27 * h),
               (0.034, 0.55 * h), (r * 0.95, 0.82 * h), (r, h)]
    return _lathe(profile, seg)


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)

    if density == "high":
        # (count, length, arch_up, tip_down, base_h, r0, max_w, n_len, n_across)
        rings = [(16, 0.185, 64, 36, 0.050, 0.030, 0.090, 14, 5),
                 (12, 0.185, 76, 18, 0.072, 0.028, 0.075, 12, 4),
                 (9,  0.200, 85, -12, 0.090, 0.026, 0.052, 12, 4)]
        whorls = [(8,  0.020, 0.075, 80, 5,  0.090, 0.034, 0.0),
                  (12, 0.034, 0.058, 72, 16, 0.082, 0.034, 0.5),
                  (16, 0.048, 0.044, 62, 24, 0.072, 0.030, 1.0)]
        nl_b, na_b, seg = 8, 3, 28
    elif density == "med":
        rings = [(11, 0.185, 64, 36, 0.050, 0.030, 0.090, 10, 4),
                 (8,  0.185, 76, 18, 0.072, 0.028, 0.075, 9, 3),
                 (6,  0.200, 85, -12, 0.090, 0.026, 0.052, 9, 3)]
        whorls = [(6,  0.020, 0.072, 80, 6,  0.090, 0.034, 0.0),
                  (9,  0.034, 0.056, 72, 16, 0.082, 0.034, 0.55),
                  (11, 0.048, 0.044, 62, 24, 0.072, 0.030, 1.0)]
        nl_b, na_b, seg = 6, 3, 18
    elif density == "low":
        rings = [(8, 0.185, 64, 36, 0.050, 0.030, 0.090, 7, 3),
                 (6, 0.195, 80, 4,  0.080, 0.027, 0.060, 6, 3)]
        whorls = [(5, 0.022, 0.070, 80, 8,  0.090, 0.034, 0.1),
                  (9, 0.044, 0.048, 66, 20, 0.075, 0.030, 0.9)]
        nl_b, na_b, seg = 5, 3, 12
    else:
        raise ValueError("density must be one of 'high', 'med', 'low'")

    lV, lF, lmeta = _build_leaves(rng, rings)
    bV, bF, bmeta = _build_bracts(rng, whorls, nl_b, na_b)
    tV, tF = _build_throat(seg)

    leaves = trimesh.Trimesh(vertices=lV, faces=lF, process=False)
    bracts = trimesh.Trimesh(vertices=bV, faces=bF, process=False)
    throat = trimesh.Trimesh(vertices=tV, faces=tF, process=True)
    leaves.metadata.update(lmeta)
    bracts.metadata.update(bmeta)

    meshes = [leaves, bracts, throat]
    mins = np.min([m.bounds[0] for m in meshes], axis=0)
    maxs = np.max([m.bounds[1] for m in meshes], axis=0)
    shift = np.array([-0.5 * (mins[0] + maxs[0]), -mins[1], -0.5 * (mins[2] + maxs[2])])
    for m in meshes:
        m.apply_translation(shift)

    scene = trimesh.Scene()
    scene.add_geometry(leaves, geom_name="leaves")
    scene.add_geometry(bracts, geom_name="bracts")
    scene.add_geometry(throat, geom_name="throat")
    return scene


# ==========================================================================
# IMAGE SAMPLING  (palette from the photo -- never invented)
# ==========================================================================
def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=float) / 255.0


def _patch_median(img, cx, cy, half):
    H, W, _ = img.shape
    x0 = int(np.clip(cx - half, 0, 1) * W); x1 = int(np.clip(cx + half, 0, 1) * W)
    y0 = int(np.clip(cy - half, 0, 1) * H); y1 = int(np.clip(cy + half, 0, 1) * H)
    x1 = max(x1, x0 + 1); y1 = max(y1, y0 + 1)
    return np.median(img[y0:y1, x0:x1].reshape(-1, 3), axis=0)


def sample_palette(img):
    centers = []
    for gx in np.linspace(0.30, 0.70, 7):
        for gy in np.linspace(0.26, 0.70, 7):
            centers.append((gx, gy))
    centers += [(0.50, 0.44), (0.47, 0.46), (0.53, 0.45),
                (0.62, 0.30), (0.33, 0.36), (0.70, 0.42),
                (0.66, 0.60), (0.36, 0.64), (0.45, 0.60), (0.55, 0.64)]

    greens, reds, yellows, deeps = [], [], [], []
    for (cx, cy) in centers:
        c = _patch_median(img, cx, cy, 0.02)
        mx, mn = float(c.max()), float(c.min())
        if mx < 1e-6 or (mx - mn) / mx < 0.14:
            continue  # neutral backdrop
        r, g, b = c
        if g >= r and (g - max(r, b)) > 0.03:
            greens.append(c)
            if c.mean() < 0.34:
                deeps.append(c)
        elif r > g:
            (yellows if (g > 0.45 and b < 0.45 and (r - g) < 0.45) else reds).append(c)

    def med(lst, fb):
        return np.median(np.array(lst), axis=0) if len(lst) >= 2 else np.array(fb)

    green = np.clip(med(greens, [0.20, 0.42, 0.11]), 0, 1)
    coral = np.clip(med(reds, [0.80, 0.24, 0.10]), 0, 1)
    yellow = np.clip(med(yellows, [0.98, 0.78, 0.14]), 0, 1)
    deep = np.clip(med(deeps, list(green * 0.55)), 0, 1)
    orange = np.clip(0.45 * coral + 0.55 * yellow, 0, 1)
    red = np.clip(coral * 0.92 + np.array([0.08, 0.0, 0.0]), 0, 1)
    return dict(green=green, coral=coral, orange=orange, yellow=yellow,
                red=red, deep=deep)


# ==========================================================================
# TEXTURE SYNTHESIS  (hue baked into the albedo atlases)
# ==========================================================================
def _value_noise(W, n, rng):
    small = (rng.random((n, n)) * 255).astype(np.uint8)
    im = Image.fromarray(small, "L").resize((W, W), Image.BILINEAR)
    return np.asarray(im, dtype=float) / 255.0


def _lance_alpha(W, base_w, peak, power, maxhalf_frac, rng):
    im = Image.new("L", (W, W), 0)
    d = ImageDraw.Draw(im)
    n = 26
    maxhalf = maxhalf_frac * W
    cx = W * 0.5
    left, right = [], []
    for i in range(n):
        v = i / (n - 1)
        if v <= peak:
            w = base_w + (1.0 - base_w) * (v / peak)
        else:
            w = (1.0 - (v - peak) / (1.0 - peak)) ** power
        w = max(w, 0.0) * maxhalf * (1.0 + rng.uniform(-0.02, 0.02))
        y = v * (W - 1)
        left.append((cx - w, y)); right.append((cx + w, y))
    pts = [(int(round(x)), int(round(y))) for x, y in (left + right[::-1])]
    d.polygon(pts, fill=255)
    return np.asarray(im)


def _organic_tile(size, kind, c_base, c_tip, c_mid, bright, rng):
    """One supersampled card tile: colored base->tip gradient + leathery
    detail + binary lance/petal alpha. Color lives HERE so it always shows."""
    SS = 4
    W = size * SS
    p = np.repeat(np.linspace(0.0, 1.0, W)[:, None], W, axis=1)   # 0 top(base)..1 bottom(tip)
    col = c_base[None, None] * (1 - p[..., None]) + c_tip[None, None] * p[..., None]
    mb = np.exp(-((p - 0.62) ** 2) / (2 * 0.16 ** 2))            # warm flush mid-outer
    col = col * (1 - 0.40 * mb[..., None]) + c_mid[None, None] * (0.40 * mb[..., None])

    col *= (0.88 + 0.20 * _value_noise(W, 26, rng))[..., None]   # mottle
    col += rng.standard_normal((W, W, 1)) * 0.02                 # fine speckle
    xs = np.linspace(-1.0, 1.0, W)[None, :]
    col *= (1.0 - 0.12 * xs ** 2)[..., None]                     # edge darkening
    col *= (1.0 - 0.12 * np.exp(-(xs ** 2) / (2 * 0.05 ** 2)))[..., None]  # midrib
    col = np.clip(col * bright, 0, 1)

    if kind == "leaf":
        alpha = _lance_alpha(W, 0.40, 0.50, 1.3, 0.46, rng)
    else:
        alpha = _lance_alpha(W, 0.42, 0.42, 1.6, 0.48, rng)

    rgba = np.dstack([(col * 255).astype(np.uint8), alpha.astype(np.uint8)])
    return Image.fromarray(rgba, "RGBA").resize((size, size), Image.LANCZOS)


def build_card_atlas(palette, rng, kind):
    """4x4 atlas (1024). Columns = warmth (green/yellow -> red), rows = sun."""
    A = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    g, coral, orange = palette["green"], palette["coral"], palette["orange"]
    yellow, red = palette["yellow"], palette["red"]
    for ti in range(16):
        r, c = ti // 4, ti % 4
        wc = c / 3.0
        bright = 0.84 + 0.16 * (r / 3.0)
        if kind == "leaf":
            c_base = g * (1 - 0.45 * wc) + coral * (0.45 * wc)
            c_tip = coral * (1 - wc) + red * wc
            c_mid = orange
        else:
            c_base = yellow * (1 - wc) + orange * wc
            c_tip = orange * (1 - wc) + red * wc
            c_mid = orange
        A.paste(_organic_tile(256, kind, np.clip(c_base, 0, 1),
                              np.clip(c_tip, 0, 1), np.clip(c_mid, 0, 1),
                              bright, rng), (c * 256, r * 256))
    return A


def _delight(crop):
    lum = 0.2126 * crop[..., 0] + 0.7152 * crop[..., 1] + 0.0722 * crop[..., 2]
    L = Image.fromarray((np.clip(lum, 0, 1) * 255).astype(np.uint8))
    rad = max(crop.shape[0], crop.shape[1]) / 6.0
    blur = np.asarray(L.filter(ImageFilter.GaussianBlur(radius=rad)), dtype=float) / 255.0 + 1e-3
    gain = np.clip(float(lum.mean()) / blur, 0.6, 1.6)
    return np.clip(crop * gain[..., None], 0, 1)


def _mirror_fold(crop):
    top = np.concatenate([crop, crop[:, ::-1, :]], axis=1)
    return np.concatenate([top, top[::-1, :, :]], axis=0)


def build_throat_swatch(img, palette):
    H, W, _ = img.shape
    crop = img[int(0.55 * H):int(0.70 * H), int(0.44 * W):int(0.56 * W)].astype(float)
    if crop.size == 0:
        crop = np.ones((32, 32, 3)) * palette["deep"][None, None, :]
    crop = _delight(crop)
    full = np.clip(_mirror_fold(crop) * 0.85, 0, 1)
    return Image.fromarray((full * 255).astype(np.uint8)).resize((512, 512), Image.LANCZOS)


# ==========================================================================
# VERTEX COLOURS  (subtle near-white sun/shade; texture carries the hue)
# ==========================================================================
def _to_rgba(col):
    c = np.empty((len(col), 4), dtype=np.uint8)
    c[:, :3] = np.clip(col * 255.0, 0, 255).astype(np.uint8)
    c[:, 3] = 255
    return c


def _leaf_vertex_color(s, u, ring, shade, jit):
    val = (0.86 + 0.14 * (1.0 - ring)) * (0.93 + 0.08 * s) * shade  # outer/tip lit
    r = 1.0 + 0.06 * np.clip(s - 0.3, 0, 1)                          # tips warmer
    b = 1.0 - 0.08 * np.clip(s, 0, 1)
    col = np.stack([val * r, val, val * b], axis=1) + (jit[:, None] - 0.5) * 0.04
    return np.clip(col, 0, 1)


def _bract_vertex_color(s, shade):
    val = (0.90 + 0.10 * s) * shade
    col = np.stack([val, val * (1.0 - 0.05 * s), val * (1.0 - 0.15 * s)], axis=1)
    return np.clip(col, 0, 1)


# ==========================================================================
# UVs + MATERIALS
# ==========================================================================
def _card_uv(tile, tu, tv):
    inset = 2.0 / 1024.0
    tr = (tile // 4).astype(float)
    tc = (tile % 4).astype(float)
    U = (tc + inset + tu * (1.0 - 2 * inset)) / 4.0
    V = (tr + inset + tv * (1.0 - 2 * inset)) / 4.0
    return np.column_stack([U, V]).astype(np.float64)


def apply_leaf_visual(m, atlas):
    md = m.metadata
    uv = _card_uv(md["tile"], (md["u"] + 1.0) * 0.5, np.clip(md["s"], 0, 1))
    mat = trimesh.visual.material.PBRMaterial(
        name="leaves", baseColorTexture=atlas, metallicFactor=0.0,
        roughnessFactor=0.8, alphaMode="MASK", alphaCutoff=0.45, doubleSided=True)
    vis = trimesh.visual.TextureVisuals(uv=uv, material=mat, image=atlas)
    vis.vertex_attributes = {"color": _to_rgba(
        _leaf_vertex_color(md["s"], md["u"], md["ring"], md["shade"], md["jit"]))}
    m.visual = vis


def apply_bract_visual(m, atlas):
    md = m.metadata
    uv = _card_uv(md["tile"], (md["u"] + 1.0) * 0.5, np.clip(md["s"], 0, 1))
    mat = trimesh.visual.material.PBRMaterial(
        name="bracts", baseColorTexture=atlas, metallicFactor=0.0,
        roughnessFactor=0.7, alphaMode="MASK", alphaCutoff=0.45, doubleSided=True)
    vis = trimesh.visual.TextureVisuals(uv=uv, material=mat, image=atlas)
    vis.vertex_attributes = {"color": _to_rgba(
        _bract_vertex_color(md["s"], md["shade"]))}
    m.visual = vis


def apply_throat_visual(m, img):
    v = m.vertices
    U = np.arctan2(v[:, 2], v[:, 0]) / (2 * np.pi) + 0.5
    y = v[:, 1]
    V = (y - y.min()) / (np.ptp(y) + 1e-9)
    uv = np.column_stack([U, V]).astype(np.float64)
    mat = trimesh.visual.material.PBRMaterial(
        name="throat", baseColorTexture=img, metallicFactor=0.0,
        roughnessFactor=0.85, doubleSided=True)
    vis = trimesh.visual.TextureVisuals(uv=uv, material=mat, image=img)
    col = np.tile([0.85, 0.82, 0.78], (len(v), 1)) * (0.45 + 0.55 * V)[:, None]
    vis.vertex_attributes = {"color": _to_rgba(np.clip(col, 0, 1))}
    m.visual = vis


# ==========================================================================
# CLI
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="Procedural textured bromeliad -> GLB")
    ap.add_argument("--image", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    img = load_image(a.image)
    scene = build_mesh(a.seed, a.density)

    trng = np.random.default_rng(a.seed + 9973)
    palette = sample_palette(img)
    leaf_atlas = build_card_atlas(palette, trng, "leaf")
    bract_atlas = build_card_atlas(palette, trng, "petal")
    throat_img = build_throat_swatch(img, palette)

    apply_leaf_visual(scene.geometry["leaves"], leaf_atlas)
    apply_bract_visual(scene.geometry["bracts"], bract_atlas)
    apply_throat_visual(scene.geometry["throat"], throat_img)

    data = scene.export(file_type="glb")
    with open(a.output, "wb") as f:
        f.write(data)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)