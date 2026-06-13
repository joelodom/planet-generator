"""Procedural tree-fern: geometry + photo-derived materials -> textured GLB.

A single stout fibrous trunk crowned by a radiating rosette of large, arching,
bipinnate fronds -- a feathery green "fountain" silhouette.

CLI:
    python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter


# ===========================================================================
#  GEOMETRY MODULE  (build_mesh)
# ===========================================================================
# MEASURED PROPORTIONS (ratios eyeballed off reference.png, ~10% accuracy)
# Whole-plant aspect height/width ~= 1.3; front content aspect (w/h) ~= 0.84.
# Trunk occupies the lower ~0.44 of height; the crown spreads many times wider.
TOTAL_HEIGHT   = 2.30          # meters, overall plant height (named size const)
TRUNK_HEIGHT   = 1.02          # ~0.44 * TOTAL_HEIGHT  (trunk top = crown origin)
TRUNK_RADIUS   = 0.22          # stout column; diameter ~0.44 m
TRUNK_FLARE    = 1.45          # basal flare multiplier (x1.45 at the foot)
TRUNK_FLARE_FRAC = 0.07        # flare confined to the bottom 7% of the trunk
TRUNK_TOP_TAPER  = 0.85        # radius at the top relative to nominal

# Foliage envelope: a lobed ellipsoid sitting on top of the trunk. Widened so
# the crown spreads broadly (front aspect ~0.84) instead of a narrow tuft.
CROWN_RX       = 1.02          # crown half-width  (crown width ~2.0 m)
CROWN_RZ       = 1.02
CROWN_RY       = 1.03          # crown half-height
ENV_CENTER_Y   = TOTAL_HEIGHT - CROWN_RY      # = 1.27
ENV_SHELL      = 0.97          # frond tips land at 97% of the shell radius
LOBES          = 4             # low-frequency radial bulges on the envelope
LOBE_AMP       = 0.08          # +/-8% radial wobble of the shell

# Frond fan: a few up-arching central fronds, many mid, a drooping outer skirt.
THETA_MIN      = np.radians(14.0)    # polar angle from straight-up (central)
THETA_MAX      = np.radians(140.0)   # outermost drooping fronds
FROND_HALFWIDTH = 0.28               # broad bipinnate blade half-width
ARCH_LIFT      = 0.55                # how much each frond bows up before its tip
GOLDEN_ANGLE   = 2.399963229728653   # radians, for even azimuthal spread


def _params(density):
    if density == "high":
        return dict(n_fronds=30, stations=32, per_side=2,
                    rachis_seg=12, rachis_sides=4,
                    trunk_sides=16, trunk_rings=14)
    if density == "med":
        return dict(n_fronds=20, stations=22, per_side=1,
                    rachis_seg=9, rachis_sides=4,
                    trunk_sides=12, trunk_rings=8)
    if density == "low":
        return dict(n_fronds=12, stations=14, per_side=1,
                    rachis_seg=6, rachis_sides=3,
                    trunk_sides=8, trunk_rings=4)
    raise ValueError("density must be 'high', 'med' or 'low'")


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _bezier(p0, p1, p2, t):
    """Quadratic Bezier point + (unit) tangent at parameter t."""
    mt = 1.0 - t
    pos = mt * mt * p0 + 2.0 * mt * t * p1 + t * t * p2
    tan = 2.0 * mt * (p1 - p0) + 2.0 * t * (p2 - p1)
    return pos, _norm(tan)


def _build_trunk(rng, sides, rings):
    octs = [(rng.uniform(2.0, 4.0), rng.uniform(0.0, 2 * np.pi), rng.uniform(3.0, 6.0)),
            (rng.uniform(5.0, 8.0), rng.uniform(0.0, 2 * np.pi), rng.uniform(6.0, 11.0)),
            (rng.uniform(9.0, 13.0), rng.uniform(0.0, 2 * np.pi), rng.uniform(10.0, 16.0))]
    amps = [0.060, 0.035, 0.018]

    angles = np.linspace(0.0, 2 * np.pi, sides, endpoint=False)
    verts = []
    for ri in range(rings + 1):
        hy = ri / rings
        y = hy * TRUNK_HEIGHT
        taper = 1.0 - (1.0 - TRUNK_TOP_TAPER) * hy
        if hy < TRUNK_FLARE_FRAC:
            f = (TRUNK_FLARE_FRAC - hy) / TRUNK_FLARE_FRAC
            taper *= 1.0 + (TRUNK_FLARE - 1.0) * f * f
        r0 = TRUNK_RADIUS * taper
        for a in angles:
            disp = 0.0
            for (freq, ph, vf), amp in zip(octs, amps):
                disp += amp * np.sin(freq * a + ph + y * vf)
            disp += rng.normal(0.0, 0.022)
            r = max(r0 * (1.0 + disp), 0.04)
            verts.append((r * np.cos(a), y, r * np.sin(a)))
    verts = np.array(verts, dtype=np.float64)

    faces = []
    for ri in range(rings):
        for si in range(sides):
            a0 = ri * sides + si
            a1 = ri * sides + (si + 1) % sides
            b0 = a0 + sides
            b1 = a1 + sides
            faces.append((a0, b0, b1))
            faces.append((a0, b1, a1))

    bc = len(verts)
    verts = np.vstack([verts, [0.0, 0.0, 0.0]])
    tc = len(verts)
    verts = np.vstack([verts, [0.0, TRUNK_HEIGHT, 0.0]])
    for si in range(sides):
        a0 = si
        a1 = (si + 1) % sides
        faces.append((bc, a1, a0))
        t0 = rings * sides + si
        t1 = rings * sides + (si + 1) % sides
        faces.append((tc, t0, t1))
    faces = np.array(faces, dtype=np.int64)

    m = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    m.fix_normals()
    return m


def _frond_geometry(rng, p):
    crown = np.array([0.0, TRUNK_HEIGHT, 0.0])
    up = np.array([0.0, 1.0, 0.0])

    stem_v, stem_f = [], []
    card_v, card_f = [], []

    fracs = (np.arange(p["n_fronds"]) + 0.5) / p["n_fronds"]
    for i in range(p["n_fronds"]):
        phi = i * GOLDEN_ANGLE + rng.normal(0.0, 0.12)
        cphi, sphi = np.cos(phi), np.sin(phi)
        h = np.array([cphi, 0.0, sphi])
        binormal = np.array([-sphi, 0.0, cphi])

        # bias toward mid angles -> a full rounded dome (not vertical spikes)
        f = float(np.clip(fracs[i] + rng.normal(0.0, 0.05), 0.0, 1.0))
        theta = THETA_MIN + (THETA_MAX - THETA_MIN) * (f ** 0.92)
        st, ct = np.sin(theta), np.cos(theta)

        lobe = 1.0 + LOBE_AMP * np.sin(LOBES * phi + rng.uniform(0, 2 * np.pi))
        tip = np.array([
            ENV_SHELL * CROWN_RX * lobe * st * cphi,
            ENV_CENTER_Y + ENV_SHELL * CROWN_RY * ct,
            ENV_SHELL * CROWN_RZ * lobe * st * sphi,
        ])

        p0 = crown + h * 0.07 + np.array([0.0, -0.02, 0.0])
        mid = 0.5 * (p0 + tip)
        lift = ARCH_LIFT * CROWN_RY * max(0.22, st)
        p1 = mid + up * lift + h * (0.14 * CROWN_RX * st)

        chord = float(np.linalg.norm(tip - p0))
        spacing = max(1e-3, chord / p["stations"])

        ts = np.linspace(0.0, 1.0, p["rachis_seg"] + 1)
        path = [_bezier(p0, p1, tip, t) for t in ts]

        # --- rachis tube (stems): thin so it hides under foliage ---
        base = len(stem_v)
        sides = p["rachis_sides"]
        ring_ang = np.linspace(0.0, 2 * np.pi, sides, endpoint=False)
        for j, (pos, tan) in enumerate(path):
            if np.linalg.norm(np.cross(tan, binormal)) < 1e-6:
                n1 = _norm(np.cross(tan, up))
            else:
                n1 = _norm(np.cross(tan, binormal))
            n2 = _norm(np.cross(tan, n1))
            rad = 0.011 * (1.0 - 0.9 * ts[j]) + 0.0022
            for a in ring_ang:
                stem_v.append(pos + rad * (np.cos(a) * n1 + np.sin(a) * n2))
        for j in range(p["rachis_seg"]):
            for si in range(sides):
                a0 = base + j * sides + si
                a1 = base + j * sides + (si + 1) % sides
                b0 = a0 + sides
                b1 = a1 + sides
                stem_f.append((a0, b0, b1))
                stem_f.append((a0, b1, a1))

        # --- pinna leaf cards: broad, overlapping -> a solid feathery blade ---
        per_side = p["per_side"]
        stations = np.linspace(0.04, 0.985, p["stations"])
        for t in stations:
            pos, tan = _bezier(p0, p1, tip, t)
            tan = _norm(tan)
            # broad width profile: quick rise, full through the middle, taper
            ramp = min(1.0, t / 0.09)
            w = FROND_HALFWIDTH * ramp * (1.0 - t ** 1.35)
            if w < 1e-3:
                continue
            hv = spacing * 0.85                       # overlap neighbours
            for s in (-1.0, 1.0):
                for k in range(per_side):
                    frac = (k + 0.5) / per_side
                    radial = w * frac * 1.05
                    center = pos + s * binormal * radial + rng.normal(0, 0.004, 3)
                    jit = rng.normal(0.0, 0.14, 3)
                    u = _norm(s * binormal + 0.30 * tan - 0.16 * up + jit)
                    v = tan - u * float(np.dot(tan, u))
                    v = _norm(v) if np.linalg.norm(v) > 1e-6 else up
                    hu = (w * (0.72 / per_side) + 0.03) * np.exp(rng.normal(0.0, 0.14))
                    b = len(card_v)
                    card_v.append(center - u * hu - v * hv)
                    card_v.append(center + u * hu - v * hv)
                    card_v.append(center + u * hu + v * hv)
                    card_v.append(center - u * hu + v * hv)
                    card_f.append((b, b + 1, b + 2))
                    card_f.append((b, b + 2, b + 3))

    stems = trimesh.Trimesh(vertices=np.array(stem_v, dtype=np.float64),
                            faces=np.array(stem_f, dtype=np.int64),
                            process=True)
    canopy = trimesh.Trimesh(vertices=np.array(card_v, dtype=np.float64),
                             faces=np.array(card_f, dtype=np.int64),
                             process=False)
    canopy.fix_normals()
    return stems, canopy


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    p = _params(density)

    trunk = _build_trunk(rng, p["trunk_sides"], p["trunk_rings"])
    stems, canopy = _frond_geometry(rng, p)

    scene = trimesh.Scene()
    scene.add_geometry(trunk, geom_name="trunk")
    scene.add_geometry(stems, geom_name="stems")
    scene.add_geometry(canopy, geom_name="canopy")

    lo = scene.bounds[0]
    if abs(lo[1]) > 1e-6:
        scene.apply_translation([0.0, -lo[1], 0.0])
    return scene


# ===========================================================================
#  COLOR SAMPLING FROM THE PHOTO
# ===========================================================================
# Sampling centers (normalized x,y) chosen by LOOKING at reference.png and
# placed WELL INSIDE the silhouette -- foliage greens (upper rosette) and
# fibrous-trunk browns (lower-centre column). Background grey is rejected.
_FOLIAGE_CENTERS = [(0.30, 0.30), (0.45, 0.25), (0.58, 0.32), (0.66, 0.42),
                    (0.40, 0.45), (0.52, 0.49), (0.33, 0.45), (0.61, 0.53),
                    (0.48, 0.20), (0.55, 0.38)]
_TRUNK_CENTERS = [(0.47, 0.72), (0.52, 0.75), (0.49, 0.80), (0.54, 0.83),
                  (0.50, 0.87), (0.45, 0.78), (0.55, 0.78), (0.50, 0.70)]


def _sample(img, centers, half):
    H, W = img.shape[:2]
    out = []
    for nx, ny in centers:
        cx, cy = int(nx * W), int(ny * H)
        x0, x1 = max(0, cx - half), min(W, cx + half + 1)
        y0, y1 = max(0, cy - half), min(H, cy + half + 1)
        patch = img[y0:y1, x0:x1].reshape(-1, 3)
        if patch.size:
            out.append(np.median(patch, axis=0))
    return np.array(out, dtype=np.float64)


def _foliage_palette(img):
    cols = _sample(img, _FOLIAGE_CENTERS, 4)
    keep = [c for c in cols if (c[1] - c[2]) > 8 and c[1] >= c[0] * 0.85]
    keep = np.array(keep) if keep else cols
    lum = keep.mean(axis=1)
    light = keep[int(np.argmax(lum))]
    dark = keep[int(np.argmin(lum))]
    if np.linalg.norm(light - dark) < 25:
        dark = np.clip(light * 0.62, 0, 255)
    # nudge toward the photo's luminous lime-green and keep it bright
    light = np.clip(light * np.array([1.05, 1.10, 0.84]) * 1.06, 0, 255)
    dark = np.clip(dark * np.array([1.02, 1.08, 0.86]), 0, 255)
    return light, dark


def _trunk_palette(img):
    cols = _sample(img, _TRUNK_CENTERS, 5)
    keep = [c for c in cols if (c[0] - c[2]) > 6 and c.mean() < 175]
    keep = np.array(keep) if keep else cols
    base = np.median(keep, axis=0)
    lum = keep.mean(axis=1)
    dark = np.clip(keep[int(np.argmin(lum))] * 0.75, 0, 255)
    if base.mean() < 70:                       # never collapse to black
        base = np.clip(base * 1.4, 0, 255)
    return np.clip(base, 0, 255), np.clip(dark, 0, 255)


# ===========================================================================
#  TEXTURE SYNTHESIS  (de-light, mirror-tile, atlas)
# ===========================================================================
def _delight(arr):
    """arr float HxWx3 in [0,1]; divide by heavily blurred luminance."""
    lum = arr.mean(axis=2)
    pim = Image.fromarray((np.clip(lum, 0, 1) * 255).astype(np.uint8))
    rad = max(4, max(arr.shape[:2]) // 4)
    blur = np.asarray(pim.filter(ImageFilter.GaussianBlur(radius=rad))).astype(np.float64) / 255.0
    blur = np.clip(blur, 1e-3, None)
    gain = np.clip(lum.mean() / blur, 0.6, 1.6)
    return np.clip(arr * gain[..., None], 0, 1)


def _mirror_tile(arr):
    top = np.hstack([arr, arr[:, ::-1]])
    return np.vstack([top, top[::-1]])


def make_trunk_texture(img, base, rng, size=512):
    H, W = img.shape[:2]
    cx, cy = int(0.50 * W), int(0.80 * H)
    hw, hh = max(14, int(0.09 * W)), max(14, int(0.06 * H))
    crop = img[max(0, cy - hh):cy + hh, max(0, cx - hw):cx + hw].astype(np.float64) / 255.0
    if crop.size < 9:
        crop = np.tile((base / 255.0).reshape(1, 1, 3), (32, 32, 1))
    crop = _delight(crop)
    cur = crop.reshape(-1, 3).mean(axis=0) + 1e-3
    crop = np.clip(crop * (base / 255.0) / cur, 0, 1)   # recolor to sampled brown
    tile = _mirror_tile(crop)
    pim = Image.fromarray((tile * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
    arr = np.asarray(pim).astype(np.float64) / 255.0
    xs = np.arange(size)
    streak = (0.86 + 0.14 * np.sin(2 * np.pi * xs / 7.0 + rng.uniform(0, 6))) * \
             (0.93 + 0.07 * np.sin(2 * np.pi * xs / 2.3 + rng.uniform(0, 6)))
    arr = np.clip(arr * streak[None, :, None], 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8), "RGB")


def make_normal_map(pim, strength=1.6):
    a = np.asarray(pim.convert("L")).astype(np.float64) / 255.0
    h = 1.0 - a
    gy, gx = np.gradient(h)
    nx, ny, nz = -gx * strength, -gy * strength, np.ones_like(h)
    l = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    n = np.stack([nx / l, ny / l, nz / l], axis=-1)
    return Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8), "RGB")


def make_stem_texture(col, rng, size=512):
    base = col / 255.0
    xs = np.linspace(0, 1, size, endpoint=False)
    ys = np.linspace(0, 1, size, endpoint=False)
    X, Y = np.meshgrid(xs, ys)
    n = (0.9 + 0.1 * np.sin(2 * np.pi * Y * 8 + rng.uniform(0, 6))) * \
        (0.95 + 0.05 * np.sin(2 * np.pi * X * 40))
    arr = np.clip(base[None, None, :] * n[..., None], 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8), "RGB")


def _leaf_color(base, rng, lit):
    j = rng.uniform(0.86, 1.14)
    r = base[0] * j * (0.92 + 0.16 * lit)
    g = base[1] * j * (0.98 + 0.05 * lit)
    b = base[2] * j * (0.86 + 0.10 * (1.0 - lit))
    return (int(np.clip(r, 0, 255)), int(np.clip(g, 0, 255)),
            int(np.clip(b, 0, 255)), 255)


def _draw_pinna(d, p_base, p_tip, leaf_len, color, rng, lit):
    axis = p_tip - p_base
    L = float(np.linalg.norm(axis))
    if L < 2:
        return
    dirv = axis / L
    perp = np.array([-dirv[1], dirv[0]])
    n = max(7, int(L / (leaf_len * 0.42)))            # dense leaflets
    d.line([tuple(p_base.astype(int)), tuple(p_tip.astype(int))],
           fill=_leaf_color(color, rng, lit * 0.6), width=max(1, int(leaf_len * 0.14)))
    for i in range(n):
        f = (i + 0.5) / n
        p = p_base + axis * f
        ll = leaf_len * (1.0 - 0.45 * f)              # leaflets shrink toward tip
        for s in (-1.0, 1.0):
            tip_l = p + perp * s * ll + dirv * ll * 0.38
            a = p - dirv * ll * 0.18
            b = p + dirv * ll * 0.22
            d.polygon([tuple(a.astype(int)), tuple(tip_l.astype(int)),
                       tuple(b.astype(int))], fill=_leaf_color(color, rng, lit))


def _draw_cluster(cell, base_rgb, rng, lit):
    ss = 4
    S = cell * ss
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    centre = np.array([S * 0.5, S * 0.5])
    for _ in range(int(rng.integers(4, 7))):          # several overlapping pinnae
        ang = rng.uniform(0, 2 * np.pi)
        v = np.array([np.cos(ang), np.sin(ang)])
        p_base = centre - v * (S * 0.5) * rng.uniform(0.85, 1.05)
        p_tip = centre + v * (S * 0.5) * rng.uniform(0.85, 1.05)
        _draw_pinna(d, p_base, p_tip, S * 0.22, base_rgb, rng, lit)
    return im.resize((cell, cell), Image.LANCZOS)


def make_canopy_atlas(light, dark, rng, size=1024, n=4):
    cell = size // n
    atlas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    for r in range(n):
        for c in range(n):
            t = 1.0 - r / (n - 1)                     # top rows sunlit/brighter
            base = dark + (light - dark) * t
            base = np.clip(base * rng.uniform(0.94, 1.10), 0, 255)
            atlas.paste(_draw_cluster(cell, base, rng, t), (c * cell, r * cell))
    return atlas


# ===========================================================================
#  UV HELPERS
# ===========================================================================
def cylindrical_uv(V, height, ru, rv):
    ang = np.arctan2(V[:, 2], V[:, 0])
    u = (ang / (2 * np.pi) + 0.5) * ru
    v = (V[:, 1] / height) * rv
    return np.column_stack([u, v]).astype(np.float64)


def fix_uv_seam(V, F, UV, ru):
    """Duplicate vertices on the atan2 wrap so the trunk has no smeared seam."""
    thr = 0.5 * ru
    Vl = [list(x) for x in V]
    UVl = [list(x) for x in UV]
    Fl = [list(f) for f in F]
    cache = {}
    for f in Fl:
        us = [UVl[i][0] for i in f]
        if max(us) - min(us) > thr:
            mx = max(us)
            for k in range(3):
                i = f[k]
                if UVl[i][0] < mx - thr:
                    if i not in cache:
                        cache[i] = len(Vl)
                        Vl.append(list(Vl[i]))
                        nu = list(UVl[i])
                        nu[0] += ru
                        UVl.append(nu)
                    f[k] = cache[i]
    return (np.array(Vl, dtype=np.float64),
            np.array(Fl, dtype=np.int64),
            np.array(UVl, dtype=np.float64))


def canopy_uv(n_verts, rng, n=4, inset=0.02):
    """Map each 4-vertex card onto a random atlas tile with 0/90/180/270 spin."""
    ncards = n_verts // 4
    tiles = rng.integers(0, n * n, ncards)
    rots = rng.integers(0, 4, ncards)
    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    uv = np.zeros((n_verts, 2), dtype=np.float64)
    for c in range(ncards):
        col, row = int(tiles[c]) % n, int(tiles[c]) // n
        rr = int(rots[c])
        for k in range(4):
            lc = corners[(k + rr) % 4]
            lu = inset + (1.0 - 2 * inset) * lc[0]
            lv = inset + (1.0 - 2 * inset) * lc[1]
            uv[4 * c + k] = ((col + lu) / n, (row + lv) / n)
    return uv


# ===========================================================================
#  VERTEX COLORS (COLOR_0)  -- multiply the texture in glTF
# ===========================================================================
def _u8(rgb):
    a = np.full((rgb.shape[0], 1), 255.0)
    return np.clip(np.hstack([rgb * 255.0, a]), 0, 255).astype(np.uint8)


def trunk_vertex_colors(V):
    tn = np.clip(V[:, 1] / TRUNK_HEIGHT, 0, 1)
    val = 0.58 + 0.42 * tn                         # AO darkening at the foot
    rgb = np.stack([val * 1.06, val * 0.99, val * 0.90], axis=1)
    return _u8(np.clip(rgb, 0, 1))


def stem_vertex_colors(V):
    sn = np.clip((V[:, 1] - 0.2) / (TOTAL_HEIGHT - 0.2), 0, 1)
    val = 0.80 + 0.20 * sn
    rgb = np.stack([val * 0.82, val * 1.0, val * 0.62], axis=1)
    return _u8(np.clip(rgb, 0, 1))


def canopy_vertex_colors(V, rng):
    nc = V.shape[0] // 4
    cen = V.reshape(nc, 4, 3).mean(axis=1)
    hy = np.clip((cen[:, 1] - 0.3) / (TOTAL_HEIGHT - 0.3), 0, 1)
    rad = np.clip(np.sqrt(cen[:, 0] ** 2 + cen[:, 2] ** 2) / CROWN_RX, 0, 1)
    bright = np.clip(0.40 * hy + 0.40 * rad + 0.20, 0, 1)   # top & outer = sunlit
    jit = rng.normal(0.0, 0.05, nc)
    val = np.clip(0.82 + 0.26 * bright + jit, 0.72, 1.14)
    r = val * (0.92 + 0.14 * bright)               # warmer/limier in sun
    g = val * 1.0
    b = val * 0.60                                  # low blue -> luminous green
    rgb = np.clip(np.stack([r, g, b], axis=1), 0, 1)
    return _u8(np.repeat(rgb, 4, axis=0))


# ===========================================================================
#  ASSEMBLY
# ===========================================================================
def textured_scene(seed, density, image_path):
    base_scene = build_mesh(seed, density)
    rng = np.random.default_rng(seed)

    img = np.asarray(Image.open(image_path).convert("RGB")).astype(np.float64)
    light_g, dark_g = _foliage_palette(img)
    trunk_base, _ = _trunk_palette(img)
    stem_col = np.clip((0.4 * trunk_base + 0.6 * dark_g) * 1.1, 0, 255)

    # --- textures ---
    trunk_tex = make_trunk_texture(img, trunk_base, rng, size=512)
    trunk_norm = make_normal_map(trunk_tex)
    stem_tex = make_stem_texture(stem_col, rng, size=512)
    atlas = make_canopy_atlas(light_g, dark_g, rng, size=1024, n=4)

    # --- materials ---
    trunk_mat = trimesh.visual.material.PBRMaterial(
        name="trunk", baseColorTexture=trunk_tex, normalTexture=trunk_norm,
        metallicFactor=0.0, roughnessFactor=0.9)
    stem_mat = trimesh.visual.material.PBRMaterial(
        name="stems", baseColorTexture=stem_tex,
        metallicFactor=0.0, roughnessFactor=0.85)
    canopy_mat = trimesh.visual.material.PBRMaterial(
        name="canopy", baseColorTexture=atlas, metallicFactor=0.0,
        roughnessFactor=0.8, alphaMode="MASK", alphaCutoff=0.45, doubleSided=True)

    out = trimesh.Scene()

    # --- trunk: cylindrical UVs (seam fixed) ---
    trunk = base_scene.geometry["trunk"]
    ru, rv = 4.0, 3.0
    uv = cylindrical_uv(trunk.vertices, TRUNK_HEIGHT, ru, rv)
    tv, tf, tuv = fix_uv_seam(trunk.vertices, trunk.faces, uv, ru)
    trunk_m = trimesh.Trimesh(vertices=tv, faces=tf, process=False)
    trunk_m.visual = trimesh.visual.TextureVisuals(uv=tuv, material=trunk_mat)
    trunk_m.visual.vertex_attributes["color"] = trunk_vertex_colors(tv)
    out.add_geometry(trunk_m, geom_name="trunk")

    # --- stems: cylindrical UVs about the world axis (texture ~uniform) ---
    stems = base_scene.geometry["stems"]
    suv = cylindrical_uv(stems.vertices, TOTAL_HEIGHT, 3.0, 4.0)
    stems.visual = trimesh.visual.TextureVisuals(uv=suv, material=stem_mat)
    stems.visual.vertex_attributes["color"] = stem_vertex_colors(stems.vertices)
    out.add_geometry(stems, geom_name="stems")

    # --- canopy: per-card atlas tiles ---
    canopy = base_scene.geometry["canopy"]
    cuv = canopy_uv(len(canopy.vertices), rng, n=4)
    canopy.visual = trimesh.visual.TextureVisuals(uv=cuv, material=canopy_mat)
    canopy.visual.vertex_attributes["color"] = canopy_vertex_colors(canopy.vertices, rng)
    out.add_geometry(canopy, geom_name="canopy")

    return out


# ===========================================================================
#  CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Procedural textured tree fern -> GLB")
    ap.add_argument("--image", required=True, help="source reference image path")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        scene = textured_scene(args.seed, args.density, args.image)
        glb = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(glb)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())