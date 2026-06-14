#!/usr/bin/env python3
"""
Procedural creeping-thyme cushion: geometry + photo-derived materials + GLB.

Builds a low rounded cushion subshrub (a fuzzy green mound densely speckled
with pink/white flower tufts), derives matte tileable/atlas materials by
SAMPLING the reference photo, applies per-surface UVs, exports a textured .glb.

Only numpy + trimesh + PIL + stdlib.  +Y up, base at y=0, meters, deterministic.

    python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb
"""

import argparse
import sys
import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial
from PIL import Image, ImageDraw, ImageFilter


# ===========================================================================
# GEOMETRY
# ===========================================================================
# Proportion FIX: the photo cushion is only slightly broader than tall
# (content aspect ~1.14).  Previous build was too flat (1.57), so the dome is
# now much taller relative to width and a little narrower.
CROWN_WIDTH = 0.40                      # full width of the dome across XZ (m)
CROWN_RADIUS = CROWN_WIDTH * 0.5

#   envelope height / width ~= 1.0  -> rendered extents land near aspect ~1.15
HEIGHT_OVER_WIDTH = 1.00
CROWN_HEIGHT = CROWN_WIDTH * HEIGHT_OVER_WIDTH

LEAF_CARD_FRAC = 0.050                  # leaf card half-size ~5% of crown width
FLOWER_CARD_FRAC = 0.028               # blossoms read a touch smaller
FOCUS_Y_FRAC = 0.12                    # interior focus for outward card normals
BASE_SHRINK = 0.86                     # smooth shell well inside -> cards are the skin

FOLIAGE_COLOR = np.array([86, 138, 52, 255], dtype=np.uint8)
FLOWER_COLOR = np.array([229, 176, 198, 255], dtype=np.uint8)

# Element counts: many more leaf + flower cards so foliage dominates and the
# dome is densely speckled like the photo.
_DENSITY = {
    "high": dict(lat=24, lon=44, clumps=26, cards=6000, ftufts=80, flowers=2600),
    "med":  dict(lat=16, lon=30, clumps=16, cards=2400, ftufts=34, flowers=1000),
    "low":  dict(lat=10, lon=18, clumps=10, cards=700,  ftufts=12, flowers=280),
}


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([0.0, 1.0, 0.0])
    return v / n


def _jitter_dir(n, max_deg, rng):
    t = rng.normal(size=3)
    t = t - n * np.dot(t, n)
    tn = np.linalg.norm(t)
    if tn < 1e-12:
        return n
    t /= tn
    ang = np.deg2rad(rng.uniform(0.0, max_deg))
    return _normalize(np.cos(ang) * n + np.sin(ang) * t)


def _tangent_basis(n, rng):
    ref = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = _normalize(ref - n * np.dot(ref, n))
    v = np.cross(n, u)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    cu, su = np.cos(phi), np.sin(phi)
    u2 = cu * u + su * v
    v2 = -su * u + cu * v
    return u2, v2


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    cfg = _DENSITY.get(density, _DENSITY["high"])

    # gentle low-frequency lobes (small amplitude -> clean, controlled width)
    n_lobes = int(rng.integers(3, 6))
    lobe_m = rng.integers(2, 5, size=n_lobes)
    lobe_a = rng.uniform(0.02, 0.05, size=n_lobes)
    lobe_p = rng.uniform(0.0, 2.0 * np.pi, size=n_lobes)

    def lobe(theta):
        theta = np.asarray(theta, dtype=float)
        s = np.ones_like(theta)
        for m, a, p in zip(lobe_m, lobe_a, lobe_p):
            s = s + a * np.cos(m * theta + p)
        return s

    def envelope(theta, phi, radial=1.0):
        L = lobe(theta)
        r = CROWN_RADIUS * np.sin(phi) * L * radial
        y = CROWN_HEIGHT * np.cos(phi) * radial
        x = r * np.cos(theta)
        z = r * np.sin(theta)
        return np.stack([x, y, z], axis=-1)

    focus = np.array([0.0, CROWN_HEIGHT * FOCUS_Y_FRAC, 0.0])

    def outward_normal(p):
        return _normalize(p - focus)

    # ---- smooth base shell dome (dark matte interior mass, mostly hidden) ---
    nlat, nlon = cfg["lat"], cfg["lon"]
    phis = np.linspace(0.0, np.pi / 2.0, nlat + 1)
    ring_phis = phis[1:]
    thetas = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False)

    sverts = [envelope(0.0, 0.0, BASE_SHRINK)]
    for phi in ring_phis:
        for th in thetas:
            sverts.append(envelope(th, phi, BASE_SHRINK))

    def ridx(r, j):
        return 1 + r * nlon + (j % nlon)

    sfaces = []
    for j in range(nlon):
        sfaces.append([0, ridx(0, j + 1), ridx(0, j)])
    for r in range(nlat - 1):
        for j in range(nlon):
            a = ridx(r, j)
            b = ridx(r, j + 1)
            c = ridx(r + 1, j + 1)
            d = ridx(r + 1, j)
            sfaces.append([a, b, c])
            sfaces.append([a, c, d])
    center_idx = len(sverts)
    sverts.append(np.array([0.0, 0.0, 0.0]))
    last = nlat - 1
    for j in range(nlon):
        sfaces.append([center_idx, ridx(last, j), ridx(last, j + 1)])

    shell = trimesh.Trimesh(vertices=np.array(sverts, dtype=float),
                            faces=np.array(sfaces, dtype=np.int64),
                            process=True)
    shell.fix_normals()

    # ---- clumped leaf cards (the fuzzy outer foliage skin) -----------------
    leaf_h = LEAF_CARD_FRAC * CROWN_WIDTH
    n_clumps = cfg["clumps"]
    n_interior = max(2, n_clumps // 6)
    per_clump = max(1, cfg["cards"] // n_clumps)

    lv, lf = [], []

    def add_card(store_v, store_f, center, u_scaled, v_scaled):
        b = len(store_v)
        store_v.append(center - u_scaled - v_scaled)
        store_v.append(center + u_scaled - v_scaled)
        store_v.append(center + u_scaled + v_scaled)
        store_v.append(center - u_scaled + v_scaled)
        store_f.append([b, b + 1, b + 2])
        store_f.append([b, b + 2, b + 3])

    for k in range(n_clumps):
        theta_c = rng.uniform(0.0, 2.0 * np.pi)
        phi_c = (np.pi / 2.0) * np.sqrt(rng.random()) * 0.99
        interior = k >= (n_clumps - n_interior)
        base_radial = rng.uniform(0.6, 0.78) if interior else 1.0
        ang_spread = 0.20
        n_cards = max(6, int(round(per_clump * rng.uniform(0.7, 1.3))))
        for _ in range(n_cards):
            th = theta_c + rng.normal(0.0, ang_spread)
            ph = np.clip(phi_c + rng.normal(0.0, ang_spread), 0.0, np.pi / 2.0 * 0.999)
            radial = base_radial * rng.uniform(0.95, 1.0)
            p = envelope(th, ph, radial)
            n = _jitter_dir(outward_normal(p), 32.0, rng)   # more tilt -> fuzz
            u, v = _tangent_basis(n, rng)
            h = leaf_h * np.exp(rng.normal(0.0, 0.3))
            add_card(lv, lf, p, u * h, v * h)

    leaves = trimesh.Trimesh(vertices=np.array(lv, dtype=float),
                             faces=np.array(lf, dtype=np.int64),
                             process=False)

    # ---- clumped flower cards (dense pink/white speckle over whole dome) ----
    flower_h = FLOWER_CARD_FRAC * CROWN_WIDTH
    n_ftufts = cfg["ftufts"]
    per_tuft = max(1, cfg["flowers"] // n_ftufts)

    fv, ff = [], []
    for _ in range(n_ftufts):
        theta_c = rng.uniform(0.0, 2.0 * np.pi)
        # **1.3 -> spread across the dome but still denser toward the crown
        phi_c = (np.pi / 2.0) * (rng.random() ** 1.3) * 0.99
        n_fl = max(3, int(round(per_tuft * rng.uniform(0.7, 1.3))))
        for _ in range(n_fl):
            th = theta_c + rng.normal(0.0, 0.18)
            ph = np.clip(phi_c + rng.normal(0.0, 0.18), 0.0, np.pi / 2.0 * 0.999)
            radial = rng.uniform(1.03, 1.08)        # sit proud of the foliage
            p = envelope(th, ph, radial)
            n = _jitter_dir(outward_normal(p), 28.0, rng)
            u, v = _tangent_basis(n, rng)
            h = flower_h * np.exp(rng.normal(0.0, 0.25))
            add_card(fv, ff, p, u * h, v * h)

    flowers = trimesh.Trimesh(vertices=np.array(fv, dtype=float),
                              faces=np.array(ff, dtype=np.int64),
                              process=False)

    # ---- normalize: lowest point at y=0, centered in X/Z -------------------
    allv = np.vstack([shell.vertices, leaves.vertices, flowers.vertices])
    mn = allv.min(axis=0)
    mx = allv.max(axis=0)
    shift = np.array([-(mn[0] + mx[0]) * 0.5, -mn[1], -(mn[2] + mx[2]) * 0.5])
    shell.apply_translation(shift)
    leaves.apply_translation(shift)
    flowers.apply_translation(shift)

    scene = trimesh.Scene()
    scene.add_geometry(shell, geom_name="foliage_base")
    scene.add_geometry(leaves, geom_name="foliage")
    scene.add_geometry(flowers, geom_name="flowers")
    return scene


# ===========================================================================
# COLOR SAMPLING
# ===========================================================================
def sample_palette(img, rng):
    H, W, _ = img.shape
    greens, pinks, whites = [], [], []
    xs = np.linspace(0.24, 0.90, 13)
    ys = np.linspace(0.34, 0.92, 13)
    ps = max(2, int(min(H, W) * 0.012))
    for fy in ys:
        for fx in xs:
            cx, cy = int(fx * W), int(fy * H)
            a = img[max(0, cy - ps):cy + ps + 1,
                    max(0, cx - ps):cx + ps + 1].reshape(-1, 3)
            if a.shape[0] < 4:
                continue
            med = np.median(a, axis=0).astype(float)
            r, g, b = med
            sat = med.max() - med.min()
            br = med.mean()
            if sat < 20 and br > 140:            # neutral, bright -> background
                continue
            if g >= r - 4 and g >= b and (g - b) > 6:
                greens.append(med)
            elif r > g and r >= b and sat > 10 and br < 235:
                pinks.append(med)
            elif br > 185 and (r - b) > 2 and sat < 45:
                whites.append(med)

    green = np.median(greens, axis=0) if greens else np.array([95., 142., 56.])
    pink = np.median(pinks, axis=0) if pinks else np.array([214., 150., 168.])
    white = (np.median(whites, axis=0) if whites
             else np.clip(pink * 1.15 + 20, 0, 255))

    # freshen the foliage a touch toward vivid yellow-green (photo character)
    green = np.clip(green * 1.04 + np.array([4., 10., -4.]), 0, 255)
    # brighten blossoms so they read as confetti
    pink = np.clip(pink * 1.05 + np.array([8., 4., 10.]), 0, 255)
    white = np.clip(np.maximum(white, pink * 1.12) + 6, 0, 255)
    rose = np.clip(pink * 0.55 + white * 0.45, 0, 255)
    center = np.clip(pink * 0.55 + np.array([55., 45., 0.]), 0, 255)  # warm eye
    return dict(green=green, pink=pink, white=white, rose=rose, center=center)


def _adj(col, factor):
    c = np.asarray(col, float) * factor
    w = (factor - 1.0)
    c = c + np.array([12.0, 4.0, -12.0]) * w * 3.0
    return np.clip(c, 0, 255)


# ===========================================================================
# MATTE BASE SWATCH  (photo LUMINANCE recolored to clean green -> no brown)
# ===========================================================================
def build_base_swatch(img, palette, rng, S=512):
    H, W, _ = img.shape
    x0, x1 = int(0.40 * W), int(0.66 * W)
    y0, y1 = int(0.62 * H), int(0.86 * H)
    if x1 - x0 < 8 or y1 - y0 < 8:
        x0, y0, x1, y1 = int(0.4 * W), int(0.4 * H), int(0.6 * W), int(0.6 * H)
    crop = img[y0:y1, x0:x1].astype(np.float32)

    # de-light luminance, then drive a CLEAN dark green (kills brown blotches)
    lum = crop.mean(axis=2)
    lum_img = Image.fromarray(np.clip(lum, 0, 255).astype(np.uint8), "L")
    radius = max(8, (x1 - x0) // 4)
    blur = np.asarray(lum_img.filter(ImageFilter.GaussianBlur(radius)),
                      dtype=np.float32) + 1e-3
    gain = np.clip(blur.mean() / blur, 0.6, 1.6)
    lum = lum * gain
    ratio = np.clip(lum / (lum.mean() + 1e-3), 0.65, 1.35)

    green_dark = np.clip(palette["green"] * 0.60, 0, 255)   # shadowed underbody
    out = np.clip(green_dark[None, None, :] * ratio[..., None], 0, 255).astype(np.uint8)

    # mirror-fold into a seamless tile
    half = Image.fromarray(out, "RGB").resize((S // 2, S // 2), Image.LANCZOS)
    ha = np.asarray(half)
    top = np.hstack([ha, ha[:, ::-1]])
    full = np.vstack([top, top[::-1, :]])
    return Image.fromarray(full.astype(np.uint8), "RGB")


# ===========================================================================
# CARD ATLASES  (4x4, PIL polygon silhouettes, supersampled, binary alpha)
# ===========================================================================
def _leaf_poly(px, py, L, W, ang):
    ts = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    x = (L / 2.0) * np.cos(ts)
    y = (W / 2.0) * np.sin(ts) * (1.0 - 0.30 * np.abs(np.cos(ts)))
    ca, sa = np.cos(ang), np.sin(ang)
    xr = px + x * ca - y * sa
    yr = py + x * sa + y * ca
    return list(zip(xr.tolist(), yr.tolist()))


def draw_leaf_tile(px_out, base_col, factor, rng, ss=4):
    D = px_out * ss
    img = Image.new("RGBA", (D, D), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    cx = cy = D / 2.0
    col0 = _adj(base_col, factor)
    n = int(rng.integers(120, 170))            # dense -> near-opaque leafy patch
    for _ in range(n):
        px = cx + rng.normal(0.0, 0.24 * D)
        py = cy + rng.normal(0.0, 0.24 * D)
        L = rng.uniform(0.11, 0.20) * D
        Wd = L * rng.uniform(0.30, 0.46)
        ang = rng.uniform(0.0, 2.0 * np.pi)
        m = rng.uniform(0.80, 1.18)            # per-leaf highlight / shadow
        c = np.clip(col0 * m + rng.uniform(-7, 7, 3), 0, 255).astype(int)
        dr.polygon(_leaf_poly(px, py, L, Wd, ang),
                   fill=(int(c[0]), int(c[1]), int(c[2]), 255))
    return img.resize((px_out, px_out), Image.LANCZOS)


def _draw_blossom(dr, cx, cy, R, petal, center, rng):
    n = 5
    a0 = rng.uniform(0.0, 2.0 * np.pi)
    for i in range(n):
        a = a0 + i * 2.0 * np.pi / n + rng.uniform(-0.1, 0.1)
        ox = cx + np.cos(a) * R * 0.5
        oy = cy + np.sin(a) * R * 0.5
        c = np.clip(petal + rng.uniform(-8, 8, 3), 0, 255).astype(int)
        dr.polygon(_leaf_poly(ox, oy, R * 1.00, R * 0.66, a),
                   fill=(int(c[0]), int(c[1]), int(c[2]), 255))
    cc = np.clip(center, 0, 255).astype(int)
    rr = R * 0.20
    dr.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
               fill=(int(cc[0]), int(cc[1]), int(cc[2]), 255))


def draw_flower_tile(px_out, palette, factor, rng, ss=4):
    D = px_out * ss
    img = Image.new("RGBA", (D, D), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    cx = cy = D / 2.0
    # a little green behind the cluster so tufts sit in foliage
    leafcol = _adj(palette["green"], factor * 0.9)
    for _ in range(int(rng.integers(3, 7))):
        px = cx + rng.normal(0.0, 0.20 * D)
        py = cy + rng.normal(0.0, 0.20 * D)
        L = rng.uniform(0.07, 0.13) * D
        ang = rng.uniform(0.0, 2.0 * np.pi)
        c = np.clip(leafcol * rng.uniform(0.85, 1.10), 0, 255).astype(int)
        dr.polygon(_leaf_poly(px, py, L, L * 0.4, ang),
                   fill=(int(c[0]), int(c[1]), int(c[2]), 255))
    # many bright blossoms filling the tile (confetti read)
    for _ in range(int(rng.integers(5, 9))):
        bx = cx + rng.normal(0.0, 0.22 * D)
        by = cy + rng.normal(0.0, 0.22 * D)
        R = rng.uniform(0.18, 0.30) * D
        ch = rng.random()
        pc = palette["pink"] if ch < 0.38 else (palette["rose"] if ch < 0.66
                                                else palette["white"])
        pc = _adj(pc, factor)
        _draw_blossom(dr, bx, by, R, pc, palette["center"], rng)
    return img.resize((px_out, px_out), Image.LANCZOS)


def build_atlas(kind, palette, rng, atlas_px=1024, n=4):
    tile = atlas_px // n
    atlas = Image.new("RGBA", (atlas_px, atlas_px), (0, 0, 0, 0))
    for r in range(n):
        for c in range(n):
            f = float(rng.uniform(0.84, 1.16))   # per-tile sun/shade variety
            if kind == "leaf":
                t = draw_leaf_tile(tile, palette["green"], f, rng)
            else:
                t = draw_flower_tile(tile, palette, f, rng)
            atlas.paste(t, (c * tile, r * tile))
    return atlas


# ===========================================================================
# UV + MATERIAL ASSIGNMENT
# ===========================================================================
def _card_uvs(n_cards, atlas_n, rng, inset):
    base = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    step = 1.0 / atlas_n
    uv = np.zeros((n_cards * 4, 2), dtype=np.float64)
    for i in range(n_cards):
        col = int(rng.integers(0, atlas_n))
        row = int(rng.integers(0, atlas_n))
        k = int(rng.integers(0, 4))
        corners = np.roll(base, k, axis=0)
        u0, u1 = col * step + inset, (col + 1) * step - inset
        v0, v1 = row * step + inset, (row + 1) * step - inset
        uv[i * 4:i * 4 + 4, 0] = u0 + corners[:, 0] * (u1 - u0)
        uv[i * 4:i * 4 + 4, 1] = v0 + corners[:, 1] * (v1 - v0)
    return uv


def _set_cards(mesh, atlas, rng, kind):
    v = mesh.vertices
    n_cards = len(v) // 4
    uv = _card_uvs(n_cards, 4, rng, inset=0.003)

    ymax = max(float(v[:, 1].max()), 1e-6)
    cols = np.zeros((len(v), 4), dtype=np.uint8)
    for i in range(n_cards):
        seg = v[i * 4:i * 4 + 4]
        t = float(np.clip(seg[:, 1].mean() / ymax, 0.0, 1.0))   # 0 base..1 crown
        if kind == "leaf":
            # fresh & bright; crown warmer/brighter, base cooler/darker
            b = float(np.clip(0.78 + 0.22 * t + rng.uniform(-0.05, 0.05), 0.55, 1.0))
            rgb = np.array([b * 1.04, b * 1.02, b * 0.90])
        else:
            b = float(np.clip(0.90 + 0.10 * t + rng.uniform(-0.04, 0.04), 0.7, 1.0))
            rgb = np.array([b * 1.0, b * 0.99, b * 1.0])
        cols[i * 4:i * 4 + 4, 0:3] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    cols[:, 3] = 255

    rough = 0.9 if kind == "leaf" else 0.85
    mat = PBRMaterial(name=("foliage" if kind == "leaf" else "flowers"),
                      baseColorTexture=atlas,
                      metallicFactor=0.0,
                      roughnessFactor=rough,
                      alphaMode="MASK",
                      alphaCutoff=0.4,
                      doubleSided=True)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    mesh.visual.vertex_attributes["color"] = cols


def _set_shell(mesh, base_tex):
    v = mesh.vertices
    ang = np.arctan2(v[:, 2], v[:, 0])
    u = (ang / (2.0 * np.pi)) % 1.0
    ymax = max(float(v[:, 1].max()), 1e-6)
    tv = v[:, 1] / ymax
    tiles = 2.5
    uv = np.column_stack([u * tiles, tv * tiles])

    t = np.clip(v[:, 1] / ymax, 0.0, 1.0)          # shaded underbody gradient
    b = 0.50 + 0.32 * t
    rgb = np.clip(np.column_stack([b * 1.0, b * 1.03, b * 0.88]) * 255, 0, 255)
    cols = np.column_stack([rgb, np.full(len(v), 255.0)]).astype(np.uint8)

    mat = PBRMaterial(name="foliage_base",
                      baseColorTexture=base_tex,
                      metallicFactor=0.0,
                      roughnessFactor=1.0,            # fully matte -> no sheen
                      alphaMode="OPAQUE",
                      doubleSided=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    mesh.visual.vertex_attributes["color"] = cols


def texture_scene(scene, base_tex, leaf_atlas, flower_atlas, seed):
    rng = np.random.default_rng((seed ^ 0x5DEECE66) & 0xFFFFFFFF)
    _set_shell(scene.geometry["foliage_base"], base_tex)
    _set_cards(scene.geometry["foliage"], leaf_atlas, rng, kind="leaf")
    _set_cards(scene.geometry["flowers"], flower_atlas, rng, kind="flower")


# ===========================================================================
# CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Procedural creeping-thyme cushion -> GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    img = np.asarray(Image.open(args.image).convert("RGB"))

    tex_rng = np.random.default_rng((args.seed ^ 0x9E3779B9) & 0xFFFFFFFF)
    palette = sample_palette(img, tex_rng)
    base_tex = build_base_swatch(img, palette, tex_rng, S=512)
    leaf_atlas = build_atlas("leaf", palette, tex_rng, atlas_px=1024, n=4)
    flower_atlas = build_atlas("flower", palette, tex_rng, atlas_px=1024, n=4)

    scene = build_mesh(args.seed, args.density)
    texture_scene(scene, base_tex, leaf_atlas, flower_atlas, args.seed)

    data = scene.export(file_type="glb")
    with open(args.output, "wb") as f:
        f.write(data)
    print("wrote {} ({} bytes)".format(args.output, len(data)))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stderr.write("ERROR: {}\n".format(exc))
        sys.exit(1)