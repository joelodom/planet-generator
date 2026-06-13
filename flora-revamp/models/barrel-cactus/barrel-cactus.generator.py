"""
Golden barrel cactus -- procedural geometry + photo-derived materials + GLB export.

    python cactus.py --image PATH --seed INT --density {high,med,low} --output OUT.glb

Ribbed green globe, starburst gold spines, and a woolly crown of alpha-cutout
cards.  Materials are derived by sampling colours from the reference photo.
Only numpy / trimesh / PIL / stdlib.  Deterministic given --seed.
+Y up, lowest vertex at y=0, metres.
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter


# ===========================================================================
# GEOMETRY
# ===========================================================================

# ---- MEASURED PROPORTIONS (read by eye off reference.png) -----------------
# Photo content aspect (w/h) ~= 0.95 -> the globe is very slightly TALLER than
# wide (nearly round).  So height/width is a touch over 1.
HEIGHT_OVER_WIDTH = 1.05          # total height / total width
BODY_WIDTH = 0.45                 # metres, full horizontal diameter
A = BODY_WIDTH * 0.5             # horizontal semi-axis (X/Z)
B = A * HEIGHT_OVER_WIDTH         # vertical semi-axis (Y)

RIB_DEPTH = 0.075                # groove depth as fraction of A (crest->groove)
SPINE_LEN_FRAC = 0.14            # central spine length / A
SPINE_SIDE_FRAC = 0.095          # splayed side spine length / A
SPINE_BASE_R = 0.0038            # spine base radius in metres
CROWN_DIMPLE = 0.08              # apical depression depth as fraction of A
WOOL_CARD_HW = A * 0.06          # wool card half-width

_PRESETS = {
    "high": dict(n_ribs=24, seg_per_rib=6, n_lat=58,
                 areoles_per_rib=10, spines_per_areole=11,
                 cone_seg=3, n_wool=150),
    "med":  dict(n_ribs=20, seg_per_rib=4, n_lat=34,
                 areoles_per_rib=7, spines_per_areole=8,
                 cone_seg=3, n_wool=80),
    "low":  dict(n_ribs=15, seg_per_rib=2, n_lat=16,
                 areoles_per_rib=4, spines_per_areole=5,
                 cone_seg=3, n_wool=28),
}


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _basis_from_normal(n):
    """Two unit tangents spanning the plane orthogonal to n."""
    ref = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = _normalize(np.cross(n, ref))
    t2 = _normalize(np.cross(n, t1))
    return t1, t2


def _cone(base_center, axis, length, base_r, nseg, phase):
    """A closed tapered needle standing on base_center, pointing along axis."""
    axis = _normalize(np.asarray(axis, dtype=float))
    t1, t2 = _basis_from_normal(axis)
    ang = np.linspace(0.0, 2.0 * np.pi, nseg, endpoint=False) + phase
    ring = (base_center[None, :]
            + base_r * (np.cos(ang)[:, None] * t1[None, :]
                        + np.sin(ang)[:, None] * t2[None, :]))
    tip = base_center + axis * length
    verts = np.vstack([ring, tip[None, :]])
    tip_i = nseg
    faces = []
    for i in range(nseg):
        j = (i + 1) % nseg
        faces.append([i, j, tip_i])
    for i in range(1, nseg - 1):
        faces.append([0, i + 1, i])
    return verts, np.asarray(faces, dtype=np.int64)


def _rib_taper(phi):
    return 0.35 + 0.65 * (np.cos(phi) ** 0.6)


def _surface_point(theta, phi, n_ribs, noise_terms):
    rib = np.cos(n_ribs * theta)
    disp = RIB_DEPTH * rib * _rib_taper(phi)
    wob = 0.0
    for (af, ff, gf, ph) in noise_terms:
        wob += af * np.cos(ff * theta + gf * phi + ph)
    rad_xz = A * np.cos(phi) * (1.0 + disp + wob)
    y = B * np.sin(phi)
    t = max(0.0, (phi - 0.55 * np.pi * 0.5) / (np.pi * 0.5 - 0.55 * np.pi * 0.5))
    y -= CROWN_DIMPLE * A * (t ** 2)
    x = rad_xz * np.cos(theta)
    z = rad_xz * np.sin(theta)
    return np.array([x, y, z])


def _build_body(cfg, rng):
    n_ribs = cfg["n_ribs"]
    n_long = n_ribs * cfg["seg_per_rib"]
    n_lat = cfg["n_lat"]

    noise_terms = []
    for _ in range(3):
        af = rng.uniform(0.006, 0.014)
        ff = float(rng.integers(2, 5))
        gf = float(rng.integers(1, 4))
        ph = rng.uniform(0.0, 2.0 * np.pi)
        noise_terms.append((af, ff, gf, ph))

    thetas = np.linspace(0.0, 2.0 * np.pi, n_long, endpoint=False)
    phis = np.linspace(-np.pi * 0.5, np.pi * 0.5, n_lat + 1)
    inner_phis = phis[1:-1]

    verts = []
    grid = np.zeros((len(inner_phis), n_long), dtype=np.int64)
    for i, phi in enumerate(inner_phis):
        for j, th in enumerate(thetas):
            grid[i, j] = len(verts)
            verts.append(_surface_point(th, phi, n_ribs, noise_terms))

    bottom_i = len(verts)
    verts.append(_surface_point(0.0, -np.pi * 0.5, n_ribs, noise_terms))
    top_i = len(verts)
    verts.append(_surface_point(0.0, np.pi * 0.5, n_ribs, noise_terms))

    faces = []
    rings = len(inner_phis)
    for i in range(rings - 1):
        for j in range(n_long):
            jn = (j + 1) % n_long
            a = grid[i, j]
            b = grid[i, jn]
            c = grid[i + 1, jn]
            d = grid[i + 1, j]
            faces.append([a, b, c])
            faces.append([a, c, d])
    for j in range(n_long):
        jn = (j + 1) % n_long
        faces.append([bottom_i, grid[0, jn], grid[0, j]])
    for j in range(n_long):
        jn = (j + 1) % n_long
        faces.append([top_i, grid[rings - 1, j], grid[rings - 1, jn]])

    mesh = trimesh.Trimesh(vertices=np.asarray(verts, dtype=float),
                           faces=np.asarray(faces, dtype=np.int64),
                           process=True)
    mesh.fix_normals()
    return mesh, noise_terms


def _ellipsoid_normal(p):
    g = np.array([p[0] / (A * A), p[1] / (B * B), p[2] / (A * A)])
    return _normalize(g)


def _build_spines(cfg, rng, noise_terms):
    """Solid gold needle cones radiating from areoles on the rib crests."""
    n_ribs = cfg["n_ribs"]
    nseg = cfg["cone_seg"]
    per_rib = cfg["areoles_per_rib"]
    per_areole = cfg["spines_per_areole"]

    sp_verts, sp_faces = [], []

    def emit(tv, tf, base, axis, length, base_r, phase):
        v, f = _cone(base, axis, length, base_r, nseg, phase)
        off = len(tv)
        tv.extend(v.tolist())
        tf.extend((f + off).tolist())

    base_phis = np.linspace(-0.42 * np.pi, 0.40 * np.pi, per_rib)
    step = (base_phis[-1] - base_phis[0]) / max(per_rib - 1, 1)

    for k in range(n_ribs):
        theta = 2.0 * np.pi * k / n_ribs
        stagger = 0.5 * step if (k % 2) else 0.0
        for phi0 in base_phis:
            phi = float(np.clip(phi0 + stagger, -0.46 * np.pi, 0.45 * np.pi))
            p = _surface_point(theta, phi, n_ribs, noise_terms)
            n = _ellipsoid_normal(p)
            t1, t2 = _basis_from_normal(n)
            base = p + n * 0.002

            Lc = SPINE_LEN_FRAC * A * (1.0 + 0.16 * rng.standard_normal())
            emit(sp_verts, sp_faces, base, n, max(Lc, 0.01),
                 SPINE_BASE_R, rng.uniform(0, np.pi))

            n_side = per_areole - 1
            for s in range(n_side):
                alpha = np.radians(36.0 + 18.0 * rng.random())
                beta = 2.0 * np.pi * s / max(n_side, 1) + rng.uniform(-0.3, 0.3)
                d = (np.cos(alpha) * n
                     + np.sin(alpha) * (np.cos(beta) * t1 + np.sin(beta) * t2))
                d = _normalize(d)
                Ls = SPINE_SIDE_FRAC * A * (1.0 + 0.20 * rng.standard_normal())
                emit(sp_verts, sp_faces, base, d, max(Ls, 0.008),
                     SPINE_BASE_R * 0.82, rng.uniform(0, np.pi))

    return trimesh.Trimesh(vertices=np.asarray(sp_verts, dtype=float),
                           faces=np.asarray(sp_faces, dtype=np.int64),
                           process=True)


def _build_wool_cards(cfg, rng, noise_terms):
    """Woolly crown tuft as flat alpha-cutout CARDS mapped to a 4x4 atlas."""
    n_ribs = cfg["n_ribs"]
    apex = _surface_point(0.0, 0.49 * np.pi, n_ribs, noise_terms)
    crown_r = A * 0.26
    n_cards = cfg["n_wool"]

    verts, faces, uvs = [], [], []
    for _ in range(n_cards):
        rr = crown_r * np.sqrt(rng.random())
        aa = rng.uniform(0.0, 2.0 * np.pi)
        c = apex + np.array([rr * np.cos(aa),
                             rng.uniform(-0.01, 0.03) * A,
                             rr * np.sin(aa)])
        nn = _normalize(rng.standard_normal(3))
        e1, e2 = _basis_from_normal(nn)
        hw = WOOL_CARD_HW * float(np.exp(0.3 * rng.standard_normal()))
        hh = hw * rng.uniform(0.8, 1.4)

        p0 = c - e1 * hw - e2 * hh
        p1 = c + e1 * hw - e2 * hh
        p2 = c + e1 * hw + e2 * hh
        p3 = c - e1 * hw + e2 * hh
        base = len(verts)
        verts.extend([p0, p1, p2, p3])
        faces.append([base, base + 1, base + 2])
        faces.append([base, base + 2, base + 3])

        tx = int(rng.integers(0, 4))
        ty = int(rng.integers(0, 4))
        u0, u1 = tx / 4.0, (tx + 1) / 4.0
        v0, v1 = ty / 4.0, (ty + 1) / 4.0
        quad = [(u0, v1), (u1, v1), (u1, v0), (u0, v0)]
        rot = int(rng.integers(0, 4))
        quad = quad[rot:] + quad[:rot]
        uvs.extend(quad)

    mesh = trimesh.Trimesh(vertices=np.asarray(verts, dtype=float),
                           faces=np.asarray(faces, dtype=np.int64),
                           process=False)
    mesh.metadata["uv"] = np.asarray(uvs, dtype=float)
    return mesh


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    if density not in _PRESETS:
        density = "high"
    cfg = _PRESETS[density]
    rng = np.random.default_rng(seed)

    body, noise_terms = _build_body(cfg, rng)
    spines = _build_spines(cfg, rng, noise_terms)
    wool = _build_wool_cards(cfg, rng, noise_terms)

    miny = min(float(m.vertices[:, 1].min()) for m in (body, spines, wool))
    offset = np.array([0.0, -miny, 0.0])
    for m in (body, spines, wool):
        m.apply_translation(offset)

    scene = trimesh.Scene()
    scene.add_geometry(body, geom_name="body")
    scene.add_geometry(spines, geom_name="spines")
    scene.add_geometry(wool, geom_name="wool")
    return scene


# ===========================================================================
# TEXTURING  (colours sampled from the reference photo)
# ===========================================================================

_GREEN_CENTERS = [(0.50, 0.58), (0.43, 0.64), (0.57, 0.64),
                  (0.50, 0.70), (0.45, 0.50), (0.55, 0.50)]
_GOLD_CENTERS = [(0.50, 0.16), (0.44, 0.20), (0.56, 0.20),
                 (0.50, 0.24), (0.40, 0.40), (0.60, 0.40)]


def _to_np(img):
    return np.asarray(img.convert("RGB"), dtype=np.float64)


def _patch_medians(arr, centers, half=0.03):
    H, W, _ = arr.shape
    out = []
    for (cx, cy) in centers:
        x0 = max(0, int((cx - half) * W)); x1 = min(W, int((cx + half) * W))
        y0 = max(0, int((cy - half) * H)); y1 = min(H, int((cy + half) * H))
        if x1 <= x0 or y1 <= y0:
            continue
        patch = arr[y0:y1, x0:x1, :].reshape(-1, 3)
        out.append(np.median(patch, axis=0))
    return np.array(out) if out else np.zeros((0, 3))


def _sample_green(arr):
    cands = _patch_medians(arr, _GREEN_CENTERS, half=0.035)
    if len(cands) == 0:
        return np.array([70.0, 125.0, 80.0])
    m = (cands[:, 1] >= cands[:, 0]) & (cands[:, 1] >= cands[:, 2]) \
        & (cands[:, 1] - cands[:, 2] > 6)
    sel = cands[m] if m.any() else cands
    return np.median(sel, axis=0)


def _sample_gold(arr):
    cands = _patch_medians(arr, _GOLD_CENTERS, half=0.02)
    if len(cands) == 0:
        return np.array([220.0, 190.0, 120.0])
    m = (cands[:, 0] >= cands[:, 2]) & (cands[:, 0] > 110) & (cands[:, 1] > 90)
    sel = cands[m] if m.any() else cands
    return np.median(sel, axis=0)


def _mirror_tile(arr, res):
    h = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)) \
        .resize((res // 2, res // 2), Image.LANCZOS)
    h = np.asarray(h, dtype=np.float64)
    top = np.concatenate([h, h[:, ::-1, :]], axis=1)
    full = np.concatenate([top, top[::-1, :, :]], axis=0)
    return full


def _mottle(res, rng, scale=8, amp=14.0):
    low = rng.normal(0.0, 1.0, (scale, scale))
    low = (low - low.min()) / (np.ptp(low) + 1e-9)
    up = np.asarray(
        Image.fromarray((low * 255).astype(np.uint8)).resize((res, res), Image.BILINEAR),
        dtype=np.float64) / 255.0
    return (up - 0.5) * 2.0 * amp


def _swatch(color, res, rng, noise=8.0, mottle_amp=12.0, mottle_scale=8):
    base = np.ones((res, res, 3), dtype=np.float64) * np.asarray(color)[None, None, :]
    base = base + _mottle(res, rng, scale=mottle_scale, amp=mottle_amp)[:, :, None]
    base = base + rng.normal(0.0, noise, (res, res, 3))
    return np.clip(base, 0, 255)


def _body_texture(arr, green, rng, res=768):
    """Clean, bright, saturated green flesh swatch (no photo-crop streaks)."""
    g = np.clip(green * 1.15, 0, 255)
    # push saturation toward green so it reads vivid, not muddy
    g = np.array([g[0] * 0.88, min(g[1] * 1.10, 255.0), g[2] * 0.82])
    sw = _swatch(g, res, rng, noise=4.0, mottle_amp=9.0, mottle_scale=12)
    full = _mirror_tile(sw, res)               # make it seamlessly tileable
    return Image.fromarray(np.clip(full, 0, 255).astype(np.uint8))


def _normal_from_albedo(pil, strength=1.2):
    g = np.asarray(pil.convert("L"), dtype=np.float64) / 255.0
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    nx = -gx * strength; ny = -gy * strength; nz = np.ones_like(g)
    l = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    n = np.stack([nx / l, ny / l, nz / l], axis=-1)
    return Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8))


def _wool_atlas(gold, rng, res=1024):
    """4x4 atlas of fuzzy wool tufts (binary alpha, supersampled then LANCZOS)."""
    SS = 4
    tile = (res // 4) * SS
    big = tile * 4
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    base = 0.45 * np.asarray(gold) + 0.55 * np.array([255.0, 248.0, 210.0])
    for ty in range(4):
        for tx in range(4):
            cx = tx * tile + tile * 0.5
            cy = ty * tile + tile * 0.5
            lit = 1.0 - ty / 3.0
            warm = np.array([1.03, 1.0, 0.88])
            tcol = np.clip(base * (0.78 + 0.45 * lit) * warm, 0, 255)
            n_spk = int(rng.integers(22, 34))
            for _ in range(n_spk):
                ang = rng.uniform(0.0, 2.0 * np.pi)
                length = tile * 0.46 * rng.uniform(0.45, 1.0)
                wbase = tile * 0.032
                tip = (cx + np.cos(ang) * length, cy + np.sin(ang) * length)
                perp = (-np.sin(ang), np.cos(ang))
                b1 = (cx + perp[0] * wbase, cy + perp[1] * wbase)
                b2 = (cx - perp[0] * wbase, cy - perp[1] * wbase)
                cc = np.clip(tcol + rng.normal(0.0, 12.0, 3), 0, 255).astype(int)
                draw.polygon([b1, b2, tip],
                             fill=(int(cc[0]), int(cc[1]), int(cc[2]), 255))
    return img.resize((res, res), Image.LANCZOS)


def _make_material(name, tex, roughness, normal=None,
                   alpha_mask=False, double_sided=False):
    kw = dict(name=name, baseColorTexture=tex,
              metallicFactor=0.0, roughnessFactor=float(roughness))
    if alpha_mask:
        kw.update(alphaMode="MASK", alphaCutoff=0.45, doubleSided=True)
    elif double_sided:
        kw.update(doubleSided=True)
    mat = trimesh.visual.material.PBRMaterial(**kw)
    if normal is not None:
        mat.normalTexture = normal
    return mat


def _attach(mesh, uv, material, colors_uint8):
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    mesh.visual.vertex_attributes["color"] = colors_uint8


def _texture_scene(scene, arr, seed, density):
    rng = np.random.default_rng(seed)
    cfg = _PRESETS[density if density in _PRESETS else "high"]
    n_ribs = cfg["n_ribs"]
    green = _sample_green(arr)
    gold = _sample_gold(arr)

    body = scene.geometry["body"]
    spines = scene.geometry["spines"]
    wool = scene.geometry["wool"]

    # ---- BODY: cylindrical UVs + rib-phase sun/shade vertex tint ----
    bv = body.vertices
    ang = np.arctan2(bv[:, 2], bv[:, 0])
    ymin, ymax = bv[:, 1].min(), bv[:, 1].max()
    U_TILES, V_TILES = 3.0, 3.0
    bu = (ang / (2.0 * np.pi) + 0.5) * U_TILES
    bvv = (bv[:, 1] - ymin) / (ymax - ymin + 1e-9) * V_TILES
    body_uv = np.column_stack([bu, bvv])

    hb = (bv[:, 1] - ymin) / (ymax - ymin + 1e-9)          # 0 base .. 1 crown
    rib = np.cos(n_ribs * ang)                             # +1 crest .. -1 groove
    crest = rib * 0.5 + 0.5                                # 0 groove .. 1 crest
    # bright crests, darker cooler grooves; brighter toward the top
    bright = (0.80 + 0.20 * hb) * (0.78 + 0.22 * crest)
    bcol = np.empty((len(bv), 4))
    bcol[:, 0] = bright * (0.94 + 0.08 * crest)            # crests a touch warmer
    bcol[:, 1] = bright * 1.02
    bcol[:, 2] = bright * (1.04 - 0.10 * crest)            # grooves bluer
    bcol[:, :3] += rng.normal(0.0, 0.015, (len(bv), 3))
    bcol[:, 3] = 1.0
    bcol = np.clip(bcol, 0.0, 1.0)
    body_tex = _body_texture(arr, green, rng, res=768)
    body_nrm = _normal_from_albedo(body_tex, strength=1.0)
    _attach(body, body_uv,
            _make_material("cactus_flesh", body_tex, 0.45, normal=body_nrm),
            (bcol * 255).astype(np.uint8))

    body_center = np.array([0.0, 0.5 * (ymin + ymax), 0.0])

    # ---- SPINES: pale straw-gold; tips near-white ----
    sv = spines.vertices
    dist = np.linalg.norm(sv - body_center[None, :], axis=1)
    t = (dist - dist.min()) / (np.ptp(dist) + 1e-9)
    spine_uv = np.column_stack([(np.arange(len(sv)) % 16) / 16.0, t])
    sb = 0.72 + 0.28 * t
    scol = np.empty((len(sv), 4))
    scol[:, 0] = sb
    scol[:, 1] = sb * (0.95 + 0.04 * t)
    scol[:, 2] = sb * (0.78 + 0.18 * t)
    scol[:, :3] += rng.normal(0.0, 0.02, (len(sv), 3))
    scol[:, 3] = 1.0
    scol = np.clip(scol, 0.0, 1.0)
    gold_base = np.clip(gold * np.array([1.00, 0.95, 0.80]), 0, 255)
    spine_tex = Image.fromarray(
        _swatch(gold_base, 512, rng, noise=6.0, mottle_amp=9.0).astype(np.uint8))
    _attach(spines, spine_uv,
            _make_material("cactus_spines", spine_tex, 0.7),
            (scol * 255).astype(np.uint8))

    # ---- WOOL: alpha-cutout cards (MASK) mapped to fuzzy-tuft atlas ----
    wv = wool.vertices
    wool_uv = np.asarray(wool.metadata["uv"], dtype=float)
    wy = wv[:, 1]
    wt = (wy - wy.min()) / (np.ptp(wy) + 1e-9)
    wb = 0.85 + 0.15 * wt
    wcol = np.empty((len(wv), 4))
    wcol[:, 0] = wb
    wcol[:, 1] = wb * 0.99
    wcol[:, 2] = wb * 0.88
    wcol[:, :3] += rng.normal(0.0, 0.015, (len(wv), 3))
    wcol[:, 3] = 1.0
    wcol = np.clip(wcol, 0.0, 1.0)
    wool_atlas = _wool_atlas(gold, rng, res=1024)
    _attach(wool, wool_uv,
            _make_material("cactus_wool", wool_atlas, 0.9,
                           alpha_mask=True, double_sided=True),
            (wcol * 255).astype(np.uint8))

    return scene


# ===========================================================================
# CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Procedural golden barrel cactus -> GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        arr = _to_np(Image.open(args.image))
        scene = build_mesh(args.seed, args.density)
        scene = _texture_scene(scene, arr, args.seed, args.density)
        scene.export(args.output)
    except Exception as exc:  # noqa: BLE001
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1
    print("wrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())