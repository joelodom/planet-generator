"""
Procedural cushion-plant (moss-phlox / dianthus character) generator + texturer.

A low domed mound of mossy green foliage almost completely smothered by hundreds
of small pink five-petalled flowers. Materials are derived from a reference photo,
per-surface UVs are applied, and a textured GLB is exported.

CLI:
    python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial


# ==========================================================================
# GEOMETRY
# ==========================================================================
OVERALL_WIDTH = 0.40                 # meters, full diameter of the cushion
DOME_HEIGHT_RATIO = 0.80             # height / width  -> front aspect ~1.25 (photo ~1.21)
TUCK_ANGLE_DEG = 100.0               # just past hemisphere -> gently tucked base
FLOWER_RADIUS_FRAC = 0.050           # flower saucer radius / crown width (big, overlapping)
CARD_HALF_FRAC = 0.030               # leaf-card half size / crown width
CLUMP_RADIUS_FRAC = 0.085            # foliage clump radius / crown width

_PARAMS = {
    "high": dict(L=64, R=20, n_flowers=1500, n_cards=1500, n_clumps=20, rim=18),
    "med":  dict(L=40, R=14, n_flowers=620,  n_cards=560,  n_clumps=14, rim=14),
    "low":  dict(L=24, R=9,  n_flowers=200,  n_cards=170,  n_clumps=8,  rim=10),
}


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def _basis_from_normal(n):
    n = _unit(n)
    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = _unit(np.cross(a, n))
    t2 = np.cross(n, t1)
    return t1, t2, n


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    if density not in _PARAMS:
        density = "high"
    P = _PARAMS[density]
    rng = np.random.default_rng(seed)

    Rx = Rz = OVERALL_WIDTH * 0.5
    phi_max = np.radians(TUCK_ANGLE_DEG)
    phi0 = np.radians(3.0)
    H = OVERALL_WIDTH * DOME_HEIGHT_RATIO
    Ry = H / (1.0 - np.cos(phi_max))

    flower_R = FLOWER_RADIUS_FRAC * OVERALL_WIDTH
    card_hs = CARD_HALF_FRAC * OVERALL_WIDTH
    clump_R = CLUMP_RADIUS_FRAC * OVERALL_WIDTH

    # gentle, damped lobes -> a clean rounded pillow (not boxy)
    n_lobes = int(rng.integers(2, 4))
    lobe_f = rng.choice(np.array([2, 3, 4]), size=n_lobes, replace=False)
    lobe_a = rng.uniform(0.015, 0.040, size=n_lobes)
    lobe_p = rng.uniform(0.0, 2 * np.pi, size=n_lobes)
    bump_ft = rng.uniform(6.0, 11.0, size=3)
    bump_fp = rng.uniform(2.0, 5.0, size=3)
    bump_a = rng.uniform(0.005, 0.015, size=3)
    bump_p = rng.uniform(0.0, 2 * np.pi, size=3)

    def _perturb(theta, phi):
        v = np.zeros_like(theta * phi * 1.0) if np.ndim(theta) or np.ndim(phi) else 0.0
        for f, a, p in zip(lobe_f, lobe_a, lobe_p):
            v = v + a * np.cos(f * theta + p)
        for ft, fp, a, p in zip(bump_ft, bump_fp, bump_a, bump_p):
            v = v + a * np.sin(ft * theta + p) * np.cos(fp * phi)
        return v

    def _surface(theta, phi):
        rs = 1.0 + np.sin(phi) * _perturb(theta, phi)
        st = np.sin(phi)
        ct = np.cos(phi)
        x = Rx * rs * st * np.cos(theta)
        z = Rz * rs * st * np.sin(theta)
        y = Ry * rs * ct
        return np.stack([x, y, z], axis=-1)

    _C = np.array([0.0, Ry * 0.25, 0.0])

    def _surface_pn(theta, phi):
        eps = 1e-3
        p = _surface(theta, phi)
        pt = _surface(theta + eps, phi)
        pp = _surface(theta, phi + eps)
        n = _unit(np.cross(pt - p, pp - p))
        if np.dot(n, p - _C) < 0.0:
            n = -n
        return p, n

    def _clamp_env(v, lim):
        """Pull stray card vertices back onto the foliage envelope (silhouette guard)."""
        f = (v[:, 0] / Rx) ** 2 + (v[:, 1] / Ry) ** 2 + (v[:, 2] / Rz) ** 2
        s = np.where(f > lim, np.sqrt(lim / np.maximum(f, 1e-9)), 1.0)
        return v * s[:, None]

    # ----- cushion dome ---------------------------------------------------
    L, R = P["L"], P["R"]
    thetas = np.linspace(0.0, 2 * np.pi, L, endpoint=False)
    phis = np.linspace(phi0, phi_max, R)
    TH, PH = np.meshgrid(thetas, phis)
    grid = _surface(TH, PH).reshape(-1, 3)

    pole = _surface(np.array(0.0), np.array(0.0)).reshape(3)
    bottom = grid[(R - 1) * L:].mean(axis=0)
    bottom[0] = 0.0
    bottom[2] = 0.0

    cushion_v = np.vstack([pole[None, :], grid, bottom[None, :]])
    pole_idx = 0
    bc_idx = cushion_v.shape[0] - 1

    def gidx(i, j):
        return 1 + i * L + (j % L)

    faces = []
    j = np.arange(L)
    for jj in j:
        faces.append([pole_idx, gidx(0, jj), gidx(0, jj + 1)])
    ii = np.arange(R - 1)[:, None]
    jcol = np.arange(L)[None, :]
    a = 1 + ii * L + jcol
    b = 1 + ii * L + (jcol + 1) % L
    c = 1 + (ii + 1) * L + (jcol + 1) % L
    d = 1 + (ii + 1) * L + jcol
    faces.extend(np.stack([a, b, c], axis=-1).reshape(-1, 3).tolist())
    faces.extend(np.stack([a, c, d], axis=-1).reshape(-1, 3).tolist())
    for jj in j:
        faces.append([bc_idx, gidx(R - 1, jj + 1), gidx(R - 1, jj)])
    cushion_f = np.array(faces, dtype=np.int64)

    # ----- templates ------------------------------------------------------
    quad_v = np.array([[-1.0, -1.0, 0.0],
                       [1.0, -1.0, 0.0],
                       [1.0, 1.0, 0.0],
                       [-1.0, 1.0, 0.0]])
    quad_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    def _petal_template(rim):
        # full, rounded five-petalled saucer (r in [0.62 .. 1.0]) -> reads as a bloom
        t = np.linspace(0.0, 2 * np.pi, rim, endpoint=False)
        lobe = 0.5 + 0.5 * np.cos(5.0 * t)
        rnorm = 0.62 + 0.38 * lobe ** 1.4
        zr = -0.10 * rnorm ** 2                    # shallow saucer cup
        rim_v = np.stack([rnorm * np.cos(t), rnorm * np.sin(t), zr], axis=-1)
        pet_v = np.vstack([[0.0, 0.0, 0.0], rim_v])
        pet_f = np.array([[0, 1 + k, 1 + (k + 1) % rim] for k in range(rim)],
                         dtype=np.int64)
        return pet_v, pet_f

    pet_v, pet_f = _petal_template(P["rim"])

    def _placer():
        return {"v": [], "f": [], "n": 0}

    def _place(acc, tmpl_v, tmpl_f, size, pos, normal, roll):
        t1, t2, nn = _basis_from_normal(normal)
        cs, sn = np.cos(roll), np.sin(roll)
        u = cs * t1 + sn * t2
        w = -sn * t1 + cs * t2
        Rm = np.column_stack([u, w, nn])
        world = (tmpl_v * size) @ Rm.T + pos
        acc["v"].append(world)
        acc["f"].append(tmpl_f + acc["n"])
        acc["n"] += tmpl_v.shape[0]

    def _jitter_normal(n, max_deg):
        t1, t2, _ = _basis_from_normal(n)
        ang = np.radians(max_deg) * np.sqrt(rng.random())
        ph = rng.uniform(0.0, 2 * np.pi)
        tan = np.cos(ph) * t1 + np.sin(ph) * t2
        return _unit(n + np.tan(ang) * tan)

    # ----- leaves : tight clumps hugging the shell (mossy understory) ------
    leaves = _placer()
    n_clumps = max(1, P["n_clumps"])
    per_clump = int(np.ceil(P["n_cards"] / n_clumps))
    for _ in range(n_clumps):
        ctheta = rng.uniform(0.0, 2 * np.pi)
        cphi = phi_max * rng.random() ** 0.85
        cpos, cn = _surface_pn(ctheta, cphi)
        cpos = cpos * 0.97                          # seat clump just inside the shell
        ct1, ct2, _ = _basis_from_normal(cn)
        for _ in range(per_clump):
            off = (np.clip(rng.normal(0, 0.5), -1, 1) * (0.6 * clump_R) * ct1 +
                   np.clip(rng.normal(0, 0.5), -1, 1) * (0.6 * clump_R) * ct2 +
                   rng.uniform(-0.4, 0.05) * clump_R * cn)   # bias inward, never proud
            pos = cpos + off
            nrm = _jitter_normal(cn, 22.0)
            hs = card_hs * float(np.exp(rng.normal(0.0, 0.22)))
            _place(leaves, quad_v, quad_f, hs, pos, nrm, rng.uniform(0, 2 * np.pi))

    # ----- flowers : many, large, evenly covering the whole dome -----------
    petals = _placer()
    phi_flower_max = phi_max * 0.98
    cmin, cmax = np.cos(phi_flower_max), np.cos(phi0)
    for _ in range(P["n_flowers"]):
        cosv = rng.uniform(cmin, cmax)             # area-uniform -> even surface coverage
        phi = np.arccos(np.clip(cosv, -1.0, 1.0))
        theta = rng.uniform(0.0, 2 * np.pi)
        p, n = _surface_pn(theta, phi)
        n2 = _jitter_normal(n, 20.0)
        size = flower_R * float(np.exp(rng.normal(0.0, 0.26)))
        roll = rng.uniform(0.0, 2 * np.pi)
        pos = p + n2 * (0.02 * size)               # sit just proud of the foliage
        _place(petals, pet_v, pet_f, size, pos, n2, roll)

    def _finish(acc):
        if acc["n"] == 0:
            return None
        return np.concatenate(acc["v"], axis=0), np.concatenate(acc["f"], axis=0)

    leaves_r = _finish(leaves)
    petals_r = _finish(petals)
    if leaves_r is not None:
        leaves_r = (_clamp_env(leaves_r[0], 1.10), leaves_r[1])
    if petals_r is not None:
        petals_r = (_clamp_env(petals_r[0], 1.14), petals_r[1])

    # ----- ground the whole object's lowest vertex at y=0 -----------------
    ymins = [cushion_v[:, 1].min()]
    if leaves_r is not None:
        ymins.append(leaves_r[0][:, 1].min())
    if petals_r is not None:
        ymins.append(petals_r[0][:, 1].min())
    ground_y = float(min(ymins))
    shift = np.array([cushion_v[:, 0].mean(), ground_y, cushion_v[:, 2].mean()])

    cushion_v = cushion_v - shift
    cushion = trimesh.Trimesh(vertices=cushion_v, faces=cushion_f, process=True)
    cushion.fix_normals()

    scene = trimesh.Scene()
    scene.add_geometry(cushion, geom_name="cushion")

    if leaves_r is not None:
        lv, lf = leaves_r
        lm = trimesh.Trimesh(vertices=lv - shift, faces=lf, process=False)
        scene.add_geometry(lm, geom_name="leaves")

    if petals_r is not None:
        pv, pf = petals_r
        pm = trimesh.Trimesh(vertices=pv - shift, faces=pf, process=False)
        pm.metadata["uv_local"] = np.column_stack(
            [0.5 + 0.46 * pet_v[:, 0], 0.5 + 0.46 * pet_v[:, 1]])
        scene.add_geometry(pm, geom_name="petals")

    return scene


# ==========================================================================
# COLOR / TEXTURE helpers
# ==========================================================================
def _mix(a, b, t):
    t = np.asarray(t, float)[..., None]
    return np.asarray(a, float) * (1.0 - t) + np.asarray(b, float) * t


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def extract_palette(rgb, hsv):
    """Sample real body colors from WELL INSIDE the silhouette (no background)."""
    H, W = rgb.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W * 0.52, H * 0.54
    rad = 0.40 * min(H, W)
    interior = ((xx - cx) ** 2 + (yy - cy) ** 2) <= rad * rad

    hue = hsv[..., 0].astype(np.int32)
    sat = hsv[..., 1].astype(np.int32)
    val = hsv[..., 2].astype(np.int32)
    Rc = rgb[..., 0].astype(np.int32)
    Gc = rgb[..., 1].astype(np.int32)

    valid = interior & (sat > 45) & (val > 35) & (val < 248)
    pink = valid & (Rc > Gc + 8) & ((hue >= 200) | (hue <= 14))
    green = valid & (Gc >= Rc) & (hue >= 45) & (hue <= 120)

    def med(m, fallback):
        if int(m.sum()) < 60:
            return np.array(fallback, float)
        return np.median(rgb[m].astype(float), axis=0)

    pink_base = med(pink, [212, 70, 150])
    green_base = med(green, [72, 96, 54])
    return pink_base, green_base


def pick_green_patch(rgb):
    """Greenest interior patch -> real moss detail for the cushion swatch."""
    H, W = rgb.shape[:2]
    hs = max(6, int(0.08 * min(H, W)))
    best, best_score = None, -1e9
    for cyf in (0.46, 0.58, 0.70):
        for cxf in (0.34, 0.50, 0.66):
            cx, cy = int(cxf * W), int(cyf * H)
            x0, x1 = max(0, cx - hs), min(W, cx + hs)
            y0, y1 = max(0, cy - hs), min(H, cy + hs)
            patch = rgb[y0:y1, x0:x1].astype(float)
            if patch.size == 0:
                continue
            m = np.median(patch.reshape(-1, 3), axis=0)
            score = m[1] - 0.5 * (m[0] + m[2])
            if score > best_score:
                best_score, best = score, patch
    if best is None:
        best = np.full((32, 32, 3), [72, 96, 54], float)
    return best


def delight(patch):
    """Flatten baked lighting by dividing out a heavily blurred luminance."""
    p = patch.astype(float)
    lum = 0.2126 * p[..., 0] + 0.7152 * p[..., 1] + 0.0722 * p[..., 2]
    limg = Image.fromarray(np.clip(lum, 0, 255).astype(np.uint8), "L")
    r = max(4, min(limg.size) // 6)
    blur = np.asarray(limg.filter(ImageFilter.GaussianBlur(r)), float) + 1e-3
    gain = np.clip(lum.mean() / blur, 0.6, 1.6)
    return np.clip(p * gain[..., None], 0, 255)


def make_cushion_albedo(patch, green, rng, res=1024):
    """Real moss patch -> de-lit -> mirror-folded (seamless) -> tinted swatch."""
    d = delight(patch)
    top = np.concatenate([d, d[:, ::-1]], axis=1)
    full = np.concatenate([top, top[::-1]], axis=0)
    img = Image.fromarray(np.clip(full, 0, 255).astype(np.uint8)).resize(
        (res, res), Image.LANCZOS)
    arr = np.asarray(img, float)
    med = np.median(arr.reshape(-1, 3), axis=0) + 1e-3
    arr = arr * (green / med)
    arr = np.clip(arr + rng.normal(0, 7, (res, res, 1)), 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def make_foliage_atlas(green, rng, A=1024):
    """4x4 atlas of dense, rounded mossy clusters; near-binary alpha, varied light."""
    T = A // 4
    SS = 4
    atlas = Image.new("RGBA", (A, A), (0, 0, 0, 0))
    for ti in range(16):
        col, row = ti % 4, ti // 4
        lit = 0.82 + 0.36 * (((ti * 9) % 16) / 15.0)
        warm = 1.0 + 0.08 * ((((ti * 3) % 16) / 15.0) - 0.5)
        timg = Image.new("RGBA", (T * SS, T * SS), (0, 0, 0, 0))
        draw = ImageDraw.Draw(timg)
        nleaf = int(rng.integers(150, 210))
        for _ in range(nleaf):
            cx = rng.uniform(-0.05, 1.05) * T * SS
            cy = rng.uniform(-0.05, 1.05) * T * SS
            ang = rng.uniform(0.0, 2 * np.pi)
            ln = rng.uniform(0.08, 0.16) * T * SS
            wd = ln * rng.uniform(0.35, 0.60)
            dx, dy = np.cos(ang), np.sin(ang)
            px, py = -dy, dx
            pts = [(cx + dx * ln / 2, cy + dy * ln / 2),
                   (cx + px * wd / 2, cy + py * wd / 2),
                   (cx - dx * ln / 2, cy - dy * ln / 2),
                   (cx - px * wd / 2, cy - py * wd / 2)]
            jit = rng.uniform(0.80, 1.16)
            cr = int(np.clip(green[0] * jit * warm * lit, 0, 255))
            cg = int(np.clip(green[1] * jit * lit, 0, 255))
            cb = int(np.clip(green[2] * jit * lit * 0.97, 0, 255))
            draw.polygon(pts, fill=(cr, cg, cb, 255))
        timg = timg.resize((T, T), Image.LANCZOS)
        atlas.paste(timg, (col * T, row * T), timg)
    return atlas


def make_petal_atlas(pink, pale, warm, rng, A=1024):
    """4x4 atlas of radial pink flowers: pale center -> vivid mid -> warm edge."""
    T = A // 4
    atlas = np.zeros((A, A, 4), np.uint8)
    yy, xx = np.mgrid[0:T, 0:T]
    cx = cy = (T - 1) / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ang = np.arctan2(yy - cy, xx - cx)
    rad = 0.49 * T
    for ti in range(16):
        col, row = ti % 4, ti // 4
        f = 0.86 + 0.26 * (((ti * 7) % 16) / 15.0)
        warmth = 1.0 + 0.10 * ((((ti * 5) % 16) / 15.0) - 0.5)
        t = np.clip(dist / rad, 0.0, 1.0)
        c1 = _mix(pale, pink, _smooth(np.clip(t / 0.45, 0, 1)))
        c2 = _mix(c1, warm, _smooth(np.clip((t - 0.55) / 0.45, 0, 1)))
        lobe = 0.94 + 0.06 * np.cos(5.0 * ang)
        rgbcol = c2 * (f * lobe)[..., None]
        rgbcol[..., 0] *= warmth
        rgbcol = np.clip(rgbcol + rng.normal(0, 5, (T, T, 1)), 0, 255)
        alpha = np.clip((rad - dist) + 0.5, 0, 1) * 255.0     # binary disc, 1px AA
        tile = np.concatenate([rgbcol, alpha[..., None]], axis=2).astype(np.uint8)
        atlas[row * T:(row + 1) * T, col * T:(col + 1) * T] = tile
    return Image.fromarray(atlas, "RGBA")


def make_normal(img_rgb, strength=1.4):
    """Tangent-space normal map from albedo (height = inverse luminance, Sobel)."""
    a = np.asarray(img_rgb.convert("L"), float) / 255.0
    h = 1.0 - a
    gx = np.zeros_like(h)
    gy = np.zeros_like(h)
    gx[:, 1:-1] = (h[:, 2:] - h[:, :-2]) * 0.5
    gy[1:-1, :] = (h[2:, :] - h[:-2, :]) * 0.5
    nx, ny, nz = -gx * strength, -gy * strength, np.ones_like(h)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    out = np.dstack([nx / ln, ny / ln, nz / ln]) * 0.5 + 0.5
    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8), "RGB")


# ==========================================================================
# UV helpers
# ==========================================================================
_QUAD_LOCAL = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def rot_local(loc, k):
    c = loc - 0.5
    for _ in range(k % 4):
        c = np.column_stack([-c[:, 1], c[:, 0]])
    return c + 0.5


def remap_local(local, ti, m=0.03):
    col, row = ti % 4, ti // 4
    u0, u1 = (col + m) / 4.0, (col + 1 - m) / 4.0
    v0, v1 = (row + m) / 4.0, (row + 1 - m) / 4.0
    uu = u0 + local[:, 0] * (u1 - u0)
    vv = v0 + local[:, 1] * (v1 - v0)
    return np.column_stack([uu, vv])


def shade(verts, kind, rng, stride, htot):
    """Per-vertex COLOR_0 sun/shade tint (multiplies the texture in glTF)."""
    y = verts[:, 1]
    h = np.clip(y / max(htot, 1e-6), 0.0, 1.0)
    if stride > 1:
        m = len(verts) // stride
        hc = h[:m * stride].reshape(m, stride).mean(axis=1)
        hc = np.clip(hc + rng.uniform(-0.06, 0.06, m), 0.0, 1.0)
        h = np.repeat(hc, stride)
    base = 0.74 + 0.32 * h
    if kind == "petals":
        r = base * (1.0 + 0.06 * (h - 0.5))
        g = base * (1.0 - 0.05 * (h - 0.5))
        b = base * (1.0 - 0.02 * (h - 0.5))
    else:  # cushion / leaves
        r = base * (1.0 + 0.05 * (h - 0.5))
        g = base
        b = base * (1.0 - 0.05 * (h - 0.5))
    rgb = np.clip(np.column_stack([r, g, b]) * 255.0, 0, 255).astype(np.uint8)
    alpha = np.full((len(rgb), 1), 255, np.uint8)
    return np.concatenate([rgb, alpha], axis=1)


# ==========================================================================
# Assembly
# ==========================================================================
CUSH_REP_U = 6.0
CUSH_REP_V = 3.0


def texture_scene(scene, rgb, hsv, seed):
    rng = np.random.default_rng(seed)

    pink, green = extract_palette(rgb, hsv)
    pale = np.clip(0.42 * pink + 0.58 * np.array([255, 236, 242]), 0, 255)
    warm = np.clip(pink * np.array([1.06, 0.84, 0.92]) + np.array([14, 0, 8]), 0, 255)

    foliage_img = make_foliage_atlas(green, rng)
    petal_img = make_petal_atlas(pink, pale, warm, rng)
    cushion_img = make_cushion_albedo(pick_green_patch(rgb), green, rng)
    cushion_nrm = make_normal(cushion_img, 1.4)

    htot = float(scene.geometry["cushion"].vertices[:, 1].max())

    # ---- cushion : cylindrical UV (no vertical streaking) + normal map ---
    mesh = scene.geometry["cushion"]
    v = mesh.vertices
    ang = np.arctan2(v[:, 2], v[:, 0])
    uv = np.column_stack([(ang / (2 * np.pi) + 0.5) * CUSH_REP_U,
                          (v[:, 1] / max(htot, 1e-6)) * CUSH_REP_V])
    mat = PBRMaterial(name="cushion", baseColorFactor=[1, 1, 1, 1],
                      baseColorTexture=cushion_img, normalTexture=cushion_nrm,
                      metallicFactor=0.0, roughnessFactor=0.85,
                      alphaMode="OPAQUE", doubleSided=False)
    mesh.visual = TextureVisuals(uv=uv, material=mat)
    mesh.visual.vertex_attributes["color"] = shade(v, "cushion", rng, 1, htot)

    # ---- leaves : per-card atlas tile + random 90deg rotation ------------
    if "leaves" in scene.geometry:
        mesh = scene.geometry["leaves"]
        v = mesh.vertices
        nc = len(v) // 4
        uv = np.zeros((len(v), 2))
        tiles = rng.integers(0, 16, nc)
        rots = rng.integers(0, 4, nc)
        rq = {k: rot_local(_QUAD_LOCAL, k) for k in range(4)}
        for k in range(nc):
            uv[4 * k:4 * k + 4] = remap_local(rq[int(rots[k])], int(tiles[k]))
        mat = PBRMaterial(name="leaves", baseColorFactor=[1, 1, 1, 1],
                          baseColorTexture=foliage_img, metallicFactor=0.0,
                          roughnessFactor=0.8, alphaMode="MASK",
                          alphaCutoff=0.45, doubleSided=True)
        mesh.visual = TextureVisuals(uv=uv, material=mat)
        mesh.visual.vertex_attributes["color"] = shade(v, "leaves", rng, 4, htot)

    # ---- petals : per-flower onto a petal atlas tile --------------------
    if "petals" in scene.geometry:
        mesh = scene.geometry["petals"]
        v = mesh.vertices
        loc = np.asarray(mesh.metadata["uv_local"], float)
        S = len(loc)
        n = len(v) // S
        uv = np.zeros((len(v), 2))
        tiles = rng.integers(0, 16, n)
        for k in range(n):
            uv[k * S:(k + 1) * S] = remap_local(loc, int(tiles[k]))
        mat = PBRMaterial(name="petals", baseColorFactor=[1, 1, 1, 1],
                          baseColorTexture=petal_img, metallicFactor=0.0,
                          roughnessFactor=0.55, alphaMode="MASK",
                          alphaCutoff=0.45, doubleSided=True)
        mesh.visual = TextureVisuals(uv=uv, material=mat)
        mesh.visual.vertex_attributes["color"] = shade(v, "petals", rng, S, htot)

    return scene


def main():
    ap = argparse.ArgumentParser(description="Cushion-plant procedural GLB generator")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        img = Image.open(args.image).convert("RGB")
        rgb = np.asarray(img)
        hsv = np.asarray(img.convert("HSV"))

        scene = build_mesh(args.seed, args.density)
        scene = texture_scene(scene, rgb, hsv, args.seed)

        data = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(data)

        tris = sum(int(g.faces.shape[0]) for g in scene.geometry.values())
        print("wrote %s  surfaces=%s  tris=%d"
              % (args.output, list(scene.geometry.keys()), tris))
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("ERROR: %s\n" % exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())