"""
Standalone procedural asset builder for a "weathered-tree-snag" archetype.

Pipeline:
  1. build_mesh(seed, density)  -> procedural geometry (tall sun-bleached snag)
  2. sample a tileable, de-lit WOOD material palette from the reference photo
  3. synthesize a crisp vertical-grain albedo + derived tangent-normal map
  4. apply cylindrical UVs (seam-fixed) and per-vertex sun/shade COLOR_0 tints
  5. export a textured binary .glb

CLI:
  python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageFilter


# ===========================================================================
# GEOMETRY  (build_mesh -- produces the "trunk")
# ===========================================================================
TOTAL_HEIGHT_M       = 3.0    # nominal overall height (incl. tallest splinter), meters
HEIGHT_OVER_WIDTH    = 4.2    # VERY slender: width/height ~0.24 in front view
BASE_FLARE_OVER_MID  = 2.2    # flared root collar ~2.2x the slim mid-shaft width
TOP_OVER_MID_RADIUS  = 0.66   # shaft keeps tapering toward the broken top
FLARE_ZONE           = 0.095  # flare confined to the bottom ~10% (short, abrupt)
SHAFT_TOP_FRAC       = 0.86   # solid shaft reaches ~86% of height; splinters above
SPIKE_MAX_FRAC       = 0.13   # tallest splinter rises ~13% of height above shaft top
TAPER_POW_NOMINAL    = 1.25   # taper curvature of the shaft

# measured radius fractions of total height (slim driftwood column)
R_BASE_FRAC = 0.100   # nominal widest (foot), before ridge multiplier
R_MID_FRAC  = 0.044   # slim mid-shaft
R_TOP_FRAC  = 0.030   # near the broken crown

_DENSITY = {
    "high": dict(n_levels=80, n_radial=88, n_harm=7, n_micro=4, n_knot=4, micro=True),
    "med":  dict(n_levels=42, n_radial=46, n_harm=5, n_micro=3, n_knot=3, micro=True),
    "low":  dict(n_levels=18, n_radial=22, n_harm=3, n_micro=0, n_knot=0, micro=False),
}


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(int(seed))
    if density not in _DENSITY:
        density = "high"
    P = _DENSITY[density]
    n_levels = int(P["n_levels"])
    nr = int(P["n_radial"])

    # ----- per-instance dimensions (slender; small abrupt foot) -----
    H = TOTAL_HEIGHT_M * rng.uniform(0.9, 1.12)
    r_base = R_BASE_FRAC * H * rng.uniform(0.92, 1.06)
    r_low = R_MID_FRAC * H * rng.uniform(0.9, 1.08)        # shaft radius above the flare
    top_radius = R_TOP_FRAC * H * rng.uniform(0.85, 1.1)   # radius near the break
    flare_peak = max(0.0, r_base - r_low)                  # extra radius at the very foot
    flare_zone = FLARE_ZONE * rng.uniform(0.85, 1.15)
    taper_pow = TAPER_POW_NOMINAL * rng.uniform(0.9, 1.15)
    shaft_top_y = H * SHAFT_TOP_FRAC
    spike_max = H * SPIKE_MAX_FRAC * rng.uniform(0.8, 1.2)

    # ----- vertical grain ridges/furrows (constant in theta -> run full height) -----
    freqs = np.sort(rng.choice(np.arange(4, 20), size=P["n_harm"], replace=False))
    amp_raw = rng.uniform(0.5, 1.0, P["n_harm"]) / (freqs / freqs.min())
    ridge_total = rng.uniform(0.05, 0.09)                  # subtle, keeps the slim profile
    amps = amp_raw / np.sum(amp_raw) * ridge_total
    phases = rng.uniform(0.0, 2.0 * np.pi, P["n_harm"])
    drifts = rng.uniform(-0.4, 0.4, P["n_harm"])

    # ----- root-buttress lobes confined to the very base -----
    n_b = int(rng.integers(3, 6))
    b_ang = rng.uniform(0.0, 2.0 * np.pi, n_b)
    b_amp = rng.uniform(0.06, 0.13, n_b)
    b_w = rng.uniform(0.22, 0.45, n_b)
    b_zone = rng.uniform(0.07, 0.13)

    # ----- knots / broken stubs scattered up the trunk -----
    n_k = int(P["n_knot"])
    if n_k > 0:
        k_th = rng.uniform(0.0, 2.0 * np.pi, n_k)
        k_t = rng.uniform(0.15, 0.80, n_k)
        k_a = rng.uniform(0.04, 0.10, n_k)
        k_wth = rng.uniform(0.15, 0.30, n_k)
        k_hth = rng.uniform(0.03, 0.07, n_k)

    # ----- fissured-heartwood micro noise -----
    if P["micro"]:
        nm = int(P["n_micro"])
        m_fth = rng.integers(2, 9, nm)
        m_ft = rng.uniform(0.5, 3.0, nm)
        m_ph = rng.uniform(0.0, 2.0 * np.pi, nm)
        m_a = rng.uniform(0.4, 1.0, nm)
        m_a = m_a / m_a.sum()
        micro_amp = rng.uniform(0.015, 0.035)

    # ----- gentle lean / axis wander (nearly vertical, like the photo) -----
    lean_dir = rng.uniform(0.0, 2.0 * np.pi)
    lean_amt = rng.uniform(0.010, 0.035) * H
    wf = rng.uniform(0.8, 1.6)
    wp = rng.uniform(0.0, 2.0 * np.pi)
    aw = rng.uniform(0.005, 0.015) * H

    def axis(t):
        s = np.sin(wf * np.pi * t + wp)
        cx = lean_amt * np.cos(lean_dir) * t ** 1.3 + aw * np.cos(lean_dir + 1.5) * s
        cz = lean_amt * np.sin(lean_dir) * t ** 1.3 + aw * np.sin(lean_dir + 1.5) * s
        return cx, cz

    cx0, cz0 = axis(0.0)

    # ----- splintered crown: a few tall spear-like slivers, low jagged rim between -----
    raw = rng.random(nr)
    s_top = raw ** rng.uniform(2.5, 4.0)
    n_sliver = max(2, nr // 12)
    sl_idx = rng.choice(nr, size=n_sliver, replace=False)
    s_top[sl_idx] = rng.uniform(0.7, 1.0, n_sliver)
    spikes = spike_max * s_top

    theta = 2.0 * np.pi * np.arange(nr) / nr
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    verts = np.zeros((n_levels * nr + 2, 3), dtype=np.float64)
    for k in range(n_levels):
        t = k / (n_levels - 1)
        ridge = 1.0 + np.sum(
            amps[:, None] * np.cos(freqs[:, None] * theta[None, :] + (phases + drifts * t)[:, None]),
            axis=0,
        )
        env = np.exp(-(t / b_zone) ** 2)
        butt = np.zeros(nr)
        for i in range(n_b):
            d = (theta - b_ang[i] + np.pi) % (2.0 * np.pi) - np.pi
            butt += b_amp[i] * np.exp(-(d / b_w[i]) ** 2)
        rm = ridge + butt * env
        if n_k > 0:
            for i in range(n_k):
                d = (theta - k_th[i] + np.pi) % (2.0 * np.pi) - np.pi
                rm += k_a[i] * np.exp(-(d / k_wth[i]) ** 2 - ((t - k_t[i]) / k_hth[i]) ** 2)
        if P["micro"]:
            mn = np.sum(
                m_a[:, None] * np.cos(m_fth[:, None] * theta[None, :]
                                      + 2.0 * np.pi * m_ft[:, None] * t + m_ph[:, None]),
                axis=0,
            )
            rm = rm * (1.0 + micro_amp * mn)
        R0 = (top_radius + (r_low - top_radius) * (1.0 - t) ** taper_pow
              + flare_peak * np.exp(-(t / flare_zone) ** 2))
        if t > 0.88:
            R0 = R0 * (1.0 - 0.18 * ((t - 0.88) / 0.12))
        r = R0 * rm
        cx, cz = axis(t)
        cx -= cx0
        cz -= cz0
        y = np.full(nr, shaft_top_y * t)
        if k == n_levels - 1:
            y = y + spikes
        base = k * nr
        verts[base:base + nr, 0] = cx + r * cos_t
        verts[base:base + nr, 1] = y
        verts[base:base + nr, 2] = cz + r * sin_t

    base_center_idx = n_levels * nr
    apex_idx = base_center_idx + 1
    verts[base_center_idx] = [0.0, 0.0, 0.0]
    cxt, czt = axis(1.0)
    verts[apex_idx] = [cxt - cx0, shaft_top_y + 0.3 * spike_max, czt - cz0]

    faces = []
    for k in range(n_levels - 1):
        base = k * nr
        nbase = (k + 1) * nr
        for j in range(nr):
            jn = (j + 1) % nr
            a = base + j
            b = base + jn
            c = nbase + jn
            d = nbase + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    for j in range(nr):
        faces.append((base_center_idx, (j + 1) % nr, j))
    top = (n_levels - 1) * nr
    for j in range(nr):
        faces.append((apex_idx, top + j, top + (j + 1) % nr))
    faces = np.asarray(faces, dtype=np.int64)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    b = mesh.bounds
    cx_mid = 0.5 * (b[0, 0] + b[1, 0])
    cz_mid = 0.5 * (b[0, 2] + b[1, 2])
    mesh.apply_translation([-cx_mid, -b[0, 1], -cz_mid])

    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="trunk")
    return scene


# ===========================================================================
# IMAGE / TEXTURE HELPERS
# ===========================================================================
def _blur2d(arr2d, radius):
    im = Image.fromarray(np.clip(arr2d * 255.0, 0, 255).astype(np.uint8))
    im = im.filter(ImageFilter.GaussianBlur(max(1, int(radius))))
    return np.asarray(im).astype(np.float64) / 255.0


def _resize_rgb(arr, size):
    im = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    im = im.resize((size, size), Image.LANCZOS)
    return np.asarray(im).astype(np.float64) / 255.0


def _center_square(img):
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


def _mirror_tile(img):
    """Mirror-fold a square patch into a seamless 2x2 reflection (tileable)."""
    top = np.concatenate([img, img[:, ::-1]], axis=1)
    full = np.concatenate([top, top[::-1, :]], axis=0)
    return full


def _lum(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def sample_wood(path, tex_size):
    """Sample a de-lit, brightness-normalized palette + tileable photo base
    from WELL INSIDE the snag (a narrow central column never hits the grey
    background). Result is forced to a bright, warm driftwood tone."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float64) / 255.0
    H, W = arr.shape[:2]

    x0, x1 = int(0.44 * W), int(0.56 * W)
    y0, y1 = int(0.18 * H), int(0.84 * H)
    strip = arr[y0:y1, x0:x1].copy()
    if strip.size == 0:
        strip = arr.copy()

    # de-light: divide by blurred luminance, clamp gain to [0.6, 1.6]
    lum = _lum(strip)
    rad = max(4, int(0.25 * min(strip.shape[0], strip.shape[1])))
    blur = _blur2d(lum, rad)
    target = float(np.median(lum))
    gain = np.clip(np.where(blur > 1e-4, target / np.maximum(blur, 1e-4), 1.0), 0.6, 1.6)
    delit = np.clip(strip * gain[..., None], 0.0, 1.0)

    # robust palette from the de-lit body pixels
    flat = delit.reshape(-1, 3)
    l2 = _lum(flat)
    p20, p80 = np.percentile(l2, [20, 80])
    dmask = flat[l2 <= p20]
    lmask = flat[l2 >= p80]
    mid = np.median(flat, axis=0)
    dark = dmask.mean(axis=0) if len(dmask) else mid * 0.7
    light = lmask.mean(axis=0) if len(lmask) else mid * 1.25

    if (_lum(light[None])[0] - _lum(dark[None])[0]) < 0.12:
        light = np.clip(mid * 1.30, 0, 1)
        dark = np.clip(mid * 0.68, 0, 1)

    # brightness-normalize to a bright sun-bleached driftwood value
    cur = float(_lum(mid[None])[0])
    scale = float(np.clip(0.66 / (cur + 1e-6), 0.8, 2.5))
    light = light * scale
    mid = mid * scale
    dark = dark * scale
    base = _resize_rgb(_mirror_tile(_center_square(delit)), tex_size) * scale

    # warm silvery undertone (kill the cool/blue cast)
    warm = np.array([1.04, 1.00, 0.94])
    light = np.clip(light * warm, 0, 1)
    mid = np.clip(mid * warm, 0, 1)
    dark = np.clip(dark * warm, 0, 1)
    base = np.clip(base * warm, 0, 1)

    return light, mid, dark, base


def make_wood_albedo(size, light, mid, dark, photo_base, rng):
    """Crisp, fully tileable, mostly-straight vertical-grain wood albedo."""
    lin = np.linspace(0.0, 1.0, size, endpoint=False)
    yy, xx = np.meshgrid(lin, lin, indexing="ij")

    # very gentle meander so grain reads as straight vertical checks, not water
    warp = (0.006 * np.sin(2 * np.pi * (3 * yy + rng.random()))
            + 0.004 * np.sin(2 * np.pi * (7 * yy + rng.random()))
            + 0.0025 * np.sin(2 * np.pi * (13 * yy + rng.random())))
    xw = xx + warp

    fur = np.zeros((size, size))
    for f in rng.choice(np.arange(6, 24), 4, replace=False):
        fur += (1.0 / np.sqrt(f)) * np.sin(2 * np.pi * int(f) * xw + 2 * np.pi * rng.random())
    fib = np.zeros((size, size))
    for f in rng.choice(np.arange(40, 130), 5, replace=False):
        fib += (1.0 / np.sqrt(f)) * np.sin(2 * np.pi * int(f) * xw + 2 * np.pi * rng.random())
    fur = (fur - fur.min()) / (np.ptp(fur) + 1e-9)
    fib = (fib - fib.min()) / (np.ptp(fib) + 1e-9)

    # mild vertical-biased blotches (avoid horizontal banding)
    blo = np.zeros((size, size))
    for _ in range(4):
        fx = int(rng.integers(1, 4))
        fy = int(rng.integers(1, 3))
        blo += np.sin(2 * np.pi * (fx * xx + fy * yy) + 2 * np.pi * rng.random())
    blo = (blo - blo.min()) / (np.ptp(blo) + 1e-9)

    t = np.clip(0.50 * fur + 0.35 * fib + 0.15 * blo, 0.0, 1.0)[..., None]
    col = np.where(t < 0.5,
                   dark + (mid - dark) * (t / 0.5),
                   mid + (light - mid) * ((t - 0.5) / 0.5))

    # deep, near-straight longitudinal cracks/checks
    crack = np.zeros((size, size))
    for _ in range(int(rng.integers(5, 9))):
        cx = rng.random()
        width = rng.uniform(0.003, 0.010)
        d = np.abs(((xw - cx + 0.5) % 1.0) - 0.5)
        crack = np.maximum(crack, np.exp(-(d / width) ** 2) * rng.uniform(0.6, 1.0))
    col = col * (1.0 - 0.55 * crack[..., None])

    col = 0.62 * col + 0.38 * photo_base
    return np.clip(col, 0.0, 1.0)


def make_normal(albedo, strength=3.0):
    """Tangent-space normal map from albedo luminance (Sobel, wrap edges)."""
    h = _lum(albedo)
    gx = np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)
    gy = np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(h)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    n = np.stack([nx / ln, ny / ln, nz / ln], axis=-1)
    return np.clip(n * 0.5 + 0.5, 0.0, 1.0)


# ===========================================================================
# UV + VERTEX COLOR
# ===========================================================================
def cylindrical_uv(V, v_tiles):
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    ang = np.arctan2(z, x)
    u = (ang / (2.0 * np.pi)) + 0.5
    ymin, ymax = y.min(), y.max()
    v = (y - ymin) / (ymax - ymin + 1e-9) * v_tiles
    return np.stack([u, v], axis=1)


def vertex_tints(V, N, rng):
    """Subtle sun/shade COLOR_0 (kept near white so it tints, not darkens)."""
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    ymin, ymax = y.min(), y.max()
    hf = (y - ymin) / (ymax - ymin + 1e-9)

    r = np.sqrt(x * x + z * z)
    nb = 24
    bidx = np.clip(((y - ymin) / (ymax - ymin + 1e-9) * nb).astype(int), 0, nb - 1)
    counts = np.bincount(bidx, minlength=nb).astype(np.float64)
    sums = np.bincount(bidx, weights=r, minlength=nb)
    mean_r = sums / np.maximum(counts, 1.0)
    dev = r - mean_r[bidx]
    dev_norm = np.clip(dev / (np.std(dev) + 1e-6), -1.0, 1.0)

    ny = N[:, 1]
    jitter = rng.uniform(-0.03, 0.03, len(V))
    val = 0.86 + 0.12 * hf + 0.05 * np.clip(ny, 0, 1) + 0.08 * dev_norm + jitter
    val = np.clip(val, 0.72, 1.0)

    rgb = val[:, None] * np.array([1.0, 0.98, 0.95])
    out = np.empty((len(V), 4), dtype=np.uint8)
    out[:, :3] = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    out[:, 3] = 255
    return out


def fix_cyl_seam(V, F, uv, col):
    """Duplicate the wrap-seam vertices so cylindrical UVs don't smear."""
    uorig = uv[:, 0]
    newV = [v for v in V]
    newUV = [u for u in uv]
    newC = [c for c in col]
    F = F.copy()
    dup = {}
    for fi in range(len(F)):
        tri = F[fi]
        us = uorig[tri]
        if us.max() - us.min() > 0.5:
            for k in range(3):
                vi = int(tri[k])
                if uorig[vi] < 0.5:
                    if vi not in dup:
                        dup[vi] = len(newV)
                        newV.append(V[vi])
                        nu = uv[vi].copy()
                        nu[0] = nu[0] + 1.0
                        newUV.append(nu)
                        newC.append(col[vi])
                    F[fi, k] = dup[vi]
    return (np.asarray(newV, dtype=np.float64), F,
            np.asarray(newUV, dtype=np.float64), np.asarray(newC, dtype=np.uint8))


# ===========================================================================
# ASSEMBLY
# ===========================================================================
def build_textured_scene(image_path, seed, density):
    tex_size = 1024 if density == "high" else (768 if density == "med" else 512)
    rng = np.random.default_rng(int(seed) ^ 0x5DEECE)

    geo_scene = build_mesh(seed, density)
    mesh = next(iter(geo_scene.geometry.values()))
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    N = np.asarray(mesh.vertex_normals, dtype=np.float64)

    light, mid, dark, photo_base = sample_wood(image_path, tex_size)
    albedo = make_wood_albedo(tex_size, light, mid, dark, photo_base, rng)
    normal = make_normal(albedo, strength=3.0)
    alb_img = Image.fromarray((albedo * 255.0).astype(np.uint8), mode="RGB")
    nrm_img = Image.fromarray((normal * 255.0).astype(np.uint8), mode="RGB")

    height = float(V[:, 1].max() - V[:, 1].min())
    mean_r = float(np.mean(np.sqrt(V[:, 0] ** 2 + V[:, 2] ** 2)))
    u_tiles = max(1, int(round((2.0 * np.pi * mean_r) / 0.6)))
    v_tiles = max(2, int(round(height / 0.6)))

    uv = cylindrical_uv(V, v_tiles)
    col = vertex_tints(V, N, rng)

    nV, nF, nUV, nC = fix_cyl_seam(V, F, uv, col)
    nUV[:, 0] = nUV[:, 0] * u_tiles

    out_mesh = trimesh.Trimesh(vertices=nV, faces=nF, process=False)
    out_mesh.fix_normals()

    mat = trimesh.visual.material.PBRMaterial(
        name="weathered_wood",
        baseColorTexture=alb_img,
        normalTexture=nrm_img,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.92,
    )
    out_mesh.visual = trimesh.visual.TextureVisuals(uv=nUV, material=mat, image=alb_img)
    out_mesh.visual.vertex_attributes["color"] = nC

    scene = trimesh.Scene()
    scene.add_geometry(out_mesh, geom_name="trunk")
    return scene


def main():
    ap = argparse.ArgumentParser(description="Procedural weathered-tree-snag GLB builder")
    ap.add_argument("--image", required=True, help="source reference photo")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        scene = build_textured_scene(args.image, args.seed, args.density)
        glb = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(glb)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("ERROR: {}\n".format(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())