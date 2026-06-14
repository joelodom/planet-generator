"""
grey-slate-spire : procedural geometry + photo-derived materials -> textured GLB

A tall, upright shard of foliated metamorphic stone (slate / schist) that
narrows from a broad base to a single sharp summit, its mass built from a few
large, flat cleaved facets rather than a fuzzy crumble. Materials are derived
from a reference photo (de-lit, mirror-tiled swatch + procedural
fracture/striation/crevice detail) plus a Sobel normal map. Triplanar UVs are
baked per-vertex.

CLI:
    python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb

Only numpy / trimesh / PIL / stdlib.
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageFilter
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial


# ===========================================================================
# GEOMETRY
# ===========================================================================

HEIGHT_OVER_WIDTH = 2.78    # slender wedge -> front aspect ~0.39 (matches photo)
DEPTH_RATIO       = 0.70    # flattened slab -> depth (Z) < width (X)
TAPER_POWER       = 1.0     # near-straight flanks, sharp tip
APEX_OFFSET       = 0.16    # apex shifted off the central axis (frac of base R)

SPIRE_HEIGHT = 1.6                                   # meters (standing-stone scale)
BASE_WIDTH   = SPIRE_HEIGHT / HEIGHT_OVER_WIDTH      # ~0.576 m
BASE_RADIUS  = 0.5 * BASE_WIDTH                      # ~0.288 m

# surface character: keep noise LOW so flat cleaved facets dominate
DISP_AMP   = 0.14   # gentle chipped grain (was fuzzy at 0.32)
CREV_AMP   = 0.07   # shallow crevices (was 0.18)
BASE_FLARE = 0.05   # almost no crumbly foot (was 0.18 -> dripping skirt)
RAD_SCALE  = 4.0    # noise cells around the ring (vertical fracture columns)
VERT_SCALE = 3.0    # noise cells up the height (< RAD_SCALE -> vertical streaks)


class _ValueNoise3D:
    def __init__(self, rng, grid=24):
        self.g = int(grid)
        self.vals = rng.standard_normal((self.g, self.g, self.g)).astype(np.float64)

    def sample(self, pts):
        g = self.g
        xi = np.floor(pts).astype(np.int64)
        xf = pts - xi
        u = xf * xf * (3.0 - 2.0 * xf)

        x0, y0, z0 = (xi[:, 0] % g), (xi[:, 1] % g), (xi[:, 2] % g)
        x1, y1, z1 = ((xi[:, 0] + 1) % g), ((xi[:, 1] + 1) % g), ((xi[:, 2] + 1) % g)
        v = self.vals

        c000 = v[x0, y0, z0]; c100 = v[x1, y0, z0]
        c010 = v[x0, y1, z0]; c110 = v[x1, y1, z0]
        c001 = v[x0, y0, z1]; c101 = v[x1, y0, z1]
        c011 = v[x0, y1, z1]; c111 = v[x1, y1, z1]

        ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
        x00 = c000 + (c100 - c000) * ux
        x10 = c010 + (c110 - c010) * ux
        x01 = c001 + (c101 - c001) * ux
        x11 = c011 + (c111 - c011) * ux
        y0_ = x00 + (x10 - x00) * uy
        y1_ = x01 + (x11 - x01) * uy
        return y0_ + (y1_ - y0_) * uz


def _fbm(noise, pts, octaves, freq=1.0, gain=0.5, lac=2.1):
    total = np.zeros(len(pts))
    amp, f, norm = 1.0, float(freq), 0.0
    for o in range(int(octaves)):
        total += amp * noise.sample(pts * f + o * 13.37)
        norm += amp
        amp *= gain
        f *= lac
    return total / max(norm, 1e-9)


def _density_params(density):
    d = str(density).lower()
    if d == "low":
        return dict(n_rings=30,  n_radial=18, octaves=2, n_cuts=3, grid=14)
    if d == "med":
        return dict(n_rings=64,  n_radial=34, octaves=3, n_cuts=4, grid=20)
    return dict(n_rings=120, n_radial=60, octaves=4, n_cuts=6, grid=26)


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    P = _density_params(density)
    n_rings, n_radial = P["n_rings"], P["n_radial"]
    noise = _ValueNoise3D(rng, P["grid"])

    t = np.linspace(0.0, 0.985, n_rings)
    y = t * SPIRE_HEIGHT
    taper = (1.0 - t) ** TAPER_POWER
    taper = taper * (1.0 + BASE_FLARE * np.exp(-t * 6.0))

    col = rng.uniform(0.88, 1.12, n_radial)
    col = np.convolve(np.r_[col[-1], col, col[0]], [0.25, 0.5, 0.25], mode="valid")

    twist = np.cumsum(rng.normal(0.0, 0.025, n_rings))
    lean_dir = rng.uniform(0.0, 2.0 * np.pi)
    lean_mag = BASE_RADIUS * (APEX_OFFSET + rng.uniform(-0.03, 0.05))
    ramp = t ** 1.4
    axis_x = lean_mag * np.cos(lean_dir) * ramp
    axis_z = lean_mag * np.sin(lean_dir) * ramp

    theta = 2.0 * np.pi * np.arange(n_radial) / n_radial
    ang = theta[None, :] + twist[:, None]
    t2 = t[:, None] * np.ones((1, n_radial))

    pc = np.stack([np.cos(ang) * RAD_SCALE,
                   t2 * VERT_SCALE,
                   np.sin(ang) * RAD_SCALE], axis=-1).reshape(-1, 3)
    disp  = _fbm(noise, pc,             P["octaves"]).reshape(n_rings, n_radial)
    disp2 = _fbm(noise, pc * 2.0 + 7.0, P["octaves"]).reshape(n_rings, n_radial)

    rad = taper[:, None] * BASE_RADIUS * col[None, :]
    r = rad * (1.0 + DISP_AMP * disp) * (1.0 - CREV_AMP * np.abs(disp2))
    r = np.maximum(r, rad * 0.30)

    X = axis_x[:, None] + r * np.cos(ang)
    Z = axis_z[:, None] + r * np.sin(ang) * DEPTH_RATIO
    Y = y[:, None] * np.ones((1, n_radial))
    verts_side = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    # --- large planar facet cuts spread around the circumference ----------
    # These are the dominant shape driver: they cleave the noisy tube into a
    # cluster of big flat slab faces with crisp straight edges (= cleaved stone).
    cut_floor = 0.05 * SPIRE_HEIGHT
    n_cuts = P["n_cuts"]
    for k in range(n_cuts):
        az = 2.0 * np.pi * k / n_cuts + rng.uniform(-0.35, 0.35)
        tilt = rng.uniform(-0.18, 0.35)           # vertical tilt -> diagonal faces
        nrm = np.array([np.cos(az), tilt, np.sin(az)])
        nrm /= np.linalg.norm(nrm)
        midy = rng.uniform(0.20, 0.72) * SPIRE_HEIGHT
        off = rng.uniform(0.30, 0.50) * BASE_RADIUS
        anchor = np.array([
            lean_mag * np.cos(lean_dir) * (midy / SPIRE_HEIGHT) ** 1.4,
            midy,
            lean_mag * np.sin(lean_dir) * (midy / SPIRE_HEIGHT) ** 1.4])
        p0 = anchor + nrm * off
        d = (verts_side - p0) @ nrm
        m = (d > 0.0) & (verts_side[:, 1] > cut_floor)
        verts_side[m] -= np.outer(d[m], nrm)

    apex = np.array([lean_mag * np.cos(lean_dir) + rng.uniform(-1, 1) * 0.01 * BASE_RADIUS,
                     SPIRE_HEIGHT,
                     lean_mag * np.sin(lean_dir) + rng.uniform(-1, 1) * 0.01 * BASE_RADIUS])
    center = np.array([0.0, 0.0, 0.0])
    verts = np.vstack([verts_side, apex[None, :], center[None, :]])

    apex_idx = n_rings * n_radial
    center_idx = apex_idx + 1

    i = np.arange(n_rings - 1)[:, None]
    j = np.arange(n_radial)[None, :]
    jn = (j + 1) % n_radial
    a = i * n_radial + j
    b = i * n_radial + jn
    c = (i + 1) * n_radial + jn
    dd = (i + 1) * n_radial + j
    f_side = np.concatenate([
        np.stack([a, b, c], axis=-1).reshape(-1, 3),
        np.stack([a, c, dd], axis=-1).reshape(-1, 3),
    ], axis=0)

    jj = np.arange(n_radial)
    jjn = (jj + 1) % n_radial
    top = (n_rings - 1) * n_radial
    f_top = np.stack([top + jj, top + jjn, np.full(n_radial, apex_idx)], axis=-1)
    f_bot = np.stack([jj, np.full(n_radial, center_idx), jjn], axis=-1)

    faces = np.concatenate([f_side, f_top, f_bot], axis=0).astype(np.int64)

    base_idx = np.arange(n_radial)
    verts[:, 0] -= verts[base_idx, 0].mean()
    verts[:, 2] -= verts[base_idx, 2].mean()
    verts[:, 1] -= verts[:, 1].min()

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.fix_normals()

    base_rgb = np.array([140, 143, 144])
    vc = np.clip(base_rgb + rng.integers(-10, 11, (len(mesh.vertices), 3)), 0, 255)
    mesh.visual.vertex_colors = np.concatenate(
        [vc, np.full((len(mesh.vertices), 1), 255)], axis=1).astype(np.uint8)

    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="rock")
    return scene


# ===========================================================================
# TEXTURING  (photo-derived, deterministic, tileable)
# ===========================================================================

ROCK_RES = 1024
WORK_RES = 512
TEX_TILES_PER_M = 1.2     # larger texture features (was 1.7 -> too fine)
ALBEDO_TARGET_MEAN = 0.60  # silvery body brightness


def _to_pil(arr):
    a = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(a, "L" if a.ndim == 2 else "RGB")


def _from_pil(img):
    return np.asarray(img).astype(np.float64) / 255.0


def _smooth(t):
    return t * t * (3.0 - 2.0 * t)


def _vnoise2d(rng, res, cx, cy):
    cx, cy = max(1, int(round(cx))), max(1, int(round(cy)))
    grid = rng.random((cy, cx))
    xs = (np.arange(res) / res) * cx
    ys = (np.arange(res) / res) * cy
    x0 = np.floor(xs).astype(int) % cx; x1 = (x0 + 1) % cx; fx = _smooth(xs - np.floor(xs))
    y0 = np.floor(ys).astype(int) % cy; y1 = (y0 + 1) % cy; fy = _smooth(ys - np.floor(ys))
    g00 = grid[np.ix_(y0, x0)]; g01 = grid[np.ix_(y0, x1)]
    g10 = grid[np.ix_(y1, x0)]; g11 = grid[np.ix_(y1, x1)]
    fx2, fy2 = fx[None, :], fy[:, None]
    top = g00 * (1 - fx2) + g01 * fx2
    bot = g10 * (1 - fx2) + g11 * fx2
    return top * (1 - fy2) + bot * fy2


def _fbm2d(rng, res, octaves, cx, cy, gain=0.5, lac=2.0):
    out = np.zeros((res, res))
    amp, norm, ccx, ccy = 1.0, 0.0, float(cx), float(cy)
    for _ in range(int(octaves)):
        out += amp * _vnoise2d(rng, res, ccx, ccy)
        norm += amp
        amp *= gain
        ccx *= lac
        ccy *= lac
    return out / max(norm, 1e-9)


def _sample_palette(img):
    arr = np.asarray(img.convert("RGB")).astype(np.float64)
    H, W = arr.shape[:2]
    centers = [(0.47, 0.28), (0.44, 0.40), (0.50, 0.41), (0.42, 0.55),
               (0.52, 0.55), (0.46, 0.67), (0.55, 0.65), (0.48, 0.78), (0.41, 0.72)]
    hs = max(4, int(min(H, W) * 0.025))
    meds = []
    for fx, fy in centers:
        cx, cy = int(fx * W), int(fy * H)
        x0, x1 = max(0, cx - hs), min(W, cx + hs)
        y0, y1 = max(0, cy - hs), min(H, cy + hs)
        patch = arr[y0:y1, x0:x1].reshape(-1, 3)
        if patch.size:
            meds.append(np.median(patch, axis=0))
    meds = np.array(meds)
    glob = np.median(meds, axis=0)
    keep = meds[np.linalg.norm(meds - glob, axis=1) < 55.0]
    return np.median(keep, axis=0) if len(keep) else glob


def _crop_swatch(img):
    arr = np.asarray(img.convert("RGB")).astype(np.float64) / 255.0
    H, W = arr.shape[:2]
    x0, x1 = int(0.40 * W), int(0.58 * W)
    y0, y1 = int(0.55 * H), int(0.82 * H)
    crop = arr[y0:y1, x0:x1]
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        crop = arr[H // 3:2 * H // 3, W // 3:2 * W // 3]
    pil = _to_pil(crop).resize((WORK_RES, WORK_RES), Image.LANCZOS)
    return _from_pil(pil)


def _delight(rgb):
    lum = rgb.mean(axis=2)
    radius = max(4, WORK_RES // 6)
    blur = _from_pil(_to_pil(lum).filter(ImageFilter.GaussianBlur(radius=radius)))
    blur = np.clip(blur, 1e-3, None)
    gain = np.clip(blur.mean() / blur, 0.6, 1.6)
    return np.clip(rgb * gain[..., None], 0.0, 1.0)


def _mirror_tile(rgb):
    top = np.concatenate([rgb, rgb[:, ::-1, :]], axis=1)
    return np.concatenate([top, top[::-1, :, :]], axis=0)


def _build_rock_albedo(img, rng):
    body = _sample_palette(img) / 255.0
    swatch = _delight(_crop_swatch(img))
    base = _mirror_tile(swatch)
    R = base.shape[0]

    cur = base.reshape(-1, 3).mean(0)
    tint = np.clip(body / (cur + 1e-3), 0.5, 1.8)
    base = np.clip(base * tint[None, None, :], 0.0, 1.0)

    # --- procedural cleaved-stone detail (light touch; breaks mirror symmetry) ---
    mottle = _fbm2d(rng, R, 4, 5, 5)
    striae = _fbm2d(rng, R, 3, 40, 6)
    fine   = _fbm2d(rng, R, 2, 130, 5)
    crack1 = _fbm2d(rng, R, 4, 18, 6)
    crack2 = _fbm2d(rng, R, 4, 8, 11)

    base *= (0.95 + 0.10 * mottle)[..., None]
    base *= (0.96 + 0.08 * striae)[..., None]
    base *= (0.985 + 0.03 * fine)[..., None]

    dk1 = np.exp(-((crack1 - 0.5) ** 2) / (2.0 * 0.013 ** 2))
    dk2 = np.exp(-((crack2 - 0.5) ** 2) / (2.0 * 0.020 ** 2))
    dark = np.clip(dk1 + 0.6 * dk2, 0.0, 1.0)
    base *= (1.0 - 0.32 * dark)[..., None]                   # shallower crevices

    lum = base.mean(axis=2)
    shade = np.clip((0.38 - lum) / 0.38, 0.0, 1.0)
    cool = np.array([0.95, 1.00, 0.96])                      # gentle green-grey
    base = base * (1.0 + shade[..., None] * (cool - 1.0))

    hi = np.clip((lum - 0.58) / 0.42, 0.0, 1.0)
    base = base + hi[..., None] * 0.10                       # silvery highlights

    # lift to a light, silvery target; mild contrast; slight cool cast
    base = np.clip(base, 0.0, 1.0)
    m = base.mean()
    base *= np.clip(ALBEDO_TARGET_MEAN / (m + 1e-3), 0.85, 1.7)
    base = (base - 0.5) * 1.06 + 0.5
    base *= np.array([0.985, 1.0, 1.0])
    return np.clip(base, 0.0, 1.0)


def _albedo_to_normal(albedo, strength=2.0):
    lum = albedo.mean(axis=2)
    lum = _from_pil(_to_pil(lum).filter(ImageFilter.GaussianBlur(radius=1.0)))
    gy, gx = np.gradient(lum)
    nx, ny = -gx * strength, -gy * strength
    nz = np.ones_like(lum)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    normal = np.stack([nx * inv, ny * inv, nz * inv], axis=-1)
    return normal * 0.5 + 0.5


def _triplanar_uv(mesh, scale):
    v = mesh.vertices
    n = mesh.vertex_normals
    an = np.abs(n)
    uv = np.zeros((len(v), 2))
    domx = (an[:, 0] >= an[:, 1]) & (an[:, 0] >= an[:, 2])
    domy = (an[:, 1] > an[:, 0]) & (an[:, 1] >= an[:, 2])
    domz = ~(domx | domy)
    uv[domx, 0] = v[domx, 2]; uv[domx, 1] = v[domx, 1]
    uv[domy, 0] = v[domy, 0]; uv[domy, 1] = v[domy, 2]
    uv[domz, 0] = v[domz, 0]; uv[domz, 1] = v[domz, 1]
    return uv * scale


def _vertex_tints(mesh, rng):
    v = mesh.vertices
    n = mesh.vertex_normals
    h = v[:, 1]
    tn = h / max(h.max(), 1e-6)

    ao = 0.86 + 0.14 * np.clip(tn * 1.1, 0.0, 1.0)          # light AO near ground
    ao *= 1.0 - 0.12 * np.clip(-n[:, 1], 0.0, 1.0)          # gentle underside dark

    nz = _ValueNoise3D(rng, 16)
    mott = np.clip(0.5 + 0.4 * nz.sample(v * 4.0 + 1.3), 0.0, 1.0)

    low_c  = np.array([0.90, 0.94, 0.91])                   # cool foot
    high_c = np.array([1.00, 1.00, 1.00])                   # silvery crown
    tint = low_c[None, :] + (high_c - low_c)[None, :] * tn[:, None]

    col = ao[:, None] * tint * (0.94 + 0.10 * mott)[:, None]
    col = np.clip(col, 0.62, 1.0)
    rgba = np.concatenate([col, np.ones((len(v), 1))], axis=1)
    return (rgba * 255.0).astype(np.uint8)


def texture_scene(seed, density, image_path):
    rng = np.random.default_rng(seed)
    scene_in = build_mesh(seed, density)
    mesh = list(scene_in.geometry.values())[0]
    _ = mesh.vertex_normals

    img = Image.open(image_path).convert("RGB")

    albedo = _build_rock_albedo(img, rng)
    normal = _albedo_to_normal(albedo)
    albedo_img = _to_pil(albedo)
    normal_img = _to_pil(normal)

    uv = _triplanar_uv(mesh, TEX_TILES_PER_M)
    tints = _vertex_tints(mesh, rng)

    material = PBRMaterial(
        name="rock",
        baseColorTexture=albedo_img,
        normalTexture=normal_img,
        metallicFactor=0.0,
        roughnessFactor=0.95,
        doubleSided=False,
    )
    tv = TextureVisuals(uv=uv, material=material)
    tv.vertex_attributes["color"] = tints
    mesh.visual = tv

    scene_out = trimesh.Scene()
    scene_out.add_geometry(mesh, geom_name="rock")
    return scene_out


# ===========================================================================
# CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="Build a textured grey-slate-spire GLB.")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        scene = texture_scene(args.seed, args.density, args.image)
        glb = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(glb)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())