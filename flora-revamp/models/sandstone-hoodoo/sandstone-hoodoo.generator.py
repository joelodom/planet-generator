"""
Procedural WEATHERED SANDSTONE HOODOO -- geometry + photo-derived material + GLB export.

A slender, top-heavy spire of layered sedimentary rock: a broad blocky foot, a
ribbed tapering neck of irregular stacked beds, and a thin, tilted, overhanging
mushroom capstone. Geometry is an asymmetric surface of revolution displaced
with coherent value noise. The single "rock" surface is textured from a tileable
sandstone albedo derived from the reference photo (de-lit, mirror-folded) with
procedural bedding laminations + cracks, a derived tangent-space normal map, and
per-vertex COLOR_0 tints.

CLI:
    python hoodoo.py --image PATH --seed INT --density {high,med,low} --output OUT.glb
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageFilter


# ===========================================================================
# GEOMETRY
# ===========================================================================
OVERALL_HEIGHT = 6.0                                   # meters, tip to ground
HEIGHT_OVER_WIDTH = 1.85                               # tall, slender spire
BASE_RADIUS = OVERALL_HEIGHT / (2.0 * HEIGHT_OVER_WIDTH)  # ~1.62 m half-width

BASE_TOP_FRAC = 0.40        # broad blocky base occupies the lower ~40%
CAP_BOTTOM_FRAC = 0.855     # thin overhanging capstone starts up high

# Normalised radius profile (peak == 1.0), scaled by BASE_RADIUS.
# Flatter / wider mushroom slab than before; neck pinches thin under it.
_T_CTRL = np.array([0.00, 0.10, 0.25, 0.38, 0.42, 0.55, 0.68,
                    0.80, 0.855, 0.875, 0.90, 0.945, 0.975, 1.00])
_R_CTRL = np.array([0.95, 1.00, 0.96, 0.66, 0.57, 0.50, 0.38,
                    0.28, 0.24, 0.72, 0.78, 0.74, 0.45, 0.14])


def _density_params(density):
    if density == "high":
        return dict(n_theta=120, n_rings=190, n_beds=11, n_noise=26, n_dents=2)
    if density == "med":
        return dict(n_theta=64, n_rings=110, n_beds=9, n_noise=16, n_dents=2)
    if density == "low":
        return dict(n_theta=30, n_rings=50, n_beds=6, n_noise=8, n_dents=1)
    raise ValueError("density must be 'high', 'med' or 'low'")


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _make_noise(rng, k):
    dirs = rng.normal(size=(k, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
    freqs = np.geomspace(0.55, 4.2, k)
    amps = (freqs[0] / freqs) ** 0.9
    phases = rng.uniform(0.0, 2.0 * np.pi, size=k)
    return dirs, freqs, phases, amps


def _sample_noise(points, noise):
    dirs, freqs, phases, amps = noise
    proj = points @ dirs.T
    vals = np.sin(proj * freqs + phases) * amps
    return vals.sum(axis=1) / amps.sum()


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    p = _density_params(density)
    T, R = p["n_theta"], p["n_rings"]

    r_ctrl = _R_CTRL * rng.uniform(0.93, 1.07, size=_R_CTRL.shape)
    H = OVERALL_HEIGHT * rng.uniform(0.92, 1.10)

    lean_dir = rng.uniform(0.0, 2.0 * np.pi)
    lean_mag = rng.uniform(0.22, 0.48)
    cap_dir = rng.uniform(0.0, 2.0 * np.pi)
    cap_mag = rng.uniform(0.18, 0.40)
    tilt_dir = rng.uniform(0.0, 2.0 * np.pi)
    tilt_mag = rng.uniform(0.12, 0.26)              # lateral shear -> tilted slab
    wob_ph = rng.uniform(0.0, 2.0 * np.pi)

    strata_ph = rng.uniform(0.0, 2.0 * np.pi)
    asym_ph = rng.uniform(0.0, 2.0 * np.pi, size=3)

    ledge_amp = 0.16 * BASE_RADIUS                  # projecting bed lip scale
    strata_amp = 0.014 * BASE_RADIUS                # subtle fine laminations
    asym_amp = 0.12

    t = np.linspace(0.0, 1.0, R)
    y = t * H

    lean = lean_mag * _smoothstep(0.10, 1.0, t)
    capw = _smoothstep(CAP_BOTTOM_FRAC - 0.03, CAP_BOTTOM_FRAC + 0.03, t)
    cap_shear = np.clip(t - CAP_BOTTOM_FRAC, 0.0, None) / (1.0 - CAP_BOTTOM_FRAC) * tilt_mag
    wobble = 0.06 * BASE_RADIUS * np.sin(t * 5.0 + wob_ph)
    cx = (lean * np.cos(lean_dir) + cap_mag * capw * np.cos(cap_dir)
          + cap_shear * np.cos(tilt_dir) + wobble * np.cos(lean_dir))
    cz = (lean * np.sin(lean_dir) + cap_mag * capw * np.sin(cap_dir)
          + cap_shear * np.sin(tilt_dir) + wobble * np.sin(lean_dir))

    theta = np.linspace(0.0, 2.0 * np.pi, T, endpoint=False)
    TH, TT = np.meshgrid(theta, t)

    r0 = np.interp(TT, _T_CTRL, r_ctrl) * BASE_RADIUS

    asym = (np.sin(2.0 * TH + asym_ph[0] + 2.0 * TT)
            + 0.6 * np.sin(3.0 * TH + asym_ph[1] - 1.5 * TT)
            + 0.4 * np.sin(5.0 * TH + asym_ph[2])) / 2.0
    r = r0 * (1.0 + asym_amp * asym)

    # --- IRREGULAR HORIZONTAL BEDS (not a periodic helix) -------------------
    nb = p["n_beds"]
    edges = np.sort(rng.uniform(0.18, 0.86, size=nb))   # bed-top heights (in t)
    bed_amp = (rng.uniform(0.0, 1.0, size=nb) ** 1.6) * ledge_amp  # many small
    bed_dir = rng.uniform(0.0, 2.0 * np.pi, size=nb)    # each lip favours a side
    idx = np.clip(np.searchsorted(edges, t, side="right"), 1, nb - 1)
    lo, hi = edges[idx - 1], edges[idx]
    frac = np.clip((t - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    shelf = frac ** 2.5                                  # sharp lip near bed top
    amp_i = (bed_amp[idx] * shelf)[:, None]
    cosfac = 0.5 + 0.5 * np.cos(TH - bed_dir[idx][:, None])
    azi = 0.35 + 0.65 * cosfac                           # one-sided shelf
    bed_zone = (_smoothstep(0.16, 0.30, t)
                * (1.0 - _smoothstep(0.84, 0.92, t)))[:, None]
    r = r + amp_i * azi * bed_zone

    # subtle fine laminations everywhere
    r = r + strata_amp * np.sin(TT * 60.0 + strata_ph + 0.3 * TH)
    r = np.maximum(r, 0.05)

    cx2, cz2 = cx[:, None], cz[:, None]
    X = cx2 + r * np.cos(TH)
    Y = np.broadcast_to(y[:, None], (R, T)).copy()
    Z = cz2 + r * np.sin(TH)
    ring_v = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    bottom_c = np.array([cx[0], 0.0, cz[0]])
    top_c = np.array([cx[-1], y[-1] + 0.05 * BASE_RADIUS, cz[-1]])
    verts = np.vstack([ring_v, bottom_c, top_c])
    n_ring = R * T
    bi, ti = n_ring, n_ring + 1

    faces = []
    for i in range(R - 1):
        a = i * T + np.arange(T)
        b = i * T + (np.arange(T) + 1) % T
        c = (i + 1) * T + np.arange(T)
        d = (i + 1) * T + (np.arange(T) + 1) % T
        faces.append(np.stack([a, c, b], axis=1))
        faces.append(np.stack([b, c, d], axis=1))
    j = np.arange(T)
    faces.append(np.stack([np.full(T, bi), (j + 1) % T, j], axis=1))
    faces.append(np.stack([np.full(T, ti), (R - 1) * T + j,
                           (R - 1) * T + (j + 1) % T], axis=1))
    faces = np.vstack(faces)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.fix_normals()

    normals = mesh.vertex_normals.copy()
    P = mesh.vertices.copy()
    noise = _make_noise(rng, p["n_noise"])

    relief = 0.055 * BASE_RADIUS
    local = 0.5 + 0.5 * np.linalg.norm(P[:, [0, 2]] - bottom_c[[0, 2]], axis=1) / BASE_RADIUS
    s = relief * local * _sample_noise(P * np.array([1.0, 0.6, 1.0]), noise)

    for _ in range(p["n_dents"]):
        dt = rng.uniform(0.08, 0.34)
        dth = rng.uniform(0.0, 2.0 * np.pi)
        rr = np.interp(dt, _T_CTRL, r_ctrl) * BASE_RADIUS
        c = np.array([rr * np.cos(dth), dt * H, rr * np.sin(dth)])
        depth = rng.uniform(0.22, 0.42)
        sig = rng.uniform(0.35, 0.55)
        d2 = np.sum((P - c) ** 2, axis=1)
        s -= depth * np.exp(-d2 / (2.0 * sig * sig))

    new = P + normals * s[:, None]
    new[bi] = bottom_c
    new[ti] = top_c
    new[:T, 1] = 0.0
    mesh.vertices = new

    mesh.vertices[:, 0] -= bottom_c[0]
    mesh.vertices[:, 2] -= bottom_c[2]
    mesh.vertices[:, 1] -= mesh.vertices[:, 1].min()
    mesh.fix_normals()

    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="rock")
    return scene


# ===========================================================================
# TEXTURING
# ===========================================================================
def _body_mask(arr):
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return (r - b > 12.0) & (r > 70.0) & (r < 252.0) & (r - g > -4.0)


def _sample_palette(arr, mask):
    body = arr[mask]
    if body.shape[0] < 200:
        mid = np.array([182.0, 112.0, 72.0])
        return mid, mid * 0.66, np.array([226.0, 176.0, 132.0])
    mid = np.median(body, axis=0)
    dark = np.percentile(body, 22, axis=0)
    light = np.percentile(body, 90, axis=0)
    return mid, dark, light


def _delight(crop):
    lum = 0.299 * crop[..., 0] + 0.587 * crop[..., 1] + 0.114 * crop[..., 2]
    radius = max(4.0, max(crop.shape[:2]) / 6.0)
    blur = np.asarray(
        Image.fromarray(lum.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius))
    ).astype(np.float64)
    gain = np.clip(lum.mean() / (blur + 1e-3), 0.6, 1.6)
    return np.clip(crop * gain[..., None], 0, 255)


def _photo_tile(arr, mask, palette, size):
    mid = palette[0]
    rows = mask.sum(axis=1)
    if rows.max() < 20:
        return np.ones((size, size, 3)) * mid
    yb = int(np.argmax(rows))
    cols = np.where(mask[yb])[0]
    xc = int((cols[0] + cols[-1]) / 2)
    w = cols[-1] - cols[0]
    half = max(8, int(min(w * 0.5, arr.shape[0] * 0.25, arr.shape[1] * 0.4) / 2))
    y0, y1 = max(0, yb - half), min(arr.shape[0], yb + half)
    x0, x1 = max(0, xc - half), min(arr.shape[1], xc + half)
    crop = arr[y0:y1, x0:x1].copy()
    cmask = mask[y0:y1, x0:x1]
    crop[~cmask] = mid
    crop = _delight(crop)

    sq = Image.fromarray(crop.astype(np.uint8)).resize((size // 2, size // 2), Image.LANCZOS)
    sq = sq.filter(ImageFilter.GaussianBlur(0.8))          # soften repeat blotches
    a = np.asarray(sq).astype(np.float64)
    m = a.mean(axis=(0, 1), keepdims=True)
    a = m + (a - m) * 0.72                                 # tame contrast
    top = np.concatenate([a, a[:, ::-1]], axis=1)          # mirror-fold seamless
    full = np.concatenate([top, top[::-1]], axis=0)
    return full


def _tile_noise(size, rng, max_freq, n_terms):
    c = (np.arange(size) + 0.5) / size
    U, V = np.meshgrid(c, c)
    out = np.zeros((size, size))
    tot = 0.0
    for _ in range(n_terms):
        fx = int(rng.integers(1, max_freq + 1))
        fy = int(rng.integers(1, max_freq + 1))
        ph = rng.uniform(0.0, 2.0 * np.pi)
        amp = 1.0 / np.hypot(fx, fy)
        out += amp * np.sin(2.0 * np.pi * (fx * U + fy * V) + ph)
        tot += amp
    return out / max(tot, 1e-6)


def _build_albedo_and_normal(arr, mask, seed, size=1024):
    rng = np.random.default_rng(seed * 7919 + 13)
    mid, dark, light = _sample_palette(arr, mask)

    base = _photo_tile(arr, mask, (mid, dark, light), size)
    c = (np.arange(size) + 0.5) / size
    _, V = np.meshgrid(c, c)

    mottle = _tile_noise(size, rng, 6, 10)
    colmix = (light[None, None, :] * (0.5 + 0.5 * mottle)[..., None]
              + dark[None, None, :] * (0.5 - 0.5 * mottle)[..., None])
    alb = 0.78 * base * (1.0 + 0.12 * mottle)[..., None] + 0.22 * colmix

    # wavy horizontal bedding -> V is world height under triplanar
    warp = 0.06 * _tile_noise(size, rng, 4, 6)
    bands = np.sin(2.0 * np.pi * 6.0 * (V + warp))
    strata = 0.90 + 0.10 * (0.5 + 0.5 * bands)
    parting = _smoothstep(0.94, 0.99, np.abs(bands))
    strata *= (1.0 - 0.14 * parting)
    alb *= strata[..., None]

    # pale buff / sun-bleached streaks toward the sampled light tone
    pale = _smoothstep(0.55, 0.92, bands) * (0.6 + 0.4 * (0.5 + 0.5 * mottle))
    alb = alb * (1.0 - 0.32 * pale)[..., None] + (light * 1.04)[None, None, :] * (0.32 * pale)[..., None]

    # sparse cracks / veins
    ridge = 1.0 - np.abs(_tile_noise(size, rng, 9, 12))
    crack = _smoothstep(0.93, 0.99, ridge)
    alb *= (1.0 - 0.28 * crack)[..., None]

    # fine grit + overall lift toward sunlit sandstone
    grit = _tile_noise(size, rng, 20, 10)
    alb *= (1.0 + 0.04 * grit)[..., None]
    alb = np.clip(alb * 1.10, 0, 255).astype(np.uint8)

    # ---- tangent-space normal from inverse-luminance height ----------------
    g = (0.299 * alb[..., 0] + 0.587 * alb[..., 1] + 0.114 * alb[..., 2]) / 255.0
    height = 1.0 - g
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = kx.T

    def _conv(im, k):
        acc = np.zeros_like(im)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc += k[dy + 1, dx + 1] * np.roll(np.roll(im, dy, 0), dx, 1)
        return acc

    strength = 1.5
    nx = -_conv(height, kx) * strength
    ny = -_conv(height, ky) * strength
    nz = np.ones_like(height)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    nmap = np.stack([nx / norm, ny / norm, nz / norm], axis=-1)
    nmap = ((nmap * 0.5 + 0.5) * 255.0).astype(np.uint8)

    return Image.fromarray(alb, "RGB"), Image.fromarray(nmap, "RGB"), (mid, dark, light)


def _triplanar_uv(mesh, tile=2.0):
    P = mesh.vertices
    n = mesh.vertex_normals
    ax = np.argmax(np.abs(n), axis=1)
    u = np.where(ax == 0, P[:, 2], P[:, 0])
    v = np.where(ax == 1, P[:, 2], P[:, 1])
    return np.stack([u / tile, v / tile], axis=1)


def _vertex_tints(mesh, seed):
    rng = np.random.default_rng(seed * 2654435761 % (2**32) + 5)
    P = mesh.vertices
    maxy = max(P[:, 1].max(), 1e-6)
    hf = np.clip(P[:, 1] / maxy, 0.0, 1.0)

    noise = _make_noise(rng, 18)
    nz = np.clip(_sample_noise(P * 0.8, noise), -1.0, 1.0)
    ground = _smoothstep(0.0, 0.12, hf)
    cap = _smoothstep(0.84, 0.96, hf)

    b = np.ones_like(hf)
    b *= (0.90 + 0.10 * hf)                 # crown a touch brighter
    b *= (0.88 + 0.12 * ground)             # gentle near-ground AO
    b *= (0.94 + 0.06 * (0.5 + 0.5 * nz))   # mottle
    b *= (0.98 + 0.02 * cap)

    warm = 0.5 + 0.5 * nz
    r = b * 1.0
    g = b * (0.97 - 0.015 * warm)
    bl = b * (0.91 - 0.04 * warm)

    col = np.clip(np.stack([r, g, bl], axis=1), 0.0, 1.0)
    out = np.empty((len(P), 4), dtype=np.uint8)
    out[:, :3] = (col * 255.0).astype(np.uint8)
    out[:, 3] = 255
    return out


def texture_scene(scene, image_path, seed):
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img).astype(np.float64)
    mask = _body_mask(arr)

    albedo, normal, _ = _build_albedo_and_normal(arr, mask, seed, size=1024)

    mesh = list(scene.geometry.values())[0]
    uv = _triplanar_uv(mesh, tile=2.0)

    material = trimesh.visual.material.PBRMaterial(
        name="sandstone",
        baseColorTexture=albedo,
        normalTexture=normal,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0,
        roughnessFactor=0.95,
        doubleSided=False,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    mesh.visual.vertex_attributes["color"] = _vertex_tints(mesh, seed)

    out = trimesh.Scene()
    out.add_geometry(mesh, geom_name="rock")
    return out


# ===========================================================================
# CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Procedural sandstone hoodoo -> textured GLB")
    ap.add_argument("--image", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    try:
        scene = build_mesh(args.seed, args.density)
        scene = texture_scene(scene, args.image, args.seed)
        scene.export(args.output)
    except Exception as exc:                          # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())