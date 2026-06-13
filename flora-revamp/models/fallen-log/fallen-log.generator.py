#!/usr/bin/env python3
"""
Weathered-driftwood asset: procedural geometry + photo-derived tileable wood
material + UVs by surface type, exported as a textured GLB.

CLI:
    python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb

Only numpy, trimesh, PIL (Pillow) and the Python stdlib are used.
+Y up, object rests on the XZ plane (min y == 0), units are metres,
deterministic given --seed.
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageFilter


# ==========================================================================
# GEOMETRY  (build_mesh)
# ==========================================================================
LENGTH = 1.06                 # m, tip-to-tip of the main body (slimmer/longer
RADIUS = 0.098                # m, nominal body radius -> raises front aspect)
WIDTH_OVER_HEIGHT = 2.6       # image bbox ratio (length+splay vs. max height)
LENGTH_OVER_DIAM = LENGTH / (2.0 * RADIUS)   # ≈ 5.4, a long-ish fragment
CROWN_FRAC = 0.30             # left-hand fraction occupied by the splay
LEFT_FLARE = 0.42             # body fattens strongly toward the left crown
RIGHT_FLARE = 0.03            # the right snapped break stays NARROW (clean)
MID_PINCH = 0.08              # gentle waisting through the middle
END_TAPER = 0.30             # overall thinning from fat left to narrow right


DENSITY = {
    "high": dict(N=56, L=112, crown=18, butt=6, pseg=5, prad=5, oct=4),
    "med":  dict(N=36, L=66,  crown=12, butt=4, pseg=4, prad=5, oct=3),
    "low":  dict(N=18, L=30,  crown=6,  butt=2, pseg=3, prad=4, oct=2),
}


def _wiggle(t, rng, n_terms, amp=1.0):
    """Smooth pseudo-random 1-D signal over t in [0,1], output ~[-amp, amp]."""
    out = np.zeros_like(t)
    total = 0.0
    for _ in range(n_terms):
        f = rng.uniform(0.7, 4.0)
        ph = rng.uniform(0.0, 2.0 * np.pi)
        a = rng.uniform(0.4, 1.0)
        out += a * np.sin(2.0 * np.pi * f * t + ph)
        total += a
    return amp * out / max(total, 1e-6)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _perp(d):
    ref = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    return _unit(np.cross(d, ref))


def _grain_profile(thetas, rng):
    """Per-angle radial multiplier: ridges + a few SHALLOW checking cracks.

    Cracks are kept shallow/wide so the body reads as a continuous worn
    surface, not a stack of planks.
    """
    ridge = np.zeros_like(thetas)
    total = 0.0
    for _ in range(5):
        freq = float(rng.integers(4, 16))
        ph = rng.uniform(0.0, 2.0 * np.pi)
        a = rng.uniform(0.3, 1.0)
        ridge += a * np.cos(freq * thetas + ph)
        total += a
    ridge /= max(total, 1e-6)
    prof = 1.0 + 0.075 * ridge

    n_crack = int(rng.integers(2, 4))
    for _ in range(n_crack):
        ca = rng.uniform(0.0, 2.0 * np.pi)
        d = np.angle(np.exp(1j * (thetas - ca)))
        width = rng.uniform(0.09, 0.16)
        depth = rng.uniform(0.07, 0.15)
        prof -= depth * np.exp(-(d / width) ** 2)
    return prof


def _radius_along(s):
    """Fat flared left crown, tapering to a narrow clean break on the right."""
    left = LEFT_FLARE * np.exp(-((s - 0.0) / 0.20) ** 2)
    right = RIGHT_FLARE * np.exp(-((s - 1.0) / 0.10) ** 2)
    taper = 1.0 - END_TAPER * s                       # thins toward the right
    waist = 1.0 - MID_PINCH * np.exp(-((s - 0.5) / 0.30) ** 2)
    return (taper + left + right) * waist


def _build_body(cfg, rng):
    N, L = cfg["N"], cfg["L"]
    s = np.linspace(0.0, 1.0, L + 1)
    thetas = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)

    cx = (s - 0.5) * LENGTH
    cy = _wiggle(s, rng, 3, amp=0.05) * RADIUS - 0.02 * RADIUS
    cz = _wiggle(s, rng, 3, amp=0.05) * RADIUS
    centre = np.stack([cx, cy, cz], axis=1)

    grain = _grain_profile(thetas, rng)
    lump = 1.0 + 0.05 * _wiggle(s, rng, 4, amp=1.0)
    base_r = RADIUS * _radius_along(s) * lump

    u = np.array([0.0, 1.0, 0.0])
    w = np.array([0.0, 0.0, 1.0])
    ring = np.cos(thetas)[:, None] * u + np.sin(thetas)[:, None] * w

    drift = 1.0 + 0.035 * _wiggle(s, rng, 5, amp=1.0)[:, None]
    r_full = (base_r[:, None] * grain[None, :] * drift)

    verts = centre[:, None, :] + r_full[:, :, None] * ring[None, :, :]
    verts = verts.reshape(-1, 3)

    i = np.arange(L)[:, None]
    j = np.arange(N)[None, :]
    a = i * N + j
    b = i * N + (j + 1) % N
    c = (i + 1) * N + (j + 1) % N
    d = (i + 1) * N + j
    faces = np.concatenate(
        [np.stack([a, b, c], -1).reshape(-1, 3),
         np.stack([a, c, d], -1).reshape(-1, 3)], axis=0)

    cl = len(verts)
    cr = cl + 1
    verts = np.vstack([verts, centre[0], centre[-1]])
    jj = np.arange(N)
    left_cap = np.stack([np.full(N, cl), (jj + 1) % N, jj], -1)
    base_r0 = (L) * N
    right_cap = np.stack([np.full(N, cr), base_r0 + jj, base_r0 + (jj + 1) % N], -1)
    faces = np.vstack([faces, left_cap, right_cap])

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return mesh, centre, base_r, grain, ring, thetas


def _build_prong(root, direction, length, base_rad, pseg, prad, rng,
                 flat=1.0, tippow=1.0, bend_amt=0.15):
    """A frayed splinter: tapered, slightly curled, flattened cross-section."""
    direction = _unit(direction)
    p1 = _perp(direction)
    p2 = _unit(np.cross(direction, p1))
    bend = (rng.uniform(-1, 1) * p1 + rng.uniform(-1, 1) * p2)
    bend = _unit(bend) * bend_amt * length
    twist = rng.uniform(-0.6, 0.6)

    f = np.linspace(0.0, 1.0, pseg + 1)
    centre = (root[None, :]
              + np.outer(f, direction * length)
              + np.outer(f ** 2, bend))
    radii = base_rad * (1.0 - f) ** tippow
    radii[0] = base_rad

    ang = np.linspace(0.0, 2.0 * np.pi, prad, endpoint=False)
    rings = []
    for k in range(pseg + 1):
        a = ang + twist * f[k]
        circ = np.cos(a)[:, None] * p1 + np.sin(a)[:, None] * p2 * flat
        rings.append(centre[k] + radii[k] * circ)
    verts = np.vstack(rings)

    faces = []
    for i in range(pseg):
        for j in range(prad):
            a0 = i * prad + j
            b0 = i * prad + (j + 1) % prad
            c0 = (i + 1) * prad + (j + 1) % prad
            d0 = (i + 1) * prad + j
            faces.append([a0, b0, c0])
            faces.append([a0, c0, d0])
    tip = len(verts)
    verts = np.vstack([verts, centre[-1] + direction * radii[-2] * 0.5])
    last = pseg * prad
    for j in range(prad):
        faces.append([tip, last + j, last + (j + 1) % prad])
    return verts, np.array(faces)


def _build_splinters(cfg, centre, base_r, grain, ring, thetas, rng):
    pseg, prad = cfg["pseg"], cfg["prad"]
    L = len(centre) - 1
    pieces = []

    def seat(idx, jang):
        c = centre[idx]
        r = base_r[idx] * grain[jang % len(thetas)]
        p = c + r * ring[jang % len(thetas)]
        out = _unit(p - c)
        return p, out

    # ---- left crown: a BROAD, DENSE fan of flat blunt splinters ----
    # damp the vertical (Y) spread so the crown reads as a wide frayed broom
    # rather than tall spikes (also keeps the front-view height down).
    i_root = max(1, int(0.05 * L))
    for _ in range(cfg["crown"]):
        jang = int(rng.integers(0, len(thetas)))
        root, out = seat(i_root, jang)
        axial = np.array([-1.0, 0.0, 0.0])
        jit = rng.normal(0, 0.12, 3)
        direction = 0.85 * out + 0.9 * axial + jit
        direction[1] *= 0.40                     # flatten the fan vertically
        direction = _unit(direction)
        length = rng.uniform(0.13, 0.24) * (LENGTH / 1.06)
        base_rad = rng.uniform(0.022, 0.040)
        v, f = _build_prong(root, direction, length, base_rad, pseg, prad, rng,
                            flat=0.4, tippow=0.95, bend_amt=0.12)
        pieces.append(trimesh.Trimesh(vertices=v, faces=f, process=False))

    # ---- right break: a few short stray fibres at the clean snapped end ----
    i_butt = min(L - 1, int(0.97 * L))
    for _ in range(cfg["butt"]):
        jang = int(rng.integers(0, len(thetas)))
        root, out = seat(i_butt, jang)
        axial = np.array([1.0, 0.0, 0.0])
        jit = rng.normal(0, 0.10, 3)
        direction = 0.4 * out + 0.95 * axial + jit
        direction[1] *= 0.5
        direction = _unit(direction)
        length = rng.uniform(0.04, 0.11) * (LENGTH / 1.06)
        base_rad = rng.uniform(0.009, 0.017)
        v, f = _build_prong(root, direction, length, base_rad, pseg, prad, rng,
                            flat=0.45, tippow=1.1, bend_amt=0.10)
        pieces.append(trimesh.Trimesh(vertices=v, faces=f, process=False))

    return trimesh.util.concatenate(pieces) if pieces else None


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    cfg = DENSITY.get(density, DENSITY["high"])

    body, centre, base_r, grain, ring, thetas = _build_body(cfg, rng)
    splinters = _build_splinters(cfg, centre, base_r, grain, ring, thetas, rng)

    meshes = [body] + ([splinters] if splinters is not None else [])
    allv = np.vstack([m.vertices for m in meshes])
    cx = 0.5 * (allv[:, 0].min() + allv[:, 0].max())
    cz = 0.5 * (allv[:, 2].min() + allv[:, 2].max())
    miny = allv[:, 1].min()
    shift = np.array([cx, miny, cz])
    for m in meshes:
        m.apply_translation(-shift)
        m.fix_normals()

    scene = trimesh.Scene()
    scene.add_geometry(body, geom_name="log")
    if splinters is not None:
        scene.add_geometry(splinters, geom_name="splinters")
    return scene


# ==========================================================================
# TEXTURING  (photo-derived, tileable, de-lit, BLEACHED-bright)
# ==========================================================================
# Probe centres chosen by LOOKING at reference.png: the pale wood body runs as
# a horizontal band across the middle.  All sit WELL INSIDE the silhouette.
_BODY_PROBES = [
    (0.50, 0.48), (0.58, 0.50), (0.66, 0.53), (0.45, 0.46),
    (0.62, 0.58), (0.54, 0.43), (0.72, 0.55), (0.40, 0.50),
    (0.22, 0.45), (0.30, 0.52),   # toward the frayed left crown
]
_SWATCH_BOX = (0.44, 0.45, 0.78, 0.62)   # clean central body crop
BLEACH_LUM = 0.66                        # target mean albedo value (pale)


def _load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def _lum(arr):
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def _sample_palette(arr):
    """Median color from several small in-silhouette patches; reject outliers."""
    H, W, _ = arr.shape
    half = max(2, int(0.02 * min(H, W)))
    meds = []
    for fx, fy in _BODY_PROBES:
        cx, cy = int(fx * W), int(fy * H)
        x0, x1 = max(0, cx - half), min(W, cx + half)
        y0, y1 = max(0, cy - half), min(H, cy + half)
        patch = arr[y0:y1, x0:x1].reshape(-1, 3)
        if patch.size:
            meds.append(np.median(patch, axis=0))
    meds = np.array(meds)
    glob = np.median(meds, axis=0)
    dist = np.linalg.norm(meds - glob[None, :], axis=1)
    keep = meds[dist < 0.22]
    if len(keep) < 3:
        keep = meds
    pal = np.clip(np.median(keep, axis=0), 0.0, 1.0)
    # lift toward a pale, sun-bleached value while keeping the sampled HUE
    pl = max(float(_lum(pal)), 1e-3)
    pal = np.clip(pal * min(0.62 / pl, 1.9), 0.0, 1.0)
    return pal


def _crop_swatch(arr):
    H, W, _ = arr.shape
    x0, y0, x1, y1 = _SWATCH_BOX
    crop = arr[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)].copy()
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        crop = arr[H // 3:2 * H // 3, W // 4:3 * W // 4].copy()
    # photo grain runs horizontally; rotate so it runs vertically (=length/V)
    return np.rot90(crop, k=1)


def _delight(arr):
    """Divide out a heavily blurred luminance; clamp gain to [0.6, 1.6]."""
    pim = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    rad = max(8, int(0.25 * min(pim.size)))
    blur = np.asarray(pim.filter(ImageFilter.GaussianBlur(rad)),
                      dtype=np.float32) / 255.0
    lum = np.maximum(_lum(blur), 1e-3)
    target = float(np.median(lum))
    gain = np.clip(target / lum, 0.6, 1.6)[..., None]
    return np.clip(arr * gain, 0.0, 1.0)


def _mirror_tile(arr, size):
    """Reflect-fold into a seamless tile, then resize to (size, size)."""
    top = np.concatenate([arr, arr[:, ::-1]], axis=1)
    full = np.concatenate([top, top[::-1, :]], axis=0)
    pim = Image.fromarray((np.clip(full, 0, 1) * 255).astype(np.uint8))
    return np.asarray(pim.resize((size, size), Image.LANCZOS),
                      dtype=np.float32) / 255.0


def _smooth1d(x, k):
    ker = np.ones(k) / k
    return np.convolve(np.concatenate([x[-k:], x, x[:k]]), ker, mode="same")[k:-k]


def _grain_overlay(size, rng):
    """Lengthwise (vertical) fibre striation + a few SHALLOW checking lines."""
    cols = _smooth1d(rng.standard_normal(size), 9)
    cols = _smooth1d(cols, 5)
    cols = cols / (np.abs(cols).max() + 1e-6)
    fac = 1.0 + 0.10 * cols
    overlay = np.repeat(fac[None, :], size, axis=0)
    # gentle vertical break-up so fibres aren't ruler-straight (no plank look)
    rows = 1.0 + 0.035 * _smooth1d(rng.standard_normal(size), 17)
    overlay *= rows[:, None]
    for _ in range(int(rng.integers(3, 6))):
        c = int(rng.integers(0, size))
        w = rng.uniform(1.5, 3.5)
        depth = rng.uniform(0.12, 0.22)
        xx = np.arange(size)
        d = np.minimum(np.abs(xx - c), size - np.abs(xx - c))
        overlay *= (1.0 - depth * np.exp(-(d / w) ** 2))[None, :]
    return overlay


def _build_wood_albedo(arr, palette, size, rng):
    """Tileable PALE bleached-wood albedo grounded in the photo's colors."""
    base = _mirror_tile(_delight(_crop_swatch(arr)), size)
    base_mean = base.reshape(-1, 3).mean(axis=0)
    base = np.clip(base * (palette / np.maximum(base_mean, 1e-3))[None, None, :],
                   0.0, 1.0)
    albedo = np.clip(base * _grain_overlay(size, rng)[..., None], 0.0, 1.0)
    # faint silvery-tan: keep near-neutral, a touch warm
    albedo[..., 2] *= 0.985
    # lift overall to a dry, sun-bleached value (only brighten, keep range)
    m = max(float(_lum(albedo).mean()), 1e-3)
    albedo = np.clip(albedo * np.clip(BLEACH_LUM / m, 1.0, 2.2), 0.0, 1.0)
    return albedo


def _normal_from_albedo(albedo, strength=1.0):
    """Tangent-space normal map: height = luminance (ridges high). Gentle."""
    lum = _lum(albedo).astype(np.float32)
    gx = np.zeros_like(lum)
    gy = np.zeros_like(lum)
    gx[:, 1:-1] = (lum[:, 2:] - lum[:, :-2]) * 0.5
    gy[1:-1, :] = (lum[2:, :] - lum[:-2, :]) * 0.5
    nx, ny = -gx * strength, -gy * strength
    nz = np.ones_like(lum)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    out = np.stack([nx / norm, ny / norm, nz / norm], axis=-1)
    return ((out * 0.5 + 0.5) * 255).astype(np.uint8)


def _to_pil(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


# --------------------------------------------------------------------------
# UVs + per-vertex tints
# --------------------------------------------------------------------------
def _cylindrical_uv(mesh, u_rep, v_rep):
    """Cylindrical UVs about the X axis; V follows the length (the grain)."""
    v = mesh.vertices
    cy = float(np.median(v[:, 1]))
    cz = float(np.median(v[:, 2]))
    ang = np.arctan2(v[:, 2] - cz, v[:, 1] - cy)
    u = (ang / (2.0 * np.pi) + 0.5) * u_rep
    xmin, xmax = float(v[:, 0].min()), float(v[:, 0].max())
    vv = (v[:, 0] - xmin) / max(xmax - xmin, 1e-6) * v_rep
    return np.stack([u, vv], axis=1).astype(np.float32)


def _vertex_tints(mesh, rng, tip_axis=None):
    """COLOR_0: sun-bleached tops bright, shaded undersides only mildly darker.

    Kept HIGH and tight so the asset stays pale (no muddy dark underside).
    """
    v = mesh.vertices
    ymin, ymax = float(v[:, 1].min()), float(v[:, 1].max())
    t = (v[:, 1] - ymin) / max(ymax - ymin, 1e-6)
    bright = 0.90 + 0.14 * t
    if tip_axis is not None:
        xmin, xmax = float(v[:, 0].min()), float(v[:, 0].max())
        xn = (v[:, 0] - xmin) / max(xmax - xmin, 1e-6)
        bright = bright + 0.10 * (xn if tip_axis > 0 else (1.0 - xn))
    bright = bright + rng.normal(0.0, 0.03, size=len(v))
    bright = np.clip(bright, 0.80, 1.15)
    base = np.array([1.0, 0.99, 0.96])             # faint warm silver
    rgb = np.clip(bright[:, None] * base[None, :], 0.0, 1.0)
    a = np.ones((len(v), 1))
    return (np.concatenate([rgb, a], axis=1) * 255).astype(np.uint8)


def _make_material(albedo_pil, normal_pil, rough=0.92):
    return trimesh.visual.material.PBRMaterial(
        name="weathered_wood",
        baseColorTexture=albedo_pil,
        normalTexture=normal_pil,
        metallicFactor=0.0,
        roughnessFactor=rough,
        baseColorFactor=[255, 255, 255, 255],
        doubleSided=False,
    )


# ==========================================================================
# ASSEMBLY
# ==========================================================================
def texture_scene(scene, image_path, seed):
    rng = np.random.default_rng(seed + 9173)
    arr = _load_image(image_path)
    palette = _sample_palette(arr)

    SIZE = 768
    albedo = _build_wood_albedo(arr, palette, SIZE, rng)
    albedo_pil = _to_pil(albedo)
    normal_pil = Image.fromarray(_normal_from_albedo(albedo))
    material = _make_material(albedo_pil, normal_pil, rough=0.92)

    for name, mesh in scene.geometry.items():
        if name == "splinters":
            uv = _cylindrical_uv(mesh, u_rep=1.5, v_rep=4.0)
            tints = _vertex_tints(mesh, rng, tip_axis=-1)
        else:  # "log" body
            v = mesh.vertices
            circ = 2.0 * np.pi * RADIUS
            length = float(v[:, 0].max() - v[:, 0].min())
            u_rep = max(1.0, circ / 0.40)
            v_rep = max(1.0, length / 0.40)
            uv = _cylindrical_uv(mesh, u_rep=u_rep, v_rep=v_rep)
            tints = _vertex_tints(mesh, rng)
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        mesh.visual.vertex_attributes["color"] = tints

    return scene


def main():
    ap = argparse.ArgumentParser(description="Procedural weathered driftwood -> GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        scene = build_mesh(args.seed, args.density)
        scene = texture_scene(scene, args.image, args.seed)
        glb = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(glb)
    except Exception as exc:  # noqa: BLE001
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1
    print("wrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())