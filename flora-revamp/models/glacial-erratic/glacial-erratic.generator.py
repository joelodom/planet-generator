#!/usr/bin/env python3
"""
Procedural weathered fieldstone boulder (granite) on a small snow mound:
geometry + photo-derived seamless materials + UVs -> textured GLB.

Fixes over the previous revision:
  * snow is now a SMALL, bright-white, low mound cupping the base
    (was a huge flat grey plate that dominated the silhouette)
  * granite albedo is SEAMLESS PROCEDURAL noise (color sampled from the
    photo) -- removes the mirror/kaleidoscope symmetry artifact
  * rounder dome: one gentle shoulder facet, minimal bottom flatten,
    height/width ~0.78 to match the photographed aspect (~1.28)

Only numpy, trimesh, PIL and the stdlib are used. Deterministic given --seed.
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageFilter, ImageDraw
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial


# ===========================================================================
# GEOMETRY
# ===========================================================================
BOULDER_WIDTH_M    = 1.40   # overall horizontal extent (X), a chunky fieldstone
HEIGHT_OVER_WIDTH  = 0.78   # squat dome; ~matches photo content aspect (1.28)
DEPTH_OVER_WIDTH   = 0.84   # slightly less deep (Z) than wide -> not a sphere

RX, RY, RZ = 0.70, 0.58, 0.62   # base ellipsoid radii (normalized)

NOISE_AMP          = 0.12   # water-worn lobes (gentle -> smooth fieldstone)
FINE_AMP           = 0.025
BASE_FREQ          = 1.70
ANISO              = 0.30   # mild asymmetry, not lopsided
RIDGE_AMP          = 0.04   # faint crown ridge
RIDGE_W            = 0.55
BOTTOM_FLATTEN_FRAC= 0.06   # only barely flatten the underside -> sides curl in

# Small snow mound that cups the base
SNOW_RADIUS  = 0.45 * BOULDER_WIDTH_M   # just a touch wider than the rock base
SNOW_HEIGHT  = 0.13                       # low mound (center height)
SNOW_OUTLINE = 0.12                       # irregular crystalline outline
ROCK_LIFT    = 0.0                        # rock base sits on the ground (y=0)

DENSITY = {
    "high": dict(rock_sub=4, snow_sub=3, octaves=5, facets=1, fine=True),
    "med":  dict(rock_sub=3, snow_sub=2, octaves=4, facets=1, fine=False),
    "low":  dict(rock_sub=2, snow_sub=1, octaves=3, facets=1, fine=False),
}


def _fbm(points, rng, octaves, base_freq, lacunarity=2.0, gain=0.5, waves=6):
    n = len(points)
    total = np.zeros(n, dtype=float)
    amp = 1.0
    freq = base_freq
    for _ in range(octaves):
        acc = np.zeros(n, dtype=float)
        for _w in range(waves):
            dvec = rng.normal(size=3)
            dvec /= (np.linalg.norm(dvec) + 1e-12)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            acc += np.sin(freq * (points @ dvec) + phase)
        total += amp * acc / np.sqrt(waves)
        amp *= gain
        freq *= lacunarity
    return total / (np.max(np.abs(total)) + 1e-12)


def _build_rock(rng, sub, octaves, n_facets, fine):
    ico = trimesh.creation.icosphere(subdivisions=sub, radius=1.0)
    dirs = np.array(ico.vertices, dtype=float)
    faces = np.array(ico.faces)
    rad = np.array([RX, RY, RZ])

    disp = NOISE_AMP * _fbm(dirs, rng, octaves=octaves, base_freq=BASE_FREQ)
    if fine:
        disp = disp + FINE_AMP * _fbm(dirs, rng, octaves=2,
                                      base_freq=BASE_FREQ * 4.0)

    aniso_axis = np.array([1.0, -0.25, 0.20])
    aniso_axis /= np.linalg.norm(aniso_axis)
    aniso = np.clip(1.0 + ANISO * (dirs @ aniso_axis), 0.55, 1.45)
    disp = disp * aniso

    perp = np.array([0.20, 0.0, 1.0]); perp /= np.linalg.norm(perp)
    top_weight = np.clip(dirs[:, 1], 0.0, 1.0) ** 2
    across = dirs @ perp
    ridge = RIDGE_AMP * top_weight * np.exp(-(across / RIDGE_W) ** 2)

    factor = 1.0 + disp + ridge
    P = dirs * rad * factor[:, None]

    # ONE gentle planar facet -> the "slightly flatter, angular" right shoulder
    for i in range(n_facets):
        if i == 0:
            nrm = np.array([1.0, -0.22, 0.20]); q = 0.90   # subtle, outer 10%
        else:
            nrm = np.array([-0.6, 0.15, -0.8]); q = 0.92
        nrm = nrm / np.linalg.norm(nrm)
        proj = P @ nrm
        d0 = np.quantile(proj, q)
        over = proj > d0
        P[over] -= (proj[over] - d0)[:, None] * nrm

    # barely flatten the underside so it rests (sides keep curling in)
    ylo = P[:, 1].min()
    H = P[:, 1].max() - ylo
    cut = ylo + BOTTOM_FLATTEN_FRAC * H
    P[P[:, 1] < cut, 1] = cut

    # enforce the measured photo proportions on the final extents
    ext = P.max(axis=0) - P.min(axis=0)
    P[:, 0] *= BOULDER_WIDTH_M / ext[0]
    P[:, 2] *= (BOULDER_WIDTH_M * DEPTH_OVER_WIDTH) / ext[2]
    P[:, 1] *= (BOULDER_WIDTH_M * HEIGHT_OVER_WIDTH) / ext[1]

    P[:, 0] -= 0.5 * (P[:, 0].max() + P[:, 0].min())
    P[:, 2] -= 0.5 * (P[:, 2].max() + P[:, 2].min())
    P[:, 1] -= P[:, 1].min()

    mesh = trimesh.Trimesh(vertices=P, faces=faces, process=False)
    mesh.fix_normals()
    return mesh


def _build_snow(rng, sub):
    """Low, irregular white mound: domed top, flat feathered ground contact."""
    ico = trimesh.creation.icosphere(subdivisions=sub, radius=1.0)
    d = np.array(ico.vertices, dtype=float)
    faces = np.array(ico.faces)

    radf = 1.0 + SNOW_OUTLINE * _fbm(d, rng, octaves=2, base_freq=2.4)

    P = d.copy()
    P[:, 0] *= SNOW_RADIUS * radf
    P[:, 2] *= SNOW_RADIUS * radf
    P[:, 1] *= SNOW_HEIGHT
    P[:, 1] = np.maximum(P[:, 1], 0.0)   # flat bottom, dome feathering to 0 at rim
    P[:, 1] -= P[:, 1].min()

    mesh = trimesh.Trimesh(vertices=P, faces=faces, process=False)
    mesh.fix_normals()
    return mesh


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    cfg = DENSITY.get(density, DENSITY["high"])
    rng = np.random.default_rng(seed)

    rock = _build_rock(rng, cfg["rock_sub"], cfg["octaves"],
                       cfg["facets"], cfg["fine"])
    snow = _build_snow(rng, cfg["snow_sub"])

    if ROCK_LIFT:
        rock.apply_translation([0.0, ROCK_LIFT, 0.0])

    scene = trimesh.Scene()
    scene.add_geometry(rock, geom_name="rock")
    scene.add_geometry(snow, geom_name="snow")
    return scene


# ===========================================================================
# TEXTURING
# ===========================================================================
ROCK_TEX = 1024
SNOW_TEX = 512
ROCK_UV_SCALE = 1.5      # texture repeats per meter (~2 tiles across boulder)
SNOW_UV_SCALE = 1.6

# sampling regions in NORMALIZED image coords, WELL INSIDE the granite body
ROCK_SAMPLES = [(0.42, 0.30), (0.52, 0.40), (0.62, 0.42), (0.38, 0.52),
                (0.55, 0.55), (0.48, 0.60), (0.50, 0.24), (0.66, 0.50)]
SNOW_SAMPLES = [(0.40, 0.86), (0.50, 0.88), (0.33, 0.85), (0.58, 0.85)]


def _gauss(arr2d, radius):
    a = np.clip(arr2d, 0.0, 255.0).astype(np.uint8)
    out = Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(out).astype(np.float32)


def _sample_patches(arr, centers, half=8):
    h, w = arr.shape[:2]
    out = []
    for (u, v) in centers:
        x, y = int(u * w), int(v * h)
        x0, x1 = max(0, x - half), min(w, x + half + 1)
        y0, y1 = max(0, y - half), min(h, y + half + 1)
        patch = arr[y0:y1, x0:x1].reshape(-1, 3)
        if len(patch):
            out.append(np.median(patch, axis=0))
    return np.array(out, dtype=np.float32)


def _robust_color(samples, tol=70.0, fallback=(100, 102, 106)):
    if len(samples) == 0:
        return np.array(fallback, dtype=np.float32)
    m = np.median(samples, axis=0)
    d = np.linalg.norm(samples - m, axis=1)
    keep = samples[d < tol]
    if len(keep) == 0:
        keep = samples
    return np.median(keep, axis=0).astype(np.float32)


def _tile_noise(res, rng, octaves, base_cells, waves=4, gain=0.5):
    """Seamless (period-1) value noise from integer-frequency sine waves.

    Tileable because all frequencies are integers; non-symmetric because of
    randomized diagonal directions, signs and phases. Range ~[-1, 1].
    """
    u = np.linspace(0.0, 1.0, res, endpoint=False, dtype=np.float32)
    U, V = np.meshgrid(u, u)
    field = np.zeros((res, res), dtype=np.float32)
    amp = 1.0
    cells = base_cells
    for _o in range(octaves):
        for _w in range(waves):
            fx = int(rng.integers(1, cells + 1))
            fy = int(rng.integers(1, cells + 1))
            sgn = 1.0 if rng.random() < 0.5 else -1.0
            ph = rng.uniform(0.0, 2.0 * np.pi)
            field += amp * np.sin(2.0 * np.pi * (fx * U + sgn * fy * V) + ph)
        amp *= gain
        cells *= 2
    return field / (np.max(np.abs(field)) + 1e-9)


def _normal_from_albedo(albedo, strength=2.2):
    h = albedo.mean(axis=2) / 255.0
    h = _gauss(h * 255.0, 0.6) / 255.0
    gy, gx = np.gradient(h)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(h)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    out = np.stack([(nx * inv * 0.5 + 0.5),
                    (ny * inv * 0.5 + 0.5),
                    (nz * inv * 0.5 + 0.5)], axis=2)
    return (out * 255.0).astype(np.uint8)


def _add_cracks(albedo, rng, n=5):
    """A few faint, wandering hairline cracks for character."""
    h, w = albedo.shape[:2]
    im = Image.fromarray(np.clip(albedo, 0, 255).astype(np.uint8)).convert("RGBA")
    draw = ImageDraw.Draw(im, "RGBA")
    for _ in range(n):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.uniform(0, 2 * np.pi)
        pts = [(x, y)]
        for _s in range(int(rng.integers(7, 15))):
            ang += rng.uniform(-0.5, 0.5)
            step = rng.uniform(0.02, 0.05) * w
            x += np.cos(ang) * step
            y += np.sin(ang) * step
            pts.append((x, y))
        a = int(rng.integers(40, 85))
        draw.line(pts, fill=(24, 24, 28, a), width=1)
    return np.asarray(im.convert("RGB")).astype(np.float32)


def build_rock_textures(arr, rng):
    """Seamless procedural granite; base tone sampled from the photo."""
    rock_color = _robust_color(_sample_patches(arr, ROCK_SAMPLES),
                               fallback=(100, 102, 106))
    res = ROCK_TEX
    base = np.ones((res, res, 3), np.float32) * rock_color

    # multi-scale mottling -> dove-grey patches through to charcoal
    big = _tile_noise(res, rng, octaves=2, base_cells=3)
    mid = _tile_noise(res, rng, octaves=3, base_cells=6)
    grain = _tile_noise(res, rng, octaves=2, base_cells=22)
    val = 0.16 * big + 0.10 * mid + 0.07 * grain
    alb = base * (1.0 + val)[..., None]

    # mineral flecks: scattered lighter (quartz) and darker (mica) speckles
    spk = _tile_noise(res, rng, octaves=1, base_cells=40)
    light = np.clip((spk - 0.45) * 3.0, 0.0, 1.0)
    dark = np.clip((-spk - 0.45) * 3.0, 0.0, 1.0)
    alb += (light * 42.0)[..., None]
    alb -= (dark * 30.0)[..., None]

    # faint lichen-like blotches (cool greenish grey)
    lich = _tile_noise(res, rng, octaves=2, base_cells=4)
    lmask = np.clip((lich - 0.55) * 3.0, 0.0, 1.0)[..., None]
    lichen_col = np.array([120, 128, 112], np.float32)
    alb = alb * (1.0 - 0.20 * lmask) + lichen_col[None, None, :] * (0.20 * lmask)

    alb = np.clip(alb, 0.0, 255.0)
    alb = _add_cracks(alb, rng, n=5)
    alb = np.clip(alb, 0, 255).astype(np.uint8)

    normal = _normal_from_albedo(alb.astype(np.float32))
    return Image.fromarray(alb, "RGB"), Image.fromarray(normal, "RGB")


def build_snow_texture(arr, rng):
    """Bright crystalline snow; base tone sampled then forced toward white."""
    snow_samples = _sample_patches(arr, SNOW_SAMPLES)
    if len(snow_samples):
        lum = snow_samples.mean(axis=1)
        snow_samples = snow_samples[lum > 150.0]
    snow_color = _robust_color(snow_samples, tol=60.0, fallback=(235, 238, 244))
    snow_color = np.maximum(snow_color, np.array([228, 231, 237], np.float32))

    res = SNOW_TEX
    base = np.ones((res, res, 3), np.float32) * snow_color
    spk = rng.normal(0.0, 1.0, (res, res)).astype(np.float32)
    spk = _gauss((spk * 0.5 + 0.5) * 255.0, 0.6) / 255.0 - 0.5
    base = base * (1.0 + 0.045 * spk)[..., None]          # crystalline sparkle
    drift = _tile_noise(res, rng, octaves=2, base_cells=6)
    base[..., 2] *= (1.0 + 0.02 * drift)                  # faint cool cast
    base = np.clip(base, 0, 255).astype(np.uint8)
    return Image.fromarray(base, "RGB")


# ---------------------------------------------------------------------------
# UVs and per-vertex tints
# ---------------------------------------------------------------------------
def triplanar_uv(vertices, normals, scale):
    n = np.abs(normals)
    ax = np.argmax(n, axis=1)
    uv = np.zeros((len(vertices), 2), dtype=np.float64)
    mx = ax == 0
    my = ax == 1
    mz = ax == 2
    uv[mx] = vertices[mx][:, [2, 1]]
    uv[my] = vertices[my][:, [0, 2]]
    uv[mz] = vertices[mz][:, [0, 1]]
    return uv * scale


def rock_vertex_colors(vertices, rng):
    """Silvery cooler crown -> darker charcoal lower flanks; subtle mottle."""
    y = vertices[:, 1]
    yn = (y - y.min()) / (np.ptp(y) + 1e-9)
    yn = yn * yn * (3.0 - 2.0 * yn)
    bright = 0.74 + 0.36 * yn
    mottle = 1.0 + np.clip(rng.normal(0.0, 0.03, len(y)), -0.08, 0.08)
    bright = bright * mottle
    r = bright * (1.0 - 0.04 * yn)
    g = bright
    b = bright * (1.0 + 0.05 * yn)
    rgb = np.clip(np.stack([r, g, b], axis=1) * 255.0, 0, 255)
    a = np.full((len(y), 1), 255.0)
    return np.concatenate([rgb, a], axis=1).astype(np.uint8)


def snow_vertex_colors(vertices, rng):
    """Bright near-white; only a touch of shading, faint blue cast."""
    y = vertices[:, 1]
    yn = (y - y.min()) / (np.ptp(y) + 1e-9)
    bright = 0.93 + 0.07 * yn
    mottle = 1.0 + np.clip(rng.normal(0.0, 0.015, len(y)), -0.04, 0.04)
    bright = bright * mottle
    r = bright
    g = bright
    b = np.minimum(bright * 1.02, 1.0)
    rgb = np.clip(np.stack([r, g, b], axis=1) * 255.0, 0, 255)
    a = np.full((len(y), 1), 255.0)
    return np.concatenate([rgb, a], axis=1).astype(np.uint8)


def texture_scene(scene, image_path, seed):
    arr = np.asarray(Image.open(image_path).convert("RGB")).astype(np.float32)
    rng = np.random.default_rng(seed + 1000)

    rock_albedo, rock_normal = build_rock_textures(arr, rng)
    snow_albedo = build_snow_texture(arr, rng)

    rock = scene.geometry["rock"]
    snow = scene.geometry["snow"]

    rock_uv = triplanar_uv(rock.vertices, rock.vertex_normals, ROCK_UV_SCALE)
    rock_mat = PBRMaterial(
        name="granite",
        baseColorTexture=rock_albedo,
        normalTexture=rock_normal,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.95,
    )
    rock.visual = TextureVisuals(uv=rock_uv, material=rock_mat)
    rock.visual.vertex_attributes["color"] = rock_vertex_colors(rock.vertices, rng)

    snow_uv = triplanar_uv(snow.vertices, snow.vertex_normals, SNOW_UV_SCALE)
    snow_mat = PBRMaterial(
        name="snow",
        baseColorTexture=snow_albedo,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.55,
    )
    snow.visual = TextureVisuals(uv=snow_uv, material=snow_mat)
    snow.visual.vertex_attributes["color"] = snow_vertex_colors(snow.vertices, rng)

    return scene


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate a textured granite-boulder GLB from a reference photo.")
    parser.add_argument("--image", required=True, help="source reference image")
    parser.add_argument("--seed", type=int, required=True, help="random seed")
    parser.add_argument("--density", choices=["high", "med", "low"],
                        default="high", help="geometry/texture detail level")
    parser.add_argument("--output", required=True, help="output .glb path")
    args = parser.parse_args()

    scene = build_mesh(args.seed, args.density)
    scene = texture_scene(scene, args.image, args.seed)

    glb = scene.export(file_type="glb")
    with open(args.output, "wb") as f:
        f.write(glb)
    print("wrote {} ({} bytes)".format(args.output, len(glb)))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("ERROR: {}\n".format(exc))
        sys.exit(1)