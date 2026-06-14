"""Procedural lichen-covered glacial-erratic boulder: geometry + photo-derived
textures + textured GLB export.

Usage:
    python this.py --image reference.png --seed 7 --density high --output rock.glb

Only numpy, trimesh, PIL and the stdlib are used. Deterministic given --seed.
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageFilter


# ===========================================================================
# GEOMETRY  (build_mesh)
# ===========================================================================
OVERALL_WIDTH_M = 1.6          # plausible alpine field-stone span across XZ
HEIGHT_OVER_WIDTH = 0.92       # chunkier rounded mass -> front aspect ~1.26
WIDEST_AT_HEIGHT_FRAC = 0.45   # bulges widest a little below mid-height

BASE_NOISE_AMP = 0.20          # overall lumpiness of the lobed body
NOISE_ANISOTROPY = 0.55        # how much relief amplitude varies by direction
BASE_FLATTEN_STRENGTH = 0.78   # how hard the underside is pressed into a plane
FACET_STRENGTH = 0.40          # how flat the worn side facets read


def _density_params(density: str):
    if density == "high":
        return dict(subdivisions=5, octaves=4, n_waves=7, side_facets=2)  # 20480 tris
    if density == "med":
        return dict(subdivisions=4, octaves=3, n_waves=6, side_facets=2)  # 5120 tris
    if density == "low":
        return dict(subdivisions=3, octaves=2, n_waves=5, side_facets=1)  # 1280 tris
    raise ValueError(f"density must be 'high', 'med' or 'low', got {density!r}")


def _normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n == 0.0, 1.0, n)
    return v / n


def _value_noise(points, rng, octaves, n_waves, base_freq=1.6):
    """Smooth, deterministic coherent noise as a sum of random plane waves."""
    total = np.zeros(len(points))
    norm = 0.0
    freq = base_freq
    amp = 1.0
    for _ in range(octaves):
        dirs = _normalize(rng.normal(size=(n_waves, 3)))
        phases = rng.uniform(0.0, 2.0 * np.pi, size=n_waves)
        wave = np.sin(points @ dirs.T * freq + phases).mean(axis=1)
        total += amp * wave
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return total / norm


def _flatten_against_plane(points, normal, offset, strength):
    normal = normal / np.linalg.norm(normal)
    signed = points @ normal - offset
    outside = signed > 0.0
    points[outside] -= np.outer(signed[outside] * strength, normal)
    return points


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    p = _density_params(density)

    ico = trimesh.creation.icosphere(subdivisions=p["subdivisions"], radius=1.0)
    v = np.array(ico.vertices, dtype=np.float64)
    faces = np.array(ico.faces)

    dirs = _normalize(v)

    amp_field = _value_noise(dirs * 0.9, rng, octaves=2, n_waves=4)
    amp = BASE_NOISE_AMP * (1.0 - NOISE_ANISOTROPY * 0.5
                            + NOISE_ANISOTROPY * (amp_field * 0.5 + 0.5))

    relief = _value_noise(dirs, rng, octaves=p["octaves"], n_waves=p["n_waves"])
    radius = 1.0 + amp * relief
    v = dirs * radius[:, None]

    v[:, 1] *= HEIGHT_OVER_WIDTH
    bulge = 1.0 + 0.06 * np.cos(np.clip(v[:, 1], -1.0, 1.0) * np.pi
                                * WIDEST_AT_HEIGHT_FRAC)
    v[:, 0] *= bulge
    v[:, 2] *= bulge

    for _ in range(p["side_facets"]):
        n = rng.normal(size=3)
        n[1] = abs(n[1]) * 0.35 + 0.15
        n = n / np.linalg.norm(n)
        reach = np.max(v @ n)
        offset = reach * rng.uniform(0.74, 0.9)
        _flatten_against_plane(v, n, offset, FACET_STRENGTH)

    down = np.array([0.0, -1.0, 0.0])
    bottom_reach = np.max(v @ down)
    base_offset = bottom_reach * 0.80
    _flatten_against_plane(v, down, base_offset, BASE_FLATTEN_STRENGTH)

    span_x = np.ptp(v[:, 0])
    span_z = np.ptp(v[:, 2])
    scale = OVERALL_WIDTH_M / max(span_x, span_z)
    v *= scale

    v[:, 0] -= 0.5 * (v[:, 0].min() + v[:, 0].max())
    v[:, 2] -= 0.5 * (v[:, 2].min() + v[:, 2].max())
    v[:, 1] -= v[:, 1].min()

    rock = trimesh.Trimesh(vertices=v, faces=faces, process=False)
    rock.fix_normals()
    rock.vertex_normals

    scene = trimesh.Scene()
    scene.add_geometry(rock, geom_name="rock")
    return scene


# ===========================================================================
# PHOTO SAMPLING  (palette comes from the image, never from memory)
# ===========================================================================
GRANITE_FALLBACK = np.array([158.0, 160.0, 165.0])
ORANGE_FALLBACK = np.array([198.0, 120.0, 52.0])
GREEN_FALLBACK = np.array([158.0, 170.0, 116.0])


def _delight(arr):
    """Divide out a heavily blurred luminance so baked-in lighting flattens."""
    gray = Image.fromarray(arr.astype(np.uint8)).convert("L")
    radius = max(8, int(max(arr.shape[0], arr.shape[1]) / 8))
    blur = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius))).astype(np.float64)
    blur = np.clip(blur, 1.0, None)
    target = float(blur.mean())
    gain = np.clip(target / blur, 0.6, 1.6)
    return np.clip(arr * gain[..., None], 0, 255)


def _luma(c):
    return (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255.0


def _retarget_luma(c, target):
    """Scale a color so its luminance hits `target` (0..1), keeping its hue."""
    l = max(_luma(c), 0.15)
    return np.clip(c * (target / l), 0, 255)


def sample_palette(image_path):
    """Sample granite / orange-lichen / green-lichen colors from inside the
    boulder silhouette (central crop avoids the grey background)."""
    img = Image.open(image_path).convert("RGB")
    arr = _delight(np.asarray(img).astype(np.float64))

    h, w = arr.shape[:2]
    crop = arr[int(0.20 * h):int(0.82 * h), int(0.16 * w):int(0.84 * w)]
    px = crop.reshape(-1, 3)
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    mx = px.max(axis=1)
    mn = px.min(axis=1)
    sat = mx - mn

    def med(mask, fallback):
        sel = px[mask]
        if len(sel) < 60:
            return fallback.copy()
        return np.median(sel, axis=0)

    orange_mask = (r > g + 18) & (g >= b) & (r - b > 35) & (r > 110)
    green_mask = (g > r + 6) & (g > b + 12) & (mx < 215)
    granite_mask = (sat < 26) & (mx > 80) & (mx < 235)

    granite = med(granite_mask, GRANITE_FALLBACK)
    orange = med(orange_mask, ORANGE_FALLBACK)
    green = med(green_mask, GREEN_FALLBACK)

    if np.ptp(orange) < 28:
        orange = ORANGE_FALLBACK.copy()
    if np.ptp(green) < 18:
        green = GREEN_FALLBACK.copy()

    # Keep granite a LIGHT ash grey so the surface never collapses to black and
    # so multiplied lichen tints stay vivid.
    granite = _retarget_luma(granite, 0.64)
    orange = _retarget_luma(orange, 0.46)
    green = _retarget_luma(green, 0.50)
    return granite, orange, green


# ===========================================================================
# TILEABLE TEXTURE SYNTHESIS  (periodic noise -> inherently seamless)
# ===========================================================================
def _periodic_noise(res, rng, octaves, base_freq, n_waves):
    """Seamless [0,1] field built from integer-frequency plane waves."""
    xs = np.linspace(0.0, 1.0, res, endpoint=False)
    X, Y = np.meshgrid(xs, xs)
    field = np.zeros((res, res))
    amp = 1.0
    freq = int(base_freq)
    for _ in range(octaves):
        for _ in range(n_waves):
            fx = int(rng.integers(-freq, freq + 1))
            fy = int(rng.integers(-freq, freq + 1))
            ph = float(rng.uniform(0.0, 2.0 * np.pi))
            field += amp * np.sin(2.0 * np.pi * (fx * X + fy * Y) + ph)
        amp *= 0.55
        freq = max(1, freq * 2)
    field -= field.min()
    field /= max(field.max(), 1e-6)
    return field


def _normal_from_height(gray, strength=1.8):
    """Tangent-space normal map from a height field (height = inverse luma)."""
    h = 1.0 - gray
    gy, gx = np.gradient(h)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(h)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    out = np.stack([nx / norm, ny / norm, nz / norm], axis=-1) * 0.5 + 0.5
    return Image.fromarray((out * 255).astype(np.uint8), "RGB")


def make_granite_textures(res, rng, granite):
    """Granite albedo: mottled crystalline grey with sparse charcoal pitting and
    a few genuine hairline cracks (NOT the old scribble field). Luminance is
    centered at 1.0 so the albedo stays near the (light) granite color."""
    mott = _periodic_noise(res, rng, 4, 3, 6)      # broad crystalline mottle
    grain = _periodic_noise(res, rng, 2, 14, 5)    # fine speckle grain
    spk = _periodic_noise(res, rng, 1, 36, 4)      # charcoal pits

    # Sparse, sharp cracks: very few low-freq waves so contours are long &
    # rare rather than a dense field of squiggles; thin and only mildly dark.
    crk = _periodic_noise(res, rng, 1, 4, 3)
    ridge = np.abs(crk - 0.5) * 2.0
    crack = np.clip(1.0 - ridge / 0.012, 0.0, 1.0)

    lum = 1.0 + 0.20 * (mott - 0.5) + 0.10 * (grain - 0.5)
    lum -= 0.22 * np.clip(0.16 - spk, 0.0, 1.0)    # occasional dark pit
    lum *= (1.0 - 0.28 * crack)                    # subtle hairline cracks
    lum = np.clip(lum, 0.55, 1.25)

    base = granite / 255.0
    cool = np.array([0.94, 0.97, 1.05])            # cool the darker mottle
    tint = base[None, None, :] * (1.0 + (0.5 - mott)[..., None] * 0.08 * cool)
    albedo = np.clip(tint * lum[..., None], 0.0, 1.0)

    albedo_img = Image.fromarray((albedo * 255).astype(np.uint8), "RGB")
    gray = np.asarray(albedo_img.convert("L")).astype(np.float64) / 255.0
    normal_img = _normal_from_height(gray, strength=1.8)
    return albedo_img, normal_img


# ===========================================================================
# UV + VERTEX-COLOR LICHEN ZONES
# ===========================================================================
UV_SCALE = 1.8   # ~3 granite tiles across the ~1.6 m boulder


def triplanar_uv(mesh):
    """Bake a triplanar projection into a single per-vertex UV set."""
    n = mesh.vertex_normals
    p = mesh.vertices * UV_SCALE
    ax = np.argmax(np.abs(n), axis=1)
    uv = np.zeros((len(p), 2))
    uv[ax == 0] = p[ax == 0][:, [2, 1]]
    uv[ax == 1] = p[ax == 1][:, [0, 2]]
    uv[ax == 2] = p[ax == 2][:, [0, 1]]
    uv -= np.floor(uv.min(axis=0))
    return uv


def lichen_vertex_colors(mesh, rng, granite, orange, green):
    """COLOR_0 tints that multiply the granite texture: vivid orange crustose
    lichen heavy on the upper-left crown, sage-green pooling on lower flanks."""
    pos = mesh.vertices
    nrm = mesh.vertex_normals
    miny, maxy = pos[:, 1].min(), pos[:, 1].max()
    h = (pos[:, 1] - miny) / max(maxy - miny, 1e-6)
    xext = max(np.abs(pos[:, 0]).max(), 1e-6)
    leftness = np.clip(-pos[:, 0] / xext, -1.0, 1.0) * 0.5 + 0.5

    blob = _value_noise(pos * 2.2, rng, 3, 6) * 0.5 + 0.5
    blob2 = _value_noise(pos * 4.2 + 11.0, rng, 3, 5) * 0.5 + 0.5
    upface = np.clip(nrm[:, 1], 0.0, 1.0)

    # Generous coverage so the lichen actually reads (it dominates the photo).
    orange_w = np.clip(1.05 * h + 0.7 * leftness + 1.3 * blob - 1.05, 0.0, 1.0)
    green_w = np.clip(1.0 * (1.0 - h) + 1.25 * blob2 - 0.80, 0.0, 1.0)
    green_w *= (1.0 - 0.6 * orange_w)              # orange wins on overlap

    gl = max(_luma(granite), 0.30)
    granite_tint = np.array([1.0, 1.0, 1.0])
    orange_tint = np.clip(orange / 255.0 / gl, 0.0, 1.0)
    green_tint = np.clip(green / 255.0 / gl, 0.0, 1.0)

    gw = np.full(len(pos), 0.85)                   # baseline granite always present
    denom = (gw + orange_w + green_w)[:, None]
    col = (granite_tint[None, :] * gw[:, None]
           + orange_tint[None, :] * orange_w[:, None]
           + green_tint[None, :] * green_w[:, None]) / denom

    # Gentle sun/shade + crevice darkening with a little per-vertex jitter.
    bright = 0.86 + 0.18 * h + 0.12 * upface
    bright += (rng.random(len(pos)) - 0.5) * 0.05
    bright = np.clip(bright, 0.74, 1.12)
    col = np.clip(col * bright[:, None], 0.0, 1.0)

    rgba = np.empty((len(pos), 4), dtype=np.uint8)
    rgba[:, :3] = (col * 255).astype(np.uint8)
    rgba[:, 3] = 255
    return rgba


# ===========================================================================
# ASSEMBLY
# ===========================================================================
def build_textured_scene(seed, density, image_path):
    rng = np.random.default_rng(seed)
    granite, orange, green = sample_palette(image_path)

    mesh = build_mesh(seed, density).geometry["rock"]

    albedo_img, normal_img = make_granite_textures(1024, rng, granite)
    uv = triplanar_uv(mesh)
    vcolors = lichen_vertex_colors(mesh, rng, granite, orange, green)

    material = trimesh.visual.material.PBRMaterial(
        name="rock",
        baseColorTexture=albedo_img,
        normalTexture=normal_img,
        metallicFactor=0.0,
        roughnessFactor=0.95,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    mesh.visual.vertex_attributes["color"] = vcolors

    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="rock")
    return scene


def main():
    ap = argparse.ArgumentParser(description="Generate a textured lichen boulder GLB.")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    scene = build_textured_scene(args.seed, args.density, args.image)
    glb = scene.export(file_type="glb")
    with open(args.output, "wb") as fh:
        fh.write(glb)
    print(f"wrote {args.output}: {len(glb)} bytes, "
          f"{len(scene.geometry['rock'].faces)} tris")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)