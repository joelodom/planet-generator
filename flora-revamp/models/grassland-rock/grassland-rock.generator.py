"""
Procedural weathered-granite boulder: geometry + photo-derived tileable
materials + UVs + textured GLB export.

Pipeline
--------
1. build_mesh(seed, density)         -> faceted icosphere boulder (one "rock"
                                        surface, +Y up, base at y=0, meters)
2. derive granite materials from the reference photo:
     - sample body / sunlit / shaded tones from WELL INSIDE the silhouette
     - de-light a clean crop, make it tileable by mirror-fold
     - overlay procedural mottling, fracture cracks, mineral speckle, lichen
     - bake a tangent-space normal map from the albedo
3. triplanar per-vertex UVs + per-vertex COLOR_0 (ground/crevice/sun tints)
4. export an embedded-texture GLB.

CLI:
    python thisscript.py --image PATH --seed INT \
        --density {high,med,low} --output OUT.glb
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageFilter


# ===========================================================================
# GEOMETRY  (validated module, reused; UVs/material attached downstream)
# ===========================================================================

# Photo silhouette is BROADER than tall: content aspect width/height ~1.15.
HEIGHT_OVER_WIDTH = 0.88          # total height / base width (measured by eye)
TARGET_ASPECT = 1.13             # enforced front-view width / height
TARGET_HEIGHT = 2.6              # meters, tip-to-ground (squat landmark stone)

TAPER_STRENGTH = 0.34             # gentle taper -> broad blunt dome (not a cone)
VERTICAL_STRETCH = 0.92           # squat the form toward the photo proportion
LEAN_FRACTION = 0.06              # subtle lean of the peak
BASE_FLATTEN_FRACTION = 0.05      # bottom ~5% flattened so it sits on ground

BASE_FREQ = 3                     # lowest noise frequency (big lumps/lobes)
ROUGH_MIN = 0.05                  # min radial noise amplitude (smooth flanks)
ROUGH_MAX = 0.22                  # max radial noise amplitude (rough side)


def _density_params(density: str) -> dict:
    # Triangle caps: high <= 80000, med <= 25000, low <= 8000.
    # icosphere faces = 20 * 4**subdivisions.
    table = {
        "high": dict(subdiv=5, octaves=5, facets=4),  # 20480 tris
        "med":  dict(subdiv=4, octaves=4, facets=3),  #  5120 tris
        "low":  dict(subdiv=3, octaves=3, facets=2),  #  1280 tris
    }
    if density not in table:
        density = "high"
    return table[density]


def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def _value_noise(d: np.ndarray, rng: np.random.Generator, res: int) -> np.ndarray:
    """Smooth 3D value noise sampled at directions d (in [-1,1]^3)."""
    res = int(res)
    lo, hi = -1.05, 1.05
    coords = (d - lo) / (hi - lo) * res          # -> [0, res]
    i0 = np.floor(coords).astype(np.int64)
    i0 = np.clip(i0, 0, res)                      # i0+1 stays within grid
    f = coords - i0
    s = _smoothstep(f)

    grid = rng.random((res + 2, res + 2, res + 2))
    ix, iy, iz = i0[:, 0], i0[:, 1], i0[:, 2]
    fx, fy, fz = s[:, 0], s[:, 1], s[:, 2]

    def g(dx, dy, dz):
        return grid[ix + dx, iy + dy, iz + dz]

    c00 = g(0, 0, 0) * (1 - fx) + g(1, 0, 0) * fx
    c10 = g(0, 1, 0) * (1 - fx) + g(1, 1, 0) * fx
    c01 = g(0, 0, 1) * (1 - fx) + g(1, 0, 1) * fx
    c11 = g(0, 1, 1) * (1 - fx) + g(1, 1, 1) * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    c = c0 * (1 - fz) + c1 * fz
    return c * 2.0 - 1.0


def _fbm(d: np.ndarray, rng: np.random.Generator, octaves: int) -> np.ndarray:
    """Fractal sum of value-noise octaves, normalized to ~[-1, 1]."""
    total = np.zeros(len(d))
    amp, norm = 1.0, 0.0
    freq = BASE_FREQ
    for _ in range(octaves):
        total += amp * _value_noise(d, rng, freq)
        norm += amp
        amp *= 0.5
        freq *= 2
    return total / norm


def _facet_cut(V: np.ndarray, rng: np.random.Generator) -> None:
    """Flatten the outer slab beyond a random plane onto that plane (in place)."""
    n = rng.normal(size=3)
    n[1] *= 0.4                                   # bias horizontal -> cliff faces
    n = n / (np.linalg.norm(n) + 1e-9)
    dist = V @ n
    thresh = np.quantile(dist, rng.uniform(0.78, 0.90))
    mask = dist > thresh
    V[mask] -= (dist[mask] - thresh)[:, None] * n


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    params = _density_params(density)

    # --- base icosphere: unit directions + topology ------------------------
    ico = trimesh.creation.icosphere(subdivisions=params["subdiv"], radius=1.0)
    d = np.asarray(ico.vertices, dtype=np.float64).copy()
    d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    faces = np.asarray(ico.faces).copy()

    # --- direction-dependent roughness (asymmetry = realism) ---------------
    asym_dir = rng.normal(size=3)
    asym_dir /= (np.linalg.norm(asym_dir) + 1e-9)
    amp_field = 0.5 * (ROUGH_MIN + ROUGH_MAX) \
        + 0.5 * (ROUGH_MAX - ROUGH_MIN) * (d @ asym_dir)

    noise = _fbm(d, rng, params["octaves"])
    R = 1.0 + amp_field * noise               # rocky radial displacement
    V = d * R[:, None]

    # --- broad dome-topped silhouette with asymmetric flanks ---------------
    hf = (d[:, 1] + 1.0) * 0.5                 # height fraction in [0, 1]
    flank_phase = rng.uniform(0.0, 2.0 * np.pi)
    az = np.arctan2(V[:, 2], V[:, 0])
    taper = 1.0 - TAPER_STRENGTH * _smoothstep(hf)
    taper *= 1.0 + 0.15 * hf * np.cos(az - flank_phase)   # one flank steeper
    V[:, 0] *= taper
    V[:, 2] *= taper

    # --- slight lean of the peak ------------------------------------------
    lean_phase = rng.uniform(0.0, 2.0 * np.pi)
    V[:, 0] += LEAN_FRACTION * np.cos(lean_phase) * hf
    V[:, 2] += LEAN_FRACTION * np.sin(lean_phase) * hf

    V[:, 1] *= VERTICAL_STRETCH

    # --- carve planar facets / fracture faces ------------------------------
    for _ in range(params["facets"]):
        _facet_cut(V, rng)

    # --- flatten the underside so it sits on the ground --------------------
    ymin = V[:, 1].min()
    yspan = np.ptp(V[:, 1])
    base_level = ymin + BASE_FLATTEN_FRACTION * yspan
    V[V[:, 1] < base_level, 1] = base_level

    # --- center in X/Z, drop to y=0, scale to real-world height ------------
    V[:, 0] -= V[:, 0].mean()
    V[:, 2] -= V[:, 2].mean()
    V[:, 1] -= V[:, 1].min()
    cur_h = np.ptp(V[:, 1])
    if cur_h > 1e-9:
        V *= (TARGET_HEIGHT / cur_h)
    V[:, 1] -= V[:, 1].min()                  # guarantee lowest point at y=0

    # --- enforce the broad photo proportion (width/height ~ TARGET_ASPECT) -
    cur_w = np.ptp(V[:, 0])
    if cur_w > 1e-9:
        sx = (TARGET_ASPECT * np.ptp(V[:, 1])) / cur_w
        V[:, 0] *= sx
        V[:, 2] *= sx
    V[:, 0] -= V[:, 0].mean()
    V[:, 2] -= V[:, 2].mean()

    # --- assemble -----------------------------------------------------------
    mesh = trimesh.Trimesh(vertices=V, faces=faces, process=False)
    mesh.merge_vertices()
    mesh.fix_normals()

    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="rock")
    return scene


# ===========================================================================
# PHOTO SAMPLING  (colors taken from WELL INSIDE the boulder silhouette)
# ===========================================================================

# The boulder fills the central ~70% of the frame against a flat grey
# background. This safe box stays clear of every edge / background pixel.
SAFE_X = (0.32, 0.70)
SAFE_Y = (0.34, 0.72)

_LUM = np.array([0.2126, 0.7152, 0.0722])

# Light-granite target: keep the albedo PALE so it does not render dark.
TARGET_ALB_LUM = 0.62


def _load_rgb(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im).astype(np.float64) / 255.0


def _resize_rgb(arr: np.ndarray, size: int) -> np.ndarray:
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    im = im.resize((size, size), Image.LANCZOS)
    return np.asarray(im).astype(np.float64) / 255.0


def sample_palette(img: np.ndarray) -> dict:
    """Median body tone plus sunlit / shaded tones, all from inside the rock."""
    H, W = img.shape[:2]
    r = max(2, int(0.018 * min(H, W)))
    cols = []
    nx, ny = 6, 6
    for i in range(nx):
        for j in range(ny):
            cx = SAFE_X[0] + (SAFE_X[1] - SAFE_X[0]) * (i + 0.5) / nx
            cy = SAFE_Y[0] + (SAFE_Y[1] - SAFE_Y[0]) * (j + 0.5) / ny
            px, py = int(cx * W), int(cy * H)
            patch = img[max(0, py - r):py + r, max(0, px - r):px + r]
            if patch.size == 0:
                continue
            cols.append(np.median(patch.reshape(-1, 3), axis=0))

    cols = np.array(cols) if cols else np.array([[0.66, 0.66, 0.64]])
    med = np.median(cols, axis=0)
    dist = np.linalg.norm(cols - med, axis=1)
    keep = cols[dist < (np.median(dist) * 2.5 + 0.05)]
    if len(keep) < 4:
        keep = cols

    lum = keep @ _LUM
    order = np.argsort(lum)
    k = max(1, len(keep) // 4)
    shade = keep[order[:k]].mean(axis=0)
    light = keep[order[-k:]].mean(axis=0)
    body = np.median(keep, axis=0)

    # Warm the sunlit tone toward tan/buff; cool the shaded tone slightly.
    light = np.clip(light * np.array([1.05, 1.00, 0.92]), 0, 1)
    shade = np.clip(shade * np.array([0.97, 0.99, 1.04]), 0, 1)
    return dict(body=body, light=light, shade=shade)


def _delight(patch: np.ndarray) -> np.ndarray:
    """Divide out heavily blurred luminance; clamp gain to [0.6, 1.6]."""
    lum = patch @ _LUM
    im = Image.fromarray((np.clip(lum, 0, 1) * 255).astype(np.uint8))
    rad = max(8, int(min(patch.shape[:2]) * 0.25))
    blur = np.asarray(im.filter(ImageFilter.GaussianBlur(rad))).astype(np.float64) / 255.0
    target = float(lum.mean()) + 1e-3
    gain = np.clip(target / (blur + 1e-3), 0.6, 1.6)
    return np.clip(patch * gain[..., None], 0, 1)


def _mirror_tile(p: np.ndarray) -> np.ndarray:
    """Mirror-fold into a seamless tile (no rolled seam through the middle)."""
    top = np.concatenate([p, p[:, ::-1]], axis=1)
    return np.concatenate([top, top[::-1]], axis=0)


def _soften_folds(tile: np.ndarray, band: int = 8) -> np.ndarray:
    """Lightly blur only thin bands across the two fold lines."""
    im = Image.fromarray((np.clip(tile, 0, 1) * 255).astype(np.uint8))
    blur = np.asarray(im.filter(ImageFilter.GaussianBlur(2))).astype(np.float64) / 255.0
    arr = np.asarray(im).astype(np.float64) / 255.0
    H, W, _ = arr.shape
    mask = np.zeros((H, W))
    cy, cx = H // 2, W // 2
    mask[:, max(0, cx - band):cx + band] = 1.0
    mask[max(0, cy - band):cy + band, :] = 1.0
    m = mask[..., None]
    return arr * (1 - m) + blur * m


# ===========================================================================
# TILEABLE 2D NOISE  (for procedural granite detail overlays)
# ===========================================================================

def _tileable_value_noise(size: int, rng: np.random.Generator, res: int) -> np.ndarray:
    res = max(2, int(res))
    grid = rng.random((res, res))
    t = np.linspace(0.0, res, size, endpoint=False)
    gx, gy = np.meshgrid(t, t)
    x0 = np.floor(gx).astype(np.int64) % res
    y0 = np.floor(gy).astype(np.int64) % res
    x1 = (x0 + 1) % res
    y1 = (y0 + 1) % res
    fx = _smoothstep(gx - np.floor(gx))
    fy = _smoothstep(gy - np.floor(gy))
    v00 = grid[y0, x0]; v10 = grid[y0, x1]
    v01 = grid[y1, x0]; v11 = grid[y1, x1]
    top = v00 * (1 - fx) + v10 * fx
    bot = v01 * (1 - fx) + v11 * fx
    return top * (1 - fy) + bot * fy


def _fbm2d(size: int, rng: np.random.Generator, octaves: int = 4, res: int = 4) -> np.ndarray:
    total = np.zeros((size, size))
    amp, norm, r = 1.0, 0.0, res
    for _ in range(octaves):
        total += amp * _tileable_value_noise(size, rng, int(round(r)))
        norm += amp
        amp *= 0.5
        r *= 2
    return total / norm


def _crack_field(size: int, rng: np.random.Generator) -> np.ndarray:
    """Sparse thin ridged lines reading as fracture cracks / clefts."""
    cracks = np.zeros((size, size))
    for _ in range(2):
        n = _fbm2d(size, rng, octaves=4, res=int(rng.integers(3, 6)))
        ridge = 1.0 - np.abs(2.0 * n - 1.0)
        ridge = np.clip((ridge - 0.80) / 0.20, 0.0, 1.0) ** 1.7
        cracks = np.maximum(cracks, ridge)
    return cracks


# ===========================================================================
# GRANITE ALBEDO + NORMAL
# ===========================================================================

def build_albedo(img: np.ndarray, palette: dict, rng: np.random.Generator,
                 size: int = 1024) -> np.ndarray:
    """Photo-grounded, tileable, PALE granite albedo with subtle features."""
    # --- tileable photo base (real grain), de-lit -------------------------
    H, W = img.shape[:2]
    x0, x1 = int(SAFE_X[0] * W), int(SAFE_X[1] * W)
    y0, y1 = int(SAFE_Y[0] * H), int(SAFE_Y[1] * H)
    patch = img[y0:y1, x0:x1]
    s = min(patch.shape[0], patch.shape[1])
    patch = patch[:s, :s] if s > 0 else img
    patch = _resize_rgb(patch, size // 2)
    patch = _delight(patch)
    base = _soften_folds(_mirror_tile(patch))          # size x size x 3

    body = palette["body"]
    light = palette["light"]
    shade = palette["shade"]

    # --- gentle tonal mottling (stays light: only mildly cooled lows) ------
    dark = body * 0.6 + shade * 0.4                     # mild, not deep shade
    m = _fbm2d(size, rng, octaves=4, res=3)[..., None]
    tone = dark[None, None, :] * (1 - m) + light[None, None, :] * m
    ratio = tone / (body[None, None, :] + 1e-3)
    alb = base * ratio                                 # keep grain, vary tone

    # --- fracture cracks (mild darken) ------------------------------------
    cracks = _crack_field(size, rng)[..., None]
    alb *= (1.0 - 0.30 * cracks)

    # --- coarse mineral speckle (sparse light + dark flecks) --------------
    sp = rng.random((size, size))
    alb += (sp > 0.994).astype(np.float64)[..., None] * 0.12
    alb -= (sp < 0.006).astype(np.float64)[..., None] * 0.10

    # --- faint lichen / dusty residue in clustered zones ------------------
    lk = _fbm2d(size, rng, octaves=3, res=2)
    lichen = np.clip((lk - 0.66) / 0.20, 0.0, 1.0)[..., None]
    moss = np.array([0.42, 0.44, 0.34])
    alb = alb * (1 - 0.09 * lichen) + moss[None, None, :] * 0.09 * lichen

    # --- normalize to a pale light-granite luminance ----------------------
    alb = np.clip(alb, 0, 1)
    cur = float((alb @ _LUM).mean()) + 1e-4
    alb *= np.clip(TARGET_ALB_LUM / cur, 0.8, 2.6)
    return (np.clip(alb, 0, 1) * 255).astype(np.uint8)


def albedo_to_normal(alb: np.ndarray, strength: float = 1.1) -> np.ndarray:
    """Gentle tangent-space normal map from albedo (height = luminance)."""
    a = alb.astype(np.float64) / 255.0
    h = a @ np.array([0.299, 0.587, 0.114])            # bright = raised
    gx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.5
    gy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 0.5
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(h)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    n = np.stack([nx * inv, ny * inv, nz * inv], axis=-1)
    return ((n * 0.5 + 0.5) * 255).astype(np.uint8)


# ===========================================================================
# UVs + PER-VERTEX COLOR
# ===========================================================================

# Large tile so the texture repeats ~once across the rock (kills the lattice
# tiling artifact from the earlier small-tile triplanar mapping).
TEXTURE_TILE_METERS = 2.2


def triplanar_uv(mesh: trimesh.Trimesh) -> np.ndarray:
    """Per-vertex triplanar UVs: project on the dominant-normal axis plane."""
    n = np.abs(mesh.vertex_normals)
    v = np.asarray(mesh.vertices)
    axis = np.argmax(n, axis=1)
    uv = np.zeros((len(v), 2))
    mx = axis == 0
    my = axis == 1
    mz = axis == 2
    uv[mx] = v[mx][:, [2, 1]]
    uv[my] = v[my][:, [0, 2]]
    uv[mz] = v[mz][:, [0, 1]]
    return uv / TEXTURE_TILE_METERS


def _spatial_field(p: np.ndarray, rng: np.random.Generator, k: int = 6,
                   fmin: float = 0.6, fmax: float = 2.4) -> np.ndarray:
    """Smooth deterministic scalar field over positions, ~[-1, 1]."""
    val = np.zeros(len(p))
    for _ in range(k):
        dirv = rng.normal(size=3)
        dirv /= (np.linalg.norm(dirv) + 1e-9)
        f = rng.uniform(fmin, fmax)
        ph = rng.uniform(0.0, 2.0 * np.pi)
        val += np.sin(p @ dirv * f + ph)
    return val / k


def vertex_colors(mesh: trimesh.Trimesh, rng: np.random.Generator) -> np.ndarray:
    """COLOR_0 tints: near white, with mild ground/crevice/sun modulation."""
    v = np.asarray(mesh.vertices)
    nrm = np.asarray(mesh.vertex_normals)
    y = v[:, 1]
    h = (y - y.min()) / (np.ptp(y) + 1e-9)

    tint = np.ones((len(v), 3))

    # lower zones: faint dusty / mossy darkening near the ground
    low = np.clip((0.30 - h) / 0.30, 0.0, 1.0)
    tint *= (1.0 - 0.07 * low)[:, None]
    tint[:, 1] *= (1.0 + 0.02 * low)               # whisper of green at base

    # crevice darkening (recessed = lower spatial field)
    crev = np.clip(-_spatial_field(v, rng, k=6, fmin=0.8, fmax=3.0), 0.0, 1.0)
    tint *= (1.0 - 0.05 * crev)[:, None]

    # warm tan staining on the sun-facing planes
    sun = np.array([-0.4, 0.82, 0.4])
    sun /= np.linalg.norm(sun)
    facing = np.clip(nrm @ sun, 0.0, 1.0)
    tint[:, 0] *= (1.0 + 0.05 * facing)
    tint[:, 2] *= (1.0 - 0.04 * facing)

    # subtle per-vertex variation
    var = _spatial_field(v, rng, k=5, fmin=1.5, fmax=4.0)
    tint *= (1.0 + 0.02 * var)[:, None]

    rgb = np.clip(tint, 0.0, 1.0)
    out = np.empty((len(v), 4), dtype=np.uint8)
    out[:, :3] = (rgb * 255).astype(np.uint8)
    out[:, 3] = 255
    return out


# ===========================================================================
# ASSEMBLY
# ===========================================================================

def build_textured_scene(image_path: str, seed: int, density: str) -> trimesh.Scene:
    scene = build_mesh(seed, density)
    mesh = scene.geometry["rock"]

    img = _load_rgb(image_path)
    palette = sample_palette(img)

    tex_rng = np.random.default_rng(seed + 9173)
    albedo = build_albedo(img, palette, tex_rng, size=1024)
    normal = albedo_to_normal(albedo)

    material = trimesh.visual.material.PBRMaterial(
        name="weathered_granite",
        baseColorTexture=Image.fromarray(albedo),
        normalTexture=Image.fromarray(normal),
        metallicFactor=0.0,
        roughnessFactor=0.95,
    )

    uv = triplanar_uv(mesh)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)

    col_rng = np.random.default_rng(seed + 4421)
    mesh.visual.vertex_attributes["color"] = vertex_colors(mesh, col_rng)

    return scene


def main() -> int:
    ap = argparse.ArgumentParser(description="Procedural granite boulder -> GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        scene = build_textured_scene(args.image, args.seed, args.density)
        data = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(data)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())