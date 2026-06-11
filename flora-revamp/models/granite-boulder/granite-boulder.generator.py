"""Procedural granite boulder: geometry + photo-derived material -> textured GLB.

Builds a squat, weathered granite erratic, derives a tileable granite albedo and
normal map from a reference photo, projects triplanar UVs, bakes sun/shade/AO
vertex colors, and exports a single-material GLB.

CLI:
    python thisscript --image PATH --seed INT --density {high,med,low} --output OUT.glb
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageFilter


# ===========================================================================
# GEOMETRY
# ===========================================================================

# --- Measured proportions (read off the reference image) --------------------
# Photo front aspect ~1.10 (nearly round, only slightly wider than tall).
HEIGHT_OVER_WIDTH = 1.05          # overall height / overall width (raised to hit 1.10)
DEPTH_OVER_WIDTH = 0.95           # nearly equal girth front-to-back
WIDEST_AT_HEIGHT_FRAC = 0.45      # widest girth sits low; top gently flattened

# --- Real-world scale -------------------------------------------------------
# A "seat-sized" erratic: roughly 0.7 m across.
OVERALL_WIDTH_M = 0.70
BASE_RADIUS_M = OVERALL_WIDTH_M * 0.5      # equatorial radius in X/Z


def _density_params(density: str):
    """Pick element counts from the density tier BEFORE building.

    icosphere face count = 20 * 4**subdivisions.
      sub 5 -> 20480 tris (high, <= 80000)
      sub 4 ->  5120 tris (med,  <= 25000)
      sub 2 ->   320 tris (low,  <=  8000)
    """
    if density == "high":
        return dict(subdiv=5, octaves=4, n_features=14, n_facets=2)
    if density == "med":
        return dict(subdiv=4, octaves=3, n_features=10, n_facets=2)
    if density == "low":
        return dict(subdiv=2, octaves=2, n_features=6, n_facets=1)
    raise ValueError(f"density must be high/med/low, got {density!r}")


def _value_noise(points, rng, n_features, freq):
    """Smooth, deterministic scalar value-noise sampled at `points`.

    Sum of randomly-oriented sinusoids -> a smooth field with no seams.
    Returns one value per point, roughly in [-1, 1].
    """
    dirs = rng.normal(size=(n_features, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    weights = rng.uniform(-1.0, 1.0, size=n_features)
    proj = (points @ dirs.T) * freq                      # (V, n_features)
    field = np.sin(proj + phases) * weights
    return field.mean(axis=1)


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    p = _density_params(density)

    # 1) Start from a unit icosphere; its vertices double as outward dirs.
    ico = trimesh.creation.icosphere(subdivisions=p["subdiv"], radius=1.0)
    dirs = np.asarray(ico.vertices, dtype=np.float64)
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)
    faces = np.asarray(ico.faces)

    # 2) Directional amplitude bias -> asymmetry (one side bulgier than other).
    bias_axis = rng.normal(size=3)
    bias_axis /= np.linalg.norm(bias_axis) + 1e-9
    dir_bias = 0.65 + 0.55 * _smoothstep(0.5 * (dirs @ bias_axis + 1.0))

    # 3) Multi-octave displacement along the radial direction -> bulging lobes.
    radius = np.ones(len(dirs))
    amp = 0.16          # base displacement as a fraction of unit radius
    freq = 1.7
    for _ in range(p["octaves"]):
        radius += amp * dir_bias * _value_noise(dirs, rng, p["n_features"], freq)
        amp *= 0.5
        freq *= 2.05
    radius = np.clip(radius, 0.6, 1.45)   # keep it convex-ish, no spikes

    verts = dirs * radius[:, None]

    # 4) Squash into the slightly-wider-than-tall spheroid (nearly round).
    verts[:, 0] *= BASE_RADIUS_M
    verts[:, 2] *= BASE_RADIUS_M * DEPTH_OVER_WIDTH
    verts[:, 1] *= BASE_RADIUS_M * HEIGHT_OVER_WIDTH

    # 5) Gently flatten the top (lighter touch so the boulder stays tall enough).
    y = verts[:, 1]
    y_max = y.max()
    top_t = _smoothstep((y - 0.45 * y_max) / (0.55 * y_max + 1e-9))
    verts[:, 1] = y - top_t * 0.10 * y_max

    # 6) One or two planar facet cuts -> flatter spalled scars with rolled
    #    edges (smoothstep blend keeps transitions soft, no sharp arrises).
    for _ in range(p["n_facets"]):
        n = rng.normal(size=3)
        n[1] = -abs(n[1]) * 0.6        # bias facets sideways / slightly down
        n /= np.linalg.norm(n) + 1e-9
        s = verts @ n
        thresh = np.quantile(s, rng.uniform(0.72, 0.85))
        span = max(s.max() - thresh, 1e-6)
        blend = _smoothstep((s - thresh) / span)
        push = blend * (s - thresh) * rng.uniform(0.55, 0.8)
        verts -= push[:, None] * n

    # 7) Flatten the underside slightly so it rests naturally on the ground.
    y = verts[:, 1]
    y_min = y.min()
    bottom_band = 0.30 * (y.max() - y_min)
    under_t = _smoothstep((0.0 + bottom_band - (y - y_min)) / (bottom_band + 1e-9))
    flat_plane = y_min + 0.12 * (y.max() - y_min)
    below = verts[:, 1] < flat_plane
    verts[below, 1] += under_t[below] * (flat_plane - verts[below, 1]) * 0.8

    # 8) Seat on XZ plane (lowest point at y=0) and center in X/Z.
    verts[:, 0] -= 0.5 * (verts[:, 0].min() + verts[:, 0].max())
    verts[:, 2] -= 0.5 * (verts[:, 2].min() + verts[:, 2].max())
    verts[:, 1] -= verts[:, 1].min()

    # 9) Build the mesh and clean up normals/winding.
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    mesh.vertex_normals  # force a finite normal computation

    assert np.isfinite(mesh.vertices).all() and len(mesh.faces) > 0

    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="rock")
    return scene


# ===========================================================================
# TEXTURING
# ===========================================================================

TEX_RES = 512          # albedo / normal resolution (>= 512, never 256)
TILE_M = 0.16          # world-space size of one texture repeat (finer grain)

# Target light, cool, sun-bleached granite grey (keeps it from going dark/olive).
TARGET_GREY = np.array([176.0, 176.0, 180.0])


def _luminance(arr):
    """Perceptual luminance of an (H,W,3) float array in [0,255]."""
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def _delight(arr):
    """Divide out broad lighting; clamp gain to [0.6, 1.6] so nothing blows out."""
    lum = _luminance(arr)
    h, w = lum.shape
    radius = max(8, max(h, w) // 8)
    # Blur on an 8-bit 'L' image -- Pillow filters reject mode 'F'.
    lum_img = Image.fromarray(np.clip(lum, 0, 255).astype(np.uint8), mode="L")
    blurred = np.asarray(lum_img.filter(ImageFilter.GaussianBlur(radius)),
                         dtype=np.float64)
    mean_lum = float(lum.mean())
    gain = np.clip(mean_lum / (blurred + 1e-6), 0.6, 1.6)
    out = arr * gain[..., None]
    return np.clip(out, 0.0, 255.0)


def _quilt_tileable(src, rng, res, n_grid):
    """Toroidally-tileable, NON-symmetric texture by patch quilting.

    Lay random rotated/flipped crops of `src` on an n_grid x n_grid lattice
    with the canvas treated as a torus (patches that run off the edge wrap),
    so the result tiles seamlessly without any mirror-fold symmetry.
    """
    sh, sw = src.shape[:2]
    cell = res // n_grid
    patch = int(cell * 1.5)                  # overlap neighbours to hide seams
    patch = min(patch, sh, sw)
    canvas = np.zeros((res, res, 3), dtype=np.float64)
    weight = np.zeros((res, res, 1), dtype=np.float64)

    # Radial feather so overlapping patches blend smoothly.
    yy, xx = np.mgrid[0:patch, 0:patch].astype(np.float64)
    cx = (patch - 1) / 2.0
    fy = 1.0 - np.abs(yy - cx) / (cx + 1e-6)
    fx = 1.0 - np.abs(xx - cx) / (cx + 1e-6)
    feather = (np.clip(fy, 0, 1) * np.clip(fx, 0, 1))[..., None] + 1e-3

    for gy in range(n_grid):
        for gx in range(n_grid):
            sy = int(rng.integers(0, sh - patch + 1))
            sx = int(rng.integers(0, sw - patch + 1))
            tile = src[sy:sy + patch, sx:sx + patch].astype(np.float64)
            k = int(rng.integers(0, 4))
            tile = np.rot90(tile, k)
            if rng.random() < 0.5:
                tile = tile[:, ::-1]
            oy = gy * cell - patch // 4
            ox = gx * cell - patch // 4
            ys = (np.arange(patch) + oy) % res
            xs = (np.arange(patch) + ox) % res
            canvas[np.ix_(ys, xs)] += tile * feather
            weight[np.ix_(ys, xs)] += feather

    return canvas / np.maximum(weight, 1e-6)


def _make_granite_albedo(src_img, rng):
    """Tileable, light, cool granite albedo derived from the photo's interior."""
    arr = np.asarray(src_img.convert("RGB"), dtype=np.float64)
    h, w = arr.shape[:2]

    # Crop the interior (no sky/ground/background), then de-light.
    cy0, cy1 = int(0.24 * h), int(0.76 * h)
    cx0, cx1 = int(0.24 * w), int(0.76 * w)
    crop = _delight(arr[cy0:cy1, cx0:cx1])

    # Neutralize colour cast + brighten toward target cool grey.
    chan_mean = crop.reshape(-1, 3).mean(axis=0) + 1e-6
    crop = crop * (TARGET_GREY / chan_mean)
    # Compress contrast around the mean so dark facet shadows stop reading as
    # painted blotches; keep enough range for granite mottle.
    cmean = crop.mean()
    crop = cmean + (crop - cmean) * 0.55
    crop = np.clip(crop, 0.0, 255.0)

    # Quilt into a seamless, non-symmetric 512 swatch.
    full = _quilt_tileable(crop, rng, TEX_RES, n_grid=4)

    # Fine mineral speckle: charcoal flecks + cream/white crystals.
    speckle = rng.normal(0.0, 6.0, size=full.shape[:2])
    fleck = rng.random(full.shape[:2])
    full += speckle[..., None]
    full[fleck > 0.990] *= 0.60                                   # dark mineral flecks
    full[fleck < 0.020] = np.minimum(255.0, full[fleck < 0.020] * 1.30)  # bright crystals

    full = np.clip(full, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(full, mode="RGB")


def _make_normal_map(albedo_img, strength=2.2):
    """Tangent-space normal map from albedo luminance (gradient on inverse height)."""
    arr = np.asarray(albedo_img, dtype=np.float64)
    lum = _luminance(arr) / 255.0
    height = 1.0 - lum                       # pits dark -> recessed
    gy, gx = np.gradient(height)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(nx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    out = np.stack([(nx * 0.5 + 0.5),
                    (ny * 0.5 + 0.5),
                    (nz * 0.5 + 0.5)], axis=-1)
    return Image.fromarray((out * 255.0).astype(np.uint8), mode="RGB")


def _triplanar_uv(mesh):
    """Per-vertex triplanar UVs: project onto the plane of the dominant normal axis."""
    verts = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    axis = np.argmax(np.abs(normals), axis=1)
    uv = np.zeros((len(verts), 2), dtype=np.float64)

    mx = axis == 0
    my = axis == 1
    mz = axis == 2
    uv[mx] = verts[mx][:, [2, 1]]   # X-facing -> (z, y)
    uv[my] = verts[my][:, [0, 2]]   # Y-facing -> (x, z)
    uv[mz] = verts[mz][:, [0, 1]]   # Z-facing -> (x, y)
    return uv / TILE_M


def _vertex_colors(mesh, rng):
    """Subtle sun/shade + AO tint -- light overall, no green cast on the body."""
    verts = np.asarray(mesh.vertices)
    y = verts[:, 1]
    y_max = max(y.max(), 1e-6)
    t = y / y_max                                  # 0 base -> 1 crown

    var = 1.0 + 0.04 * _value_noise(verts, rng, 8, 9.0)
    # Keep it bright: base only slightly dusky, crown sun-bleached.
    bright = (0.86 + 0.16 * _smoothstep(t)) * var
    bright = np.clip(bright, 0.0, 1.05)

    rgb = np.ones((len(verts), 3)) * bright[:, None]

    # Faint greenish-grey grime ONLY in the lowest crevice band (very subtle).
    gmask = _smoothstep((0.18 - t) / 0.18)
    rgb[:, 0] *= (1.0 - 0.05 * gmask)
    rgb[:, 1] *= (1.0 + 0.02 * gmask)
    rgb[:, 2] *= (1.0 - 0.03 * gmask)

    rgb = np.clip(rgb, 0.0, 1.0)
    out = np.empty((len(verts), 4), dtype=np.uint8)
    out[:, :3] = (rgb * 255.0).astype(np.uint8)
    out[:, 3] = 255
    return out


def texture_scene(scene, src_img, seed):
    """Attach photo-derived granite material + triplanar UVs + vertex tints."""
    rng = np.random.default_rng(seed + 101)
    mesh = list(scene.geometry.values())[0]

    albedo = _make_granite_albedo(src_img, rng)
    normal = _make_normal_map(albedo)
    uv = _triplanar_uv(mesh)

    material = trimesh.visual.material.PBRMaterial(
        name="granite",
        baseColorTexture=albedo,
        normalTexture=normal,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0,
        roughnessFactor=0.95,
        doubleSided=False,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    mesh.visual.vertex_attributes["color"] = _vertex_colors(mesh, rng)
    return scene


# ===========================================================================
# CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="Procedural granite boulder -> GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        src_img = Image.open(args.image).convert("RGB")
        scene = build_mesh(args.seed, args.density)
        scene = texture_scene(scene, src_img, args.seed)
        glb = scene.export(file_type="glb")
        with open(args.output, "wb") as f:
            f.write(glb)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())