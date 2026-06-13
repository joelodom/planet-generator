"""
Standalone procedural generator + texturer for a broad, low-lying
STRATIFIED SANDSTONE ROCK.

Builds a flattened, layered shelf-shaped weathered rock (geometry), derives a
seamless tileable sandstone material (FFT-tileable procedural grain + crisp
horizontal bedding laminae + ochre staining, all tinted by colours SAMPLED from
the reference photo), bakes triplanar UVs, adds sun/shade per-vertex tints, and
exports a textured GLB.

CLI:
    python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb

Only numpy / trimesh / PIL / stdlib. +Y up, base at y=0, metres.
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageFilter


# ===========================================================================
# GEOMETRY
# ===========================================================================
OVERALL_LENGTH_M = 2.6          # long axis (X), a hefty boulder-shelf

# Measured silhouette ratios (front view target width/height ~2.37 -> H/W ~0.42,
# but noise widens X and the buried base eats height, so the modelling H/W is set
# higher to land the rendered aspect near the photo):
HEIGHT_OVER_WIDTH = 0.60        # raised so the rendered front aspect ~2.4
DEPTH_OVER_WIDTH = 0.58         # moderately deep, less than its length

RX = 1.30                       # half-length  (widest)
RY = RX * HEIGHT_OVER_WIDTH     # half-height
RZ = RX * DEPTH_OVER_WIDTH      # half-depth

NOISE_AMP = 0.16                # large lumpy surface variation
STRATA_AMP = 0.12               # how far beds protrude/recede radially
LIP_AMP = 0.65                  # extra overhang at the bottom edge of each bed
TAPER = 0.26                    # gentler pinch -> blunter wedge end (was 0.34)
TOP_FLATTEN = 0.55              # compress the crest into a gentle plateau
TOP_THRESH = 0.40               # fraction of max height where flattening starts
BOTTOM_FLATTEN = 0.10           # fraction of height clamped flat (buried base)
FACET_STRENGTH = 0.55           # partial flattening of one large planar facet


def _make_noise_tables(rng):
    perm = rng.permutation(256).astype(np.int64)
    perm = np.concatenate([perm, perm])
    vals = rng.uniform(-1.0, 1.0, 256)
    return perm, vals


def _value_noise(points, perm, vals):
    ix = np.floor(points[:, 0]).astype(np.int64)
    iy = np.floor(points[:, 1]).astype(np.int64)
    iz = np.floor(points[:, 2]).astype(np.int64)
    fx = points[:, 0] - ix
    fy = points[:, 1] - iy
    fz = points[:, 2] - iz

    def sm(t):
        return t * t * (3.0 - 2.0 * t)

    wx, wy, wz = sm(fx), sm(fy), sm(fz)

    def H(a, b, c):
        return vals[perm[(perm[(perm[a & 255] + (b & 255)) & 255] + (c & 255)) & 255]]

    c000 = H(ix, iy, iz);     c100 = H(ix + 1, iy, iz)
    c010 = H(ix, iy + 1, iz); c110 = H(ix + 1, iy + 1, iz)
    c001 = H(ix, iy, iz + 1);     c101 = H(ix + 1, iy, iz + 1)
    c011 = H(ix, iy + 1, iz + 1); c111 = H(ix + 1, iy + 1, iz + 1)

    x00 = c000 + wx * (c100 - c000)
    x10 = c010 + wx * (c110 - c010)
    x01 = c001 + wx * (c101 - c001)
    x11 = c011 + wx * (c111 - c011)
    y0 = x00 + wy * (x10 - x00)
    y1 = x01 + wy * (x11 - x01)
    return y0 + wz * (y1 - y0)


def _fbm(points, octaves, perm, vals, base_freq=2.2, lacunarity=2.0, gain=0.5):
    total = np.zeros(len(points))
    amp = 1.0
    freq = base_freq
    norm = 0.0
    for o in range(octaves):
        total += amp * _value_noise(points * freq + o * 17.3, perm, vals)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm


def _density_params(density):
    if density == "high":
        return dict(subdiv=5, octaves=5, strata=13)   # ~20480 tris
    if density == "med":
        return dict(subdiv=4, octaves=4, strata=10)   # ~5120 tris
    if density == "low":
        return dict(subdiv=2, octaves=3, strata=6)    # ~320 tris
    raise ValueError("density must be 'high', 'med' or 'low'")


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    p = _density_params(density)

    perm, vals = _make_noise_tables(rng)
    perm2, vals2 = _make_noise_tables(rng)

    ico = trimesh.creation.icosphere(subdivisions=p["subdiv"], radius=1.0)
    v = np.asarray(ico.vertices, dtype=np.float64)
    faces = np.asarray(ico.faces)

    y_u = v[:, 1]
    up = np.clip(y_u, 0.0, 1.0)

    # lumpy surface noise, smoothed on top, broken on the flanks
    top_fac = np.clip(1.0 - 0.70 * up, 0.30, 1.0)
    side_emphasis = 1.0 + 0.30 * (-v[:, 0]) + 0.15 * v[:, 2]
    noise = _fbm(v, p["octaves"], perm, vals, base_freq=2.2)
    r_noise = NOISE_AMP * top_fac * side_emphasis * noise

    # horizontal stratification: sharp stepped beds with undercut lips
    n_strata = p["strata"]
    bed_offset = rng.uniform(-1.0, 1.0, n_strata + 2)
    bed_strength = rng.uniform(0.30, 1.0, n_strata + 2)

    t = (y_u * 0.5 + 0.5) * n_strata
    band = np.clip(np.floor(t).astype(np.int64), 0, n_strata + 1)
    frac = t - np.floor(t)

    az_mod = 0.55 + 0.45 * _fbm(v * np.array([1.0, 0.18, 1.0]),
                                2, perm2, vals2, base_freq=1.6)
    lip = (1.0 - frac) ** 1.5                                 # sharper overhang
    strata_top_fac = np.clip(1.0 - 1.15 * up, 0.0, 1.0)
    r_strata = STRATA_AMP * az_mod * strata_top_fac * (
        0.5 * bed_offset[band] + LIP_AMP * bed_strength[band] * lip
    )

    R = 1.0 + r_noise + r_strata
    pts = v * R[:, None]

    pts[:, 0] *= RX
    pts[:, 1] *= RY
    pts[:, 2] *= RZ

    # blunter wedge taper toward +X
    taper = 1.0 - TAPER * ((v[:, 0] + 1.0) * 0.5)
    pts[:, 1] *= taper
    pts[:, 2] *= taper

    # flatten the crest into a gentle plateau (shelf, not dome)
    ytop = pts[:, 1].max()
    thr = TOP_THRESH * ytop
    hi = pts[:, 1] > thr
    pts[hi, 1] = thr + (pts[hi, 1] - thr) * TOP_FLATTEN

    # one large planar facet cut for character (partial flatten)
    fn = np.array([0.55 + 0.2 * rng.uniform(-1, 1),
                   0.45 + 0.2 * rng.uniform(-1, 1),
                   0.30 + 0.2 * rng.uniform(-1, 1)])
    fn /= np.linalg.norm(fn)
    d = np.dot(pts, fn)
    cut = np.percentile(d, 88.0)
    over = d > cut
    pts[over] -= np.outer((d[over] - cut) * FACET_STRENGTH, fn)

    meters = OVERALL_LENGTH_M / (2.0 * RX)
    pts *= meters

    # flatten the underside so it sits flush (half-buried)
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    floor = ymin + BOTTOM_FLATTEN * (ymax - ymin)
    pts[:, 1] = np.maximum(pts[:, 1], floor)

    # ground it: lowest point at y=0, centred in X/Z
    pts[:, 1] -= pts[:, 1].min()
    bb_min = pts.min(axis=0)
    bb_max = pts.max(axis=0)
    pts[:, 0] -= 0.5 * (bb_min[0] + bb_max[0])
    pts[:, 2] -= 0.5 * (bb_min[2] + bb_max[2])

    mesh = trimesh.Trimesh(vertices=pts, faces=faces, process=False)
    mesh.fix_normals()

    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="rock")
    return scene


# ===========================================================================
# TEXTURING
# ===========================================================================
TEX = 1024                      # final tileable albedo (>= 512 rule)
TILE_M = 0.85                   # world metres per texture tile (triplanar)


def _luminance(arr):
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def _tileable_noise(size, beta, rng):
    """Perfectly periodic (seamless) scalar field in [0,1] via 1/f^beta spectrum."""
    f = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(f, f)
    r = np.sqrt(fx * fx + fy * fy)
    r[0, 0] = 1.0
    amp = r ** (-beta)
    amp[0, 0] = 0.0
    ph = rng.uniform(0.0, 2.0 * np.pi, (size, size))
    spec = amp * (np.cos(ph) + 1j * np.sin(ph))
    field = np.fft.ifft2(spec).real
    field -= field.min()
    m = field.max()
    if m > 1e-9:
        field /= m
    return field.astype(np.float32)


def _load_image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def _sample_palette(arr):
    """Median colour of small patches placed WELL inside the rock body.

    The reference rock fills the central band; flat grey background sits above
    (~top 30%) and below. Patches kept inside x[0.20,0.80], y[0.44,0.74];
    outliers (grey background) rejected by distance from the patch median.
    """
    h, w, _ = arr.shape
    centres = [(0.50, 0.56), (0.36, 0.52), (0.62, 0.58), (0.46, 0.66),
               (0.58, 0.50), (0.30, 0.60), (0.70, 0.62), (0.50, 0.46)]
    half = max(4, int(0.035 * min(h, w)))
    meds = []
    for fx, fy in centres:
        cx, cy = int(fx * w), int(fy * h)
        x0, x1 = max(0, cx - half), min(w, cx + half)
        y0, y1 = max(0, cy - half), min(h, cy + half)
        patch = arr[y0:y1, x0:x1].reshape(-1, 3)
        if len(patch):
            meds.append(np.median(patch, axis=0))
    meds = np.array(meds)
    glob = np.median(meds, axis=0)
    dist = np.linalg.norm(meds - glob, axis=1)
    keep = meds[dist < 55.0]
    if len(keep) < 2:
        keep = meds
    base = np.clip(np.median(keep, axis=0), 40, 230)

    # warm sandstone family derived from the sampled body colour
    light = np.clip(base * np.array([1.20, 1.15, 1.04]) + 22, 0, 255)  # sunlit bed
    dark = np.clip(base * np.array([0.55, 0.52, 0.48]), 0, 255)        # recess
    ochre = np.clip(base * np.array([1.08, 0.90, 0.62]), 0, 255)       # staining
    return dict(base=base, light=light, dark=dark, ochre=ochre)


def _build_albedo(palette, rng):
    """Seamless tileable sandstone: horizontal bedding + grain + ochre staining.

    Fully periodic by construction (no mirror-fold, no kaleidoscope artifact).
    Colours come from the photo-sampled palette; bedding runs along V so that,
    through triplanar UVs, the strata line up with world-Y on the rock's flanks.
    """
    n = TEX
    base = palette["base"]; light = palette["light"]
    dark = palette["dark"]; ochre = palette["ochre"]

    rows = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    rows = np.repeat(rows, n, axis=1)

    warp = _tileable_noise(n, 2.2, rng)        # gentle horizontal undulation
    grain = _tileable_noise(n, 1.1, rng)       # fine granular face texture
    mottle = _tileable_noise(n, 1.8, rng)      # broad tonal variation
    stain = _tileable_noise(n, 2.4, rng)       # ochre staining patches

    n_beds = 14.0
    band_coord = rows * n_beds + 0.55 * (warp - 0.5)
    band = 0.5 + 0.5 * np.cos(2.0 * np.pi * band_coord)        # 0..1 across beds

    # base colour, lit on bed faces, blended toward the recess tone in troughs
    bw = band[..., None]
    col = light[None, None, :] * bw + base[None, None, :] * (1.0 - bw)

    seam = (1.0 - band) ** 4                                    # sharp recess seams
    col = col * (1.0 - 0.45 * seam[..., None]) + dark[None, None, :] * (0.45 * seam[..., None])

    # crisp thin laminae (finer cracks between beds) -- row-based, stays tileable
    fine = 0.5 + 0.5 * np.cos(2.0 * np.pi * (rows * n_beds * 3.0 + warp))
    lam = fine ** 10
    col = col * (1.0 - 0.22 * lam[..., None])

    # granular mottling
    col = col * (0.88 + 0.24 * mottle[..., None])
    col = col * (0.94 + 0.12 * grain[..., None])

    # ochre mineral staining
    sm = (np.clip(stain - 0.45, 0.0, 1.0) * 0.6)[..., None]
    col = col * (1.0 - sm) + ochre[None, None, :] * sm

    col = np.clip(col, 0, 255).astype(np.uint8)
    return Image.fromarray(col, mode="RGB")


def _build_normal(albedo_img, strength=3.0):
    """Tangent-space normal map from albedo (height = inverse luminance)."""
    arr = np.asarray(albedo_img, dtype=np.float32)
    height = 1.0 - _luminance(arr) / 255.0
    height = np.asarray(
        Image.fromarray((height * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(0.6)), dtype=np.float32) / 255.0
    gy, gx = np.gradient(height)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(height)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    out = np.stack([nx * inv, ny * inv, nz * inv], axis=-1) * 0.5 + 0.5
    return Image.fromarray((out * 255.0).astype(np.uint8), mode="RGB")


def _triplanar_uv(mesh):
    """Bake triplanar projection into per-vertex UVs (dominant-axis pick).

    Side faces use world-Y as V so the texture's horizontal bedding lines up with
    the geometry's strata; the top (Y-dominant) uses X/Z and stays clean.
    """
    n = mesh.vertex_normals
    vtx = np.asarray(mesh.vertices)
    an = np.abs(n)
    uv = np.zeros((len(vtx), 2))
    uv[:, 0] = vtx[:, 0]        # default: Y-dominant -> (x, z)
    uv[:, 1] = vtx[:, 2]
    xdom = (an[:, 0] >= an[:, 1]) & (an[:, 0] >= an[:, 2])
    zdom = (an[:, 2] >= an[:, 0]) & (an[:, 2] > an[:, 1])
    uv[xdom, 0] = vtx[xdom, 2]; uv[xdom, 1] = vtx[xdom, 1]
    uv[zdom, 0] = vtx[zdom, 0]; uv[zdom, 1] = vtx[zdom, 1]
    return uv / TILE_M


def _vertex_tints(mesh, rng):
    """Sun/shade COLOR_0 tints: sunlit crest warmer/brighter, low crevices darker."""
    vtx = np.asarray(mesh.vertices)
    nrm = mesh.vertex_normals
    y = vtx[:, 1]
    h = (y - y.min()) / max(1e-6, np.ptp(y))
    up = np.clip(nrm[:, 1], 0.0, 1.0)

    shade = 0.80 + 0.30 * h + 0.08 * up                # higher floor -> no mud
    jitter = rng.uniform(-0.035, 0.035, len(vtx))
    shade = np.clip(shade + jitter, 0.68, 1.16)

    warm = 0.06 * (h - 0.45)
    r = shade * (1.0 + warm)
    g = shade * (1.0 + 0.25 * warm)
    b = shade * (1.0 - warm)
    col = np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)
    rgba = np.empty((len(vtx), 4), dtype=np.uint8)
    rgba[:, :3] = (col * 255.0).astype(np.uint8)
    rgba[:, 3] = 255
    return rgba


def _texture_rock(mesh, image_arr, seed):
    rng = np.random.default_rng(seed + 9173)
    palette = _sample_palette(image_arr)
    albedo = _build_albedo(palette, rng)
    normal = _build_normal(albedo)

    uv = _triplanar_uv(mesh)
    material = trimesh.visual.material.PBRMaterial(
        name="sandstone",
        baseColorTexture=albedo,
        normalTexture=normal,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicFactor=0.0,
        roughnessFactor=0.95,
        doubleSided=False,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    mesh.visual.vertex_attributes["color"] = _vertex_tints(mesh, rng)
    return mesh


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Stratified sandstone rock -> GLB")
    parser.add_argument("--image", required=True, help="reference photo path")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--density", choices=["high", "med", "low"], default="high")
    parser.add_argument("--output", required=True, help="output .glb path")
    args = parser.parse_args()

    scene = build_mesh(args.seed, args.density)
    mesh = scene.geometry["rock"]
    image_arr = _load_image(args.image)
    _texture_rock(mesh, image_arr, args.seed)

    scene.export(args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)