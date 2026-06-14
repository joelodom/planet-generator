#!/usr/bin/env python3
"""
Procedural generator + texturer for a pale-blue crystalline mineral
hand-specimen ("blue-calcite-specimen", class: rock).

Pipeline:
    build_mesh()  -> fractured, cleaved wedge of stone (geometry only)
    texturing     -> palette sampled from the reference photo mapped through
                     a cool dark->body->frost ramp on ORGANIC tileable
                     value-noise (no mirror symmetry), with organic cracks,
                     druzy sparkle, a derived normal + roughness map,
                     triplanar UVs and gentle per-vertex AO/frost tints
    export        -> a single embedded-texture GLB

Only numpy, trimesh, PIL and the stdlib are used.  A given --seed is fully
deterministic.

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
# GEOMETRY
# ===========================================================================

# --- Proportions measured (by eye) off the reference image. ----------------
OVERALL_WIDTH_M = 0.20      # widest span in X of a collector hand-specimen
HEIGHT_OVER_WIDTH = 0.62    # roughly triangular wedge, clearly wider than tall
DEPTH_OVER_WIDTH = 0.55     # noticeably shallower than it is wide
PEAK_X_FRAC = 0.30          # tall blunt peak sits ~30% in from the left
RIGHT_TAPER = 0.48          # right side thins/shortens toward a sharp edge
TARGET_ASPECT = 1.42        # front width/height to match the photo (~1.45)

# Sculpting amounts (fractions of the mean radius unless noted).
NOISE_AMP_FRAC = 0.12       # broad lumpiness (kept low so facets stay flat)
RIDGE_AMP_FRAC = 0.05       # fine crystalline crease detail
SNAP_STRENGTH = 0.52        # strong collapse to layer planes -> stepped ledges
BASE_CUT_FRAC = 0.08        # bottom slice flattened so it rests on the ground


def _make_table(rng, size=256):
    return rng.random(size)


def _value_noise(points, table):
    """Coherent value noise in ~[0,1] sampled at `points` (N,3)."""
    mask = table.shape[0] - 1
    i0 = np.floor(points).astype(np.int64)
    f = points - i0
    u = f * f * (3.0 - 2.0 * f)
    ix, iy, iz = i0[:, 0], i0[:, 1], i0[:, 2]

    def corner(cx, cy, cz):
        h = (((ix + cx) * 73856093)
             ^ ((iy + cy) * 19349663)
             ^ ((iz + cz) * 83492791))
        return table[h & mask]

    c000, c100 = corner(0, 0, 0), corner(1, 0, 0)
    c010, c110 = corner(0, 1, 0), corner(1, 1, 0)
    c001, c101 = corner(0, 0, 1), corner(1, 0, 1)
    c011, c111 = corner(0, 1, 1), corner(1, 1, 1)

    x00 = c000 + (c100 - c000) * u[:, 0]
    x10 = c010 + (c110 - c010) * u[:, 0]
    x01 = c001 + (c101 - c001) * u[:, 0]
    x11 = c011 + (c111 - c011) * u[:, 0]
    y0 = x00 + (x10 - x00) * u[:, 1]
    y1 = x01 + (x11 - x01) * u[:, 1]
    return y0 + (y1 - y0) * u[:, 2]


def _fbm(points, table, octaves, base_freq, rng):
    total = np.zeros(len(points))
    amp, freq, norm = 1.0, base_freq, 0.0
    for _ in range(octaves):
        off = rng.uniform(-100.0, 100.0, 3)
        total += amp * _value_noise(points * freq + off, table)
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return total / norm


# --- Density presets.  faces = 20 * 4**subdiv (well within budgets):
#   high: subdiv 5 -> 20480 tris;  med: subdiv 4 -> 5120;  low: subdiv 3 -> 1280
def _density_params(density):
    presets = {
        "high": dict(subdiv=5, octaves=5, base_freq=1.8, n_cuts=3, layers=7),
        "med":  dict(subdiv=4, octaves=4, base_freq=1.8, n_cuts=3, layers=6),
        "low":  dict(subdiv=3, octaves=3, base_freq=1.6, n_cuts=2, layers=5),
    }
    return presets.get(density, presets["high"])


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    p = _density_params(density)

    # --- base sphere -> ellipsoid matching the measured proportions --------
    width = OVERALL_WIDTH_M * (1.0 + rng.uniform(-0.08, 0.08))
    rx = width * 0.5
    ry = width * HEIGHT_OVER_WIDTH * 0.5
    rz = width * DEPTH_OVER_WIDTH * 0.5
    mean_r = (rx + ry + rz) / 3.0

    ico = trimesh.creation.icosphere(subdivisions=p["subdiv"], radius=1.0)
    faces = ico.faces
    unit = ico.vertices / np.linalg.norm(ico.vertices, axis=1, keepdims=True)
    V = unit * np.array([rx, ry, rz])

    # --- wedge silhouette: tall blunt left, thin tapered right -------------
    t = (V[:, 0] - V[:, 0].min()) / max(np.ptp(V[:, 0]), 1e-6)
    t = np.clip((t - PEAK_X_FRAC) / (1.0 - PEAK_X_FRAC), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    V[:, 1] *= (1.0 - RIGHT_TAPER * t)                # shorten to the right
    V[:, 2] *= (1.0 - (RIGHT_TAPER * 0.7) * t)        # thin to the right
    V[:, 1] *= (1.0 + 0.16 * (1.0 - t) * np.clip(unit[:, 1], 0.0, 1.0))

    # --- diagonal cleavage stepping (the signature feature) ----------------
    L = np.array([0.26 + rng.uniform(-0.05, 0.05),
                  1.0,
                  0.12 + rng.uniform(-0.05, 0.05)])
    L /= np.linalg.norm(L)
    proj = V @ L
    step_h = max(np.ptp(proj) / p["layers"], 1e-6)
    proj_snapped = step_h * np.round(proj / step_h)
    shift = SNAP_STRENGTH * (proj_snapped - proj)
    V += np.outer(shift, L)                           # terrace into ledges

    # --- rough crystalline surface: fBm + ridged detail --------------------
    sample = V / mean_r
    base = (_fbm(sample, _make_table(rng), p["octaves"], p["base_freq"], rng)
            - 0.5) * 2.0
    ridged_raw = _fbm(sample, _make_table(rng), max(p["octaves"] - 1, 2),
                      p["base_freq"] * 2.2, rng)
    ridged = 1.0 - np.abs(2.0 * ridged_raw - 1.0)

    low = _value_noise(sample * 0.9, _make_table(rng))
    amp_mod = (0.55 + 0.9 * low) * (1.0 + 0.35 * np.clip(unit[:, 1], 0.0, 1.0))

    disp = (base * NOISE_AMP_FRAC + ridged * RIDGE_AMP_FRAC) * amp_mod * mean_r
    norm = np.linalg.norm(V, axis=1)
    out_dir = V / np.maximum(norm, 1e-6)[:, None]
    V += out_dir * disp[:, None]

    # --- big planar facet cuts: flat broken cleavage faces (angularity) ----
    a = rng.uniform(0.0, 2.0 * np.pi)
    planes = [
        # dominant sloping cleavage face across the top, dipping to the right
        (np.array([0.45, 0.85, rng.uniform(-0.12, 0.12)]), 0.72),
        # the thin right-hand broken edge
        (np.array([1.0, -0.30, rng.uniform(-0.25, 0.25)]), 0.78),
        # a large flat cleavage face on a flank / back
        (np.array([np.cos(a) * 0.95, 0.05, np.sin(a) * 0.95]), 0.82),
    ]
    for n, q in planes[:p["n_cuts"]]:
        n = n / np.linalg.norm(n)
        d = V @ n
        off = np.quantile(d, q)
        beyond = d > off
        V[beyond] -= np.outer(d[beyond] - off, n)     # project onto the plane

    # --- force the wide, flat front aspect to match the photo --------------
    xext = np.ptp(V[:, 0])
    yext = max(np.ptp(V[:, 1]), 1e-6)
    sy = np.clip((xext / TARGET_ASPECT) / yext, 0.6, 1.05)
    V[:, 1] *= sy

    # --- flatten the underside so it rests naturally -----------------------
    y_cut = V[:, 1].min() + BASE_CUT_FRAC * np.ptp(V[:, 1])
    V[V[:, 1] < y_cut, 1] = y_cut

    # --- seat on the XZ plane, centered near the origin --------------------
    V[:, 0] -= 0.5 * (V[:, 0].max() + V[:, 0].min())
    V[:, 2] -= 0.5 * (V[:, 2].max() + V[:, 2].min())
    V[:, 1] -= V[:, 1].min()

    mesh = trimesh.Trimesh(vertices=V, faces=faces, process=False)
    mesh.fix_normals()

    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="rock")
    return scene


# ===========================================================================
# TEXTURING
# ===========================================================================

TEX_RES = 1024          # albedo / normal resolution (>= 512, never 256)
UV_REPEATS = 2.0        # how many times the swatch tiles across the rock
FALLBACK_BODY = np.array([158.0, 200.0, 212.0])   # pale cyan-blue


def _sample_palette(arr):
    """Sample the rock's body colour from patches well inside the silhouette.

    Patch centres are hand-placed on the blue body (avoiding the grey
    background corners and the white crest).  Background-like patches are
    rejected; the median of the survivors is the body colour.
    """
    H, W = arr.shape[:2]

    def patch(cx, cy, half=3):
        r = int(round(cy * H))
        c = int(round(cx * W))
        r0, r1 = max(r - half, 0), min(r + half + 1, H)
        c0, c1 = max(c - half, 0), min(c + half + 1, W)
        return np.median(arr[r0:r1, c0:c1].reshape(-1, 3), axis=0)

    bg = np.median(np.stack([patch(0.03, 0.03), patch(0.97, 0.03),
                             patch(0.03, 0.97), patch(0.97, 0.97)]), axis=0)

    centres = [(0.34, 0.50), (0.42, 0.58), (0.50, 0.52), (0.46, 0.66),
               (0.56, 0.50), (0.38, 0.62), (0.52, 0.62), (0.44, 0.45),
               (0.60, 0.56), (0.30, 0.55)]
    cols = np.stack([patch(cx, cy) for cx, cy in centres])

    guess = np.median(cols, axis=0)
    keep = [c for c in cols
            if np.linalg.norm(c - bg) > 28.0 and np.linalg.norm(c - guess) < 75.0]
    body = np.median(np.stack(keep), axis=0) if len(keep) >= 3 else guess

    if body.mean() < 60 or body.mean() > 245:
        body = FALLBACK_BODY.copy()
    # keep it cool: never let the body skew warm (red dominant)
    body[0] = min(body[0], body[2])
    return body.astype(float)


def _tile_vnoise(res, rng, base_period=7, octaves=6):
    """Organic, seamlessly tileable value-noise fBm in ~[0,1].

    Each octave interpolates a small periodic lattice with wrap-around
    indexing, so the result tiles exactly with no mirror symmetry.
    """
    out = np.zeros((res, res))
    coords = np.arange(res) / res
    amp, norm, per = 1.0, 0.0, int(base_period)
    for _ in range(octaves):
        g = rng.random((per, per))
        fx = coords * per
        i0 = np.floor(fx).astype(int) % per
        i1 = (i0 + 1) % per
        tw = fx - np.floor(fx)
        tw = tw * tw * (3.0 - 2.0 * tw)               # smoothstep

        gA = g[np.ix_(i0, i0)]
        gB = g[np.ix_(i0, i1)]
        gC = g[np.ix_(i1, i0)]
        gD = g[np.ix_(i1, i1)]
        TX = tw[None, :]
        TY = tw[:, None]
        top = gA * (1.0 - TX) + gB * TX
        bot = gC * (1.0 - TX) + gD * TX
        out += amp * (top * (1.0 - TY) + bot * TY)

        norm += amp
        amp *= 0.5
        per *= 2
    return out / norm


def _build_textures(body, rng):
    """Return (albedo, normal, metalrough) PIL images + palette tones."""
    res = TEX_RES
    body = np.clip(body, 0, 255)
    frost = np.clip(body + (255.0 - body) * 0.55, 0, 255)   # frosty highlight
    dark = np.clip(body * 0.62, 0, 255)                     # cool recess shadow
    white = np.array([238.0, 244.0, 247.0])                 # druzy sparkle

    # --- organic value-noise drives all spatial structure (no symmetry) ----
    mottle = _tile_vnoise(res, rng, base_period=7, octaves=6)
    mottle = np.clip((mottle - 0.5) * 1.5 + 0.5, 0.0, 1.0)  # widen contrast

    ridged = 1.0 - np.abs(2.0 * _tile_vnoise(res, rng, 5, 5) - 1.0)
    cracks = np.clip((ridged - 0.86) / 0.14, 0.0, 1.0)      # thin organic fissures

    speckle = np.clip((_tile_vnoise(res, rng, 34, 2) - 0.74) / 0.26, 0.0, 1.0)

    # --- albedo: map the noise through a cool dark->body->frost ramp -------
    a = mottle[..., None]
    lo = dark + (body - dark) * (a / 0.5)
    hi = body + (frost - body) * ((a - 0.5) / 0.5)
    albedo = np.where(a < 0.5, lo, hi)

    cm = (cracks * 0.5)[..., None]                          # darken fissures
    albedo = albedo * (1.0 - cm) + dark * cm
    sp = (speckle * 0.30)[..., None]                        # frosty sparkle
    albedo = albedo * (1.0 - sp) + white * sp
    albedo = np.clip(albedo, 0, 255)

    # --- normal map from inverse-luminance height (wrapped Sobel) ----------
    lum = (albedo @ np.array([0.299, 0.587, 0.114])) / 255.0
    height = 1.0 - lum
    gx = np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)
    gy = np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)
    strength = 2.4
    nx, ny, nz = -gx * strength, -gy * strength, np.ones_like(height)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    normal = np.stack([nx * inv, ny * inv, nz * inv], axis=-1)
    normal_img = np.clip((normal * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

    # --- metallic-roughness: waxy faces, rougher cracks/druzy --------------
    rough = np.clip(0.82 + 0.14 * cracks + 0.10 * speckle, 0.55, 0.96)
    mr = np.zeros((res, res, 3))
    mr[..., 1] = rough * 255.0          # G = roughness
    mr[..., 2] = 0.0                    # B = metallic (none)

    return (Image.fromarray(albedo.astype(np.uint8), mode="RGB"),
            Image.fromarray(normal_img, mode="RGB"),
            Image.fromarray(mr.astype(np.uint8), mode="RGB"),
            frost, body, dark)


def _triplanar_uv(mesh, freq):
    """Per-vertex triplanar UVs baked from world position by dominant axis."""
    v = mesh.vertices
    n = np.abs(mesh.vertex_normals)
    ax = np.argmax(n, axis=1)
    uv = np.zeros((len(v), 2))
    m0, m1, m2 = ax == 0, ax == 1, ax == 2
    uv[m0] = v[m0][:, [2, 1]]
    uv[m1] = v[m1][:, [0, 2]]
    uv[m2] = v[m2][:, [0, 1]]
    return uv * freq


def _vertex_tints(mesh, rng):
    """Gentle COLOR_0 tints: near-white crests, mild cool AO in recesses."""
    v = mesh.vertices
    nrm = mesh.vertex_normals
    h = v[:, 1] / max(v[:, 1].max(), 1e-6)
    ny = nrm[:, 1]

    exposed = np.clip(0.55 * h + 0.55 * np.clip(ny, 0.0, 1.0) - 0.10, 0.0, 1.0)
    darkw = np.clip(0.6 * np.clip(1.0 - h * 1.6, 0.0, 1.0)
                    + 0.5 * np.clip(-ny, 0.0, 1.0), 0.0, 1.0)

    shade = np.clip(0.84 + 0.16 * exposed - 0.30 * darkw, 0.58, 1.0)
    cols = np.empty((len(v), 4), dtype=np.uint8)
    r = shade * (1.0 - 0.06 * darkw)                 # recesses cooler (less red)
    g = shade
    b = np.minimum(shade * (1.0 + 0.05 * darkw), 1.0)
    jit = rng.uniform(-0.03, 0.03, len(v))
    cols[:, 0] = np.clip((r + jit) * 255.0, 0, 255).astype(np.uint8)
    cols[:, 1] = np.clip((g + jit) * 255.0, 0, 255).astype(np.uint8)
    cols[:, 2] = np.clip((b + jit) * 255.0, 0, 255).astype(np.uint8)
    cols[:, 3] = 255
    return cols


def texture_scene(scene, image_path, seed):
    """Sample the photo palette, build materials, attach them to the rock."""
    rng = np.random.default_rng(seed + 9973)

    try:
        arr = np.asarray(Image.open(image_path).convert("RGB")).astype(float)
        body = _sample_palette(arr)
    except Exception:
        body = FALLBACK_BODY.copy()

    albedo_img, normal_img, mr_img, frost, body, dark = _build_textures(body, rng)

    material = trimesh.visual.material.PBRMaterial(
        name="blue_calcite",
        baseColorTexture=albedo_img,
        normalTexture=normal_img,
        metallicRoughnessTexture=mr_img,
        metallicFactor=0.0,
        roughnessFactor=0.88,        # stone, with a faint glassy/waxy sheen
        doubleSided=False,
        alphaMode="OPAQUE",
    )

    for mesh in scene.geometry.values():
        ext = np.ptp(mesh.vertices, axis=0)
        freq = UV_REPEATS / max(ext.mean(), 1e-6)
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=_triplanar_uv(mesh, freq), material=material)
        mesh.visual.vertex_attributes["color"] = _vertex_tints(mesh, rng)

    return scene


# ===========================================================================
# CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Generate a textured blue-calcite specimen GLB.")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, required=True, help="deterministic seed")
    ap.add_argument("--density", choices=["high", "med", "low"],
                    default="high", help="polygon / detail level")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        scene = build_mesh(args.seed, args.density)
        scene = texture_scene(scene, args.image, args.seed)
        glb = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(glb)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())