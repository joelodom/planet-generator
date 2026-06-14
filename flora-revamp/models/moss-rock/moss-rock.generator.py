"""Procedural mossy granite fieldstone boulder: geometry + photo-derived
materials + textured GLB export, in one standalone script.

A single squat, weathered, dome-like boulder. All geometry is returned under
the single semantic surface name "rock"; the three materials called out in the
description -- bare pale granite, a vivid green moss blanket on the upper
hemisphere, and scattered chalky lichen rosettes -- are produced by:
  * a tileable, PALE granite albedo (de-lit, recoloured from a real photo crop,
    with procedural mineral mottle, cracks and PIL-drawn lichen rosettes),
  * a gentle derived tangent-space normal map,
  * per-vertex COLOR_0 tints that paint the macro moss/stone distribution
    (vivid green on the top and in hollows, bare grey on the lower / right
    flanks), multiplying the granite texture in glTF.

CLI:
    python thisscript.py --image PATH --seed INT --density {high,med,low} \
        --output OUT.glb
"""

import argparse
import math
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter


# ==========================================================================
# GEOMETRY  (build_mesh)
# ==========================================================================
BOULDER_WIDTH = 1.20            # meters, overall left-right extent (X)
HEIGHT_OVER_WIDTH = 1.00       # near-round dome; final aspect is forced below
DEPTH_OVER_WIDTH = 0.94        # front-back (Z) slightly less than width
BASE_TAPER = 0.10              # base ~10% narrower than the bulging top
UNDERSIDE_CUT = 0.12          # bottom 12% gets flattened so it sits flat
UNDERSIDE_COMPRESS = 0.45     # how hard that bottom slab is squashed

# Force the FRONT silhouette to match the photo (width/height ~= 1.09).
# Kept a touch above 1.0 so it still reads "broader than tall".
TARGET_FRONT_ASPECT = 1.10

BOULDER_HEIGHT = BOULDER_WIDTH * HEIGHT_OVER_WIDTH
BOULDER_DEPTH = BOULDER_WIDTH * DEPTH_OVER_WIDTH

RADIUS_NOISE = 0.20
DIR_ASYMMETRY = 0.35          # asymmetry: amplitude varies by direction
FACET_STRENGTH = 0.20         # shallow worn shoulders, NOT glassy gem facets

# How many meters of surface one texture tile spans (triplanar projection).
TILE_METERS = 0.50

_PRESETS = {
    "high": dict(subdiv=5, octaves=4, facets=1),
    "med":  dict(subdiv=4, octaves=3, facets=0),
    "low":  dict(subdiv=3, octaves=2, facets=0),
}


def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def _value_noise(points, freq, rng):
    """Tileable trilinear value noise sampled at `points`, returns [0,1]."""
    res = int(freq)
    grid = rng.random((res, res, res))
    p = points * freq
    p0 = np.floor(p).astype(np.int64)
    f = _smoothstep(p - p0)

    ix0 = p0[:, 0] % res
    iy0 = p0[:, 1] % res
    iz0 = p0[:, 2] % res
    ix1 = (p0[:, 0] + 1) % res
    iy1 = (p0[:, 1] + 1) % res
    iz1 = (p0[:, 2] + 1) % res
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]

    c000 = grid[ix0, iy0, iz0]
    c100 = grid[ix1, iy0, iz0]
    c010 = grid[ix0, iy1, iz0]
    c110 = grid[ix1, iy1, iz0]
    c001 = grid[ix0, iy0, iz1]
    c101 = grid[ix1, iy0, iz1]
    c011 = grid[ix0, iy1, iz1]
    c111 = grid[ix1, iy1, iz1]

    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


def _fbm(points, octaves, rng, base_freq=2):
    """Fractal value noise -> roughly [-0.5, 0.5]."""
    total = np.zeros(len(points))
    amp = 1.0
    freq = base_freq
    norm = 0.0
    for o in range(octaves):
        total += amp * (_value_noise(points + (o + 1) * 7.31, freq, rng) - 0.5)
        norm += amp
        amp *= 0.5
        freq *= 2
    return total / norm


def _apply_facet(verts, normal, strength, rng):
    """Gently flatten only the outermost cap past a plane -> a worn shoulder.
    Deliberately shallow so it never reads as a cut-gem facet."""
    n = normal / np.linalg.norm(normal)
    proj = verts @ n
    d = np.quantile(proj, 0.85)        # touch only the outermost ~15%
    s = proj - d
    mask = s > 0
    if mask.any():
        smax = s[mask].max() + 1e-9
        ramp = _smoothstep(np.clip(s[mask] / smax, 0.0, 1.0))   # soft, no crease
        verts[mask] -= np.outer(s[mask] * strength * ramp, n)
    return verts


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    cfg = _PRESETS.get(density, _PRESETS["high"])

    # 1. Base sphere; its unit-length vertices double as surface directions.
    ico = trimesh.creation.icosphere(subdivisions=cfg["subdiv"], radius=1.0)
    dirs = np.asarray(ico.vertices, dtype=np.float64)
    faces = np.asarray(ico.faces)

    # 2. Lumpy radial displacement with direction-dependent amplitude so the
    #    mass is asymmetric (one flank bulges more than the other).
    asym_axis = rng.normal(size=3)
    asym_axis /= np.linalg.norm(asym_axis)
    dir_gain = 1.0 + DIR_ASYMMETRY * (dirs @ asym_axis)
    radius = 1.0 + RADIUS_NOISE * _fbm(dirs, cfg["octaves"], rng) * dir_gain
    V = dirs * radius[:, None]

    # 3. One or two SHALLOW facet cuts for worn flats (bias away from the
    #    underside so we never flatten the face the boulder rests on).
    for _ in range(cfg["facets"]):
        n = rng.normal(size=3)
        if n[1] < -0.25:
            n[1] = -n[1]
        V = _apply_facet(V, n, FACET_STRENGTH, rng)

    # 4. Squash the sphere to roughly the dome proportions.
    V[:, 0] *= BOULDER_WIDTH * 0.5
    V[:, 1] *= BOULDER_HEIGHT * 0.5
    V[:, 2] *= BOULDER_DEPTH * 0.5

    # 5. Vertical taper: base slightly narrower than the bulging top.
    y = V[:, 1]
    span = np.ptp(y)
    t = (y - y.min()) / (span + 1e-9)          # 0 at base, 1 at top
    taper = 1.0 - BASE_TAPER * (1.0 - t)
    V[:, 0] *= taper
    V[:, 2] *= taper

    # 6. Flatten the underside slightly so it sits naturally on the ground.
    y = V[:, 1]
    ymin = y.min()
    span = np.ptp(y)
    cut = ymin + UNDERSIDE_CUT * span
    below = y < cut
    V[below, 1] = cut - (cut - y[below]) * UNDERSIDE_COMPRESS

    # 7. Force the front silhouette aspect (width/height) to match the photo,
    #    then stand on the XZ plane centered in X/Z.
    ext = V.max(axis=0) - V.min(axis=0)
    desired_h = ext[0] / TARGET_FRONT_ASPECT
    V[:, 1] *= desired_h / max(ext[1], 1e-9)

    V[:, 1] -= V[:, 1].min()
    bmin = V.min(axis=0)
    bmax = V.max(axis=0)
    V[:, 0] -= 0.5 * (bmin[0] + bmax[0])
    V[:, 2] -= 0.5 * (bmin[2] + bmax[2])

    mesh = trimesh.Trimesh(vertices=V, faces=faces, process=False)
    mesh.fix_normals()

    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="rock")
    return scene


# ==========================================================================
# PHOTO SAMPLING HELPERS
# ==========================================================================
def _lum(a):
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def _crop_frac(arr, cx, cy, half):
    h, w = arr.shape[:2]
    x0 = max(0, int((cx - half) * w))
    x1 = min(w, int((cx + half) * w))
    y0 = max(0, int((cy - half) * h))
    y1 = min(h, int((cy + half) * h))
    if x1 <= x0 or y1 <= y0:
        return arr[0:1, 0:1].copy()
    return arr[y0:y1, x0:x1].copy()


def _corner_bg(arr):
    """Estimate the neutral background colour from the four image corners."""
    h, w = arr.shape[:2]
    s = max(2, min(h, w) // 18)
    patches = [arr[:s, :s], arr[:s, -s:], arr[-s:, :s], arr[-s:, -s:]]
    flat = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    return np.median(flat, axis=0)


def _sample_color(arr, centers, bg, half=0.045):
    """Median of several small in-silhouette patches, discarding any that
    land on (or near) the background colour."""
    cols = []
    for cx, cy in centers:
        p = _crop_frac(arr, cx, cy, half).reshape(-1, 3)
        if p.size == 0:
            continue
        med = np.median(p, axis=0)
        if np.linalg.norm(med - bg) < 22.0:      # likely caught background
            continue
        cols.append(med)
    if not cols:
        p = _crop_frac(arr, 0.5, 0.5, 0.12).reshape(-1, 3)
        return np.median(p, axis=0)
    return np.median(np.asarray(cols), axis=0)


def _sample_light(arr, centers, bg, half=0.04):
    """Sample the brightest texels of each patch -- used for chalky lichen."""
    vals = []
    for cx, cy in centers:
        p = _crop_frac(arr, cx, cy, half).reshape(-1, 3)
        if p.size == 0:
            continue
        lum = p.mean(axis=1)
        thr = np.percentile(lum, 75)
        bright = p[lum >= thr]
        if bright.size == 0:
            continue
        med = np.median(bright, axis=0)
        if med.mean() <= bg.mean() + 4.0:
            continue
        vals.append(med)
    if not vals:
        return np.array([208.0, 209.0, 203.0])
    return np.median(np.asarray(vals), axis=0)


# Sampling regions chosen by LOOKING at reference.png (fractions of W,H),
# all placed well inside the boulder silhouette.
MOSS_REGIONS = [(0.45, 0.30), (0.55, 0.40), (0.40, 0.46),
                (0.60, 0.32), (0.50, 0.55)]
GRANITE_REGIONS = [(0.15, 0.66), (0.20, 0.80), (0.80, 0.60),
                   (0.78, 0.76), (0.50, 0.90)]
LICHEN_REGIONS = [(0.50, 0.33), (0.45, 0.42), (0.58, 0.45), (0.40, 0.36)]


# ==========================================================================
# TEXTURE SYNTHESIS
# ==========================================================================
def _delight(patch):
    """Flatten baked-in lighting by dividing out a heavily blurred luminance,
    clamping the gain to [0.6, 1.6]."""
    lum = _lum(patch.clip(0, 255))
    radius = max(3, min(patch.shape[0], patch.shape[1]) // 3)
    blur = np.asarray(
        Image.fromarray(lum.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=radius)),
        dtype=np.float64)
    blur = np.clip(blur, 1.0, 255.0)
    gain = np.clip(blur.mean() / blur, 0.6, 1.6)
    return np.clip(patch * gain[..., None], 0, 255)


def _make_tileable(patch, res):
    """Mirror-fold a swatch into a seamless res x res tile, softening only an
    ~8px band at each fold."""
    half = max(8, res // 2)
    small = Image.fromarray(patch.clip(0, 255).astype(np.uint8)).resize(
        (half, half), Image.LANCZOS)
    a = np.asarray(small, dtype=np.float64)
    top = np.concatenate([a, a[:, ::-1]], axis=1)
    full = np.concatenate([top, top[::-1, :]], axis=0)
    if full.shape[0] != res or full.shape[1] != res:
        full = np.asarray(
            Image.fromarray(full.astype(np.uint8)).resize(
                (res, res), Image.LANCZOS), dtype=np.float64)
    blurred = np.asarray(
        Image.fromarray(full.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=2)), dtype=np.float64)
    band = 8
    c = res // 2
    full[c - band:c + band, :] = blurred[c - band:c + band, :]
    full[:, c - band:c + band] = blurred[:, c - band:c + band]
    return full


def _draw_lichen(res, rng, count, color, supersample=4):
    """Draw scattered chalky foliose-lichen rosettes onto a transparent RGBA
    layer (supersampled then LANCZOS-downscaled for clean, near-binary edges)."""
    big = res * supersample
    layer = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cr, cg, cb = int(color[0]), int(color[1]), int(color[2])
    for _ in range(count):
        cx = int(rng.integers(0, big))
        cy = int(rng.integers(0, big))
        rad = float(rng.uniform(0.010, 0.032) * big)
        alpha = int(rng.integers(85, 165))
        lobes = int(rng.integers(6, 13))
        for _l in range(lobes):
            ang = rng.random() * 2.0 * math.pi
            dist = rad * (0.2 + 0.8 * rng.random())
            bx = cx + math.cos(ang) * dist
            by = cy + math.sin(ang) * dist
            br = rad * rng.uniform(0.18, 0.42)
            jit = rng.integers(-10, 10, size=3)
            col = (int(np.clip(cr + jit[0], 0, 255)),
                   int(np.clip(cg + jit[1], 0, 255)),
                   int(np.clip(cb + jit[2], 0, 255)),
                   alpha)
            d.ellipse([bx - br, by - br, bx + br, by + br], fill=col)
    return layer.resize((res, res), Image.LANCZOS)


def _normal_from_albedo(rgb, strength=1.1):
    """Gentle tangent-space normal map: height = inverse (blurred) luminance,
    central-difference gradients -> encoded normal. Kept soft so the surface
    does not read as dark fuzz."""
    lum = _lum(rgb)
    lum = np.asarray(
        Image.fromarray(lum.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=1.4)), dtype=np.float64)
    h = 1.0 - lum / 255.0
    gx = np.zeros_like(h)
    gy = np.zeros_like(h)
    gx[:, 1:-1] = (h[:, 2:] - h[:, :-2]) * 0.5
    gy[1:-1, :] = (h[2:, :] - h[:-2, :]) * 0.5
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(h)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    out = np.empty((h.shape[0], h.shape[1], 3), dtype=np.float64)
    out[..., 0] = (nx * inv) * 0.5 + 0.5
    out[..., 1] = (ny * inv) * 0.5 + 0.5
    out[..., 2] = (nz * inv) * 0.5 + 0.5
    return Image.fromarray((out * 255.0).clip(0, 255).astype(np.uint8), "RGB")


def build_granite_textures(arr, bg, granite_rgb, lichen_rgb, seed, res):
    """Return (albedo PIL, normal PIL) for the granite surface. Kept PALE and
    fairly achromatic so the per-vertex moss tint can drive the green zones
    without multiplying down to mud."""
    rng = np.random.default_rng(seed + 101)

    # --- real photo detail, de-lit and made tileable ---
    patch = _crop_frac(arr, 0.17, 0.74, 0.12)        # clean lower-left granite
    patch = _delight(patch)
    tiled = _make_tileable(patch, res)
    lum = _lum(tiled)
    norm = np.clip(lum / max(lum.mean(), 1e-6), 0.84, 1.16)   # gentle contrast

    # --- pale, desaturated granite base colour (lifted so it never goes dark) ---
    g = granite_rgb.astype(np.float64)
    g = g * 0.50 + g.mean() * 0.50                    # strong desaturate -> grey
    g = np.clip(g * 1.12, 152.0, 208.0)              # pale cool grey
    stone = g[None, None, :] * norm[..., None]

    # --- low-frequency mineral mottle (subtle) ---
    small = rng.random((16, 16))
    mott = np.asarray(
        Image.fromarray((small * 255).astype(np.uint8)).resize(
            (res, res), Image.BILINEAR).filter(ImageFilter.GaussianBlur(10)),
        dtype=np.float64) / 255.0
    stone *= (0.92 + 0.16 * mott)[..., None]

    # --- fine granular speckle (small, so the normal map stays calm) ---
    stone += rng.normal(0.0, 4.0, size=stone.shape)
    stone = np.clip(stone, 0, 255)

    base = Image.fromarray(stone.astype(np.uint8), "RGB").convert("RGBA")

    # --- scattered chalky lichen rosettes (on stone AND moss zones) ---
    lichen_count = max(18, res // 16)
    lichen_layer = _draw_lichen(res, rng, lichen_count, lichen_rgb)
    base = Image.alpha_composite(base, lichen_layer)

    # --- a few thin dark cracks / veins ---
    crack_layer = Image.new("RGBA", (res, res), (0, 0, 0, 0))
    dc = ImageDraw.Draw(crack_layer)
    for _ in range(max(3, res // 220)):
        x = float(rng.integers(0, res))
        y = float(rng.integers(0, res))
        ang = rng.random() * 2.0 * math.pi
        pts = [(x, y)]
        for _s in range(int(rng.integers(6, 16))):
            ang += rng.normal(0.0, 0.5)
            x += math.cos(ang) * rng.uniform(4, 13)
            y += math.sin(ang) * rng.uniform(4, 13)
            pts.append((x, y))
        a = int(rng.integers(35, 80))
        dc.line(pts, fill=(40, 37, 32, a), width=int(rng.integers(1, 3)))
    base = Image.alpha_composite(base, crack_layer)

    albedo = base.convert("RGB")
    normal = _normal_from_albedo(np.asarray(albedo, dtype=np.float64))
    return albedo, normal


# ==========================================================================
# UVs, VERTEX COLOURS, MATERIAL
# ==========================================================================
def _triplanar_uv(verts, normals):
    """Per-vertex triplanar UVs: project along each vertex's dominant axis."""
    an = np.abs(normals)
    ax = np.argmax(an, axis=1)
    uv = np.zeros((len(verts), 2), dtype=np.float64)
    mx, my, mz = ax == 0, ax == 1, ax == 2
    uv[mx, 0] = verts[mx, 2]; uv[mx, 1] = verts[mx, 1]
    uv[my, 0] = verts[my, 0]; uv[my, 1] = verts[my, 2]
    uv[mz, 0] = verts[mz, 0]; uv[mz, 1] = verts[mz, 1]
    return (uv / TILE_METERS).astype(np.float32)


def _vivid_moss(moss_rgb):
    """Take the sampled photo moss colour and push it to a vivid, bright
    yellow-green so the texture multiply reads clearly as moss."""
    c = moss_rgb / 255.0
    mean = c.mean()
    c = mean + (c - mean) * 1.7                       # boost saturation
    c = np.clip(c, 0.0, 1.0)
    mx = c.max()
    if mx > 1e-6:
        c = c / mx * 0.98                             # brighten near full value
    return c


def _moss_vertex_colors(mesh, moss_rgb, seed):
    """COLOR_0 tints: vivid green moss on the top / in hollows, bare grey on
    the lower and right flanks, with gentle damp-shade AO toward the base."""
    rng = np.random.default_rng(seed + 202)
    V = mesh.vertices
    N = mesh.vertex_normals
    y = V[:, 1]
    hf = y / max(y.max(), 1e-6)                       # 0 at base, 1 at top
    up = np.clip(N[:, 1], 0.0, 1.0)
    rx = np.clip(V[:, 0] / (BOULDER_WIDTH * 0.5), -1.0, 1.0)

    noise = _fbm(V * 2.0, 3, rng)                     # wispy creep / fingers

    # moss coverage factor -- generous so the top reads predominantly green
    m = 0.45 * hf + 0.60 * up + 0.40 * noise + 0.28
    m -= 0.40 * np.clip(rx, 0.0, 1.0) * (1.0 - hf)    # bare lower-RIGHT flank
    m = np.clip(m, 0.0, 1.0)
    m = m * m * (3.0 - 2.0 * m)                       # smoothstep

    # damp, shaded grounding: only mildly darker low (keep moss bright)
    ao = np.clip(0.74 + 0.30 * hf, 0.74, 1.0)
    ao *= (0.94 + 0.06 * (noise + 0.5))

    moss01 = _vivid_moss(moss_rgb)
    stone_tint = np.array([0.97, 0.96, 0.93])         # let pale granite show

    col = stone_tint[None, :] * (1.0 - m)[:, None] + moss01[None, :] * m[:, None]
    col *= ao[:, None]
    col += rng.normal(0.0, 0.015, col.shape)          # subtle per-clump jitter
    col = np.clip(col, 0.0, 1.0)

    rgba = np.empty((len(V), 4), dtype=np.uint8)
    rgba[:, :3] = (col * 255.0).astype(np.uint8)
    rgba[:, 3] = 255
    return rgba


def texture_scene(scene, arr, seed, density):
    """Attach photo-derived granite material + moss vertex tints to the rock."""
    bg = _corner_bg(arr)
    granite_rgb = _sample_color(arr, GRANITE_REGIONS, bg)
    moss_rgb = _sample_color(arr, MOSS_REGIONS, bg)
    lichen_rgb = _sample_light(arr, LICHEN_REGIONS, bg)

    res = 1024 if density == "high" else 512
    albedo, normal = build_granite_textures(
        arr, bg, granite_rgb, lichen_rgb, seed, res)

    mesh = scene.geometry["rock"]
    uv = _triplanar_uv(mesh.vertices, mesh.vertex_normals)
    colors = _moss_vertex_colors(mesh, moss_rgb, seed)

    material = trimesh.visual.material.PBRMaterial(
        name="rock_mossy_granite",
        baseColorTexture=albedo,
        normalTexture=normal,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0,
        roughnessFactor=0.92,
        doubleSided=False,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    mesh.visual.vertex_attributes["color"] = colors   # exports as COLOR_0

    out = trimesh.Scene()
    out.add_geometry(mesh, geom_name="rock")
    return out


# ==========================================================================
# CLI
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate a textured mossy granite boulder GLB.")
    parser.add_argument("--image", required=True, help="reference photo path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--density", choices=["high", "med", "low"],
                        default="high")
    parser.add_argument("--output", required=True, help="output .glb path")
    args = parser.parse_args()

    try:
        img = Image.open(args.image).convert("RGB")
        arr = np.asarray(img, dtype=np.float64)

        scene = build_mesh(args.seed, args.density)
        scene = texture_scene(scene, arr, args.seed, args.density)

        glb = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(glb)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()