"""
Procedural granite boulder: geometry + photo-derived tileable material + GLB.

Pipeline
--------
1. build_mesh(seed, density)  -> faceted, trapezoidal grey-stone chunk (+Y up,
   base at y=0, meters).
2. From the SOURCE PHOTO derive a de-lit, EDGE-seamless (non-symmetric) stone
   albedo at the photo's own grey, then add subtle procedural mottling /
   hairline cracks / mineral specks.  A tangent-space normal map is derived.
3. Triplanar per-vertex UVs project the swatch onto the rock; a COLOR_0
   vertex tint deepens crevices and the ground-line for soft AO.
4. Export a single textured binary .glb.

Only numpy, trimesh, PIL and the stdlib are used.

CLI:
    python thisscript --image PATH --seed INT --density {high,med,low} --output OUT.glb
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter


# ===========================================================================
# GEOMETRY
# ===========================================================================

# ---------------------------------------------------------------------------
# Proportions measured by eye off the reference image (~10% accuracy).
# Front-view target width/height aspect ~= 1.03 (photo content aspect).
# ---------------------------------------------------------------------------
OVERALL_HEIGHT = 0.80          # meters; a chunky landscaping / scree boulder
HEIGHT_OVER_WIDTH = 0.97       # silhouette is a touch wider than tall (~1.03 w/h)
DEPTH_OVER_WIDTH = 0.90        # a touch shallower front-to-back than wide
CROWN_TAPER = 0.80             # crown ~80% of base width: broad, blocky top (not a point)
BASE_FLATTEN_FRACTION = 0.12   # lowest ~12% of height sheared to a flat base

# Base, untapered axis scales applied to the unit sphere before noise/taper.
# Widened from the prior pass to lift front aspect 0.92 -> ~1.03.
SCALE_X = 1.03
SCALE_Y = 1.00
SCALE_Z = 0.92


def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def _value_noise(points, rng, lattice):
    """Smooth value noise sampled at `points` (expected roughly in [-1, 1]).

    A fresh random lattice is drawn from `rng` on every call, so call order
    is what makes the result deterministic for a given seed.
    """
    L = int(lattice)
    grid = rng.standard_normal((L + 1, L + 1, L + 1))

    p = (points * 0.5 + 0.5) * L
    p = np.clip(p, 0.0, L - 1e-6)
    i0 = np.floor(p).astype(np.int64)
    f = p - i0
    i1 = i0 + 1
    w = _smoothstep(f)

    x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
    x1, y1, z1 = i1[:, 0], i1[:, 1], i1[:, 2]

    c000 = grid[x0, y0, z0]; c100 = grid[x1, y0, z0]
    c010 = grid[x0, y1, z0]; c110 = grid[x1, y1, z0]
    c001 = grid[x0, y0, z1]; c101 = grid[x1, y0, z1]
    c011 = grid[x0, y1, z1]; c111 = grid[x1, y1, z1]

    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
    c00 = c000 * (1 - wx) + c100 * wx
    c10 = c010 * (1 - wx) + c110 * wx
    c01 = c001 * (1 - wx) + c101 * wx
    c11 = c011 * (1 - wx) + c111 * wx
    c0 = c00 * (1 - wy) + c10 * wy
    c1 = c01 * (1 - wy) + c11 * wy
    return c0 * (1 - wz) + c1 * wz


def _density_params(density):
    # More facets than the prior pass -> blockier, flatter, more angular reads.
    if density == "low":
        return dict(subdivisions=3, octaves=3, hollows=1, facets=3)   # 1280 tris
    if density == "med":
        return dict(subdivisions=4, octaves=4, hollows=2, facets=4)   # 5120 tris
    return dict(subdivisions=5, octaves=5, hollows=3, facets=5)       # 20480 tris


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    if density not in ("high", "med", "low"):
        density = "high"
    p = _density_params(density)

    ico = trimesh.creation.icosphere(subdivisions=p["subdivisions"], radius=1.0)
    unit = np.asarray(ico.vertices, dtype=np.float64)
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    faces = np.asarray(ico.faces)

    dir_amp = 1.0 + 0.55 * _value_noise(unit, rng, lattice=2)
    r = np.ones(len(unit))
    amp = 0.11
    lattice = 2
    for _ in range(p["octaves"]):
        r += dir_amp * amp * _value_noise(unit, rng, lattice=lattice)
        amp *= 0.52
        lattice *= 2

    for _ in range(p["hollows"]):
        cdir = rng.standard_normal(3)
        cdir /= np.linalg.norm(cdir)
        ang = np.arccos(np.clip(unit @ cdir, -1.0, 1.0))
        width = rng.uniform(0.40, 0.62)
        depth = rng.uniform(0.06, 0.13)
        r -= depth * np.exp(-(ang / width) ** 2)

    r = np.maximum(r, 0.45)
    V = unit * r[:, None]

    V[:, 0] *= SCALE_X
    V[:, 1] *= SCALE_Y
    V[:, 2] *= SCALE_Z

    ymin, ymax = V[:, 1].min(), V[:, 1].max()
    h = (V[:, 1] - ymin) / max(ymax - ymin, 1e-9)
    taper = 1.0 - (1.0 - CROWN_TAPER) * h
    V[:, 0] *= taper
    V[:, 2] *= taper

    # Large planar facet cuts: bigger flats (lower percentile) and normals that
    # span horizontal->upward, so we carve broad flat SIDE faces and slanted
    # upper faces meeting at sharp arrises -- never the underside.
    for _ in range(p["facets"]):
        n = rng.standard_normal(3)
        n[1] = rng.uniform(-0.10, 0.85)              # mostly side / upward
        n /= np.linalg.norm(n)
        proj = V @ n
        d = np.percentile(proj, rng.uniform(55.0, 72.0))
        mask = proj > d
        if np.any(mask):
            V[mask] -= (proj[mask] - d)[:, None] * n

    # Explicit slanted-crown shear: turns the pointed top into a broad, tilted
    # flat face like the photographed boulder.
    cn = np.array([rng.uniform(-0.4, 0.4), 1.0, rng.uniform(-0.4, 0.4)])
    cn /= np.linalg.norm(cn)
    proj = V @ cn
    d = np.percentile(proj, rng.uniform(58.0, 70.0))
    mask = proj > d
    if np.any(mask):
        V[mask] -= (proj[mask] - d)[:, None] * cn

    ymin, ymax = V[:, 1].min(), V[:, 1].max()
    cut_y = ymin + BASE_FLATTEN_FRACTION * (ymax - ymin)
    V[V[:, 1] < cut_y, 1] = cut_y

    a = rng.uniform(0.0, 2.0 * np.pi)
    ca, sa = np.cos(a), np.sin(a)
    rot = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
    V = V @ rot.T

    k = OVERALL_HEIGHT / max(V[:, 1].max() - V[:, 1].min(), 1e-9)
    V *= k
    xmin, xmax = V[:, 0].min(), V[:, 0].max()
    zmin, zmax = V[:, 2].min(), V[:, 2].max()
    V[:, 0] -= 0.5 * (xmin + xmax)
    V[:, 2] -= 0.5 * (zmin + zmax)
    V[:, 1] -= V[:, 1].min()

    mesh = trimesh.Trimesh(vertices=V, faces=faces, process=False)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="rock")
    return scene


# ===========================================================================
# TEXTURING
# ===========================================================================

STONE_RES = {"high": 1024, "med": 768, "low": 512}  # all >= 512
TILES_PER_METER = 1.6      # ~1.3 repeats across the boulder: minimal visible tiling
DELIGHT_CLAMP = (0.6, 1.6)  # gain clamp so nothing washes out
AO_STRENGTH = 0.28          # crevice darkening on COLOR_0
GROUND_DARK = 0.86          # base-line darkening on COLOR_0


def _blur2d(channel, radius):
    """Gaussian blur of a 2D float array (0..255) via PIL."""
    im = Image.fromarray(np.clip(channel, 0, 255).astype(np.uint8), "L")
    im = im.filter(ImageFilter.GaussianBlur(float(radius)))
    return np.asarray(im).astype(np.float64)


def _seamless(img, band):
    """Make a tile wrap seamlessly by blending ONLY the outer `band` px at each
    edge against the opposite edge.  Interior stays pristine and asymmetric --
    no mirror symmetry, no center seam."""
    img = img.astype(np.float64).copy()
    H, W = img.shape[0], img.shape[1]
    b = int(band)
    if b < 2:
        return img
    for i in range(b):
        w = 0.5 * (1.0 - i / b)                  # 0.5 at the very edge -> 0 inward
        Li = img[:, i].copy()
        Ri = img[:, W - 1 - i].copy()
        img[:, i] = Li * (1.0 - w) + Ri * w
        img[:, W - 1 - i] = Ri * (1.0 - w) + Li * w
    for j in range(b):
        w = 0.5 * (1.0 - j / b)
        Tj = img[j, :].copy()
        Bj = img[H - 1 - j, :].copy()
        img[j, :] = Tj * (1.0 - w) + Bj * w
        img[H - 1 - j, :] = Bj * (1.0 - w) + Tj * w
    return img


def _sample_body_color(arr, rng):
    """Median grey of several small patches placed WELL INSIDE the silhouette.

    Patches landing on the neutral backdrop (far from the robust median) are
    discarded so the body colour comes only from the rock.
    """
    H, W = arr.shape[:2]
    half = max(3, int(0.03 * min(H, W)))
    centres = [
        (0.42, 0.40), (0.50, 0.45), (0.58, 0.42), (0.40, 0.55),
        (0.52, 0.58), (0.62, 0.55), (0.45, 0.66), (0.57, 0.68),
        (0.50, 0.52), (0.36, 0.48), (0.64, 0.46),
    ]
    meds = []
    for fx, fy in centres:
        x = int(fx * W)
        y = int(fy * H)
        patch = arr[max(0, y - half):y + half, max(0, x - half):x + half]
        if patch.size:
            meds.append(np.median(patch.reshape(-1, 3), axis=0))
    meds = np.asarray(meds, dtype=np.float64)
    body = np.median(meds, axis=0)
    dist = np.linalg.norm(meds - body, axis=1)
    keep = dist < (np.median(dist) * 2.0 + 15.0)
    if keep.sum() >= 3:
        body = np.median(meds[keep], axis=0)
    return np.clip(body, 12.0, 243.0)


def _delight(crop):
    """Flatten baked lighting: divide by a heavily blurred luminance, clamp gain."""
    H, W = crop.shape[:2]
    lum = crop.mean(axis=2)
    blur = _blur2d(lum, max(8, min(H, W) // 8))
    gain = blur.mean() / np.clip(blur, 1e-3, None)
    gain = np.clip(gain, DELIGHT_CLAMP[0], DELIGHT_CLAMP[1])
    return np.clip(crop * gain[..., None], 0, 255)


def make_albedo(arr, rng, res):
    """De-lit, edge-seamless, NON-symmetric stone albedo keyed to the photo grey,
    with subtle mottling / hairline cracks / mineral specks."""
    H, W = arr.shape[:2]
    body = _sample_body_color(arr, rng)

    # Central, fully-interior crop, used full-frame (no mirror fold -> no kaleidoscope).
    cx0, cx1 = int(0.28 * W), int(0.72 * W)
    cy0, cy1 = int(0.28 * H), int(0.72 * H)
    crop = arr[cy0:cy1, cx0:cx1].astype(np.uint8)
    crop = np.asarray(
        Image.fromarray(crop).resize((res, res), Image.LANCZOS)
    ).astype(np.float64)

    crop = _delight(crop)

    # Re-key to the sampled body colour, then add mild contrast for tonal range
    # (pale lit grains vs darker recesses) so it doesn't read flat.
    cmean = np.clip(crop.reshape(-1, 3).mean(axis=0), 1.0, None)
    crop = crop * (body / cmean)[None, None, :]
    gmean = crop.mean()
    crop = np.clip(gmean + (crop - gmean) * 1.15, 0, 255)

    band = max(6, res // 16)
    base = _seamless(crop, band)

    # Smooth low-frequency mottling (full-res random -> blur; no ring artifacts).
    rnd = rng.random((res, res))
    mott = _blur2d(rnd * 255.0, res / 22.0) / 255.0
    mr = np.ptp(mott)
    mott = (mott - mott.min()) / (mr if mr > 1e-6 else 1.0)
    mott = _seamless(mott, band)
    base = np.clip(base * (0.93 + 0.14 * mott)[..., None], 0, 255)

    # A few subtle, fairly straight hairline fractures (kept inside an inset).
    img = Image.fromarray(base.astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)
    inset = int(0.10 * res)
    crack_col = tuple(int(v) for v in np.clip(body * 0.62, 0, 255))
    for _ in range(int(rng.integers(2, 5))):
        x = float(rng.integers(inset, res - inset))
        y = float(rng.integers(inset, res - inset))
        ang = rng.uniform(0.0, 2.0 * np.pi)
        pts = [(x, y)]
        for _ in range(int(rng.integers(6, 12))):
            ang += rng.uniform(-0.35, 0.35)
            step = rng.uniform(0.02, 0.045) * res
            x = float(np.clip(x + np.cos(ang) * step, inset, res - inset))
            y = float(np.clip(y + np.sin(ang) * step, inset, res - inset))
            pts.append((x, y))
        draw.line(pts, fill=crack_col, width=int(rng.integers(1, 3)))

    out = np.asarray(img).astype(np.float64)

    # Mineral specks: many dark grains plus a sparse scatter of bright mica.
    nspeck = (res * res) // 450
    sx = rng.integers(0, res, nspeck)
    sy = rng.integers(0, res, nspeck)
    dark = np.clip(body * 0.42, 0, 255)
    light = np.clip(body * 1.55, 0, 255)
    bright = rng.random(nspeck) < 0.15
    out[sy, sx] = np.where(bright[:, None], light[None, :], dark[None, :])

    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB"), body


def make_normal(albedo_img, strength=1.6):
    """Tangent-space normal map derived from albedo luminance (bright = raised)."""
    arr = np.asarray(albedo_img).astype(np.float64) / 255.0
    height = arr.mean(axis=2)
    gy, gx = np.gradient(height)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(height)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    rgb = np.stack(
        [(nx * 0.5 + 0.5), (ny * 0.5 + 0.5), (nz * 0.5 + 0.5)], axis=2
    )
    return Image.fromarray((rgb * 255).astype(np.uint8), "RGB")


def triplanar_uv(mesh):
    """Bake triplanar projection into per-vertex UVs (meters * tiles-per-meter)."""
    V = np.asarray(mesh.vertices)
    N = np.asarray(mesh.vertex_normals)
    axis = np.argmax(np.abs(N), axis=1)        # dominant world axis per vertex
    uv = np.zeros((len(V), 2), dtype=np.float64)
    mx = axis == 0
    my = axis == 1
    mz = axis == 2
    uv[mx] = V[mx][:, [2, 1]]                   # x-faces: project (z, y)
    uv[my] = V[my][:, [0, 2]]                   # y-faces: project (x, z)
    uv[mz] = V[mz][:, [0, 1]]                   # z-faces: project (x, y)
    return uv * TILES_PER_METER


def vertex_tint(mesh, rng):
    """COLOR_0 soft-AO tint: darker in concavities, along the ground, with a
    little per-vertex grain.  Multiplies the albedo in glTF, so it stays near 1."""
    V = np.asarray(mesh.vertices)
    N = np.asarray(mesh.vertex_normals)
    n = len(V)

    edges = np.asarray(mesh.edges_unique)
    nb_sum = np.zeros_like(V)
    nb_cnt = np.zeros(n)
    np.add.at(nb_sum, edges[:, 0], V[edges[:, 1]])
    np.add.at(nb_sum, edges[:, 1], V[edges[:, 0]])
    np.add.at(nb_cnt, edges[:, 0], 1.0)
    np.add.at(nb_cnt, edges[:, 1], 1.0)
    nb_mean = nb_sum / np.clip(nb_cnt, 1.0, None)[:, None]
    lap = nb_mean - V
    concavity = np.sum(lap * N, axis=1)         # >0 in pits/seams
    cstd = concavity.std() + 1e-6
    ao = np.clip(concavity / (2.5 * cstd), 0.0, 1.0)

    y = V[:, 1]
    yn = (y - y.min()) / max(np.ptp(y), 1e-9)
    ground = GROUND_DARK + (1.0 - GROUND_DARK) * np.clip(yn / 0.40, 0.0, 1.0)

    grain = 1.0 + rng.uniform(-0.04, 0.04, n)

    val = (1.0 - AO_STRENGTH * ao) * ground * grain
    val = np.clip(val, 0.5, 1.0)
    rgb = np.clip(val[:, None] * 255.0, 0, 255).astype(np.uint8)
    rgb = np.repeat(rgb, 3, axis=1)
    alpha = np.full((n, 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=1)


def texture_rock(mesh, image_path, seed, density):
    """Attach photo-derived albedo + normal, triplanar UVs and a COLOR_0 tint."""
    rng = np.random.default_rng(seed * 2 + 1)
    src = np.asarray(Image.open(image_path).convert("RGB")).astype(np.float64)

    res = STONE_RES.get(density, 1024)
    albedo, _body = make_albedo(src, rng, res)
    normal = make_normal(albedo)

    material = trimesh.visual.material.PBRMaterial(
        name="granite",
        baseColorTexture=albedo,
        normalTexture=normal,
        metallicFactor=0.0,
        roughnessFactor=0.95,        # matte, weathered stone
        doubleSided=False,
        alphaMode="OPAQUE",
    )

    uv = triplanar_uv(mesh)
    colors = vertex_tint(mesh, rng)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    # COLOR_0 vertex tint (multiplies the albedo in glTF) -- set after init.
    mesh.visual.vertex_attributes["color"] = colors
    return mesh


# ===========================================================================
# CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="Procedural granite boulder -> GLB.")
    ap.add_argument("--image", required=True, help="source reference photo")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    scene = build_mesh(args.seed, args.density)
    mesh = scene.geometry["rock"]
    texture_rock(mesh, args.image, args.seed, args.density)

    glb = scene.export(file_type="glb")
    with open(args.output, "wb") as fh:
        fh.write(glb)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - surface any failure as non-zero exit
        sys.stderr.write("ERROR: {}\n".format(exc))
        sys.exit(1)