#!/usr/bin/env python3
"""
Snow-capped boulder: procedural geometry + photo-derived materials -> textured GLB.

Pipeline:
  1. build_mesh(seed, density) -> Scene with "rock" and "snow" geometry.
  2. Sample colors and detail swatches from the reference photo (well inside the
     silhouette), de-light them, make them tileable, and synthesize albedo +
     normal maps for stone (mottled, cracked, gritty) and snow (granular, with
     cool blue-grey shadowing).
  3. Bake UVs per surface (triplanar for stone, planar top-down for the snow
     cap), attach PBR materials + per-vertex COLOR_0 tints.
  4. Export a binary .glb.

Only numpy / trimesh / PIL / stdlib are used.

CLI:
  python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter


# ===========================================================================
# GEOMETRY
# ===========================================================================
OVERALL_WIDTH_M   = 0.85   # boulder roughly 0.85 m across -> a chunky field stone
HEIGHT_OVER_WIDTH = 0.82   # taller than the first pass: front aspect target ~1.26
DEPTH_OVER_WIDTH  = 0.90   # slightly less deep than wide -> gentle asymmetry
SNOW_LINE_FRAC    = 0.46   # snow covers ~ top half of height (scalloped lower)

RX = OVERALL_WIDTH_M * 0.5
RZ = RX * DEPTH_OVER_WIDTH
RY = OVERALL_WIDTH_M * HEIGHT_OVER_WIDTH * 0.5

AMP          = 0.11
DIR_VAR      = 0.50
FACET_STRENGTH = 0.40
BOTTOM_FLATTEN = 0.10

CROWN_THICK  = 0.055
EDGE_THICK   = 0.025
LIP_OUT      = 0.020
LIP_DOWN     = 0.050
SCALLOP_FRAC = 0.09

# rock_sub: subdivisions; oct: noise octaves; facets: planar cuts;
# rsm/ssm: Taubin smoothing iterations for rock / snow.
_PRESETS = {
    "high": dict(rock_sub=5, oct=4, facets=2, rsm=2, ssm=4),
    "med":  dict(rock_sub=4, oct=3, facets=2, rsm=2, ssm=3),
    "low":  dict(rock_sub=3, oct=2, facets=1, rsm=1, ssm=2),
}


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _fbm(points, rng, octaves, base_freq, lacunarity=2.0, gain=0.55, n_waves=8):
    """Smooth, deterministic value noise as a sum of random sine waves."""
    pts = np.asarray(points, dtype=float)
    total = np.zeros(len(pts))
    freq, amp, norm = base_freq, 1.0, 0.0
    for _ in range(octaves):
        dirs = rng.normal(size=(n_waves, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
        phase = rng.uniform(0.0, 2.0 * np.pi, size=n_waves)
        proj = pts @ dirs.T * freq + phase
        total += amp * np.sin(proj).mean(axis=1)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / (norm + 1e-9)


def _boundary_edges(faces):
    e = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return uniq[counts == 1]


def _taubin(V, faces, iterations, lamb=0.5, mu=-0.53):
    """Pure-numpy Taubin (low-pass) smoothing -> removes facets without
    shrinkage, and rounds the whole-face snow boundary into a soft scallop."""
    if iterations <= 0:
        return V.astype(float)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    n = len(V)
    deg = np.maximum(np.bincount(edges[:, 0], minlength=n), 1)[:, None]
    src, dst = edges[:, 0], edges[:, 1]
    V = V.astype(float).copy()
    for it in range(iterations * 2):
        step = lamb if (it % 2 == 0) else mu
        s = np.zeros_like(V)
        np.add.at(s, src, V[dst])
        V += step * (s / deg - V)
    return V


def _settle(v):
    """Centre in X/Z and drop the lowest point to y = 0."""
    v[:, 0] -= 0.5 * (v[:, 0].min() + v[:, 0].max())
    v[:, 2] -= 0.5 * (v[:, 2].min() + v[:, 2].max())
    v[:, 1] -= v[:, 1].min()
    return v


def _build_rock(rng, cfg):
    ico = trimesh.creation.icosphere(subdivisions=cfg["rock_sub"], radius=1.0)
    unit = ico.vertices.copy()
    faces = ico.faces.copy()

    amp_dir = 1.0 + DIR_VAR * _fbm(unit, rng, octaves=2, base_freq=1.2)
    disp = _fbm(unit, rng, octaves=cfg["oct"], base_freq=2.0)
    radius = 1.0 + AMP * amp_dir * disp

    v = unit * radius[:, None]
    v[:, 0] *= RX
    v[:, 1] *= RY
    v[:, 2] *= RZ

    for _ in range(cfg["facets"]):
        nrm = rng.normal(size=3)
        nrm[1] = -abs(nrm[1]) * 0.6
        nrm /= np.linalg.norm(nrm) + 1e-9
        proj = v @ nrm
        d = np.quantile(proj, 0.70)
        over = proj > d
        v[over] -= np.outer((proj[over] - d) * FACET_STRENGTH, nrm)

    # Smooth away facets, THEN flatten the underside and settle on the ground.
    v = _taubin(v, faces, cfg["rsm"])
    ylo, yhi = v[:, 1].min(), v[:, 1].max()
    plane = ylo + BOTTOM_FLATTEN * (yhi - ylo)
    v[v[:, 1] < plane, 1] = plane
    v = _settle(v)

    rock = trimesh.Trimesh(vertices=v, faces=faces, process=False)
    rock.fix_normals()
    return rock


def _build_snow(rock, rng, cfg):
    rv = rock.vertices
    rf = rock.faces
    rn = rock.vertex_normals
    H = rv[:, 1].max()

    horiz = rv.copy()
    horiz[:, 1] = 0.0
    hdir = horiz / (np.linalg.norm(horiz, axis=1, keepdims=True) + 1e-6)
    scallop = _fbm(hdir, rng, octaves=2, base_freq=1.8)   # big organic lobes

    frac = SNOW_LINE_FRAC
    snowline = frac * H + SCALLOP_FRAC * H * scallop
    snowy = rv[:, 1] > snowline
    sfaces = rf[snowy[rf].all(axis=1)]
    while len(sfaces) == 0 and frac > 0.1:
        frac -= 0.05
        snowline = frac * H + SCALLOP_FRAC * H * scallop
        snowy = rv[:, 1] > snowline
        sfaces = rf[snowy[rf].all(axis=1)]
    if len(sfaces) == 0:
        return None

    used = np.unique(sfaces)
    remap = -np.ones(len(rv), dtype=np.int64)
    remap[used] = np.arange(len(used))

    denom = np.maximum(H - snowline, 1e-6)
    t = np.clip((rv[:, 1] - snowline) / denom, 0.0, 1.0)
    thick = EDGE_THICK + (CROWN_THICK - EDGE_THICK) * _smoothstep(t)

    outer = rv + rn * thick[:, None]
    outer_used = outer[used]
    # Gentle low-frequency mounding so the cap is not a perfect dome.
    lump = 0.018 * _fbm(rv[used], rng, octaves=2, base_freq=3.0)
    outer_used = outer_used + rn[used] * lump[:, None]

    top_faces = remap[sfaces]

    bnd_edges = _boundary_edges(sfaces)
    bnd_verts = np.unique(bnd_edges)
    lipmap = -np.ones(len(rv), dtype=np.int64)
    lipmap[bnd_verts] = np.arange(len(bnd_verts))

    lipb = rv[bnd_verts].copy()
    lh = lipb.copy()
    lh[:, 1] = 0.0
    lh /= np.linalg.norm(lh, axis=1, keepdims=True) + 1e-6
    lipb += lh * LIP_OUT
    lipb[:, 1] -= LIP_DOWN

    snow_vertices = np.vstack([outer_used, lipb])
    base = len(outer_used)

    a, b = bnd_edges[:, 0], bnd_edges[:, 1]
    at, bt = remap[a], remap[b]
    al, bl = base + lipmap[a], base + lipmap[b]
    skirt = np.empty((len(bnd_edges) * 2, 3), dtype=np.int64)
    skirt[0::2] = np.stack([at, bt, bl], axis=1)
    skirt[1::2] = np.stack([at, bl, al], axis=1)

    snow_faces = np.vstack([top_faces, skirt])
    # Smooth the cap + soften the boundary sawtooth into a draped scallop.
    snow_vertices = _taubin(snow_vertices, snow_faces, cfg["ssm"])

    snow = trimesh.Trimesh(vertices=snow_vertices, faces=snow_faces, process=False)
    snow.remove_unreferenced_vertices()
    snow.fix_normals()
    return snow


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    cfg = _PRESETS.get(density, _PRESETS["high"])
    rng = np.random.default_rng(seed)

    rock = _build_rock(rng, cfg)
    snow = _build_snow(rock, rng, cfg)

    scene = trimesh.Scene()
    scene.add_geometry(rock, geom_name="rock")
    if snow is not None and len(snow.faces) > 0:
        scene.add_geometry(snow, geom_name="snow")
    return scene


# ===========================================================================
# TEXTURE SYNTHESIS HELPERS
# ===========================================================================
ROCK_TEX = 1024
SNOW_TEX = 1024
ROCK_TILE_M = 0.28
SNOW_TILE_M = 0.40


def _to_img(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _luminance(arr):
    return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114


def _sample_color(img, boxes):
    h, w = img.shape[:2]
    meds = []
    for (x0, y0, x1, y1) in boxes:
        a, b = int(x0 * w), int(y0 * h)
        c, d = int(x1 * w), int(y1 * h)
        patch = img[b:d, a:c].reshape(-1, 3)
        if patch.size:
            meds.append(np.median(patch, axis=0))
    if not meds:
        return np.array([128.0, 128.0, 128.0])
    meds = np.array(meds)
    overall = np.median(meds, axis=0)
    dist = np.linalg.norm(meds - overall, axis=1)
    thresh = np.median(dist) * 2.5 + 25.0
    keep = meds[dist <= thresh]
    return np.median(keep, axis=0) if len(keep) else overall


def _crop_fraction(img, box):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    a, b = int(x0 * w), int(y0 * h)
    c, d = int(x1 * w), int(y1 * h)
    crop = img[b:d, a:c].astype(float)
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        crop = np.tile(np.median(img.reshape(-1, 3), axis=0), (64, 64, 1))
    n = min(crop.shape[0], crop.shape[1])
    return crop[:n, :n]


def _delight(sw):
    lum = _luminance(sw)
    limg = Image.fromarray(np.clip(lum, 0, 255).astype(np.uint8), "L")
    radius = max(4, min(sw.shape[:2]) // 6)
    blur = np.asarray(limg.filter(ImageFilter.GaussianBlur(radius)), dtype=float)
    target = np.median(blur)
    gain = np.clip(target / np.maximum(blur, 1e-3), 0.6, 1.6)
    return np.clip(sw * gain[..., None], 0, 255)


def _make_tileable(a):
    top = np.concatenate([a, a[:, ::-1]], axis=1)
    full = np.concatenate([top, top[::-1, :]], axis=0)
    return full


def _resize(a, size):
    return np.asarray(_to_img(a).resize((size, size), Image.LANCZOS), dtype=float)


def _value_noise(size, cells, rng):
    cells = max(2, int(cells))
    small = (rng.random((cells, cells)) * 255).astype(np.uint8)
    up = Image.fromarray(small, "L").resize((size, size), Image.BICUBIC)
    return np.asarray(up, dtype=float) / 255.0


def _fbm2d(size, rng, base_cells=5, octaves=4):
    out = np.zeros((size, size))
    amp, tot, cells = 1.0, 0.0, base_cells
    for _ in range(octaves):
        out += amp * _value_noise(size, cells, rng)
        tot += amp
        amp *= 0.5
        cells *= 2
    return out / tot


def _height_to_normal(height, strength):
    gy, gx = np.gradient(height.astype(float))
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(height)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    out = np.stack([nx * inv * 0.5 + 0.5,
                    ny * inv * 0.5 + 0.5,
                    nz * inv * 0.5 + 0.5], axis=-1)
    return _to_img(out * 255.0)


def _draw_cracks(albedo, rng, n_cracks, color):
    ss = 2
    size = albedo.shape[0]
    img = _to_img(albedo).resize((size * ss, size * ss), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    big = size * ss
    for _ in range(n_cracks):
        x, y = rng.uniform(0, big), rng.uniform(0, big)
        ang = rng.uniform(0, 2 * np.pi)
        steps = rng.integers(14, 30)
        step = big / rng.uniform(18, 40)
        pts = [(x, y)]
        for _ in range(int(steps)):
            ang += rng.uniform(-0.5, 0.5)
            x += np.cos(ang) * step
            y += np.sin(ang) * step
            pts.append((x, y))
        w = int(rng.integers(ss, ss * 2 + 1))
        draw.line(pts, fill=tuple(int(c) for c in color), width=w)
    img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=float)


# ===========================================================================
# MATERIAL BUILDERS
# ===========================================================================
def _build_rock_material(img, rng):
    size = ROCK_TEX
    rock_boxes = [
        (0.40, 0.70, 0.55, 0.80),
        (0.24, 0.62, 0.38, 0.74),
        (0.62, 0.60, 0.76, 0.72),
        (0.44, 0.55, 0.58, 0.65),
    ]
    med = np.clip(_sample_color(img, rock_boxes), 40, 200)
    dark = np.clip(med * 0.64, 0, 255)
    warm = np.clip(med * np.array([1.10, 1.00, 0.84]), 0, 255)
    cool = np.clip(med * np.array([0.90, 0.95, 1.05]), 0, 255)

    sw = _crop_fraction(img, (0.30, 0.55, 0.64, 0.84))
    detail = _luminance(_resize(_make_tileable(_delight(sw)), size))
    detail = detail / max(np.mean(detail), 1e-3)
    detail = np.clip(detail, 0.70, 1.30)

    m_low = _fbm2d(size, rng, base_cells=5, octaves=4)
    m_dark = _fbm2d(size, rng, base_cells=3, octaves=3)

    col = np.ones((size, size, 3)) * med[None, None, :]
    w_warm = np.clip((m_low - 0.50) * 2.0, 0, 1)[..., None]
    col = col * (1 - w_warm) + warm[None, None, :] * w_warm
    w_cool = np.clip((0.50 - m_low) * 2.0, 0, 1)[..., None]
    col = col * (1 - w_cool) + cool[None, None, :] * w_cool
    w_dark = (np.clip((m_dark - 0.55) / 0.45, 0, 1) * 0.55)[..., None]
    col = col * (1 - w_dark) + dark[None, None, :] * w_dark

    col *= detail[..., None]
    grain = 1.0 + (rng.random((size, size)) - 0.5) * 0.18
    col *= grain[..., None]
    fleck = rng.random((size, size))
    col[fleck > 0.986] *= 1.55
    col[fleck < 0.010] *= 0.70
    col = np.clip(col, 0, 255)
    col = _draw_cracks(col, rng, n_cracks=10, color=np.clip(dark * 0.55, 0, 255))

    albedo = _to_img(col)
    height = np.clip(_luminance(col) / 255.0, 0, 1)
    normal = _height_to_normal(height, strength=3.0)

    return trimesh.visual.material.PBRMaterial(
        name="rock", baseColorTexture=albedo, normalTexture=normal,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0, roughnessFactor=0.95, doubleSided=False,
    )


def _build_snow_material(img, rng):
    size = SNOW_TEX
    snow_boxes = [
        (0.40, 0.22, 0.55, 0.32),
        (0.30, 0.27, 0.42, 0.36),
        (0.55, 0.25, 0.68, 0.34),
        (0.45, 0.17, 0.56, 0.25),
    ]
    med = np.clip(_sample_color(img, snow_boxes), 205, 255)
    shadow = np.clip(med * np.array([0.88, 0.92, 1.00]), 0, 255)

    sw = _crop_fraction(img, (0.34, 0.18, 0.62, 0.40))
    detail = _luminance(_resize(_make_tileable(_delight(sw)), size))
    detail = detail / max(np.mean(detail), 1e-3)
    detail = np.clip(detail, 0.94, 1.06)

    m_low = _fbm2d(size, rng, base_cells=4, octaves=4)
    col = np.ones((size, size, 3)) * med[None, None, :]
    w_shadow = (np.clip((m_low - 0.50) / 0.50, 0, 1) * 0.28)[..., None]
    col = col * (1 - w_shadow) + shadow[None, None, :] * w_shadow

    col *= detail[..., None]
    grain = 1.0 + (rng.random((size, size)) - 0.5) * 0.06
    col *= grain[..., None]
    crumbs = rng.random((size, size))
    col[crumbs > 0.992] *= 0.92
    col = np.clip(col, 0, 255)

    albedo = _to_img(col)
    height = np.clip(_luminance(col) / 255.0, 0, 1)
    normal = _height_to_normal(height, strength=1.0)

    return trimesh.visual.material.PBRMaterial(
        name="snow", baseColorTexture=albedo, normalTexture=normal,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0, roughnessFactor=0.65, doubleSided=False,
    )


# ===========================================================================
# UVs + per-vertex tints
# ===========================================================================
def _triplanar_uv(verts, normals, scale):
    n = np.abs(normals)
    ax = np.argmax(n, axis=1)
    uv = np.zeros((len(verts), 2))
    mx, my, mz = ax == 0, ax == 1, ax == 2
    uv[mx] = verts[mx][:, [2, 1]]
    uv[my] = verts[my][:, [0, 2]]
    uv[mz] = verts[mz][:, [0, 1]]
    return uv * scale


def _planar_uv(verts, scale):
    return verts[:, [0, 2]] * scale          # seamless top-down for the snow cap


def _rock_vertex_colors(mesh, rng):
    v = mesh.vertices
    ymax = max(v[:, 1].max(), 1e-6)
    ynorm = v[:, 1] / ymax
    shade = 0.62 + 0.38 * _smoothstep(ynorm / 0.55)
    drift = rng.normal(0.0, 0.05, size=len(v))
    r = shade * (1.0 + drift + 0.04)
    g = shade * (1.0 + drift)
    b = shade * (1.0 + drift - 0.05)
    col = np.clip(np.stack([r, g, b], axis=1) * 255.0, 0, 255)
    out = np.empty((len(v), 4), dtype=np.uint8)
    out[:, :3] = col.astype(np.uint8)
    out[:, 3] = 255
    return out


def _snow_vertex_colors(mesh, rng):
    v = mesh.vertices
    ylo, yhi = v[:, 1].min(), v[:, 1].max()
    t = np.clip((v[:, 1] - ylo) / max(yhi - ylo, 1e-6), 0, 1)
    shade = 0.90 + 0.10 * _smoothstep(t)        # bright crown, gently cooler lip
    jitter = rng.normal(0.0, 0.015, size=len(v))
    r = np.clip(shade + jitter, 0, 1)
    g = np.clip(shade + jitter + 0.004, 0, 1)
    b = np.clip(shade * 1.02 + jitter + 0.015, 0, 1)
    col = np.clip(np.stack([r, g, b], axis=1) * 255.0, 0, 255)
    out = np.empty((len(v), 4), dtype=np.uint8)
    out[:, :3] = col.astype(np.uint8)
    out[:, 3] = 255
    return out


def _apply_surface(mesh, material, uv, vcolors):
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    mesh.visual.vertex_attributes["color"] = vcolors


# ===========================================================================
# ASSEMBLE
# ===========================================================================
def build_textured_scene(image_path, seed, density):
    img = np.asarray(Image.open(image_path).convert("RGB"), dtype=float)
    rng = np.random.default_rng(seed)

    scene = build_mesh(seed, density)
    rock_mat = _build_rock_material(img, rng)
    snow_mat = _build_snow_material(img, rng)

    for name, mesh in scene.geometry.items():
        if name.startswith("snow"):
            _apply_surface(mesh, snow_mat,
                           _planar_uv(mesh.vertices, 1.0 / SNOW_TILE_M),
                           _snow_vertex_colors(mesh, rng))
        else:
            _apply_surface(mesh, rock_mat,
                           _triplanar_uv(mesh.vertices, mesh.vertex_normals,
                                         1.0 / ROCK_TILE_M),
                           _rock_vertex_colors(mesh, rng))
    return scene


def main():
    ap = argparse.ArgumentParser(description="Snow-capped boulder -> textured GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        scene = build_textured_scene(args.image, args.seed, args.density)
        glb = scene.export(file_type="glb")
        with open(args.output, "wb") as f:
            f.write(glb)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())