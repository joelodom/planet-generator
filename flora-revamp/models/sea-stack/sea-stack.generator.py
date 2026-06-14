"""
Standalone procedural generator + texturer for a rounded, weathered boulder.

Builds geometry (a squat ovoid stone with broad lobes, gentle facets, sinuous
crown cracks and a thin CONNECTED granular debris collar that overlaps the
stone), derives TILEABLE materials by sampling colors from a reference photo,
applies triplanar UVs per surface, bakes per-vertex AO/tint into COLOR_0, and
exports an embedded-texture GLB.

CLI:
    python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb

Only numpy / trimesh / PIL / stdlib. +Y up, base at y=0, meters. Deterministic.
"""

import sys
import argparse

import numpy as np
import trimesh
from PIL import Image, ImageFilter

from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial


LUM = np.array([0.299, 0.587, 0.114])


# ===========================================================================
#  GEOMETRY  (build_mesh)
# ===========================================================================
# Aspect tuned so the WHOLE object (stone + tight collar) reads ~1.33 wide:1
# tall in front view, matching the photo's 1.34 content aspect.
HEIGHT_OVER_WIDTH  = 0.76   # squat but clearly rounded / domed
WIDEST_AT_FRAC     = 0.50   # maximum girth sits near mid-height
GRAVEL_COLLAR_FRAC = 0.12   # debris is a THIN low band hugging the base

BOULDER_WIDTH  = 1.25                                 # meters (seat-sized)
BOULDER_HEIGHT = BOULDER_WIDTH * HEIGHT_OVER_WIDTH    # ~0.95 m


def _make_noise_grid(rng, n=16):
    return rng.random((n, n, n))


def _sample_noise(grid, pts):
    """Trilinear smoothstep value noise in [0,1], periodic over the grid."""
    n = grid.shape[0]
    p = np.mod(pts, n)
    i0 = np.floor(p).astype(np.int64)
    f = p - i0
    i0 = np.mod(i0, n)
    i1 = np.mod(i0 + 1, n)
    u = f * f * (3.0 - 2.0 * f)

    x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
    x1, y1, z1 = i1[:, 0], i1[:, 1], i1[:, 2]
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]

    c000 = grid[x0, y0, z0]; c100 = grid[x1, y0, z0]
    c010 = grid[x0, y1, z0]; c110 = grid[x1, y1, z0]
    c001 = grid[x0, y0, z1]; c101 = grid[x1, y0, z1]
    c011 = grid[x0, y1, z1]; c111 = grid[x1, y1, z1]

    c00 = c000 * (1 - ux) + c100 * ux
    c10 = c010 * (1 - ux) + c110 * ux
    c01 = c001 * (1 - ux) + c101 * ux
    c11 = c011 * (1 - ux) + c111 * ux
    c0 = c00 * (1 - uy) + c10 * uy
    c1 = c01 * (1 - uy) + c11 * uy
    return c0 * (1 - uz) + c1 * uz


def _fbm(grid, dirs, octaves, offset):
    disp = np.zeros(len(dirs))
    amp, amp_sum = 1.0, 0.0
    for o in range(octaves):
        freq = 3.0 * (2 ** o)
        coords = dirs * freq + offset + o * 11.7
        disp += (_sample_noise(grid, coords) - 0.5) * amp
        amp_sum += amp
        amp *= 0.5
    return disp / amp_sum


def _build_boulder(rng, subdiv, octaves, crack_count):
    ico = trimesh.creation.icosphere(subdivisions=subdiv, radius=1.0)
    faces = ico.faces
    d = ico.vertices.astype(np.float64)
    d /= np.linalg.norm(d, axis=1, keepdims=True)

    pts = d * np.array([1.0, HEIGHT_OVER_WIDTH, 1.0])    # squat ellipsoid

    grid = _make_noise_grid(rng, 16)
    disp = _fbm(grid, d, octaves, rng.random(3) * 17.0)
    aniso = 0.18 + 0.07 * d[:, 0] + 0.04 * d[:, 2]       # gentle lopsided lumps
    pts *= (1.0 + disp * aniso)[:, None]

    for _ in range(2):                                   # subtle planar facets
        nrm = rng.normal(size=3)
        nrm[1] = abs(nrm[1]) * 0.4 + 0.1
        nrm /= np.linalg.norm(nrm)
        proj = pts @ nrm
        cut = np.quantile(proj, 0.90)
        over = proj > cut
        pts[over] -= np.outer((proj[over] - cut) * 0.5, nrm)

    h = pts[:, 1]                                         # sinuous crown cracks
    top_frac = (h - h.min()) / (np.ptp(h) + 1e-9)
    unit = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    for _ in range(crack_count):
        ang = rng.uniform(0, np.pi)
        along = np.array([np.cos(ang), 0.0, np.sin(ang)])
        across = np.array([-np.sin(ang), 0.0, np.cos(ang)])
        wig = 0.18 * np.sin((pts @ along) * rng.uniform(3.0, 5.0)
                            + rng.uniform(0, 6.28))
        dist = np.abs((pts @ across) - wig)
        width = rng.uniform(0.04, 0.07)
        depth = rng.uniform(0.025, 0.05)
        groove = np.exp(-(dist / width) ** 2) * np.clip((top_frac - 0.45) / 0.55, 0, 1)
        pts -= unit * (groove * depth)[:, None]

    ymin = pts[:, 1].min()                               # flatten the underside
    yspan = np.ptp(pts[:, 1]) + 1e-9
    flat_cut = ymin + 0.16 * yspan
    below = pts[:, 1] < flat_cut
    pts[below, 1] = flat_cut - (flat_cut - pts[below, 1]) * 0.28

    horiz = 0.5 * (np.ptp(pts[:, 0]) + np.ptp(pts[:, 2]))
    pts *= BOULDER_WIDTH / horiz
    pts[:, 0] -= 0.5 * (pts[:, 0].max() + pts[:, 0].min())
    pts[:, 2] -= 0.5 * (pts[:, 2].max() + pts[:, 2].min())
    pts[:, 1] -= pts[:, 1].min()

    rock = trimesh.Trimesh(vertices=pts, faces=faces, process=False)
    rock.fix_normals()
    return rock


def _build_gravel(rng, n_theta, n_rad, inner_r, outer_r, collar_top):
    """A SINGLE connected, low, lumpy debris band. Its inner rim sits deep
    inside the boulder's lower flank (collar is physically attached), it is
    tallest where it banks against the stone and tapers to ground at a tight
    outer rim -- a thin collar, NOT a flared skirt."""
    grid = _make_noise_grid(rng, 16)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    radial = np.linspace(0.0, 1.0, n_rad + 1)
    TH, S = np.meshgrid(thetas, radial)                  # (n_rad+1, n_theta)
    cos, sin = np.cos(TH), np.sin(TH)

    c1 = np.stack([cos * 6.0, sin * 6.0, S * 4.0], axis=-1).reshape(-1, 3) + 3.0
    c2 = np.stack([cos * 17.0, sin * 17.0, S * 9.0], axis=-1).reshape(-1, 3) + 11.0
    n1 = _sample_noise(grid, c1).reshape(S.shape)
    n2 = _sample_noise(grid, c2).reshape(S.shape)

    R = inner_r + (outer_r - inner_r) * S
    R = R + (n1 - 0.5) * (outer_r - inner_r) * 0.05 * (0.3 + S)    # mild rim
    prof = collar_top * np.clip(1.0 - S, 0.0, 1.0) ** 1.5          # banked inner
    bump = (n2 - 0.5) * collar_top * 0.35 + (n1 - 0.5) * collar_top * 0.20
    y = np.maximum(prof + bump, 0.0)

    verts = np.stack([R * cos, y, R * sin], axis=-1).reshape(-1, 3)

    faces = []
    for i in range(n_rad):
        for j in range(n_theta):
            a = i * n_theta + j
            b = (i + 1) * n_theta + j
            c = (i + 1) * n_theta + (j + 1) % n_theta
            d = i * n_theta + (j + 1) % n_theta
            faces.append((a, b, c))
            faces.append((a, c, d))
    faces = np.array(faces, dtype=np.int64)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if mesh.face_normals[:, 1].mean() < 0.0:             # orient upward
        faces = faces[:, ::-1]
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return mesh


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)

    if density == "low":
        subdiv, octaves, cracks = 2, 2, 1
        g_theta, g_rad = 44, 3
    elif density == "med":
        subdiv, octaves, cracks = 3, 3, 2
        g_theta, g_rad = 72, 4
    else:  # "high"
        subdiv, octaves, cracks = 4, 4, 3
        g_theta, g_rad = 120, 6

    rock = _build_boulder(rng, subdiv, octaves, cracks)

    b = rock.bounds
    height = b[1][1] - b[0][1]
    base_radius = 0.25 * ((b[1][0] - b[0][0]) + (b[1][2] - b[0][2]))
    collar_top_y = GRAVEL_COLLAR_FRAC * height
    gravel = _build_gravel(rng, g_theta, g_rad,
                           inner_r=base_radius * 0.55,   # buried in the flank
                           outer_r=base_radius * 1.03,   # hugs the base tightly
                           collar_top=collar_top_y)

    scene = trimesh.Scene()
    scene.add_geometry(rock, geom_name="rock")
    scene.add_geometry(gravel, geom_name="gravel")
    return scene


# ===========================================================================
#  TEXTURING
# ===========================================================================
ROCK_RES   = 1024
GRAVEL_RES = 512
ROCK_TILE_M   = 0.70   # larger -> fewer repeats / seams, calmer stone
GRAVEL_TILE_M = 0.10   # fine grit

# Photo sampling regions (normalized x,y) -- placed WELL INSIDE the silhouette,
# from reference.png: the dark central stone mass, and the LIGHT granular band
# along the very bottom. No sky/ground/background.
ROCK_PATCHES = [
    (0.40, 0.30), (0.50, 0.28), (0.60, 0.32),
    (0.38, 0.42), (0.50, 0.44), (0.62, 0.42),
    (0.45, 0.54), (0.55, 0.54), (0.50, 0.22),
]
GRAVEL_PATCHES = [
    (0.32, 0.80), (0.42, 0.82), (0.52, 0.83), (0.62, 0.81),
    (0.38, 0.78), (0.58, 0.78), (0.50, 0.85),
]


def _load_image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def _to_img(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGB")


def _smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a + 1e-9), 0, 1)
    return t * t * (3.0 - 2.0 * t)


def _sample_patches(img, centers):
    """Median color of several small interior patches; drop luminance
    outliers (background/shadow hits) so only true body colors survive."""
    H, W = img.shape[:2]
    half = max(3, int(min(H, W) * 0.02))
    cols = []
    for cx, cy in centers:
        x, y = int(cx * W), int(cy * H)
        x0, x1 = max(0, x - half), min(W, x + half + 1)
        y0, y1 = max(0, y - half), min(H, y + half + 1)
        patch = img[y0:y1, x0:x1].reshape(-1, 3)
        if len(patch):
            cols.append(np.median(patch, axis=0))
    cols = np.array(cols)
    if len(cols) >= 4:
        lum = cols @ LUM
        keep = np.abs(lum - np.median(lum)) < 0.22
        if keep.sum() >= 3:
            cols = cols[keep]
    return np.clip(cols, 0.03, 1.0)


def _extract(img, cx, cy, frac):
    H, W = img.shape[:2]
    s = max(8, int(min(H, W) * frac))
    x = int(cx * W - s / 2); y = int(cy * H - s / 2)
    x = max(0, min(W - s, x)); y = max(0, min(H - s, y))
    return img[y:y + s, x:x + s].copy()


def _delight(patch):
    """Flatten large-scale lighting: divide by a heavily blurred luminance,
    gain clamped to [0.6, 1.6] so nothing washes out."""
    lum = patch @ LUM
    im = Image.fromarray((np.clip(lum, 0, 1) * 255).astype(np.uint8), "L")
    rad = max(2, patch.shape[0] // 6)
    bl = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=rad)),
                    dtype=np.float64) / 255.0
    gain = np.clip(bl.mean() / (bl + 1e-4), 0.6, 1.6)
    return np.clip(patch * gain[..., None], 0, 1)


def _mirror_tile(field, res):
    """Seamless tileable texture by reflect-folding a swatch, then resample."""
    top = np.concatenate([field, field[:, ::-1]], axis=1)
    full = np.concatenate([top, top[::-1, :]], axis=0)
    im = Image.fromarray((np.clip(full, 0, 1) * 255).astype(np.uint8), "L")
    im = im.resize((res, res), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64) / 255.0


def _tileable_fractal(rng, res, beta=2.2):
    """Periodic (inherently tileable) fractal field in [0,1] via spectral
    shaping of white noise -- never blurs a seam through the detail."""
    w = rng.normal(size=(res, res))
    F = np.fft.fft2(w)
    f = np.fft.fftfreq(res)
    FX, FY = np.meshgrid(f, f)
    mag = np.sqrt(FX ** 2 + FY ** 2)
    mag[0, 0] = 1.0
    F *= 1.0 / (mag ** (beta / 2.0))
    out = np.fft.ifft2(F).real
    out -= out.min()
    out /= (np.ptp(out) + 1e-9)
    return out


def _normal_map(alb_uint8, strength=1.5):
    """Tangent-space normal map from albedo: height = inverse luminance,
    periodic (rolled) Sobel so the map tiles."""
    g = (alb_uint8[..., :3].astype(np.float64) / 255.0) @ LUM
    h = 1.0 - g
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
    nx, ny = -dx * strength, -dy * strength
    nz = np.ones_like(h)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    nm = np.stack([nx / norm, ny / norm, nz / norm], axis=-1)
    return Image.fromarray(((nm * 0.5 + 0.5) * 255).astype(np.uint8), "RGB")


# --- albedo builders -------------------------------------------------------
def _rock_albedo(rng, img, res):
    cols = _sample_patches(img, ROCK_PATCHES)
    lum = cols @ LUM
    order = np.argsort(lum)
    k = max(1, len(cols) // 3)
    dark = np.clip(cols[order[:k]].mean(axis=0), 0.04, 1.0)
    mid = np.median(cols, axis=0)
    pale = cols[order[-k:]].mean(axis=0)
    # Keep a real value range for the near-black stone (lightest ~2x darkest),
    # but stay anchored to the dark body so it does not read marbled.
    if pale.mean() < dark.mean() * 1.8:
        pale = np.clip(dark * 1.8 + 0.03, 0, 1)
    iron = np.clip(mid * np.array([1.30, 1.05, 0.75]), 0, 1)   # warm staining

    mottle = _tileable_fractal(rng, res, beta=2.4)
    grain  = _tileable_fractal(rng, res, beta=1.1)
    stain  = _tileable_fractal(rng, res, beta=3.0)
    crackf = _tileable_fractal(rng, res, beta=4.2)         # long, sparse cracks

    base = 0.65 * dark + 0.35 * mid                        # mostly dark charcoal
    t = (_smoothstep(mottle, 0.55, 0.85) * 0.40)[..., None]   # subtle pale patches
    alb = base * (1 - t) + pale * t
    st = (_smoothstep(stain, 0.68, 0.90) * 0.28)[..., None]   # faint iron mottle
    alb = alb * (1 - st) + iron * st
    alb *= (0.92 + 0.16 * grain)[..., None]               # fine mineral grain
    cf = np.clip(1.0 - np.abs(crackf - 0.5) / 0.020, 0, 1) ** 1.5
    alb *= (1.0 - 0.40 * cf)[..., None]                   # thin sinuous cracks

    detail = _mirror_tile((_delight(_extract(img, 0.50, 0.40, 0.18)) @ LUM), res)
    alb *= np.clip(detail / (detail.mean() + 1e-6), 0.86, 1.14)[..., None]
    return _to_img(alb)


def _gravel_albedo(rng, img, res):
    cols = _sample_patches(img, GRAVEL_PATCHES)
    # Guarantee the collar reads LIGHT (tan/cream/grey), never the dark stone.
    if (cols @ LUM).mean() < 0.38:
        cols = np.clip(cols + (0.45 - (cols @ LUM).mean()), 0, 1)
    K = len(cols)
    blob  = _tileable_fractal(rng, res, beta=2.8)
    grain = _tileable_fractal(rng, res, beta=1.3)
    spec  = _tileable_fractal(rng, res, beta=0.8)

    idx = np.clip((blob * K).astype(int), 0, K - 1)        # clumps of pebbles
    alb = cols[idx]
    alb *= (0.86 + 0.22 * grain)[..., None]                # grit shading (light)
    alb *= (0.93 + 0.12 * spec)[..., None]                 # fine speckle

    detail = _mirror_tile((_delight(_extract(img, 0.50, 0.82, 0.14)) @ LUM), res)
    alb *= np.clip(detail / (detail.mean() + 1e-6), 0.88, 1.12)[..., None]
    return _to_img(alb)


# --- UVs & vertex colors ---------------------------------------------------
def _triplanar_uv(vertices, normals, scale):
    """Triplanar projection baked into per-vertex UVs (dominant-axis pick)."""
    n = np.abs(normals)
    ax = np.argmax(n, axis=1)
    uv = np.zeros((len(vertices), 2))
    vx, vy, vz = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    mx, my, mz = ax == 0, ax == 1, ax == 2
    uv[mx, 0], uv[mx, 1] = vz[mx], vy[mx]
    uv[my, 0], uv[my, 1] = vx[my], vz[my]
    uv[mz, 0], uv[mz, 1] = vx[mz], vy[mz]
    return uv * scale


def _rock_vcols(rng, mesh):
    v = mesh.vertices
    h = (v[:, 1] - v[:, 1].min()) / (np.ptp(v[:, 1]) + 1e-9)
    nz = _sample_noise(_make_noise_grid(rng, 16), v * 3.0 + 5.0)
    ao = 0.66 + 0.34 * _smoothstep(h, 0.0, 0.55)            # darker near ground
    tint = np.clip(ao * (0.93 + 0.08 * nz), 0, 1)
    rgb = np.clip(np.stack([tint * 0.99, tint * 1.00, tint * 1.03], axis=1), 0, 1)
    a = np.full((len(v), 1), 255, np.uint8)
    return np.concatenate([(rgb * 255).astype(np.uint8), a], axis=1)


def _gravel_vcols(rng, mesh):
    v = mesh.vertices
    nz = _sample_noise(_make_noise_grid(rng, 16), v * 12.0 + 9.0)
    b = np.clip(0.90 + 0.10 * nz, 0.8, 1.0)                 # keep collar bright
    rgb = np.clip(np.stack([b * 1.0, b * 0.98, b * 0.92], axis=1), 0, 1)  # warm sand
    a = np.full((len(v), 1), 255, np.uint8)
    return np.concatenate([(rgb * 255).astype(np.uint8), a], axis=1)


def _texture_scene(scene, img, rng):
    rock = scene.geometry["rock"]
    gravel = scene.geometry["gravel"]

    # --- rock: uniform dark charcoal stone, subtle mottle + thin cracks ---
    r_alb = _rock_albedo(rng, img, ROCK_RES)
    r_nrm = _normal_map(np.asarray(r_alb), strength=1.6)
    r_uv = _triplanar_uv(rock.vertices, rock.vertex_normals, 1.0 / ROCK_TILE_M)
    r_mat = PBRMaterial(name="rock", baseColorTexture=r_alb, normalTexture=r_nrm,
                        metallicFactor=0.0, roughnessFactor=0.95)
    rock.visual = TextureVisuals(uv=r_uv, material=r_mat)
    rock.visual.vertex_attributes["color"] = _rock_vcols(rng, rock)

    # --- gravel: light tan/cream/grey granular collar (double-sided) ---
    g_alb = _gravel_albedo(rng, img, GRAVEL_RES)
    g_nrm = _normal_map(np.asarray(g_alb), strength=1.4)
    g_uv = _triplanar_uv(gravel.vertices, gravel.vertex_normals, 1.0 / GRAVEL_TILE_M)
    g_mat = PBRMaterial(name="gravel", baseColorTexture=g_alb, normalTexture=g_nrm,
                        metallicFactor=0.0, roughnessFactor=0.95, doubleSided=True)
    gravel.visual = TextureVisuals(uv=g_uv, material=g_mat)
    gravel.visual.vertex_attributes["color"] = _gravel_vcols(rng, gravel)

    return scene


# ===========================================================================
#  CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Procedural textured boulder -> GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    img = _load_image(args.image)
    scene = build_mesh(args.seed, args.density)
    tex_rng = np.random.default_rng(args.seed * 2 + 1)
    _texture_scene(scene, img, tex_rng)

    data = scene.export(file_type="glb")
    with open(args.output, "wb") as fh:
        fh.write(data)
    print("wrote", args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(1)