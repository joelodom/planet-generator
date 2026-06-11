"""Procedural saguaro-cactus asset: geometry + photo-derived materials -> GLB.

Builds a ribbed columnar trunk with two upward-curving arms, a corky woody
foot, and alpha-cutout SPINE CARDS (the faint bristled fuzz), derives tileable
materials from a reference photo, applies UVs per surface, and exports a
textured binary GLB.

CLI:
    python saguaro.py --image PATH --seed INT --density {high,med,low} --output OUT.glb

Only numpy / trimesh / PIL / stdlib.
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------------------
# Measured proportions (estimated by eye off the reference, ~10% accuracy).
# Front-view width/height of the photo subject is ~0.67 -- a STOUT column, so
# the trunk is thick relative to its height and the arms span wide.
# +Y is UP. Object stands on the XZ plane (lowest point y=0), axis near origin.
# ---------------------------------------------------------------------------
TOTAL_HEIGHT = 3.2          # m, plausible saguaro height (overall size)
HEIGHT_OVER_WIDTH = 1.5     # stout column (trunk h / overall width-ish)
R_TRUNK = 0.42              # m, thick trunk radius, broadest through lower-middle
BASE_H_FRAC = 0.075         # woody basal stem ~7.5% of total height
ARM_R_BRANCH_FRAC = 0.62    # right (higher) arm branches at ~62% of height
ARM_L_BRANCH_FRAC = 0.50    # left (lower) arm branches at ~50% of height
ARM_REACH_FRAC = 0.24       # arms reach out ~24% of height before turning up
ARM_RISE_FRAC = 0.40        # arms then rise, topping out well below the apex
ARM_TOP_FRAC = 0.88         # arm tips reach at most ~88% of total height
RIB_AMP = 0.06              # radial rib modulation (~6% of local radius)

BODY_V_PERIOD = 0.16        # metres per vertical texture tile (body)
BASE_V_PERIOD = 0.10        # metres per vertical texture tile (woody foot)


def _density_params(density: str) -> dict:
    """Choose element counts BEFORE building so we generate at target LOD."""
    table = {
        "high": dict(sides=72, n_ribs=18, trunk_rings=44, arm_rings=26,
                     base_rings=8, rib_amp=RIB_AMP, spines=1200),
        "med":  dict(sides=42, n_ribs=14, trunk_rings=28, arm_rings=16,
                     base_rings=6, rib_amp=RIB_AMP, spines=480),
        "low":  dict(sides=24, n_ribs=12, trunk_rings=16, arm_rings=10,
                     base_rings=4, rib_amp=0.05, spines=150),
    }
    if density not in table:
        density = "high"
    return table[density]


# ---------------------------------------------------------------------------
# Small vector helpers
# ---------------------------------------------------------------------------
def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return v / n


def _parallel_frames(P: np.ndarray):
    """Rotation-minimizing-ish frames (parallel transport) along a polyline."""
    M = len(P)
    T = np.zeros((M, 3))
    T[1:-1] = P[2:] - P[:-2]
    T[0] = P[1] - P[0]
    T[-1] = P[-1] - P[-2]
    T = _normalize(T)

    seed = np.array([1.0, 0.0, 0.0]) if abs(T[0, 0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    N = np.zeros((M, 3))
    N[0] = _normalize(np.cross(T[0], seed))
    for i in range(1, M):
        n = N[i - 1] - np.dot(N[i - 1], T[i]) * T[i]
        if np.linalg.norm(n) < 1e-8:
            n = np.cross(T[i], seed)
        N[i] = _normalize(n)
    B = _normalize(np.cross(T, N))
    return T, N, B


def _grid_faces(n_rings: int, ncol: int) -> np.ndarray:
    """Triangulated quad grid; columns are NOT wrapped (seam is duplicated)."""
    idx = np.arange(n_rings * ncol).reshape(n_rings, ncol)
    faces = []
    for i in range(n_rings - 1):
        a = idx[i, :-1]
        b = idx[i, 1:]
        c = idx[i + 1, :-1]
        d = idx[i + 1, 1:]
        faces.append(np.stack([a, c, b], axis=1))
        faces.append(np.stack([b, c, d], axis=1))
    return np.vstack(faces)


def _build_column(centerline, radii, sides, n_ribs, rib_amp, rib_phase,
                  cap_bottom, cap_top, u_tiles, v_period):
    """Swept generalized cylinder with a ribbed cross-section + cylindrical UVs.

    The seam column is duplicated (ncol = sides+1) with u = u_tiles so the
    texture wraps seamlessly under REPEAT addressing. Returns verts, faces, uvs.
    """
    M = len(centerline)
    _, N, B = _parallel_frames(centerline)

    seg = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])           # arc length per ring

    ncol = sides + 1
    jj = np.arange(ncol)
    theta = 2.0 * np.pi * (jj % sides) / sides
    cos_t = np.cos(theta)[:, None]
    sin_t = np.sin(theta)[:, None]
    rib = 1.0 + rib_amp * np.cos(n_ribs * theta + rib_phase)
    u_row = (jj / sides) * u_tiles

    verts = np.empty((M * ncol, 3))
    uvs = np.empty((M * ncol, 2))
    for i in range(M):
        rr = (radii[i] * rib)[:, None]
        offset = rr * (cos_t * N[i] + sin_t * B[i])
        verts[i * ncol:(i + 1) * ncol] = centerline[i] + offset
        uvs[i * ncol:(i + 1) * ncol, 0] = u_row
        uvs[i * ncol:(i + 1) * ncol, 1] = s[i] / v_period

    faces = [_grid_faces(M, ncol)]

    if cap_bottom:
        c_idx = len(verts)
        verts = np.vstack([verts, centerline[0][None, :]])
        uvs = np.vstack([uvs, [[0.5 * u_tiles, s[0] / v_period]]])
        ring = np.arange(sides)
        faces.append(np.stack([ring, ring + 1, np.full(sides, c_idx)], axis=1))

    if cap_top:
        t_last = _normalize(centerline[-1] - centerline[-2])
        tip = float(radii[-1]) * 0.9
        apex = centerline[-1] + t_last * tip
        a_idx = len(verts)
        verts = np.vstack([verts, apex[None, :]])
        uvs = np.vstack([uvs, [[0.5 * u_tiles, (s[-1] + tip) / v_period]]])
        base0 = (M - 1) * ncol + np.arange(sides)
        faces.append(np.stack([base0, np.full(sides, a_idx), base0 + 1], axis=1))

    return verts, np.vstack(faces), uvs


def _bezier(P0, P1, P2, P3, n):
    t = np.linspace(0.0, 1.0, n)[:, None]
    mt = 1.0 - t
    return (mt**3) * P0 + 3 * (mt**2) * t * P1 + 3 * mt * (t**2) * P2 + (t**3) * P3


def _trunk_radius(t: np.ndarray) -> np.ndarray:
    taper = 1.0 - 0.20 * t
    bulge = 1.0 + 0.06 * np.exp(-((t - 0.25) / 0.18) ** 2)
    flare = 1.0 + 0.20 * np.exp(-(t / 0.05) ** 2)
    dome = np.where(t > 0.9,
                    np.sqrt(np.clip(1.0 - ((t - 0.9) / 0.1) ** 2, 0.0, 1.0)),
                    1.0)
    return R_TRUNK * taper * bulge * flare * np.clip(dome, 0.12, 1.0)


def _arm_radius(s: np.ndarray, r0: float) -> np.ndarray:
    taper = 1.0 - 0.40 * s
    dome = np.where(s > 0.85,
                    np.sqrt(np.clip(1.0 - ((s - 0.85) / 0.15) ** 2, 0.0, 1.0)),
                    1.0)
    return r0 * taper * np.clip(dome, 0.18, 1.0)


def _build_spine_cards(body: trimesh.Trimesh, density: str, rng) -> trimesh.Trimesh:
    """Alpha-cutout spine clusters (the bristled fuzz) as small quad cards
    seated on the body surface, each mapped to a random tile of a 4x4 atlas."""
    n_want = _density_params(density)["spines"]
    V = body.vertices
    Nv = body.vertex_normals
    ylo = V[:, 1].min()
    span = max(np.ptp(V[:, 1]), 1e-6)
    cand = np.where(V[:, 1] > ylo + 0.05 * span)[0]
    if len(cand) == 0:
        cand = np.arange(len(V))
    n = int(min(n_want, len(cand)))
    idx = rng.choice(cand, size=n, replace=False)

    p = V[idx]
    nr = _normalize(Nv[idx])
    up = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(nr, up)
    bad = np.linalg.norm(t1, axis=1) < 1e-6
    if np.any(bad):
        t1[bad] = np.cross(nr[bad], np.array([1.0, 0.0, 0.0]))
    t1 = _normalize(t1)
    t2 = _normalize(np.cross(nr, t1))

    half = 0.024 * np.exp(rng.normal(0.0, 0.3, size=n))[:, None]
    half = np.clip(half, 0.012, 0.050)
    base = p + nr * half[:, 0:1] * 0.3

    c0 = base + (-t1 - t2) * half
    c1 = base + (t1 - t2) * half
    c2 = base + (t1 + t2) * half
    c3 = base + (-t1 + t2) * half
    verts = np.empty((n * 4, 3))
    verts[0::4] = c0
    verts[1::4] = c1
    verts[2::4] = c2
    verts[3::4] = c3

    inset = 0.012
    s = 0.25 - 2 * inset
    tcol = rng.integers(0, 4, n)
    trow = rng.integers(0, 4, n)
    rot = rng.integers(0, 4, n)
    u0 = tcol * 0.25 + inset
    v0 = trow * 0.25 + inset
    q = np.zeros((n, 4, 2))
    q[:, 0] = np.stack([u0, v0], axis=1)
    q[:, 1] = np.stack([u0 + s, v0], axis=1)
    q[:, 2] = np.stack([u0 + s, v0 + s], axis=1)
    q[:, 3] = np.stack([u0, v0 + s], axis=1)
    uv = np.empty((n * 4, 2))
    for i in range(n):
        r = int(rot[i])
        for k in range(4):
            uv[i * 4 + k] = q[i, (k + r) % 4]

    faces = np.empty((n * 2, 3), dtype=np.int64)
    bi = np.arange(n) * 4
    faces[0::2, 0] = bi
    faces[0::2, 1] = bi + 1
    faces[0::2, 2] = bi + 2
    faces[1::2, 0] = bi
    faces[1::2, 1] = bi + 2
    faces[1::2, 2] = bi + 3

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv, material=trimesh.visual.material.PBRMaterial())
    tint = np.clip(rng.uniform(0.85, 1.0, size=(n * 4, 1)), 0, 1) * np.array([1.0, 1.0, 0.96])
    mesh.visual.vertex_attributes["color"] = np.clip(
        np.hstack([tint, np.ones((n * 4, 1))]) * 255, 0, 255).astype(np.uint8)
    return mesh


# ===========================================================================
# GEOMETRY
# ===========================================================================
def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    P = _density_params(density)
    sides = P["sides"]
    n_ribs = P["n_ribs"]
    rib_amp = P["rib_amp"]

    H = TOTAL_HEIGHT * float(rng.uniform(0.92, 1.08))
    base_h = H * BASE_H_FRAC
    green_bottom = base_h * 0.6
    green_top = H
    rib_phase = float(rng.uniform(0.0, 2.0 * np.pi))

    body_v, body_f, body_uv = [], [], []
    voff = 0

    def add(v, f, uv):
        nonlocal voff
        body_v.append(v)
        body_f.append(f + voff)
        body_uv.append(uv)
        voff += len(v)

    # --- Trunk -------------------------------------------------------------
    n_tr = P["trunk_rings"]
    ty = np.linspace(green_bottom, green_top, n_tr)
    t = np.linspace(0.0, 0.98, n_tr)
    lean_x = float(rng.uniform(-0.03, 0.03)) * H
    lean_z = float(rng.uniform(-0.02, 0.02)) * H
    wob = 0.008 * H
    tx = lean_x * (t ** 1.3) + wob * np.sin(t * 3.1 + rng.uniform(0, 6.28))
    tz = lean_z * (t ** 1.3) + wob * np.sin(t * 2.7 + rng.uniform(0, 6.28))
    trunk_cl = np.column_stack([tx, ty, tz])
    trunk_r = _trunk_radius(t)
    v, f, uv = _build_column(trunk_cl, trunk_r, sides, n_ribs, rib_amp,
                             rib_phase, True, True, n_ribs, BODY_V_PERIOD)
    add(v, f, uv)

    # --- Two asymmetric arms (thick, topping out below the apex) -----------
    r_arm = R_TRUNK / np.sqrt(2.0) * 0.92
    n_arm = P["arm_rings"]

    def trunk_axis_at(frac):
        i = int(np.clip(frac, 0.0, 1.0) * (n_tr - 1))
        return trunk_cl[i]

    for direction, branch_frac, taller in [(+1.0, ARM_R_BRANCH_FRAC, True),
                                            (-1.0, ARM_L_BRANCH_FRAC, False)]:
        bf = branch_frac + float(rng.uniform(-0.04, 0.04))
        start = trunk_axis_at(bf)
        reach = direction * H * ARM_REACH_FRAC * float(rng.uniform(0.9, 1.15))
        rise = H * ARM_RISE_FRAC * (1.1 if taller else 0.95) * float(rng.uniform(0.9, 1.1))
        z_off = float(rng.uniform(-0.04, 0.04)) * H
        y0 = start[1]
        y_top = min(y0 + rise, green_top * ARM_TOP_FRAC)

        # mostly-vertical arm with a rounded elbow (keeps ribs from swirling)
        P0 = np.array([start[0], y0, start[2]])
        P1 = np.array([start[0] + reach * 0.65, y0 + 0.02 * H, start[2] + 0.3 * z_off])
        P2 = np.array([start[0] + reach, y0 + 0.14 * H, start[2] + z_off])
        P3 = np.array([start[0] + reach, y_top, start[2] + z_off])
        arm_cl = _bezier(P0, P1, P2, P3, n_arm)
        sp = np.linspace(0.0, 0.97, n_arm)
        arm_r = _arm_radius(sp, r_arm)
        v, f, uv = _build_column(arm_cl, arm_r, sides, n_ribs, rib_amp,
                                 rib_phase + float(rng.uniform(-0.5, 0.5)),
                                 False, True, n_ribs, BODY_V_PERIOD)
        add(v, f, uv)

    body = trimesh.Trimesh(vertices=np.vstack(body_v), faces=np.vstack(body_f),
                           process=False)
    body.fix_normals()
    body_uv = np.vstack(body_uv)

    # --- Woody / corky basal stem -----------------------------------------
    n_b = P["base_rings"]
    by_ = np.linspace(0.0, base_h, n_b)
    bt = np.linspace(0.0, 1.0, n_b)
    base_r = R_TRUNK * (0.97 + 0.10 * np.exp(-(bt / 0.25) ** 2))
    base_cl = np.column_stack([trunk_cl[0, 0] * bt, by_, trunk_cl[0, 2] * bt])
    base_u_tiles = max(8, n_ribs // 2)
    vb, fb, uvb = _build_column(base_cl, base_r, sides, max(8, n_ribs // 2),
                                rib_amp * 0.6, rib_phase, True, False,
                                base_u_tiles, BASE_V_PERIOD)
    base = trimesh.Trimesh(vertices=vb, faces=fb, process=False)
    base.fix_normals()

    # --- Per-vertex COLOR_0 tints (multiply texture in glTF) ---------------
    # Brighter, warmer yellow-green up high; dulling to olive low down.
    by = body.vertices[:, 1]
    fy = (by - by.min()) / max(np.ptp(by), 1e-6)
    jit = rng.uniform(-0.035, 0.035, size=len(by))
    bright = np.clip(0.92 + 0.18 * fy + jit, 0.0, 1.15)
    bcol = np.empty((len(by), 4))
    bcol[:, 0] = bright * (1.00 + 0.04 * fy)      # R (toward yellow-green)
    bcol[:, 1] = bright                            # G (lead)
    bcol[:, 2] = bright * (0.80 + 0.05 * (1 - fy))  # B
    bcol[:, 3] = 1.0
    body.visual = trimesh.visual.TextureVisuals(
        uv=body_uv, material=trimesh.visual.material.PBRMaterial())
    body.visual.vertex_attributes["color"] = np.clip(bcol * 255, 0, 255).astype(np.uint8)

    vy = base.vertices[:, 1]
    fb_y = (vy - vy.min()) / max(np.ptp(vy), 1e-6)
    ao = np.clip(0.65 + 0.35 * fb_y, 0.0, 1.0)
    ccol = np.stack([ao, ao, ao, np.ones_like(ao)], axis=1)
    base.visual = trimesh.visual.TextureVisuals(
        uv=uvb, material=trimesh.visual.material.PBRMaterial())
    base.visual.vertex_attributes["color"] = np.clip(ccol * 255, 0, 255).astype(np.uint8)

    # --- Spine cards (alpha-cutout fuzz seated on the body) ----------------
    spines = _build_spine_cards(body, density, rng)

    # --- Assemble; rest on y=0, centered in X/Z ---------------------------
    scene = trimesh.Scene()
    scene.add_geometry(body, geom_name="body")
    scene.add_geometry(base, geom_name="base")
    scene.add_geometry(spines, geom_name="spines")

    lo, hi = scene.bounds
    scene.apply_translation([-0.5 * (lo[0] + hi[0]), -lo[1], -0.5 * (lo[2] + hi[2])])
    return scene


# ===========================================================================
# TEXTURING (derive tileable materials from the reference photo)
# ===========================================================================
def _load_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def _crop_frac(arr, x0, y0, x1, y1) -> np.ndarray:
    h, w = arr.shape[:2]
    cx0, cx1 = int(x0 * w), int(x1 * w)
    cy0, cy1 = int(y0 * h), int(y1 * h)
    cx0, cx1 = max(0, min(cx0, w - 2)), max(1, min(cx1, w - 1))
    cy0, cy1 = max(0, min(cy0, h - 2)), max(1, min(cy1, h - 1))
    if cx1 <= cx0:
        cx1 = cx0 + 1
    if cy1 <= cy0:
        cy1 = cy0 + 1
    return arr[cy0:cy1, cx0:cx1].copy()


def _saturation(rgb):
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    return (mx - mn) / (mx + 1e-6)


def _delight(arr: np.ndarray) -> np.ndarray:
    """Divide out a heavily blurred luminance; clamp gain to [0.6, 1.6]."""
    h, w = arr.shape[:2]
    lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    radius = max(2, int(max(h, w) / 4))
    lum_img = Image.fromarray(np.clip(lum * 255, 0, 255).astype(np.uint8), "L")
    blur = np.asarray(lum_img.filter(ImageFilter.GaussianBlur(radius)),
                      dtype=np.float64) / 255.0
    blur = np.maximum(blur, 1e-3)
    gain = np.clip(lum.mean() / blur, 0.6, 1.6)
    return np.clip(arr * gain[..., None], 0.0, 1.0)


def _mirror_tile(arr: np.ndarray, size: int) -> np.ndarray:
    """Reflect-pad into a seamless tile (mirror fold), then resize."""
    hm = np.concatenate([arr, arr[:, ::-1]], axis=1)
    vm = np.concatenate([hm, hm[::-1, :]], axis=0)
    img = Image.fromarray(np.clip(vm * 255, 0, 255).astype(np.uint8)).resize(
        (size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0


def _normal_map(albedo: np.ndarray, strength: float) -> Image.Image:
    """Tangent-space normal map from albedo luminance (height = luminance)."""
    lum = 0.2126 * albedo[..., 0] + 0.7152 * albedo[..., 1] + 0.0722 * albedo[..., 2]
    gy, gx = np.gradient(lum)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(lum)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    out = np.stack([nx / norm, ny / norm, nz / norm], axis=-1) * 0.5 + 0.5
    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8), "RGB")


def _sample_body_color(arr) -> np.ndarray:
    """Median of small patches well inside the green body; drop background."""
    centers = [(0.46, 0.30), (0.50, 0.42), (0.47, 0.55),
               (0.52, 0.36), (0.45, 0.48), (0.51, 0.60)]
    cols = []
    for cx, cy in centers:
        patch = _crop_frac(arr, cx - 0.02, cy - 0.03, cx + 0.02, cy + 0.03)
        m = np.median(patch.reshape(-1, 3), axis=0)
        if m[1] >= m[0] and m[1] >= m[2] and m.mean() < 0.85 and _saturation(m[None])[0] > 0.10:
            cols.append(m)
    if not cols:
        cols = [np.array([0.45, 0.62, 0.30])]
    return np.median(np.array(cols), axis=0)


def _build_body_texture(arr, size, rng):
    """Green ribbed epidermis: photo swatch -> de-lit, tiled, rib-shaded,
    with a central column of pale areole/spine stipple (one per rib)."""
    crop = _delight(_crop_frac(arr, 0.42, 0.30, 0.57, 0.62))
    tile = _mirror_tile(crop, size)

    body_col = _sample_body_color(arr)
    tile_mean = tile.reshape(-1, 3).mean(axis=0) + 1e-3
    tile = np.clip(tile * (body_col / tile_mean) ** 0.6, 0.0, 1.0)
    tile = np.clip(tile * 1.12, 0.0, 1.0)              # brighten toward the photo

    # gentle vertical rib shading: crest (centre) bright, grooves (edges) soft
    u = np.linspace(0.0, 1.0, size)[None, :]
    shade = (0.88 + 0.12 * np.sin(np.pi * u))[..., None]
    tile = np.clip(tile * shade, 0.0, 1.0)

    img = Image.fromarray((tile * 255).astype(np.uint8), "RGB").convert("RGBA")

    ss = 3
    big = Image.new("RGBA", (size * ss, size * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    n_rows = 3
    spine = (236, 230, 212, 255)
    pad = (196, 188, 150, 200)
    seat = (150, 120, 88, 255)
    for r in range(n_rows):
        cy = (r + 0.5) / n_rows * size * ss + rng.uniform(-8, 8)
        cx = 0.5 * size * ss + rng.uniform(-6, 6)
        pr = size * ss * 0.018
        d.ellipse([cx - pr, cy - pr, cx + pr, cy + pr], fill=pad)
        sr = pr * 0.45
        d.ellipse([cx - sr, cy - sr, cx + sr, cy + sr], fill=seat)
        n_sp = 9
        for k in range(n_sp):
            a = 2 * np.pi * k / n_sp + rng.uniform(-0.15, 0.15)
            ln = size * ss * rng.uniform(0.018, 0.030)
            d.line([cx, cy, cx + np.cos(a) * ln, cy + np.sin(a) * ln],
                   fill=spine, width=max(1, int(size * ss * 0.0016)))
    big = big.resize((size, size), Image.LANCZOS)
    img = Image.alpha_composite(img, big).convert("RGB")

    albedo = np.asarray(img, dtype=np.float64) / 255.0
    return img, _normal_map(albedo, strength=2.0)


def _build_base_texture(arr, size, rng):
    """Corky woody foot: photo swatch -> de-lit, tiled, with vertical cracks."""
    crop = _delight(_crop_frac(arr, 0.45, 0.88, 0.54, 0.965))
    if crop.shape[0] < 6 or crop.shape[1] < 6:
        crop = _delight(_crop_frac(arr, 0.43, 0.84, 0.56, 0.97))
    tile = _mirror_tile(crop, size)

    img = Image.fromarray((tile * 255).astype(np.uint8), "RGB").convert("RGB")
    d = ImageDraw.Draw(img)
    for _ in range(14):
        x = rng.uniform(0, size)
        w = max(1, int(rng.uniform(1, 2.5)))
        dark = int(rng.uniform(35, 80))
        y = 0.0
        while y < size:
            nx = x + rng.uniform(-3, 3)
            d.line([x, y, nx, y + 14], fill=(dark, dark - 6, dark - 12), width=w)
            x, y = nx, y + 14
    img = img.filter(ImageFilter.GaussianBlur(0.6))

    albedo = np.asarray(img, dtype=np.float64) / 255.0
    return img, _normal_map(albedo, strength=2.6)


def _build_spine_atlas(size, rng) -> Image.Image:
    """4x4 atlas of distinct spine-cluster tiles (sunlit top rows brighter/
    warmer, shaded lower rows darker/cooler). Binary alpha, drawn supersampled."""
    ss = 4
    S = size * ss
    tile = S // 4
    big = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    for row in range(4):
        for col in range(4):
            ox, oy = col * tile, row * tile
            warm = 1.0 - row / 3.0
            bright = 0.78 + 0.22 * (1.0 - row / 3.0)
            base_col = np.array([238.0, 232.0, 214.0]) * bright * np.array(
                [1.0 + 0.03 * warm, 1.0, 0.96 - 0.04 * warm])
            base_col = tuple(int(np.clip(c, 40, 255)) for c in base_col)
            n_clusters = 1 if rng.random() < 0.6 else 2
            for _ in range(n_clusters):
                cx = ox + tile * rng.uniform(0.38, 0.62)
                cy = oy + tile * rng.uniform(0.38, 0.62)
                seat = tuple(int(c * 0.55) for c in base_col)
                sr = tile * 0.05
                d.ellipse([cx - sr, cy - sr, cx + sr, cy + sr], fill=seat + (255,))
                n_sp = int(rng.integers(12, 20))
                for k in range(n_sp):
                    a = 2 * np.pi * k / n_sp + rng.uniform(-0.12, 0.12)
                    L = tile * rng.uniform(0.18, 0.42)
                    w = max(2, int(tile * rng.uniform(0.006, 0.012)))
                    d.line([cx, cy, cx + np.cos(a) * L, cy + np.sin(a) * L],
                           fill=base_col + (255,), width=w)
    return big.resize((size, size), Image.LANCZOS)


def build_materials(image_path: str, seed: int) -> dict:
    arr = _load_rgb(image_path)
    rng = np.random.default_rng(seed ^ 0x5A6A)

    body_albedo, body_normal = _build_body_texture(arr, 1024, rng)
    base_albedo, base_normal = _build_base_texture(arr, 512, rng)
    spine_atlas = _build_spine_atlas(1024, rng)

    body_mat = trimesh.visual.material.PBRMaterial(
        name="body", baseColorTexture=body_albedo, normalTexture=body_normal,
        metallicFactor=0.0, roughnessFactor=0.55, doubleSided=True,
        alphaMode="OPAQUE")
    base_mat = trimesh.visual.material.PBRMaterial(
        name="base", baseColorTexture=base_albedo, normalTexture=base_normal,
        metallicFactor=0.0, roughnessFactor=0.92, doubleSided=True,
        alphaMode="OPAQUE")
    spine_mat = trimesh.visual.material.PBRMaterial(
        name="spines", baseColorTexture=spine_atlas, metallicFactor=0.0,
        roughnessFactor=0.8, doubleSided=True, alphaMode="MASK",
        alphaCutoff=0.45)
    return {"body": body_mat, "base": base_mat, "spines": spine_mat}


# ===========================================================================
# CLI
# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Procedural saguaro -> textured GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        scene = build_mesh(args.seed, args.density)
        mats = build_materials(args.image, args.seed)
        for name, mat in mats.items():
            if name in scene.geometry:
                scene.geometry[name].visual.material = mat

        data = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(data)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())