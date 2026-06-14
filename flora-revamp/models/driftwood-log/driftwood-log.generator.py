#!/usr/bin/env python3
"""
Standalone procedural generator + texturer + GLB exporter for a piece of
bleached, weathered driftwood (a smooth, stout tapering shaft that erupts into
a dense, rounded basket of gnarled root fingers).

Geometry  -> build_mesh(seed, density)        (+Y up, base at y=0, meters)
Materials -> derived from the reference photo (palette sampled from pixels)
Export    -> textured binary .glb

CLI:
    python thisscript.py --image PATH --seed INT --density {high,med,low} --output OUT.glb
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageFilter


# ============================================================================
# GEOMETRY MODULE
# ============================================================================
TOTAL_LENGTH      = 1.05    # m, overall span tip -> end of root tangle
HEIGHT_OVER_LENGTH = 0.50   # overall height / length (front aspect target ~1.9)
ROOT_FRACTION     = 0.30    # root tangle occupies the right ~30% of length
TRUNK_FRACTION    = 0.70    # smooth shaft is the left ~70% of length

TRUNK_LENGTH      = TOTAL_LENGTH * TRUNK_FRACTION   # ~0.735 m
ROOT_REACH        = TOTAL_LENGTH * ROOT_FRACTION    # ~0.315 m
TRUNK_MAX_RADIUS  = 0.110   # m, fat (root) end ; stout shaft
TRUNK_TIP_FRAC    = 0.34    # blunt rounded nose keeps real girth (not a spike)

GROOVE_AMP = 0.045                      # shallow longitudinal grooves
NOISE_AMP  = 0.14 * TRUNK_MAX_RADIUS    # weathered bump displacement
PIT_DEPTH  = 0.09 * TRUNK_MAX_RADIUS    # knot-pit dimples

_DENSITY = {
    "high": dict(trunk_segs=120, trunk_radial=40, n_roots=50, root_segs=14,
                 root_radial=7, fork_p=0.55, n_pits=6, waves=16, octaves=2),
    "med":  dict(trunk_segs=70,  trunk_radial=26, n_roots=30, root_segs=10,
                 root_radial=6, fork_p=0.45, n_pits=4, waves=12, octaves=2),
    "low":  dict(trunk_segs=34,  trunk_radial=14, n_roots=13, root_segs=6,
                 root_radial=5, fork_p=0.30, n_pits=2, waves=8,  octaves=1),
}


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _tangents(pts):
    pts = np.asarray(pts, float)
    T = np.empty_like(pts)
    T[1:-1] = pts[2:] - pts[:-2]
    T[0] = pts[1] - pts[0]
    T[-1] = pts[-1] - pts[-2]
    nrm = np.linalg.norm(T, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    return T / nrm


def _rmf(pts, T):
    """Rotation-minimizing frame normals along a polyline (double reflection)."""
    n = len(pts)
    N = np.zeros((n, 3))
    ref = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(ref, T[0])) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    N[0] = _normalize(np.cross(T[0], ref))
    for i in range(n - 1):
        v1 = pts[i + 1] - pts[i]
        c1 = float(np.dot(v1, v1))
        if c1 < 1e-14:
            N[i + 1] = N[i]
            continue
        rL = N[i] - (2.0 / c1) * np.dot(v1, N[i]) * v1
        tL = T[i] - (2.0 / c1) * np.dot(v1, T[i]) * v1
        v2 = T[i + 1] - tL
        c2 = float(np.dot(v2, v2))
        if c2 < 1e-14:
            N[i + 1] = _normalize(rL)
        else:
            N[i + 1] = _normalize(rL - (2.0 / c2) * np.dot(v2, rL) * v2)
    return N


def _ring_faces(n_rings, sides):
    """Quad-strip faces (two tris) for a closed tube of `n_rings` rings."""
    j = np.arange(sides)
    j2 = (j + 1) % sides
    out = []
    for i in range(n_rings - 1):
        a = i * sides
        b = (i + 1) * sides
        out.append(np.stack([a + j, a + j2, b + j2], axis=1))
        out.append(np.stack([a + j, b + j2, b + j], axis=1))
    return np.vstack(out)


def _fourier_noise(points, rng, n_waves, freq):
    """Smooth, deterministic pseudo-noise via random Fourier features (~[-1,1])."""
    pts = np.asarray(points, float)
    dirs = rng.normal(size=(n_waves, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    fr = freq * rng.uniform(0.5, 1.6, size=n_waves)
    ph = rng.uniform(0.0, 2.0 * np.pi, size=n_waves)
    proj = pts @ (dirs * fr[:, None]).T
    return np.sin(proj + ph).mean(axis=1)


def _build_trunk(rng, p):
    segs = p["trunk_segs"]
    sides = p["trunk_radial"]
    t = np.linspace(0.0, 1.0, segs)

    # spine: lay along +X, GENTLE arch and slight side bend (kept straight-ish)
    arch = rng.uniform(0.015, 0.035) * (1.0 if rng.random() < 0.5 else -1.0)
    bend = rng.uniform(-0.03, 0.03)
    spine = np.column_stack([
        t * TRUNK_LENGTH,
        arch * np.sin(np.pi * t),
        bend * np.sin(np.pi * t) + 0.02 * bend * t,
    ])

    T = _tangents(spine)
    N = _rmf(spine, T)
    B = np.cross(T, N)

    # radius profile: blunt rounded nose -> stout, gently swelling thick end
    tip_r = TRUNK_TIP_FRAC
    shape = tip_r + (1.0 - tip_r) * np.power(t, 0.5)
    shape *= 1.0 + 0.06 * np.sin(np.pi * t)
    base_r = TRUNK_MAX_RADIUS * shape

    thetas = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    groove = np.zeros(sides)
    for k in range(1, 5):
        nharm = int(rng.integers(3, 9))
        groove += (rng.uniform(0.3, 1.0) / k) * np.sin(nharm * thetas + rng.uniform(0, 2 * np.pi))
    groove = GROOVE_AMP * groove / max(np.max(np.abs(groove)), 1e-6)

    R = base_r[:, None] * (1.0 + groove[None, :])

    for _ in range(p["n_pits"]):
        pt = rng.uniform(0.18, 0.95)
        pth = rng.uniform(0.0, 2.0 * np.pi)
        dt = (t - pt) / 0.035
        dth = (((thetas - pth + np.pi) % (2.0 * np.pi)) - np.pi) / 0.45
        R -= PIT_DEPTH * np.exp(-(dt[:, None] ** 2 + dth[None, :] ** 2))
    R = np.maximum(R, 0.004)

    cos = np.cos(thetas)
    sin = np.sin(thetas)
    dirs = cos[None, :, None] * N[:, None, :] + sin[None, :, None] * B[:, None, :]
    verts = (spine[:, None, :] + R[:, :, None] * dirs).reshape(-1, 3)
    centers = np.repeat(spine, sides, axis=0)

    outward = verts - centers
    onrm = np.linalg.norm(outward, axis=1, keepdims=True)
    onrm[onrm < 1e-9] = 1.0
    outward_n = outward / onrm
    nz = _fourier_noise(verts, rng, p["waves"], 4.0)
    if p["octaves"] > 1:
        nz = nz + 0.4 * _fourier_noise(verts, rng, p["waves"] + 2, 11.0)
    verts = verts + outward_n * (nz[:, None] * NOISE_AMP)

    faces = _ring_faces(segs, sides)

    apex_nose = spine[0] - T[0] * float(R[0].mean()) * 0.7
    apex_end = spine[-1] + T[-1] * float(R[-1].mean()) * 0.15
    ai0 = len(verts)
    ai1 = ai0 + 1
    verts = np.vstack([verts, apex_nose[None, :], apex_end[None, :]])
    j = np.arange(sides)
    j2 = (j + 1) % sides
    last = (segs - 1) * sides
    cap0 = np.stack([np.full(sides, ai0), j, j2], axis=1)
    cap1 = np.stack([np.full(sides, ai1), last + j2, last + j], axis=1)
    faces = np.vstack([faces, cap0, cap1])

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.fix_normals()

    end_info = dict(center=spine[-1].copy(), T=T[-1].copy(),
                    N=N[-1].copy(), B=B[-1].copy(),
                    radius=float(TRUNK_MAX_RADIUS * shape[-1]))
    return mesh, end_info


def _make_tube(pts, radii, sides):
    pts = np.asarray(pts, float)
    radii = np.asarray(radii, float)
    n = len(pts)
    T = _tangents(pts)
    N = _rmf(pts, T)
    B = np.cross(T, N)
    thetas = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    cos = np.cos(thetas)
    sin = np.sin(thetas)
    dirs = cos[None, :, None] * N[:, None, :] + sin[None, :, None] * B[:, None, :]
    verts = (pts[:, None, :] + radii[:, None, None] * dirs).reshape(-1, 3)
    faces = _ring_faces(n, sides)

    apex0 = pts[0] - T[0] * max(radii[0], 1e-3) * 0.6
    apex1 = pts[-1] + T[-1] * max(radii[-1], 1e-3) * 0.7
    ai0 = len(verts)
    ai1 = ai0 + 1
    verts = np.vstack([verts, apex0[None, :], apex1[None, :]])
    j = np.arange(sides)
    j2 = (j + 1) % sides
    last = (n - 1) * sides
    cap0 = np.stack([np.full(sides, ai0), j, j2], axis=1)
    cap1 = np.stack([np.full(sides, ai1), last + j2, last + j], axis=1)
    faces = np.vstack([faces, cap0, cap1])
    return verts, faces


def _grow_path(rng, base, init_dir, attract, reach, nseg):
    """A short, twisting root centerline pulled in toward the bundle (compact)."""
    pts = [np.asarray(base, float).copy()]
    d = _normalize(init_dir)
    curl_axis = _normalize(rng.normal(size=3))
    curl_strength = rng.uniform(0.2, 0.6)
    step = (reach / nseg) * rng.uniform(0.7, 1.05)
    for i in range(nseg):
        sl = step * (1.0 - 0.45 * i / nseg)
        pts.append(pts[-1] + d * sl)
        turn = rng.normal(scale=0.35, size=3) + curl_axis * curl_strength
        if i > nseg * 0.35:                       # curl back into the snarl early
            back = attract - pts[-1]
            nb = np.linalg.norm(back)
            if nb > 1e-6:
                turn += (back / nb) * rng.uniform(0.3, 0.9)
        d = _normalize(d + turn * 0.5)
    pts = np.array(pts)
    pts[1:] += rng.normal(scale=reach * 0.010, size=(len(pts) - 1, 3))
    return pts


def _build_roots(rng, p, end):
    sides = p["root_radial"]
    nseg = p["root_segs"]
    end_c, end_T, end_N, end_B, end_r = (end["center"], end["T"],
                                         end["N"], end["B"], end["radius"])
    # compact rounded ball of roots just past the trunk end
    attract = end_c + end_T * ROOT_REACH * 0.32

    specs = []
    for _ in range(p["n_roots"]):
        ang = rng.uniform(0.0, 2.0 * np.pi)
        rf = np.sqrt(rng.uniform(0.0, 1.0))
        br = rf * end_r
        base = (end_c
                + br * (np.cos(ang) * end_N + np.sin(ang) * end_B)
                - end_T * rng.uniform(0.0, 0.05))
        # mostly project outward off the end, with vertical/lateral spread
        init_dir = _normalize(end_T * rng.uniform(0.3, 0.85) + rng.normal(scale=0.7, size=3))
        reach_i = ROOT_REACH * rng.uniform(0.40, 0.80)
        pts = _grow_path(rng, base, init_dir, attract, reach_i, nseg)

        # THICK, stubby roots that taper to a fat broken tip (not hair-thin)
        r0 = end_r * rng.uniform(0.20, 0.36)
        u = np.linspace(0.0, 1.0, nseg + 1)
        radii = r0 * np.power(1.0 - u, 0.6)
        radii *= 1.0 + rng.uniform(-0.18, 0.25, size=radii.shape)
        radii = np.maximum(radii, 0.0045)
        specs.append((pts, radii))

        if nseg > 5 and rng.random() < p["fork_p"]:
            jf = int(rng.integers(int(nseg * 0.25), int(nseg * 0.70)))
            cbase = pts[jf]
            cdir = _normalize((pts[jf] - pts[jf - 1]) + rng.normal(scale=0.8, size=3))
            cn = max(4, nseg - jf - 1)
            cpts = _grow_path(rng, cbase, cdir, attract, ROOT_REACH * rng.uniform(0.30, 0.55), cn)
            cu = np.linspace(0.0, 1.0, cn + 1)
            cr = max(radii[jf] * 0.72, 0.005) * np.power(1.0 - cu, 0.6)
            cr = np.maximum(cr, 0.004)
            specs.append((cpts, cr))

    all_v = []
    all_f = []
    voff = 0
    for pts, radii in specs:
        v, f = _make_tube(pts, radii, sides)
        all_v.append(v)
        all_f.append(f + voff)
        voff += len(v)

    mesh = trimesh.Trimesh(vertices=np.vstack(all_v),
                           faces=np.vstack(all_f), process=True)
    mesh.fix_normals()
    return mesh


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    if density not in _DENSITY:
        density = "high"
    p = _DENSITY[density]
    rng = np.random.default_rng(seed)

    trunk, end = _build_trunk(rng, p)
    roots = _build_roots(rng, p, end)

    s = rng.uniform(0.9, 1.1)
    trunk.apply_scale(s)
    roots.apply_scale(s)

    allv = np.vstack([trunk.vertices, roots.vertices])
    shift = np.array([
        -0.5 * (allv[:, 0].min() + allv[:, 0].max()),
        -allv[:, 1].min(),
        -0.5 * (allv[:, 2].min() + allv[:, 2].max()),
    ])
    trunk.apply_translation(shift)
    roots.apply_translation(shift)

    scene = trimesh.Scene()
    scene.add_geometry(trunk, geom_name="trunk")
    scene.add_geometry(roots, geom_name="roots")
    return scene


# ============================================================================
# TEXTURING  (palette sampled from the photo; tileable bleached-wood material)
# ============================================================================
_LUM = np.array([0.299, 0.587, 0.114])

TRUNK_BOX = (0.20, 0.42, 0.45, 0.63)
ROOTS_BOX = (0.60, 0.82, 0.42, 0.66)


def _to_img(a):
    a = np.clip(a, 0.0, 1.0)
    return Image.fromarray((a * 255.0 + 0.5).astype(np.uint8))


def sample_palette(arr):
    """Sample bleached-wood colour from the photo, then LIFT its value.

    Patches across the central region; the bleached wood is the lighter
    cluster, so keep the brighter half (rejects darker grey background). Then
    lift the value to a bone-white level while preserving the sampled HUE --
    real driftwood is near-white, far brighter than a raw average.
    """
    H, W, _ = arr.shape
    xs = np.linspace(0.16, 0.86, 11)
    ys = np.linspace(0.34, 0.70, 7)
    patch = max(4, int(min(H, W) * 0.013))
    cols = []
    for fy in ys:
        for fx in xs:
            cx, cy = int(fx * W), int(fy * H)
            x0, x1 = max(0, cx - patch), min(W, cx + patch)
            y0, y1 = max(0, cy - patch), min(H, cy + patch)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            cols.append(np.median(arr[y0:y1, x0:x1].reshape(-1, 3), axis=0))
    cols = np.array(cols)
    if len(cols) < 4:
        cols = arr.reshape(-1, 3)[::997]

    lum = cols @ _LUM
    wood = cols[lum >= np.percentile(lum, 50.0)]      # brighter half = wood
    if len(wood) < 4:
        wood = cols
    base = np.median(wood, axis=0)

    # lift value to a believable bleached level, keep hue
    blum = float(base @ _LUM)
    if blum > 1e-3:
        base = np.clip(base * (0.80 / blum), 0.0, 1.0)
    # gentle spread (light wood: lightest only ~1.5x darkest)
    light = np.clip(base * 1.10, 0.0, 1.0)
    dark = np.clip(base * 0.80, 0.0, 1.0)
    return base, light, dark


def photo_mottle(arr, box, size):
    """De-lit, mirror-folded (tileable) luminance mottle from a wood crop.

    Only the normalised luminance *variation* is used (recoloured by the
    palette), so background colour can never leak into the albedo.
    """
    H, W, _ = arr.shape
    x0, x1 = sorted((int(box[0] * W), int(box[1] * W)))
    y0, y1 = sorted((int(box[2] * H), int(box[3] * H)))
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return np.ones((size, size))
    gray = arr[y0:y1, x0:x1] @ _LUM
    g = Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8)).resize(
        (size, size), Image.LANCZOS)
    ga = np.asarray(g, float) / 255.0
    blur = np.asarray(g.filter(ImageFilter.GaussianBlur(size / 12.0)), float) / 255.0
    blur = np.clip(blur, 1e-3, None)
    gain = np.clip(ga / blur, 0.6, 1.6)          # de-light, clamp the gain
    m = 1.0 + (gain - 1.0) * 0.35                # subtle multiplier around 1.0
    m = 0.5 * (m + m[:, ::-1])                    # mirror-fold -> seamless tile
    m = 0.5 * (m + m[::-1, :])
    return m


def make_wood_albedo(size, rng, base, light, dark, kind, mottle):
    """Tileable bleached-wood albedo: subtle longitudinal grain + mottle."""
    uu = np.linspace(0.0, 1.0, size, endpoint=False)
    U, V = np.meshgrid(uu, uu)                    # U varies along columns

    grain = np.zeros((size, size))
    nharm = 5 if kind == "trunk" else 8
    hi = 16 if kind == "trunk" else 30
    for k in range(nharm):
        f = int(rng.integers(4, hi))
        amp = rng.uniform(0.3, 1.0) / (1.0 + 0.5 * k)
        wav = 0.18 * np.sin(2 * np.pi * int(rng.integers(1, 4)) * V + rng.uniform(0, 2 * np.pi))
        grain += amp * np.sin(2 * np.pi * f * U + rng.uniform(0, 2 * np.pi) + wav)

    mott = np.zeros((size, size))
    for _ in range(4):
        fu, fv = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        mott += rng.uniform(0.3, 0.8) * np.sin(2 * np.pi * fu * U + rng.uniform(0, 2 * np.pi)) \
            * np.sin(2 * np.pi * fv * V + rng.uniform(0, 2 * np.pi))

    grain = (grain - grain.min()) / (np.ptp(grain) + 1e-9)
    mott = (mott - mott.min()) / (np.ptp(mott) + 1e-9)
    if kind == "trunk":
        val = 0.5 * grain + 0.5 * mott            # smooth, sanded shaft
    else:
        val = np.power(0.6 * grain + 0.4 * mott, 1.3)   # rougher, deeper crevices
    val = np.clip(val, 0.0, 1.0)

    col = dark[None, None, :] * (1.0 - val)[..., None] + light[None, None, :] * val[..., None]
    col = 0.55 * col + 0.45 * base[None, None, :]   # bias toward the bright base
    col = col * mottle[..., None]

    nsp = 5 if kind == "trunk" else 14
    depth = (0.18, 0.34) if kind == "trunk" else (0.25, 0.5)
    yy, xx = np.mgrid[0:size, 0:size]
    mask = np.ones((size, size))
    for _ in range(nsp):
        cx, cy = int(rng.integers(0, size)), int(rng.integers(0, size))
        rad = rng.uniform(size * 0.006, size * 0.018)
        dx = np.minimum(np.abs(xx - cx), size - np.abs(xx - cx))
        dy = np.minimum(np.abs(yy - cy), size - np.abs(yy - cy))
        mask *= 1.0 - rng.uniform(*depth) * np.exp(-(dx * dx + dy * dy) / (2.0 * rad * rad))
    col = col * mask[..., None]

    col = col * np.array([1.0, 0.995, 0.975])[None, None, :]   # faint warm undertone
    return np.clip(col, 0.0, 1.0)


def albedo_to_normal(col, strength):
    """Tangent-space normal map from albedo luminance (height = luminance)."""
    h = col @ _LUM
    gy, gx = np.gradient(h)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(h)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    n = np.stack([nx * inv, ny * inv, nz * inv], axis=-1)
    return n * 0.5 + 0.5


def metallic_roughness(col, kind):
    """glTF metallic-roughness texture (G=roughness, B=metallic=0)."""
    lum = col @ _LUM
    if kind == "trunk":
        base_r, var = 0.78, 0.16          # sanded / slightly polished shaft
    else:
        base_r, var = 0.92, 0.10          # rough, fibrous, splintery roots
    rough = np.clip(base_r - var * (lum - 0.5) * 2.0, 0.45, 0.99)
    out = np.zeros(col.shape)
    out[..., 1] = rough
    return out


def trunk_uv(mesh):
    """Cylindrical UVs about the part (X) axis: u=around, v=along length."""
    v = mesh.vertices
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    cy = 0.5 * (y.min() + y.max())
    cz = 0.5 * (z.min() + z.max())
    ang = np.arctan2(z - cz, y - cy)
    u = (ang / (2.0 * np.pi) + 0.5) * 3.0
    length = max(x.max() - x.min(), 1e-6)
    vv = (x - x.min()) / length * (length / 0.33)
    return np.stack([u, vv], axis=1).astype(np.float32)


def roots_uv(mesh):
    """Triplanar projection baked into per-vertex UVs (dominant-axis)."""
    v = mesh.vertices
    n = mesh.vertex_normals
    ax = np.argmax(np.abs(n), axis=1)
    tile = 0.09
    uv = np.zeros((len(v), 2))
    m = ax == 0
    uv[m] = v[m][:, [1, 2]]
    m = ax == 1
    uv[m] = v[m][:, [0, 2]]
    m = ax == 2
    uv[m] = v[m][:, [0, 1]]
    return (uv / tile).astype(np.float32)


def vertex_colors(mesh, kind, rng):
    """COLOR_0 tints (multiply albedo): keep it BRIGHT, gentle sun/shade only."""
    v = mesh.vertices
    n = mesh.vertex_normals
    y = v[:, 1]
    yt = (y - y.min()) / (np.ptp(y) + 1e-9)
    if kind == "trunk":
        up = np.clip(n[:, 1], -1.0, 1.0)
        br = 0.93 + 0.08 * yt + 0.03 * up         # bright; top a touch sunnier
        br += rng.normal(0.0, 0.018, size=len(v))
        warm = np.array([1.0, 0.995, 0.975])
        lo, hi = 0.86, 1.05
    else:
        c = v.mean(axis=0)
        rad = np.linalg.norm(v - c, axis=1)
        rt = (rad - rad.min()) / (np.ptp(rad) + 1e-9)
        br = 0.86 + 0.12 * rt + 0.06 * yt         # outer tips brighter, inner shaded
        br += rng.normal(0.0, 0.03, size=len(v))
        warm = np.array([1.0, 0.99, 0.965])
        lo, hi = 0.78, 1.05
    br = np.clip(br, lo, hi)
    col = np.clip(br[:, None] * warm[None, :], 0.0, 1.0)
    rgba = np.concatenate([col, np.ones((len(v), 1))], axis=1)
    return (rgba * 255.0 + 0.5).astype(np.uint8)


def texture_mesh(mesh, kind, rng, palette, mottle):
    base, light, dark = palette
    size = 1024
    col = make_wood_albedo(size, rng, base, light, dark, kind, mottle)
    albedo = _to_img(col)
    normal = _to_img(albedo_to_normal(col, 2.0))
    mr = _to_img(metallic_roughness(col, kind))

    uv = trunk_uv(mesh) if kind == "trunk" else roots_uv(mesh)

    mat = trimesh.visual.material.PBRMaterial(
        name=kind,
        baseColorTexture=albedo,
        metallicRoughnessTexture=mr,
        normalTexture=normal,
        metallicFactor=1.0,        # actual metallic baked to 0 in texture B
        roughnessFactor=1.0,       # roughness comes from texture G
        doubleSided=(kind == "roots"),
    )
    vis = trimesh.visual.TextureVisuals(uv=uv, material=mat, image=albedo)
    vis.vertex_attributes["color"] = vertex_colors(mesh, kind, rng)
    mesh.visual = vis


# ============================================================================
# DRIVER
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Procedural textured driftwood -> GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    arr = np.asarray(img, dtype=float) / 255.0

    palette = sample_palette(arr)
    mottle_trunk = photo_mottle(arr, TRUNK_BOX, 1024)
    mottle_roots = photo_mottle(arr, ROOTS_BOX, 1024)

    scene = build_mesh(args.seed, args.density)
    tex_rng = np.random.default_rng(args.seed + 7919)

    texture_mesh(scene.geometry["trunk"], "trunk", tex_rng, palette, mottle_trunk)
    texture_mesh(scene.geometry["roots"], "roots", tex_rng, palette, mottle_roots)

    glb = scene.export(file_type="glb")
    with open(args.output, "wb") as fh:
        fh.write(glb)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("ERROR: {}\n".format(exc))
        sys.exit(1)