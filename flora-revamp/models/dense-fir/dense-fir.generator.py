#!/usr/bin/env python3
"""
Procedural textured Christmas-tree (balsam-fir) generator.

Builds the geometry (clumped leaf cards + thin wooden bole), derives foliage
and bark materials from a reference photo, applies per-surface UVs, and exports
a textured GLB.

CLI:
    python thisscript.py --image PATH --seed INT \
        --density {high,med,low} --output OUT.glb

Only numpy + trimesh + PIL + stdlib are used. Deterministic given --seed.
"""

import argparse
import math
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial


# ===========================================================================
# GEOMETRY  (+Y up, base at y=0, meters)
# ===========================================================================
OVERALL_HEIGHT = 2.0                      # meters; ~7 ft holiday tree
HEIGHT_OVER_WIDTH = 1.47                  # photo aspect: width/height ~= 0.68
BASE_RADIUS = OVERALL_HEIGHT / HEIGHT_OVER_WIDTH / 2.0   # ~0.68 m half-width
PROFILE_EXP = 0.80                        # <1 => convex, broad skirt
CROWN_WIDTH = 2.0 * BASE_RADIUS

TRUNK_TOP_FRAC = 0.92
TRUNK_RADIUS = 0.028
TRUNK_FLARE = 1.45
TRUNK_FLARE_FRAC = 0.06

SHELL_INSET = (0.80, 1.0)                 # clumps sit just inside the shell
N_TIERS = 11                              # whorl tiers => scalloped edge
LOBE_COUNT = 4
LOBE_AMP = 0.05
ENVELOPE_PAD = 1.01                       # max protrusion past shell (~1%)


def _density_config(density: str) -> dict:
    presets = {
        "high": dict(n_cards=4200, n_clumps=30, trunk_sides=14, branch_sides=6,
                     n_branches=30),
        "med":  dict(n_cards=1700, n_clumps=22, trunk_sides=10, branch_sides=5,
                     n_branches=16),
        "low":  dict(n_cards=520,  n_clumps=14, trunk_sides=6,  branch_sides=4,
                     n_branches=0),
    }
    if density not in presets:
        density = "high"
    return presets[density]


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([0.0, 1.0, 0.0])


def _envelope_radius(t, rng_phase=0.0):
    """Cone radius at height fraction t in [0,1] (t=0 base, t=1 apex)."""
    return BASE_RADIUS * np.power(max(0.0, 1.0 - t), PROFILE_EXP)


def _lobe_factor(theta, rng_phase):
    return 1.0 + LOBE_AMP * np.sin(LOBE_COUNT * theta + rng_phase)


def _frustum(p0, p1, r0, r1, sides):
    """A capped tapered cylinder (wood). Returns (verts, faces)."""
    axis = _normalize(p1 - p0)
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[1]) > 0.9 else np.array([0.0, 1.0, 0.0])
    u = _normalize(np.cross(axis, ref))
    w = np.cross(axis, u)
    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = np.cos(ang)[:, None] * u[None, :] + np.sin(ang)[:, None] * w[None, :]
    bottom = p0[None, :] + r0 * ring
    top = p1[None, :] + r1 * ring
    verts = np.vstack([bottom, top, p0[None, :], p1[None, :]])
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces.append([i, j, sides + j])
        faces.append([i, sides + j, sides + i])
    c0, c1 = 2 * sides, 2 * sides + 1
    for i in range(sides):
        j = (i + 1) % sides
        faces.append([c0, j, i])
        faces.append([c1, sides + i, sides + j])
    return verts, np.array(faces, dtype=np.int64)


def _accumulate(meshes):
    all_v, all_f, off = [], [], 0
    for v, f in meshes:
        all_v.append(v)
        all_f.append(f + off)
        off += len(v)
    if not all_v:
        return None
    return np.vstack(all_v), np.vstack(all_f)


def _build_wood(rng, cfg):
    H = OVERALL_HEIGHT
    sides = cfg["trunk_sides"]

    n_seg = 4
    ys = np.linspace(0.0, H * TRUNK_TOP_FRAC, n_seg + 1)
    trunk_parts = []
    for k in range(n_seg):
        y0, y1 = ys[k], ys[k + 1]
        t0, t1 = y0 / H, y1 / H
        r0 = TRUNK_RADIUS * (1.0 - 0.7 * t0)
        r1 = TRUNK_RADIUS * (1.0 - 0.7 * t1)
        if t0 < TRUNK_FLARE_FRAC:
            r0 *= 1.0 + (TRUNK_FLARE - 1.0) * (1.0 - t0 / TRUNK_FLARE_FRAC)
        if t1 < TRUNK_FLARE_FRAC:
            r1 *= 1.0 + (TRUNK_FLARE - 1.0) * (1.0 - t1 / TRUNK_FLARE_FRAC)
        p0 = np.array([0.0, y0, 0.0])
        p1 = np.array([0.0, y1, 0.0])
        trunk_parts.append(_frustum(p0, p1, r0, r1, sides))
    trunk = _accumulate(trunk_parts)

    branch_parts = []
    nb = cfg["n_branches"]
    bsides = cfg["branch_sides"]
    if nb > 0:
        for _ in range(nb):
            t = float(rng.uniform(0.05, 0.95)) ** 0.7
            y = t * H
            theta = float(rng.uniform(0.0, 2.0 * np.pi))
            shell_r = _envelope_radius(t) * _lobe_factor(theta, 0.0)
            r_par = TRUNK_RADIUS * (1.0 - 0.7 * t) + 1e-3
            start = np.array([0.0, y, 0.0])
            tip = np.array([np.cos(theta) * shell_r * 0.9,
                            y - shell_r * 0.12,
                            np.sin(theta) * shell_r * 0.9])
            r0 = min(r_par, 0.016 * (1.0 - 0.6 * t) + 0.004)
            r1 = r0 * 0.35
            branch_parts.append(_frustum(start, tip, r0, r1, bsides))
    branches = _accumulate(branch_parts)
    return trunk, branches


def _clamp_to_envelope(verts):
    """Snap every canopy vertex inside the cone envelope (radial + vertical).

    This removes straggling cards that overshoot the apex or poke past the
    shell, giving a clean, well-proportioned, scalloped cone silhouette and
    making the model's extents match the intended size.
    """
    H = OVERALL_HEIGHT
    verts[:, 1] = np.clip(verts[:, 1], 0.0, H)
    t = np.clip(verts[:, 1] / H, 0.0, 1.0)
    max_r = BASE_RADIUS * np.power(1.0 - t, PROFILE_EXP) * ENVELOPE_PAD + 0.02
    r = np.hypot(verts[:, 0], verts[:, 2])
    scale = np.where(r > max_r, max_r / np.maximum(r, 1e-6), 1.0)
    verts[:, 0] *= scale
    verts[:, 2] *= scale
    return verts


def _build_canopy(rng, cfg):
    H = OVERALL_HEIGHT
    n_cards_target = cfg["n_cards"]
    n_clumps = cfg["n_clumps"]
    lobe_phase = float(rng.uniform(0.0, 2.0 * np.pi))

    centers, shell_normals, c_radii = [], [], []
    # tiers, with extra weight at the broad lower whorls for a full skirt
    tier_t = np.linspace(0.03, 0.97, N_TIERS)
    for ci in range(n_clumps):
        # bias selection toward lower tiers (sqrt pulls index down)
        tsel = tier_t[int((1.0 - math.sqrt(rng.uniform(0.0, 1.0))) * (N_TIERS - 1))]
        t = float(np.clip(tsel + rng.normal(0.0, 0.035), 0.02, 0.97))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        shell_r = _envelope_radius(t) * _lobe_factor(theta, lobe_phase)
        r = shell_r * float(rng.uniform(*SHELL_INSET))
        centers.append(np.array([np.cos(theta) * r, t * H, np.sin(theta) * r]))
        shell_normals.append(_normalize(np.array([np.cos(theta), 0.30, np.sin(theta)])))
        c_radii.append(shell_r)
    # a few interior clumps for mass depth
    for _ in range(max(1, n_clumps // 5)):
        t = float(rng.uniform(0.1, 0.8))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        shell_r = _envelope_radius(t)
        r = shell_r * float(rng.uniform(0.2, 0.55))
        centers.append(np.array([np.cos(theta) * r, t * H, np.sin(theta) * r]))
        shell_normals.append(_normalize(np.array([np.cos(theta), 0.5, np.sin(theta)])))
        c_radii.append(shell_r)

    total_clumps = len(centers)
    cards_per_clump = max(8, n_cards_target // total_clumps)
    clump_radius = 0.08 * CROWN_WIDTH          # tighter clumps
    half_size = 0.036 * CROWN_WIDTH            # smaller cards => finer mass

    verts_list, faces_list = [], []
    voff = 0
    quad_idx = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    for c, n_shell, shell_r in zip(centers, shell_normals, c_radii):
        for _ in range(cards_per_clump):
            off = rng.normal(0.0, 1.0, 3) * (clump_radius * np.array([1.0, 0.7, 1.0]))
            center = c + off
            rad_xz = np.hypot(center[0], center[2])
            t_here = np.clip(center[1] / H, 0.0, 1.0)
            max_r = _envelope_radius(t_here) * ENVELOPE_PAD
            if rad_xz > max_r and rad_xz > 1e-6:
                s_ = max_r / rad_xz
                center[0] *= s_
                center[2] *= s_

            # nearly tangent: small normal jitter => smooth, dense shell
            n = _normalize(n_shell + rng.normal(0.0, 0.14, 3))
            up = np.array([0.0, 1.0, 0.0])
            if np.linalg.norm(np.cross(up, n)) < 1e-4:
                u = np.array([1.0, 0.0, 0.0])
            else:
                u = _normalize(np.cross(up, n))
            v = _normalize(np.cross(n, u))
            ang = float(rng.uniform(0.0, 2.0 * np.pi))
            ca, sa = np.cos(ang), np.sin(ang)
            u2 = ca * u + sa * v
            v2 = _normalize((-sa * u + ca * v) + up * 0.12)   # gentle upturn

            s = half_size * float(np.exp(rng.normal(0.0, 0.20)))
            su, sv = s, s * 1.2
            quad = np.array([
                center - u2 * su - v2 * sv,
                center + u2 * su - v2 * sv,
                center + u2 * su + v2 * sv,
                center - u2 * su + v2 * sv,
            ])
            verts_list.append(quad)
            faces_list.append(quad_idx + voff)
            voff += 4

    verts = _clamp_to_envelope(np.vstack(verts_list))
    faces = np.vstack(faces_list)
    return verts, faces


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    cfg = _density_config(density)
    scene = trimesh.Scene()

    trunk, branches = _build_wood(rng, cfg)
    trunk_mesh = trimesh.Trimesh(vertices=trunk[0], faces=trunk[1], process=True)
    scene.add_geometry(trunk_mesh, geom_name="trunk")

    if branches is not None:
        br_mesh = trimesh.Trimesh(vertices=branches[0], faces=branches[1], process=True)
        scene.add_geometry(br_mesh, geom_name="branches")

    cv, cf = _build_canopy(rng, cfg)
    canopy = trimesh.Trimesh(vertices=cv, faces=cf, process=False)
    canopy.fix_normals()
    scene.add_geometry(canopy, geom_name="canopy")

    bounds = scene.bounds
    cx = 0.5 * (bounds[0][0] + bounds[1][0])
    cz = 0.5 * (bounds[0][2] + bounds[1][2])
    miny = bounds[0][1]
    scene.apply_translation([-cx, -miny, -cz])
    return scene


# ===========================================================================
# COLOR SAMPLING FROM THE PHOTO
# ===========================================================================
def sample_foliage_palette(image_path):
    """Sample green foliage colors from WELL INSIDE the tree silhouette."""
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    H, W = arr.shape[:2]
    half = max(3, int(0.012 * min(H, W)))

    centers = [(0.50, 0.45), (0.45, 0.40), (0.55, 0.55), (0.42, 0.60),
               (0.58, 0.45), (0.50, 0.62), (0.50, 0.33), (0.47, 0.72),
               (0.53, 0.50), (0.40, 0.50), (0.60, 0.55), (0.50, 0.52)]

    pix = []
    for fx, fy in centers:
        cx, cy = int(fx * W), int(fy * H)
        x0, x1 = max(0, cx - half), min(W, cx + half)
        y0, y1 = max(0, cy - half), min(H, cy + half)
        win = arr[y0:y1, x0:x1].reshape(-1, 3)
        if win.size == 0:
            continue
        med = np.median(win, axis=0)
        r, g, b = med
        sat = med.max() - med.min()
        if sat < 22 or med.mean() > 175 or not (g >= r - 3 and g >= b - 3):
            continue
        pix.append(win)

    if pix:
        allpix = np.vstack(pix)
        dark = np.percentile(allpix, 30, axis=0)
        mid = np.percentile(allpix, 55, axis=0)
        light = np.percentile(allpix, 82, axis=0)
    else:
        dark = np.array([28, 58, 30], np.float32)
        mid = np.array([48, 88, 46], np.float32)
        light = np.array([96, 130, 70], np.float32)

    # guarantee a healthy value spread (bright tips, deep shadows)
    if light.mean() < mid.mean() * 1.30:
        light = np.clip(mid * 1.5, 0, 255)
    if dark.mean() > mid.mean() * 0.78:
        dark = np.clip(mid * 0.58, 0, 255)

    return {
        "dark": tuple(int(c) for c in dark),
        "mid": tuple(int(c) for c in mid),
        "light": tuple(int(c) for c in light),
    }


# ===========================================================================
# TEXTURE SYNTHESIS
# ===========================================================================
def _mix(a, b, t):
    return a + (b - a) * t


def _rgb_tuple(c):
    c = np.clip(c, 0, 255)
    return (int(c[0]), int(c[1]), int(c[2]), 255)


def generate_foliage_atlas(palette, rng):
    """4x4 atlas of distinct needle-cluster tiles drawn with PIL polygons.

    Solid interior body + feathery needle edge => mostly-binary alpha. Sunlit
    (upper) tiles brighter/warmer; shaded (lower) tiles darker/cooler.
    """
    TILE = 256
    SS = 4
    R = TILE * SS
    dark = np.array(palette["dark"], np.float32)
    mid = np.array(palette["mid"], np.float32)
    light = np.array(palette["light"], np.float32)
    warm = np.array([20.0, 12.0, -6.0])

    atlas = Image.new("RGBA", (TILE * 4, TILE * 4), (0, 0, 0, 0))

    for idx in range(16):
        row, col = idx // 4, idx % 4
        lit = 0.92 + 0.34 * ((3 - row) / 3.0) + float(rng.uniform(-0.05, 0.05))
        tile = Image.new("RGBA", (R, R), (0, 0, 0, 0))
        d = ImageDraw.Draw(tile)

        nsprig = int(rng.integers(8, 12))
        for _ in range(nsprig):
            bx = float(rng.uniform(0.16, 0.84)) * R
            by = float(rng.uniform(0.55, 0.96)) * R
            a = float(rng.uniform(-0.5, 0.5))
            dx, dy = math.sin(a), -math.cos(a)        # mostly upward
            length = float(rng.uniform(0.48, 0.84)) * R
            nstep = 15
            for k in range(nstep):
                f = k / (nstep - 1)
                px = bx + dx * length * f
                py = by + dy * length * f
                rad = (1.0 - f) * float(rng.uniform(0.045, 0.08)) * R + 0.012 * R
                body = _mix(dark, mid, 0.35 + 0.4 * f) * lit + warm * (lit - 0.9)
                d.ellipse([px - rad, py - rad, px + rad, py + rad],
                          fill=_rgb_tuple(body))
                for _ in range(int(rng.integers(2, 5))):
                    na = a + float(rng.uniform(-0.95, 0.95))
                    ndx, ndy = math.sin(na), -abs(math.cos(na))
                    nl = rad * float(rng.uniform(2.0, 3.6))
                    ex, ey = px + ndx * nl, py + ndy * nl
                    w = rad * 0.36
                    perp = (-ndy, ndx)
                    p1 = (px + perp[0] * w, py + perp[1] * w)
                    p2 = (px - perp[0] * w, py - perp[1] * w)
                    needle = _mix(mid, light, float(rng.uniform(0.25, 1.0))) * lit \
                        + warm * (lit - 0.9)
                    d.polygon([p1, p2, (ex, ey)], fill=_rgb_tuple(needle))

        tile_small = tile.resize((TILE, TILE), Image.LANCZOS)
        atlas.paste(tile_small, (col * TILE, row * TILE), tile_small)

    a = np.asarray(atlas).astype(np.float32)
    alpha = a[:, :, 3] / 255.0
    alpha = np.clip((alpha - 0.35) / 0.30, 0.0, 1.0)   # narrow AA band only
    a[:, :, 3] = alpha * 255.0
    return Image.fromarray(a.astype(np.uint8), "RGBA")


def generate_bark(palette, rng):
    """Warm-brown bark albedo with vertical grain and a visible value range."""
    N = 512
    shadow = np.array(palette["dark"], np.float32)
    base = np.clip(0.5 * shadow + 0.5 * np.array([82.0, 56.0, 36.0]), 0, 255)

    col = rng.normal(0.0, 1.0, N)
    kern = np.ones(9) / 9.0
    col = np.convolve(np.concatenate([col[-4:], col, col[:4]]), kern, "same")[4:-4]
    grain = np.tile(col[None, :], (N, 1))
    fine = rng.normal(0.0, 1.0, (N, N))
    fine = (fine + np.roll(fine, 1, axis=0) + np.roll(fine, 2, axis=0)) / 3.0
    streaks = np.sin(np.linspace(0, 38 * np.pi, N))[None, :]

    value = 1.0 + 0.30 * grain + 0.10 * fine + 0.10 * streaks
    value = np.clip(value, 0.6, 1.6)
    img = np.clip(base[None, None, :] * value[:, :, None], 0, 255).astype(np.uint8)
    return Image.fromarray(img, "RGB")


# ===========================================================================
# UVs + VERTEX COLOR (COLOR_0)
# ===========================================================================
def _canopy_uv_and_color(mesh, rng):
    verts = mesh.vertices
    n = len(verts)
    n_cards = n // 4

    tiles = rng.integers(0, 16, n_cards)
    rots = rng.integers(0, 4, n_cards)
    card_bright = rng.uniform(0.92, 1.10, n_cards)

    m = 0.03
    cuv = m + np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32) * (1 - 2 * m)

    uv = np.zeros((n, 2), np.float32)
    col0 = np.zeros((n, 4), np.uint8)
    H = OVERALL_HEIGHT

    for i in range(n_cards):
        row, c = int(tiles[i]) // 4, int(tiles[i]) % 4
        order = [(k + int(rots[i])) % 4 for k in range(4)]
        for k in range(4):
            uu, vv = cuv[order[k]]
            uv[4 * i + k] = ((c + uu) / 4.0, 1.0 - (row + vv) / 4.0)

            p = verts[4 * i + k]
            hf = np.clip(p[1] / H, 0.0, 1.0)
            shell = _envelope_radius(hf) + 1e-6
            depth = np.clip(np.hypot(p[0], p[2]) / shell, 0.0, 1.0)
            b = (0.74 + 0.26 * hf + 0.20 * depth) * card_bright[i]
            b = float(np.clip(b, 0.6, 1.22))
            rgb = np.clip(np.array([b, b, b * 0.95]) * 255.0, 0, 255)
            col0[4 * i + k] = (rgb[0], rgb[1], rgb[2], 255)

    return uv, col0


def _cyl_uv_and_color(mesh, v_repeat):
    verts = mesh.vertices
    ang = np.arctan2(verts[:, 2], verts[:, 0])
    u = ((ang / (2.0 * np.pi)) % 1.0 * 2.0) % 1.0
    ymin = verts[:, 1].min()
    yrange = max(1e-6, np.ptp(verts[:, 1]))
    v = (verts[:, 1] - ymin) / yrange * v_repeat
    uv = np.column_stack([u, v]).astype(np.float32)

    hf = (verts[:, 1] - ymin) / yrange
    b = np.clip(0.70 + 0.30 * hf, 0.0, 1.0)
    rgb = np.clip(np.column_stack([b, b, b * 0.97]) * 255.0, 0, 255)
    col0 = np.column_stack([rgb, np.full(len(verts), 255)]).astype(np.uint8)
    return uv, col0


# ===========================================================================
# ASSEMBLY
# ===========================================================================
def texture_scene(scene, palette, seed):
    rng = np.random.default_rng(seed ^ 0x5EED2026)

    atlas = generate_foliage_atlas(palette, rng)
    bark = generate_bark(palette, rng)

    foliage_mat = PBRMaterial(
        name="needles",
        baseColorTexture=atlas,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0,
        roughnessFactor=0.85,
        alphaMode="MASK",
        alphaCutoff=0.45,
        doubleSided=True,
    )
    bark_mat = PBRMaterial(
        name="bark",
        baseColorTexture=bark,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0,
        roughnessFactor=0.9,
        doubleSided=False,
    )

    for name, mesh in list(scene.geometry.items()):
        if name == "canopy":
            uv, col0 = _canopy_uv_and_color(mesh, rng)
            mesh.visual = TextureVisuals(uv=uv, material=foliage_mat)
            mesh.visual.vertex_attributes["color"] = col0
        elif name in ("trunk", "branches"):
            v_repeat = 3.0 if name == "trunk" else 4.0
            uv, col0 = _cyl_uv_and_color(mesh, v_repeat)
            mesh.visual = TextureVisuals(uv=uv, material=bark_mat)
            mesh.visual.vertex_attributes["color"] = col0
    return scene


def main():
    ap = argparse.ArgumentParser(description="Procedural textured conifer -> GLB")
    ap.add_argument("--image", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    try:
        palette = sample_foliage_palette(args.image)
        scene = build_mesh(args.seed, args.density)
        scene = texture_scene(scene, palette, args.seed)
        glb = scene.export(file_type="glb")
        with open(args.output, "wb") as fh:
            fh.write(glb)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    tri = sum(len(m.faces) for m in scene.geometry.values())
    print(f"Wrote {args.output}  ({tri} triangles, density={args.density})")
    return 0


if __name__ == "__main__":
    sys.exit(main())