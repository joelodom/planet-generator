#!/usr/bin/env python3
"""
Procedural KAPOK / CEIBA emergent tree -> textured GLB.

Tall columnar trunk + plank buttresses, a few primary limbs, and a broad,
flattened, continuous parasol of clumped leaf cards. Materials are derived
by SAMPLING COLORS from a reference photo; UVs are cylindrical for wood and
per-tile atlas for leaf cards; sun/shade + AO bake into per-vertex COLOR_0.

CLI:
    python thisscript --image PATH --seed INT --density {high,med,low} --output OUT.glb
"""

import argparse
import sys
import math

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter


# ===========================================================================
# GEOMETRY  (build_mesh -- proportions measured by eye off the reference)
# ===========================================================================
TREE_HEIGHT = 42.0          # meters, total height (kapok emergent scale)

TRUNK_TOP_FRAC      = 0.70  # crown starts at ~70% of total height
BUTTRESS_FRAC       = 0.12  # plank buttresses rise to ~12% of total height
BASE_FLARE_FRAC     = 0.08  # bottom ~8% of the bole flares outward
BASE_FLARE_MULT     = 1.55  # radius multiplier at the very base

TRUNK_BASE_RADIUS   = 1.05  # m, just above the buttress skirt
TRUNK_TOP_RADIUS    = 0.42  # m, where the crown limbs emerge

# Crown: a wide, flattened parasol. Widened so the front-view aspect
# (width/height) matches the photo (~0.69) instead of reading too slim.
CROWN_HALF_WIDTH    = 0.29 * TREE_HEIGHT    # broad flat parasol
CROWN_HALF_HEIGHT   = 0.105 * TREE_HEIGHT   # flattened: depth << width

BUTTRESS_OUT_MULT   = 1.9   # outer ground reach as a multiple of base radius


def _density_params(density):
    if density == "high":
        return dict(trunk_sides=16, trunk_segs=30, buttresses=8,
                    butt_samples=9, n_primary=6, branch_sides=8,
                    branch_segs=5, n_cards=3400, n_clumps=30)
    if density == "med":
        return dict(trunk_sides=10, trunk_segs=16, buttresses=6,
                    butt_samples=7, n_primary=5, branch_sides=6,
                    branch_segs=4, n_cards=1400, n_clumps=22)
    if density == "low":
        return dict(trunk_sides=7, trunk_segs=8, buttresses=5,
                    butt_samples=5, n_primary=4, branch_sides=5,
                    branch_segs=3, n_cards=430, n_clumps=14)
    raise ValueError("density must be one of 'high', 'med', 'low'")


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _frame(tangent):
    """An orthonormal (u, v) pair perpendicular to a tangent direction."""
    t = _unit(tangent)
    ref = np.array([0.0, 1.0, 0.0]) if abs(t[1]) < 0.95 else np.array([1.0, 0.0, 0.0])
    u = _unit(np.cross(ref, t))
    v = _unit(np.cross(t, u))
    return u, v


def _build_tube(centers, radii, sides, cap_bottom=True, cap_top=True,
                rng=None, bark=0.0):
    """Loft a circular tube along a polyline of centers with per-node radii."""
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    n = len(centers)
    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)

    verts = []
    for i in range(n):
        if i == 0:
            tan = centers[1] - centers[0]
        elif i == n - 1:
            tan = centers[-1] - centers[-2]
        else:
            tan = centers[i + 1] - centers[i - 1]
        u, vv = _frame(tan)
        r = radii[i]
        ring = centers[i] + r * (np.outer(ca, u) + np.outer(sa, vv))
        if bark > 0.0 and rng is not None:
            # subtle vertical-streak bark relief, kept small to not muddy form
            streak = 1.0 + bark * (0.6 * np.sin(3.0 * ang + i * 0.4)
                                   + rng.normal(0.0, 0.25, sides))
            ring = centers[i] + (r * streak)[:, None] * (
                np.outer(ca, u) + np.outer(sa, vv))
        verts.append(ring)
    verts = np.vstack(verts)

    faces = []
    for i in range(n - 1):
        a = i * sides
        b = (i + 1) * sides
        for j in range(sides):
            jn = (j + 1) % sides
            faces.append([a + j, b + j, b + jn])
            faces.append([a + j, b + jn, a + jn])

    if cap_bottom:
        c0 = len(verts)
        verts = np.vstack([verts, centers[0]])
        for j in range(sides):
            jn = (j + 1) % sides
            faces.append([c0, jn, j])
    if cap_top:
        c1 = len(verts)
        verts = np.vstack([verts, centers[-1]])
        base = (n - 1) * sides
        for j in range(sides):
            jn = (j + 1) % sides
            faces.append([c1, base + j, base + jn])

    return verts, np.asarray(faces, dtype=np.int64)


def _trunk_profile(segs):
    """Centerline heights + radii for the bole, with a basal flare."""
    top_y = TRUNK_TOP_FRAC * TREE_HEIGHT
    t = np.linspace(0.0, 1.0, segs)
    ys = t * top_y
    radii = TRUNK_BASE_RADIUS + (TRUNK_TOP_RADIUS - TRUNK_BASE_RADIUS) * t
    flare_h = BASE_FLARE_FRAC * TREE_HEIGHT
    fmask = ys < flare_h
    f = np.clip((flare_h - ys[fmask]) / max(flare_h, 1e-6), 0.0, 1.0)
    radii[fmask] *= 1.0 + (BASE_FLARE_MULT - 1.0) * (f ** 2)
    centers = np.column_stack([np.zeros(segs), ys, np.zeros(segs)])
    return centers, radii


def _buttress(azimuth, samples, thickness):
    """One thin plank buttress fin in the radial/vertical plane at `azimuth`."""
    rad_dir = np.array([np.cos(azimuth), 0.0, np.sin(azimuth)])
    tang = np.array([-np.sin(azimuth), 0.0, np.cos(azimuth)])  # thickness dir

    r_attach = TRUNK_BASE_RADIUS * 0.92          # inside the bark -> connected
    H = BUTTRESS_FRAC * TREE_HEIGHT
    R_out = TRUNK_BASE_RADIUS * BUTTRESS_OUT_MULT

    s = np.linspace(0.0, 1.0, samples)
    heights = H * s
    radii = r_attach + (R_out - r_attach) * (1.0 - s) ** 2.4
    curve = [r * rad_dir + np.array([0.0, h, 0.0])
             for r, h in zip(radii, heights)]
    inner_ground = r_attach * rad_dir
    poly = curve + [inner_ground]
    poly = np.asarray(poly)

    half = 0.5 * thickness * tang
    front = poly + half
    back = poly - half
    m = len(poly)
    verts = np.vstack([front, back])
    faces = []
    for k in range(1, m - 1):
        faces.append([0, k, k + 1])              # front
        faces.append([m, m + k + 1, m + k])      # back (reversed)
    for k in range(m):                           # side wall strip
        kn = (k + 1) % m
        faces.append([k, kn, m + kn])
        faces.append([k, m + kn, m + k])
    return verts, np.asarray(faces, dtype=np.int64)


def _branches(p, n_primary, sides, segs, r_base, rng):
    """Primary limbs fanning up-and-out from the trunk top; returns tips too.

    Flatter elevation + lengths kept well inside the crown so the foliage
    seats on the limbs as one continuous parasol instead of floating above.
    """
    meshes = []
    tips = []
    r_child = r_base / np.sqrt(n_primary)        # r_parent^2 ~= sum r_child^2
    az0 = rng.uniform(0.0, 2.0 * np.pi)
    for i in range(n_primary):
        az = az0 + 2.0 * np.pi * i / n_primary + rng.uniform(-0.25, 0.25)
        el = np.radians(rng.uniform(20.0, 42.0))  # flatter, layered tiers
        out = np.array([np.cos(az), 0.0, np.sin(az)])
        up = np.array([0.0, 1.0, 0.0])
        length = CROWN_HALF_WIDTH * rng.uniform(0.50, 0.72)
        dir0 = _unit(np.cos(el) * out + np.sin(el) * up)

        t = np.linspace(0.0, 1.0, segs)
        centers = (p[None, :]
                   + np.outer(t * length, dir0)
                   + np.outer((t ** 2) * length * 0.10, -up))
        radii = r_child * (1.0 - 0.78 * t) + 0.04
        v, f = _build_tube(centers, radii, sides,
                           cap_bottom=False, cap_top=True)
        meshes.append(trimesh.Trimesh(vertices=v, faces=f, process=False))
        tips.append(centers[-1])
    return meshes, tips


def _make_lobes(rng):
    """Low-frequency radial bulge function for the crown shell."""
    n = int(rng.integers(3, 7))
    freqs = rng.integers(2, 5, n)
    phases = rng.uniform(0.0, 2.0 * np.pi, n)
    amps = rng.uniform(0.05, 0.12, n)

    def lobe(theta):
        return 1.0 + float(np.sum(amps * np.cos(freqs * theta + phases)))
    return lobe


def _canopy(center, lobe, n_cards, n_clumps, branch_tips, rng):
    """Clumped leaf cards densely filling a wide, flattened crown envelope.

    Many overlapping clumps biased toward the rim (low elevation) read as a
    single broad parasol; clump centers are clamped onto the envelope so no
    clump can float free of the mass.
    """
    W = CROWN_HALF_WIDTH
    Hc = CROWN_HALF_HEIGHT
    radii_vec = np.array([W, Hc, W])

    clump_centers = []
    clump_normals = []
    # branch tips anchor some clumps onto the actual limbs
    for tip in branch_tips:
        d = _unit(tip - center)
        clump_centers.append(tip)
        clump_normals.append(d)

    n_interior = max(2, n_clumps // 7)
    while len(clump_centers) < n_clumps:
        theta = rng.uniform(0.0, 2.0 * np.pi)
        # bias elevation low -> spread wide around the rim (flat parasol),
        # while still doming the top.
        el = np.radians(85.0 * (rng.random() ** 1.6))
        d = np.array([np.cos(el) * np.cos(theta),
                      np.sin(el),
                      np.cos(el) * np.sin(theta)])
        shell = center + radii_vec * d * lobe(theta)
        interior = len(clump_centers) >= (n_clumps - n_interior)
        frac = rng.uniform(0.45, 0.65) if interior else rng.uniform(0.88, 1.0)
        clump_centers.append(center + (shell - center) * frac)
        clump_normals.append(_unit(shell - center))

    clump_centers = np.asarray(clump_centers)
    clump_normals = np.asarray(clump_normals)

    clump_radius = 0.16 * W        # larger -> clumps overlap into one mass
    card_half = 0.050 * W
    per = max(8, n_cards // n_clumps)

    all_v = []
    all_f = []
    vbase = 0
    for c, nrm in zip(clump_centers, clump_normals):
        for _ in range(per):
            off = rng.normal(0.0, 0.55, 3) * clump_radius
            ctr = c + off
            n = _unit(nrm + rng.normal(0.0, 0.42, 3))  # ~+/-25deg jitter
            u, v = _frame(n)
            hs = float(np.clip(rng.lognormal(np.log(card_half), 0.35),
                               card_half * 0.5, card_half * 2.0))
            quad = np.array([ctr + u * hs + v * hs,
                             ctr - u * hs + v * hs,
                             ctr - u * hs - v * hs,
                             ctr + u * hs - v * hs])
            all_v.append(quad)
            all_f.append([[vbase, vbase + 1, vbase + 2],
                          [vbase, vbase + 2, vbase + 3]])
            vbase += 4
    verts = np.vstack(all_v)
    faces = np.vstack(all_f).astype(np.int64)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    rng = np.random.default_rng(seed)
    p = _density_params(density)

    # ---- Trunk bole -------------------------------------------------------
    centers, radii = _trunk_profile(p["trunk_segs"])
    tv, tf = _build_tube(centers, radii, p["trunk_sides"],
                         cap_bottom=True, cap_top=True,
                         rng=rng, bark=0.05)
    trunk_parts = [trimesh.Trimesh(vertices=tv, faces=tf, process=False)]

    # ---- Plank buttresses (part of the "trunk" surface) -------------------
    az0 = rng.uniform(0.0, 2.0 * np.pi)
    for i in range(p["buttresses"]):
        az = az0 + 2.0 * np.pi * i / p["buttresses"] + rng.uniform(-0.12, 0.12)
        thick = TRUNK_BASE_RADIUS * rng.uniform(0.22, 0.32)
        bv, bf = _buttress(az, p["butt_samples"], thick)
        trunk_parts.append(trimesh.Trimesh(vertices=bv, faces=bf, process=False))

    trunk_mesh = trimesh.util.concatenate(trunk_parts)
    trunk_mesh.merge_vertices()
    trunk_mesh.fix_normals()

    # ---- Primary limbs ----------------------------------------------------
    p_top = np.array([0.0, TRUNK_TOP_FRAC * TREE_HEIGHT, 0.0])
    branch_meshes, tips = _branches(p_top, p["n_primary"], p["branch_sides"],
                                    p["branch_segs"], TRUNK_TOP_RADIUS, rng)
    branch_mesh = trimesh.util.concatenate(branch_meshes)
    branch_mesh.merge_vertices()
    branch_mesh.fix_normals()

    # ---- Canopy of clumped leaf cards ------------------------------------
    crown_base_y = TRUNK_TOP_FRAC * TREE_HEIGHT
    # seat the crown lower so its underside overlaps the limbs (continuous)
    crown_center = np.array([0.0, crown_base_y + CROWN_HALF_HEIGHT * 0.55, 0.0])
    lobe = _make_lobes(rng)
    canopy_mesh = _canopy(crown_center, lobe, p["n_cards"], p["n_clumps"],
                          tips, rng)

    scene = trimesh.Scene()
    scene.add_geometry(trunk_mesh, geom_name="trunk")
    scene.add_geometry(branch_mesh, geom_name="branches")
    scene.add_geometry(canopy_mesh, geom_name="canopy")

    lo = scene.bounds[0]
    scene.apply_translation([0.0, -lo[1], 0.0])
    return scene


# ===========================================================================
# TEXTURING  -- sample the photo, synthesize tileable swatches + leaf atlas
# ===========================================================================
LUM_W = np.array([0.299, 0.587, 0.114])


def _load_image(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.uint8)


def _crop_norm(img, box):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    a, b = int(x0 * W), int(y0 * H)
    c, d = int(x1 * W), int(y1 * H)
    a, c = max(0, min(a, W - 2)), min(W, max(c, a + 2))
    b, d = max(0, min(b, H - 2)), min(H, max(d, b + 2))
    return img[b:d, a:c]


def _sample_patches(img, box, n, rng, keep_green=False):
    """Median colors of many small patches inside `box`, outliers rejected."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    ph = max(2, int(0.012 * min(H, W)))
    cols = []
    for _ in range(n):
        px = int(rng.uniform(x0, x1) * W)
        py = int(rng.uniform(y0, y1) * H)
        a = np.clip(px - ph, 0, W - 1)
        c = np.clip(px + ph, a + 1, W)
        b = np.clip(py - ph, 0, H - 1)
        d = np.clip(py + ph, b + 1, H)
        patch = img[b:d, a:c].reshape(-1, 3).astype(np.float64)
        col = np.median(patch, axis=0)
        if keep_green:
            # drop grey background / sky: foliage is clearly green-dominant
            if not (col[1] > col[2] + 4 and col[1] > 28 and col[1] < 235):
                continue
        cols.append(col)
    if not cols:
        return None
    cols = np.array(cols)
    med = np.median(cols, axis=0)
    dist = np.linalg.norm(cols - med, axis=1)
    keep = cols[dist < (np.median(dist) * 2.0 + 12.0)]
    if len(keep) == 0:
        keep = cols
    return keep


def _delight(rgb):
    """Divide out a heavily blurred luminance; clamp gain to [0.6, 1.6]."""
    rgb = rgb.astype(np.float64)
    h, w = rgb.shape[:2]
    lum = rgb @ LUM_W
    rad = max(2, min(h, w) // 3)
    blur = np.asarray(Image.fromarray(lum.astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(rad))).astype(np.float64)
    blur = np.clip(blur, 1.0, 255.0)
    target = float(np.median(lum))
    gain = np.clip(target / blur, 0.6, 1.6)
    return np.clip(rgb * gain[..., None], 0, 255).astype(np.uint8)


def _mirror_tile(rgb):
    """Reflect-pad into a 2x2 mirror so the result tiles seamlessly."""
    top = np.concatenate([rgb, rgb[:, ::-1]], axis=1)
    full = np.concatenate([top, top[::-1]], axis=0)
    return full


def _make_bark(img, rng, res=1024):
    """Tileable PALE GREY-GREEN bark albedo from a trunk swatch + soft relief.

    The sampled tone is desaturated and lifted toward a silvery grey-green so
    the bole reads pale (not brown), and contrast is kept low so the
    cylindrical UV seam is not conspicuous.
    """
    base = _sample_patches(img, (0.46, 0.45, 0.54, 0.72), 36, rng)
    base_col = (np.median(base, axis=0) if base is not None
                else np.array([172.0, 174.0, 162.0]))
    # desaturate toward luminance, give a faint cool green cast, lift value
    g = float(base_col @ LUM_W)
    base_col = 0.42 * base_col + 0.58 * np.array([g, g, g])
    base_col = base_col * np.array([0.99, 1.02, 0.97])
    base_col = np.clip(base_col * np.clip(170.0 / max(g, 70.0), 0.95, 1.5),
                       70.0, 232.0)

    swatch = _crop_norm(img, (0.46, 0.55, 0.54, 0.78))
    swatch = _delight(swatch)
    tile = _mirror_tile(swatch)
    arr = np.asarray(Image.fromarray(tile).resize((res, res), Image.LANCZOS)
                     ).astype(np.float64)

    # pull tone firmly toward the pale grey-green base, keep some detail
    mean = arr.reshape(-1, 3).mean(axis=0)
    arr = 0.35 * arr + 0.65 * (arr * (base_col / np.clip(mean, 1.0, None)))

    # faint vertical streaking (periodic -> stays tileable), low amplitude
    xx = np.linspace(0.0, 2.0 * np.pi, res, endpoint=False)
    streak = 1.0 + 0.035 * np.sin(7.0 * xx)[None, :, None]
    arr = arr * streak

    # subtle lenticel speckles / silvery flecks (per-pixel -> tileable)
    spk = rng.random((res, res))
    arr[spk < 0.0035] *= 0.72
    arr[spk > 0.9975] *= 1.12

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _normal_from_albedo(pil_img, strength=1.3):
    """Tangent-space normal map from albedo luminance (periodic Sobel)."""
    arr = np.asarray(pil_img).astype(np.float64) @ LUM_W
    arr /= 255.0
    gx = (np.roll(arr, -1, axis=1) - np.roll(arr, 1, axis=1)) * 0.5
    gy = (np.roll(arr, -1, axis=0) - np.roll(arr, 1, axis=0)) * 0.5
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(arr)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    out = np.stack([nx / norm, ny / norm, nz / norm], axis=-1)
    out = ((out * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB")


def _leaf_palette(img, rng):
    greens = _sample_patches(img, (0.22, 0.06, 0.78, 0.34), 60, rng,
                             keep_green=True)
    if greens is None or len(greens) < 3:
        greens = np.array([[70, 104, 52], [96, 132, 64],
                           [54, 86, 44], [120, 150, 80]], dtype=np.float64)
    return greens


def _draw_palmate(draw, center, size, angle, color, rng):
    """A small palmate cluster of pointed leaflets as filled polygons."""
    n_leaf = int(rng.integers(5, 8))
    spread = math.radians(rng.uniform(70, 120))
    for k in range(n_leaf):
        a = angle - spread / 2 + spread * (k / max(n_leaf - 1, 1))
        a += rng.uniform(-0.08, 0.08)
        d = np.array([math.cos(a), math.sin(a)])
        pdir = np.array([-d[1], d[0]])
        L = size * rng.uniform(0.7, 1.1)
        Wd = L * rng.uniform(0.28, 0.42)
        base = np.array(center)
        pts = [base,
               base + d * 0.30 * L + pdir * 0.50 * Wd,
               base + d * 0.72 * L + pdir * 0.38 * Wd,
               base + d * L,
               base + d * 0.72 * L - pdir * 0.38 * Wd,
               base + d * 0.30 * L - pdir * 0.50 * Wd]
        jit = rng.uniform(0.85, 1.12)
        col = tuple(int(np.clip(c * jit, 0, 255)) for c in color)
        draw.polygon([tuple(pt) for pt in pts], fill=col + (255,))


def _make_leaf_atlas(img, rng, res=1024, grid=4):
    """4x4 atlas of distinct foliage cluster tiles (sunlit warm -> shaded cool)."""
    palette = _leaf_palette(img, rng)
    tile = res // grid
    ss = 4
    atlas = Image.new("RGBA", (res, res), (0, 0, 0, 0))

    for row in range(grid):
        b = 1.18 - 0.5 * (row / (grid - 1))
        warm = np.array([1.0 + 0.10 * (1 - row / (grid - 1)),
                         1.0,
                         1.0 - 0.10 * (1 - row / (grid - 1))])
        for col in range(grid):
            work = Image.new("RGBA", (tile * ss, tile * ss), (0, 0, 0, 0))
            draw = ImageDraw.Draw(work)
            n_clu = int(rng.integers(11, 17))
            for _ in range(n_clu):
                bsc = palette[rng.integers(len(palette))]
                cc = np.clip(bsc * b * warm, 0, 255)
                center = (rng.uniform(0.1, 0.9) * tile * ss,
                          rng.uniform(0.1, 0.9) * tile * ss)
                size = rng.uniform(0.22, 0.40) * tile * ss
                ang = rng.uniform(0, 2 * math.pi)
                _draw_palmate(draw, center, size, ang, cc, rng)
            work = work.resize((tile, tile), Image.LANCZOS)
            atlas.paste(work, (col * tile, row * tile), work)

    # push alpha toward binary: keep only anti-aliased edges
    a = np.asarray(atlas).copy()
    al = a[..., 3].astype(np.float64)
    al = np.where(al > 128, 255, np.where(al < 40, 0, al)).astype(np.uint8)
    a[..., 3] = al
    return Image.fromarray(a, "RGBA"), grid


# ---- UVs ------------------------------------------------------------------
def _cyl_uv(verts, v_period, u_rep):
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    u = (np.arctan2(z, x) / (2.0 * np.pi) + 0.5) * u_rep
    v = y / v_period
    return np.column_stack([u, v]).astype(np.float64)


def _card_uv(n_verts, grid, rng):
    """One atlas tile per leaf card (groups of 4 verts), random rotation."""
    n_cards = n_verts // 4
    inset = 1.0 / 2048.0
    step = 1.0 / grid
    corners = [(1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    uv = np.zeros((n_verts, 2), dtype=np.float64)
    for k in range(n_cards):
        idx = int(rng.integers(0, grid * grid))
        rot = int(rng.integers(0, 4))
        ci, ri = idx % grid, idx // grid
        u0, v0 = ci * step, ri * step
        rc = corners[rot:] + corners[:rot]
        for j in range(4):
            fu, fv = rc[j]
            uu = u0 + inset + fu * (step - 2 * inset)
            vv = v0 + inset + fv * (step - 2 * inset)
            uv[k * 4 + j] = (uu, vv)
    return uv


# ---- per-vertex COLOR_0 ---------------------------------------------------
def _canopy_colors(verts, rng):
    """Sun/shade gradient: top & outer brighter/warmer, inner/lower darker."""
    y = verts[:, 1]
    ymin, ymax = float(y.min()), float(y.max())
    h = (y - ymin) / max(ymax - ymin, 1e-6)
    cx, cz = float(verts[:, 0].mean()), float(verts[:, 2].mean())
    r = np.sqrt((verts[:, 0] - cx) ** 2 + (verts[:, 2] - cz) ** 2)
    rad = r / max(float(r.max()), 1e-6)
    factor = 0.66 + 0.34 * (0.6 * h + 0.4 * rad)
    n_cards = len(verts) // 4
    jit = np.repeat(rng.uniform(-0.05, 0.05, n_cards), 4)[:len(verts)]
    factor = np.clip(factor + jit, 0.5, 1.12)
    cr = factor * (1.0 + 0.07 * h)
    cg = factor
    cb = factor * (0.92 - 0.05 * h)
    col = np.clip(np.column_stack([cr, cg, cb]) * 255.0, 0, 255)
    a = np.full((len(verts), 1), 255.0)
    return np.hstack([col, a]).astype(np.uint8)


def _wood_colors(verts, ao_height=5.0, base=0.78):
    """Subtle AO darkening near the ground; gentle overall value variation."""
    y = verts[:, 1]
    ao = base + (1.0 - base) * np.clip(y / ao_height, 0.0, 1.0)
    col = np.clip(np.column_stack([ao, ao, ao]) * 255.0, 0, 255)
    a = np.full((len(verts), 1), 255.0)
    return np.hstack([col, a]).astype(np.uint8)


# ===========================================================================
# Assemble textured scene
# ===========================================================================
def build_textured_scene(seed, density, image_path):
    scene = build_mesh(seed, density)
    img = _load_image(image_path)
    rng = np.random.default_rng(seed)  # deterministic texture jitter

    # --- materials -------------------------------------------------------
    bark_img = _make_bark(img, rng, res=1024)
    bark_norm = _normal_from_albedo(bark_img, strength=1.3)
    bark_mat = trimesh.visual.material.PBRMaterial(
        name="bark",
        baseColorTexture=bark_img,
        normalTexture=bark_norm,
        metallicFactor=0.0,
        roughnessFactor=0.9,
    )

    atlas_img, grid = _make_leaf_atlas(img, rng, res=1024, grid=4)
    leaf_mat = trimesh.visual.material.PBRMaterial(
        name="canopy",
        baseColorTexture=atlas_img,
        metallicFactor=0.0,
        roughnessFactor=0.8,
        alphaMode="MASK",
        alphaCutoff=0.45,
        doubleSided=True,
    )

    # --- trunk (u_rep=1 -> minimal seam, pale grey-green bark) ----------
    trunk = scene.geometry["trunk"]
    uv = _cyl_uv(trunk.vertices, v_period=4.0, u_rep=1.0)
    trunk.visual = trimesh.visual.TextureVisuals(uv=uv, material=bark_mat)
    trunk.visual.vertex_attributes["color"] = _wood_colors(trunk.vertices, 5.0, 0.7)

    # --- branches --------------------------------------------------------
    branches = scene.geometry["branches"]
    uv = _cyl_uv(branches.vertices, v_period=3.0, u_rep=1.0)
    branches.visual = trimesh.visual.TextureVisuals(uv=uv, material=bark_mat)
    branches.visual.vertex_attributes["color"] = _wood_colors(
        branches.vertices, ao_height=TREE_HEIGHT, base=0.85)

    # --- canopy ----------------------------------------------------------
    canopy = scene.geometry["canopy"]
    uv = _card_uv(len(canopy.vertices), grid, rng)
    cols = _canopy_colors(canopy.vertices, rng)
    canopy.visual = trimesh.visual.TextureVisuals(uv=uv, material=leaf_mat)
    canopy.visual.vertex_attributes["color"] = cols

    return scene


def main(argv=None):
    ap = argparse.ArgumentParser(description="Procedural kapok/ceiba tree -> GLB")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args(argv)

    try:
        scene = build_textured_scene(args.seed, args.density, args.image)
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