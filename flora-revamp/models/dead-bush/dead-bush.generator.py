"""
Standalone procedural generator + texturer for a DRIED TUMBLEWEED archetype.

A tumbleweed is the skeletal remains of a Russian thistle: a slightly flattened,
roughly spherical, cloud-like tangle of fine dead twigs radiating from a denser
knot near the bottom-center and forking chaotically into a soft, ragged, see-
through fuzz at the shell.  The whole object is one wood material: dry, splintery
woody stem.

GEOMETRY: structural twigs are thin tapering tubes ("branches").  The fine
hair-thin fringe that gives the round, fuzzy, see-through silhouette is modelled
as alpha-cutout LEAF-STYLE CARDS ("fuzz"), scattered EVENLY over a thick shell
(Fibonacci directions, not clumps) so the outline reads as one filled globe and
not a few separate tufts.  COLORS for every texture are sampled from the photo.

CLI:
    python thisscript.py --image PATH --seed INT --density {high,med,low} \
                         --output OUT.glb
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw


# ============================================================================
# GEOMETRY
# ============================================================================
# Measured proportions (read off the reference image, ~10% eyeball accuracy).
CROWN_WIDTH        = 0.72          # meters, X/Z diameter of the globe
HEIGHT_OVER_WIDTH  = 0.86          # slightly flattened globe (front aspect ~1.12)
KNOT_HEIGHT_FRAC   = 0.30          # root knot sits at ~30% of total height
SHELL_OVERSHOOT    = 1.03          # tip fuzz may splay <= +3% past the envelope

RX = CROWN_WIDTH * 0.5             # horizontal radius (X)
RZ = CROWN_WIDTH * 0.5             # horizontal radius (Z)
RY = RX * HEIGHT_OVER_WIDTH        # vertical radius (flattened)

V_TILE = 0.05                      # meters of twig length per texture V repeat


def _runit(rng):
    """A uniformly random unit vector."""
    v = rng.normal(size=3)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])


def _fib_sphere(n):
    """`n` roughly-even directions on the unit sphere (Fibonacci spiral)."""
    i = np.arange(n)
    golden = (1.0 + 5.0 ** 0.5) / 2.0
    y = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = 2.0 * np.pi * i / golden
    return np.stack([r * np.cos(theta), y, r * np.sin(theta)], axis=1)


def _split_radii(r_parent, n, rng):
    """Child radii obeying r_parent^2 ~= sum(r_child^2), with jitter."""
    base = r_parent / np.sqrt(n)
    rs = base * rng.uniform(0.75, 1.25, n)
    scale = r_parent / np.sqrt(np.sum(rs * rs) + 1e-18)
    return rs * scale


def _tube(path, radii, sides):
    """
    Sweep polyline `path` (M,3) with per-vertex `radii` (M,) into a closed tube
    using a parallel-transport frame.  Returns (verts, faces, cumlen) where
    cumlen (M,) is arc length along the path (used for the V texture axis).
    """
    path = np.asarray(path, dtype=float)
    M = len(path)

    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg)])

    tang = np.zeros((M, 3))
    if M >= 3:
        tang[1:-1] = path[2:] - path[:-2]
    tang[0] = path[1] - path[0]
    tang[-1] = path[-1] - path[-2]
    tlen = np.linalg.norm(tang, axis=1, keepdims=True)
    tlen[tlen < 1e-9] = 1.0
    tang = tang / tlen

    normals = np.zeros((M, 3))
    a = tang[0]
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n0 = np.cross(a, ref)
    normals[0] = n0 / (np.linalg.norm(n0) + 1e-12)
    for i in range(1, M):
        t0, t1 = tang[i - 1], tang[i]
        v = np.cross(t0, t1)
        s = np.linalg.norm(v)
        c = float(np.dot(t0, t1))
        nprev = normals[i - 1]
        if s < 1e-9:
            ni = nprev
        else:
            v = v / s
            ni = nprev * c + np.cross(v, nprev) * s + v * np.dot(v, nprev) * (1.0 - c)
        ni = ni - tang[i] * np.dot(ni, tang[i])
        nl = np.linalg.norm(ni)
        if nl < 1e-9:
            ref = np.array([0.0, 0.0, 1.0]) if abs(tang[i][2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            ni = np.cross(tang[i], ref)
            nl = np.linalg.norm(ni)
        normals[i] = ni / nl
    binorm = np.cross(tang, normals)

    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    cos, sin = np.cos(ang), np.sin(ang)

    verts = np.empty((M * sides, 3))
    for i in range(M):
        ring = path[i][None, :] + radii[i] * (np.outer(cos, normals[i]) + np.outer(sin, binorm[i]))
        verts[i * sides:(i + 1) * sides] = ring

    faces = []
    for i in range(M - 1):
        b = i * sides
        b2 = (i + 1) * sides
        for j in range(sides):
            jn = (j + 1) % sides
            faces.append([b + j, b2 + j, b2 + jn])
            faces.append([b + j, b2 + jn, b + jn])
    for j in range(1, sides - 1):                     # start cap
        faces.append([0, j + 1, j])
    off = (M - 1) * sides
    for j in range(1, sides - 1):                     # end cap
        faces.append([off, off + j, off + j + 1])

    return verts, np.asarray(faces, dtype=np.int64), cumlen


def _params(density):
    if density == "low":
        return dict(n_main=14, max_branches=400, max_gen=4, sides=3,
                    nodes_min=2, nodes_max=3, seg_base=0.040,
                    tri_budget=8000, tube_factor=0.55, cards=340)
    if density == "med":
        return dict(n_main=22, max_branches=1100, max_gen=5, sides=3,
                    nodes_min=3, nodes_max=4, seg_base=0.027,
                    tri_budget=25000, tube_factor=0.55, cards=1300)
    return dict(n_main=30, max_branches=2600, max_gen=6, sides=4,
                nodes_min=3, nodes_max=5, seg_base=0.020,
                tri_budget=80000, tube_factor=0.62, cards=3200)


def build_mesh(seed: int, density: str = "high") -> trimesh.Scene:
    """Build tumbleweed geometry: solid twig tubes ("branches") plus an evenly
    scattered shell of alpha-card fuzz ("fuzz").  Per-vertex UV + exposure are
    carried in mesh.metadata so the texturing step can attach materials."""
    rng = np.random.default_rng(seed)
    if density not in ("high", "med", "low"):
        density = "high"
    P = _params(density)

    sides = P["sides"]
    tri_soft = int(P["tri_budget"] * P["tube_factor"])

    C = np.array([0.0, RY, 0.0])
    knot_y = KNOT_HEIGHT_FRAC * (2.0 * RY)
    K_KNOT = np.array([0.0, knot_y, 0.0])

    # gentle, low-amplitude lobes -> rounder globe (avoids the lumpy "cross")
    n_lobes = int(rng.integers(3, 6))
    lobe_dirs = np.array([_runit(rng) for _ in range(n_lobes)])
    lobe_amp = rng.uniform(-0.06, 0.06, n_lobes)

    def env_factor(u):
        return float(np.clip(1.0 + np.sum(lobe_amp * (lobe_dirs @ u)), 0.92, 1.08))

    def env_radius(u):
        inv = np.sqrt((u[0] / RX) ** 2 + (u[1] / RY) ** 2 + (u[2] / RZ) ** 2)
        inv = inv if inv > 1e-9 else 1e-9
        return env_factor(u) / inv

    OUTWARD_BIAS = 0.22      # lower -> twigs wander & fill rather than spike out
    KINK_AMP     = 0.58
    SPREAD       = 0.62
    TAPER        = 0.85
    MIN_R        = 0.0005
    R0           = 0.0055

    def ring_exposure(pp, ejit):
        """Sun exposure: outer & top bright, knot dark (floor kept high so the
        interior never collapses to muddy black)."""
        rc = pp - C
        rl = np.linalg.norm(rc)
        if rl > 1e-6:
            rf = min(1.0, rl / (env_radius(rc / rl) + 1e-9))
        else:
            rf = 0.0
        hf = pp[1] / (2.0 * RY)
        return float(np.clip(0.32 + 0.55 * rf + 0.13 * hf + ejit, 0.0, 1.0))

    def grow_branch(task):
        start = task["start"]
        d = task["dir"].astype(float)
        r = float(task["radius"])
        gen = task["gen"]
        kind = task["kind"]
        ejit = task["ejit"]

        K = 2 if kind == "stub" else int(rng.integers(P["nodes_min"], P["nodes_max"] + 1))

        pts = [start.copy()]
        rads = [max(r, MIN_R)]
        p = start.copy()
        reached = False

        for _ in range(K - 1):
            rc = p - C
            rl = np.linalg.norm(rc)
            radial = rc / rl if rl > 1e-6 else d
            d = d + OUTWARD_BIAS * radial + KINK_AMP * _runit(rng)
            d[1] -= 0.04
            nd = np.linalg.norm(d)
            d = d / nd if nd > 1e-9 else radial

            step = P["seg_base"] * rng.uniform(0.8, 1.25)
            p = p + d * step
            r *= TAPER
            if p[1] < 0.004:
                p[1] = 0.004

            rc2 = p - C
            rl2 = np.linalg.norm(rc2)
            u = rc2 / rl2 if rl2 > 1e-6 else radial
            sd = env_radius(u)
            cap = sd * (SHELL_OVERSHOOT if kind == "stub" else 1.0)
            if rl2 > cap:
                p = C + u * cap * rng.uniform(0.98, 1.0)
                if p[1] < 0.004:
                    p[1] = 0.004
                reached = True

            pts.append(p.copy())
            rads.append(max(r, MIN_R))
            if rl2 >= sd * 0.97:
                reached = True
                break

        verts, faces, cumlen = _tube(pts, rads, sides)

        u_ring = (np.arange(sides) / float(sides)).astype(np.float64)
        uv = np.empty((len(pts) * sides, 2))
        exp = np.empty(len(pts) * sides)
        for i, pp in enumerate(pts):
            uv[i * sides:(i + 1) * sides, 0] = u_ring
            uv[i * sides:(i + 1) * sides, 1] = cumlen[i] / V_TILE
            exp[i * sides:(i + 1) * sides] = ring_exposure(pp, ejit)

        children = []
        tip_p, tip_r = pts[-1], rads[-1]
        if kind != "stub" and gen < P["max_gen"]:
            if reached:
                for _ in range(int(rng.integers(0, 3))):
                    sdir = d + 0.8 * _runit(rng)
                    sdir /= (np.linalg.norm(sdir) + 1e-12)
                    children.append(dict(start=tip_p.copy(), dir=sdir,
                                         radius=max(tip_r * 0.6, MIN_R),
                                         gen=P["max_gen"], kind="stub", ejit=ejit))
            else:
                nc = int(rng.integers(2, 4))
                rs = _split_radii(tip_r, nc, rng)
                for k in range(nc):
                    cdir = d + SPREAD * _runit(rng)
                    cdir /= (np.linalg.norm(cdir) + 1e-12)
                    children.append(dict(start=tip_p.copy(), dir=cdir,
                                         radius=max(rs[k], MIN_R),
                                         gen=gen + 1, kind="cont",
                                         ejit=ejit + rng.uniform(-0.03, 0.03)))
                if len(pts) >= 3 and rng.random() < 0.45:
                    idx = int(rng.integers(1, len(pts) - 1))
                    sc = pts[idx] - C
                    sn = np.linalg.norm(sc)
                    base = sc / sn if sn > 1e-6 else d
                    sdir = base + 0.9 * _runit(rng)
                    sdir /= (np.linalg.norm(sdir) + 1e-12)
                    children.append(dict(start=pts[idx].copy(), dir=sdir,
                                         radius=max(rads[idx] * 0.55, MIN_R),
                                         gen=gen + 1, kind="cont",
                                         ejit=ejit + rng.uniform(-0.03, 0.03)))

        return verts, faces, uv, exp, children

    # ---- structural twig tubes (many spokes -> isotropic round fill) -------
    queue = []
    base_dirs = _fib_sphere(P["n_main"])
    for i in range(P["n_main"]):
        d = base_dirs[i] + 0.30 * _runit(rng)
        d /= (np.linalg.norm(d) + 1e-12)
        s = K_KNOT + _runit(rng) * rng.uniform(0.0, 0.03)
        s[1] = max(s[1], 0.01)
        queue.append(dict(start=s, dir=d, radius=R0 * rng.uniform(0.8, 1.2),
                          gen=1, kind="cont", ejit=rng.uniform(-0.05, 0.05)))

    all_v, all_f, all_uv, all_e = [], [], [], []
    voff = 0
    tri_count = 0
    n_branches = 0
    while queue and n_branches < P["max_branches"] and tri_count < tri_soft:
        task = queue.pop(0)
        v, f, uv, e, kids = grow_branch(task)
        all_v.append(v)
        all_f.append(f + voff)
        all_uv.append(uv)
        all_e.append(e)
        voff += len(v)
        tri_count += len(f)
        n_branches += 1
        queue.extend(kids)

    if not all_v:  # safety fallback (should never trigger)
        v, f, cl = _tube(np.array([[0, 0, 0.0], [0, 0.1, 0]]), np.array([0.01, 0.005]), sides)
        all_v = [v]
        all_f = [f]
        all_uv = [np.zeros((len(v), 2))]
        all_e = [np.full(len(v), 0.5)]

    Vb = np.vstack(all_v)
    Fb = np.vstack(all_f)
    UVb = np.vstack(all_uv).astype(np.float32)
    EXPb = np.concatenate(all_e).astype(np.float32)

    # ---- alpha-card fuzz: EVEN scatter over a thick shell (no big clumps) --
    def build_cards():
        cw = CROWN_WIDTH
        n = P["cards"]
        hs_base = 0.030 * cw                   # small cards -> uniform fuzz
        dirs = _fib_sphere(n)                  # even directions = round outline

        cverts, cfaces, cuv, cexp = [], [], [], []
        vo = 0
        for i in range(n):
            d = dirs[i] + 0.18 * _runit(rng)
            d /= (np.linalg.norm(d) + 1e-12)
            shell = env_radius(d)
            frac = rng.uniform(0.60, 1.0)      # thick shell -> depth, not hollow
            center = C + d * (shell * frac) + _runit(rng) * (0.02 * cw)
            if center[1] < 0.01:
                center[1] = 0.01

            rc = center - C
            rl = np.linalg.norm(rc)
            nrm = rc / rl if rl > 1e-6 else d
            nrm = nrm + 0.35 * _runit(rng)     # +/- ~25 deg
            nrm /= (np.linalg.norm(nrm) + 1e-12)

            up = np.array([0.0, 1.0, 0.0])
            ax = np.cross(up, nrm)
            if np.linalg.norm(ax) < 1e-4:
                ax = np.array([1.0, 0.0, 0.0])
            ax /= np.linalg.norm(ax)
            ay = np.cross(nrm, ax)
            ay /= (np.linalg.norm(ay) + 1e-12)

            hs = hs_base * float(np.exp(rng.normal(0.0, 0.25)))
            ang = rng.uniform(0.0, 2.0 * np.pi)
            ca, sa = np.cos(ang), np.sin(ang)
            axr = ca * ax + sa * ay
            ayr = -sa * ax + ca * ay

            c0 = center - hs * axr - hs * ayr
            c1 = center + hs * axr - hs * ayr
            c2 = center + hs * axr + hs * ayr
            c3 = center - hs * axr + hs * ayr
            cverts.extend([c0, c1, c2, c3])
            cfaces.append([vo, vo + 1, vo + 2])
            cfaces.append([vo, vo + 2, vo + 3])

            gx = int(rng.integers(0, 4))
            gy = int(rng.integers(0, 4))
            rot = int(rng.integers(0, 4))
            pad = 0.004
            u0, u1 = gx * 0.25 + pad, (gx + 1) * 0.25 - pad
            v0, v1 = gy * 0.25 + pad, (gy + 1) * 0.25 - pad
            base = [(u0, v1), (u1, v1), (u1, v0), (u0, v0)]
            cuv.extend(base[rot:] + base[:rot])

            e = ring_exposure(center, 0.0)
            cexp.extend([e, e, e, e])
            vo += 4

        return (np.asarray(cverts, dtype=float),
                np.asarray(cfaces, dtype=np.int64),
                np.asarray(cuv, dtype=np.float32),
                np.asarray(cexp, dtype=np.float32))

    Vf, Ff, UVf, EXPf = build_cards()

    # ---- stand on the XZ plane, centered in X/Z (apply to both parts) ------
    y_min = min(Vb[:, 1].min(), Vf[:, 1].min())
    x_lo = min(Vb[:, 0].min(), Vf[:, 0].min())
    x_hi = max(Vb[:, 0].max(), Vf[:, 0].max())
    z_lo = min(Vb[:, 2].min(), Vf[:, 2].min())
    z_hi = max(Vb[:, 2].max(), Vf[:, 2].max())
    cx = 0.5 * (x_lo + x_hi)
    cz = 0.5 * (z_lo + z_hi)
    for Varr in (Vb, Vf):
        Varr[:, 1] -= y_min
        Varr[:, 0] -= cx
        Varr[:, 2] -= cz

    scene = trimesh.Scene()

    mesh_b = trimesh.Trimesh(vertices=Vb, faces=Fb, process=False)
    mesh_b.metadata["uv"] = UVb
    mesh_b.metadata["exposure"] = EXPb
    scene.add_geometry(mesh_b, geom_name="branches")

    mesh_f = trimesh.Trimesh(vertices=Vf, faces=Ff, process=False)
    mesh_f.metadata["uv"] = UVf
    mesh_f.metadata["exposure"] = EXPf
    scene.add_geometry(mesh_f, geom_name="fuzz")

    return scene


# ============================================================================
# TEXTURING -- colors sampled from the reference photo
# ============================================================================
def _norm01(a):
    mn = float(a.min())
    mx = float(a.max())
    return (a - mn) / (mx - mn + 1e-9)


def _upsample_periodic(coarse, size):
    """Bilinear upsample of a (fr,fc) grid to (size,size), wrapping at the
    edges so the result tiles seamlessly."""
    fr, fc = coarse.shape
    iy = np.linspace(0, fr, size, endpoint=False)
    ix = np.linspace(0, fc, size, endpoint=False)
    y0 = np.floor(iy).astype(int) % fr
    y1 = (y0 + 1) % fr
    ty = (iy - np.floor(iy))[:, None]
    x0 = np.floor(ix).astype(int) % fc
    x1 = (x0 + 1) % fc
    tx = (ix - np.floor(ix))[None, :]
    rows = coarse[y0, :] * (1 - ty) + coarse[y1, :] * ty
    out = rows[:, x0] * (1 - tx) + rows[:, x1] * tx
    return out


def _fbm(size, rng, octaves, base=4):
    """Tileable fractal noise (periodic by construction)."""
    out = np.zeros((size, size))
    amp = 1.0
    total = 0.0
    f = base
    for _ in range(octaves):
        out += amp * _upsample_periodic(rng.random((f, f)), size)
        total += amp
        amp *= 0.5
        f *= 2
    return out / total


def _ramp(field, stops_t, stops_c):
    """Map a [0,1] scalar field through a piecewise-linear color ramp."""
    cs = [np.asarray(c, dtype=float) for c in stops_c]
    out = np.zeros(field.shape + (3,))
    for k in range(len(stops_t) - 1):
        t0, t1 = stops_t[k], stops_t[k + 1]
        m = (field >= t0) & (field <= t1) if k == 0 else (field > t0) & (field <= t1)
        if not np.any(m):
            continue
        w = (field[m] - t0) / max(1e-6, (t1 - t0))
        out[m] = cs[k] * (1 - w)[:, None] + cs[k + 1] * w[:, None]
    out[field > stops_t[-1]] = cs[-1]
    return out


def _delight(arr):
    """Divide out a heavily blurred luminance (clamped gain) to flatten the
    photo's baked lighting before color sampling."""
    H, W, _ = arr.shape
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    sw, sh = max(1, W // 32), max(1, H // 32)
    blur = np.asarray(img.resize((sw, sh), Image.LANCZOS)
                         .resize((W, H), Image.LANCZOS)).astype(float)
    blum = 0.2126 * blur[..., 0] + 0.7152 * blur[..., 1] + 0.0722 * blur[..., 2]
    target = float(np.median(blum))
    gain = np.clip(target / (blum + 1e-6), 0.6, 1.6)[..., None]
    return np.clip(arr * gain, 0, 255)


def extract_palette(image_path):
    """Sample woody colors from WELL INSIDE the silhouette, rejecting the
    neutral grey background by keeping only warm (R>B), saturated pixels.
    Then lift toward the bleached straw/tan/silver of the photo."""
    img = Image.open(image_path).convert("RGB")
    arr = _delight(np.asarray(img).astype(np.float64))
    H, W, _ = arr.shape

    crop = arr[int(0.15 * H):int(0.88 * H), int(0.15 * W):int(0.85 * W)].reshape(-1, 3)
    R, G, B = crop[:, 0], crop[:, 1], crop[:, 2]
    lum = 0.2126 * R + 0.7152 * G + 0.0722 * B
    warmth = R - B
    sat = np.max(crop, axis=1) - np.min(crop, axis=1)

    mask = (warmth > 8) & (sat > 10) & (lum > 30)
    if mask.sum() < 200:
        mask = (warmth > 3) & (lum > 25)
    if mask.sum() < 50:
        mask = lum > np.median(lum)
    wood = crop[mask]
    wlum = 0.2126 * wood[:, 0] + 0.7152 * wood[:, 1] + 0.0722 * wood[:, 2]

    def med(sub, fb):
        return np.median(sub, axis=0) if len(sub) else fb

    body = np.median(wood, axis=0)
    lit = med(wood[wlum >= np.percentile(wlum, 80)], body)
    shadow = med(wood[wlum <= np.percentile(wlum, 25)], body)
    highlight = np.percentile(wood, 96, axis=0)

    pal = dict(body=body, lit=lit, shadow=shadow, highlight=highlight)
    for k in pal:
        pal[k] = np.clip(pal[k], 0, 255)
    # the render came out too dark -> lift toward bleached straw, keep warmth
    pal["body"] = np.clip(pal["body"] * 1.18, 0, 255)
    pal["lit"] = np.clip(pal["lit"] * 1.20 + 6, 0, 255)
    pal["shadow"] = np.clip(pal["shadow"] * 1.05 * np.array([1.0, 0.94, 0.84]), 0, 255)
    pal["highlight"] = np.clip(pal["highlight"] * 1.06 + 18, 0, 255)
    return pal


def make_wood_textures(pal, rng, density):
    """Synthesize a tileable fibrous-wood albedo (grain along V) plus a derived
    normal map, using the photo-sampled palette."""
    size = 1024 if density == "high" else 512
    octaves = {"high": 6, "med": 5, "low": 4}[density]

    grain = _norm01(_upsample_periodic(rng.random((6, 64)), size))   # fibers along V
    mottle = _fbm(size, rng, octaves, base=4)
    speck = _norm01(_upsample_periodic(rng.random((128, 128)), size))

    field = _norm01(0.55 * grain + 0.45 * mottle)
    rgb = _ramp(field,
                [0.0, 0.45, 0.80, 1.0],
                [pal["shadow"], pal["body"], pal["lit"], pal["highlight"]])
    rgb[speck > 0.88] *= 0.7                        # dark splintery stubble
    fib = grain > 0.82
    rgb[fib] = np.minimum(255.0, rgb[fib] * 1.12)   # catch-light along fibers

    rgb = np.clip(rgb, 0, 255)
    albedo = Image.fromarray(rgb.astype(np.uint8), "RGB")

    lum = (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]) / 255.0
    strength = 2.2
    dx = (np.roll(lum, -1, axis=1) - np.roll(lum, 1, axis=1)) * 0.5
    dy = (np.roll(lum, -1, axis=0) - np.roll(lum, 1, axis=0)) * 0.5
    nx, ny, nz = -dx * strength, -dy * strength, np.ones_like(lum)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    nrm = np.stack([(nx * inv * 0.5 + 0.5),
                    (ny * inv * 0.5 + 0.5),
                    (nz * inv * 0.5 + 0.5)], axis=-1)
    normal = Image.fromarray(np.clip(nrm * 255, 0, 255).astype(np.uint8), "RGB")
    return albedo, normal


def _vary_color(base, rng):
    c = np.clip(base + rng.uniform(-12, 16, 3), 0, 255).astype(int)
    return (int(c[0]), int(c[1]), int(c[2]), 255)


def _draw_twig(draw, x, y, ang, length, tile, base, rng, depth):
    steps = 4
    seg = length / steps
    w = max(1, int(tile * 0.014 * rng.uniform(0.8, 1.4)))
    px, py = x, y
    a = ang
    for s in range(steps):
        a += rng.uniform(-0.35, 0.35)
        nx = px + np.cos(a) * seg
        ny = py + np.sin(a) * seg
        ww = max(1, int(w * (1.0 - s / (steps + 1.0))))
        draw.line([(px, py), (nx, ny)], fill=_vary_color(base, rng), width=ww)
        if depth > 0 and rng.random() < 0.55:
            oa = a + (1 if rng.random() < 0.5 else -1) * rng.uniform(0.5, 1.0)
            _draw_twig(draw, nx, ny, oa, length * 0.5, tile, base, rng, depth - 1)
        px, py = nx, ny


def make_card_atlas(pal, rng):
    """A 4x4 atlas of distinct dry-twig cluster tiles (RGBA, anti-aliased edge
    alpha).  Tiles are pale/bleached (cards are the sunlit outer fringe); top
    rows a touch brighter, bottom rows a touch cooler."""
    ATLAS = 1024
    SS = 4
    tilebig = (ATLAS * SS) // 4
    atlas = Image.new("RGBA", (ATLAS * SS, ATLAS * SS), (0, 0, 0, 0))

    lit = np.asarray(pal["lit"], dtype=float)
    for gy in range(4):
        for gx in range(4):
            scale = [1.05, 0.98, 0.90, 0.83][gy]
            base = lit * scale + np.array([6.0, 2.0, -6.0]) * (0.5 - gx / 3.0)
            base = np.clip(base, 0, 255)

            tile_img = Image.new("RGBA", (tilebig, tilebig), (0, 0, 0, 0))
            td = ImageDraw.Draw(tile_img)
            cx = tilebig * 0.5
            cy = tilebig * 0.5
            n_main = int(rng.integers(9, 15))
            for _ in range(n_main):
                ang = rng.uniform(0.0, 2.0 * np.pi)
                length = tilebig * rng.uniform(0.28, 0.46)
                _draw_twig(td, cx, cy, ang, length, tilebig, base, rng, depth=2)
            atlas.paste(tile_img, (gx * tilebig, gy * tilebig))

    return atlas.resize((ATLAS, ATLAS), Image.LANCZOS)


def vertex_tints(exposure):
    """Per-vertex COLOR_0: warm tan knot -> bright bleached outer twigs.
    Floor kept high so shaded interior stays legible, not muddy black."""
    shade = np.array([0.74, 0.68, 0.58])
    lit = np.array([1.00, 0.99, 0.95])
    e = np.clip(exposure, 0.0, 1.0)[:, None]
    cols = np.clip(shade * (1 - e) + lit * e, 0.0, 1.0)
    rgba = np.concatenate([cols, np.ones((len(cols), 1))], axis=1)
    return (rgba * 255.0 + 0.5).astype(np.uint8)


# ============================================================================
# ASSEMBLY + EXPORT
# ============================================================================
def texture_scene(scene, image_path, seed, density):
    pal = extract_palette(image_path)
    tex_rng = np.random.default_rng(seed + 12345)
    albedo, normal = make_wood_textures(pal, tex_rng, density)
    atlas = make_card_atlas(pal, tex_rng)

    # solid twig tubes -- opaque dry wood
    geom_b = scene.geometry["branches"]
    uv_b = np.asarray(geom_b.metadata["uv"], dtype=np.float64)
    exp_b = np.asarray(geom_b.metadata["exposure"], dtype=np.float64)
    mat_b = trimesh.visual.material.PBRMaterial(
        name="dry_woody_stem",
        baseColorFactor=[255, 255, 255, 255],
        baseColorTexture=albedo,
        normalTexture=normal,
        metallicFactor=0.0,
        roughnessFactor=0.9,
        doubleSided=True,
        alphaMode="OPAQUE",
    )
    geom_b.visual = trimesh.visual.TextureVisuals(uv=uv_b, material=mat_b)
    geom_b.visual.vertex_attributes["color"] = vertex_tints(exp_b)

    # fine peripheral fuzz -- alpha-cutout twig cards
    geom_f = scene.geometry["fuzz"]
    uv_f = np.asarray(geom_f.metadata["uv"], dtype=np.float64)
    exp_f = np.asarray(geom_f.metadata["exposure"], dtype=np.float64)
    mat_f = trimesh.visual.material.PBRMaterial(
        name="twig_fuzz_cards",
        baseColorFactor=[255, 255, 255, 255],
        baseColorTexture=atlas,
        metallicFactor=0.0,
        roughnessFactor=0.85,
        doubleSided=True,
        alphaMode="MASK",
        alphaCutoff=0.4,
    )
    geom_f.visual = trimesh.visual.TextureVisuals(uv=uv_f, material=mat_f)
    geom_f.visual.vertex_attributes["color"] = vertex_tints(exp_f)

    return scene


def main():
    ap = argparse.ArgumentParser(description="Generate a textured tumbleweed GLB.")
    ap.add_argument("--image", required=True, help="reference photo path")
    ap.add_argument("--seed", type=int, default=0, help="deterministic seed")
    ap.add_argument("--density", choices=["high", "med", "low"], default="high")
    ap.add_argument("--output", required=True, help="output .glb path")
    args = ap.parse_args()

    try:
        scene = build_mesh(args.seed, args.density)
        scene = texture_scene(scene, args.image, args.seed, args.density)
        glb = scene.export(file_type="glb")
        with open(args.output, "wb") as f:
            f.write(glb)
    except Exception as exc:  # noqa: BLE001 -- surface any failure as non-zero exit
        sys.stderr.write("ERROR: {}\n".format(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())