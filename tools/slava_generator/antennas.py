"""Radar antennas rebuilt from Kuleshov sheet 1 (side profile) and photos of Moskva / Varyag:
MR-800 "Top Pair" (main mast), Fregat (foremast top) and Argon-1164 "Front Door" (foremast front).

Reflectors are real curved surfaces carrying alpha-tested lattice textures painted for each antenna;
frames, back trusses, feeds and waveguides are thin tubes.
"""
import math
import numpy as np

from meshkit import *
import atlas as st
from kit import M, P3, truss, railing_pts, decal, slab
from detail import rbox, axis_frame


# =============================================================================================
# geometry helpers
# =============================================================================================
def _orient(P, N, UV, I):
    """flip every triangle whose winding disagrees with its vertex normals."""
    a, b, c = P[I[:, 0]], P[I[:, 1]], P[I[:, 2]]
    fn = np.cross(b - a, c - a)
    vn = N[I[:, 0]] + N[I[:, 1]] + N[I[:, 2]]
    flip = np.einsum('ij,ij->i', fn, vn) < 0
    I = I.copy()
    I[flip] = I[flip][:, ::-1]
    return P, N, UV, I


def pipe(points, r, seg=5, closed=False):
    """tube along a polyline with rotation-minimising frames (no twist on curved or closed paths)."""
    pts = [np.asarray(p, float) for p in points]
    if closed and np.linalg.norm(pts[0] - pts[-1]) < 1e-6:
        pts = pts[:-1]
    n = len(pts)
    T = []
    for i in range(n):
        if closed:
            t = normalize(pts[(i + 1) % n] - pts[i]) + normalize(pts[i] - pts[i - 1])
        elif i == 0:
            t = pts[1] - pts[0]
        elif i == n - 1:
            t = pts[-1] - pts[-2]
        else:
            t = normalize(pts[i + 1] - pts[i]) + normalize(pts[i] - pts[i - 1])
        T.append(normalize(t))
    a = np.array([0.0, 1.0, 0.0]) if abs(T[0][1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    F = [normalize(a - np.dot(a, T[0]) * T[0])]
    for i in range(1, n):
        v = F[-1] - np.dot(F[-1], T[i]) * T[i]
        F.append(normalize(v) if np.linalg.norm(v) > 1e-9 else F[-1])
    if closed:
        # spread the holonomy twist evenly so the seam closes
        v = normalize(F[-1] - np.dot(F[-1], T[0]) * T[0])
        ang = math.atan2(np.dot(v, np.cross(T[0], F[0])), np.dot(v, F[0]))
        for i in range(n):
            t = -ang * i / n
            F[i] = F[i] * math.cos(t) + np.cross(T[i], F[i]) * math.sin(t)
        pts, T, F = pts + [pts[0]], T + [T[0]], F + [F[0]]
        n += 1
    P, N, UV, I = [], [], [], []
    acc = 0.0
    for i in range(n):
        if i > 0:
            acc += np.linalg.norm(pts[i] - pts[i - 1])
        Bv = np.cross(T[i], F[i])
        for j in range(seg + 1):
            ang = 2 * math.pi * j / seg
            d = F[i] * math.cos(ang) + Bv * math.sin(ang)
            P.append(pts[i] + r * d)
            N.append(d)
            UV.append((j / seg * 2 * math.pi * r, acc))
    for i in range(n - 1):
        for j in range(seg):
            a_ = i * (seg + 1) + j
            b_ = a_ + seg + 1
            I += [(a_, a_ + 1, b_ + 1), (a_, b_ + 1, b_)]
    return _orient(np.array(P), np.array(N), np.array(UV), np.array(I))


def cyl_reflector(V, right, up, nrm, w, h, sag, vrange, us, rows=2):
    """parabolic-cylinder reflector: straight along `up`, curved across `right`, concave toward +nrm.
    S(u, v) = V + right*u + up*v + nrm*sag*(2u/w)^2 (V = middle of the vertex line).
    vrange(u) -> (vlo, vhi) gives the outline; `us` are the column positions (include its corners).
    UV = (0.5 + u/w, 0.5 - v/h), i.e. the decal's image space. Returns (geo, outline, S, normal_at)."""
    V, right, up, nrm = (np.asarray(q, float) for q in (V, right, up, nrm))

    def S(u, v):
        return V + right * u + up * v + nrm * (sag * (2.0 * u / w) ** 2)

    def nrm_at(u):
        return normalize(nrm - right * (8.0 * sag * u / (w * w)))

    P, N, UV, I = [], [], [], []
    for u in us:
        lo, hi = vrange(u)
        for j in range(rows + 1):
            v = lo + (hi - lo) * j / rows
            P.append(S(u, v)); N.append(nrm_at(u)); UV.append((0.5 + u / w, 0.5 - v / h))
    for k in range(len(us) - 1):
        for j in range(rows):
            a = k * (rows + 1) + j
            b = a + rows + 1
            I += [(a, b, b + 1), (a, b + 1, a + 1)]
    geo = _orient(np.array(P), np.array(N), np.array(UV), np.array(I))
    outline = [S(u, vrange(u)[0]) for u in us] + [S(u, vrange(u)[1]) for u in reversed(us)]
    return geo, outline, S, nrm_at


def ell_dish(V, right, up, nrm, a, b, depth, cut=1.0, nr=4, ns=20):
    """elliptical paraboloid with its vertex at V, concave toward +nrm; the rim (semi-axes a, b) lies
    `depth` along nrm. The lower edge is clipped flat at y = -cut*b. UV = (0.5 + x/2a, 0.5 - y/2b).
    Returns (geo, rim, D) with D(x, y) the surface point."""
    V, right, up, nrm = (np.asarray(q, float) for q in (V, right, up, nrm))

    def D(x, y):
        return V + right * x + up * y + nrm * (depth * ((x / a) ** 2 + (y / b) ** 2))

    def rmax(th):
        s = math.sin(th)
        return min(1.0, cut / -s) if s < -1e-6 else 1.0

    P, N, UV = [V], [nrm], [(0.5, 0.5)]
    for k in range(1, nr + 1):
        for j in range(ns):
            th = 2 * math.pi * j / ns
            rr = k / nr * rmax(th)
            x, y = a * rr * math.cos(th), b * rr * math.sin(th)
            P.append(D(x, y))
            N.append(normalize(nrm - right * (2 * depth * x / a ** 2) - up * (2 * depth * y / b ** 2)))
            UV.append((0.5 + x / (2 * a), 0.5 - y / (2 * b)))
    I = [(0, 1 + j, 1 + (j + 1) % ns) for j in range(ns)]
    for k in range(1, nr):
        for j in range(ns):
            p, q = 1 + (k - 1) * ns + j, 1 + (k - 1) * ns + (j + 1) % ns
            I += [(p, p + ns, q + ns), (p, q + ns, q)]
    geo = _orient(np.array(P), np.array(N), np.array(UV), np.array(I))
    rim = [np.asarray(P[1 + (nr - 1) * ns + j]) for j in range(ns)]
    return geo, rim, D


def _u_limit(vrange, w, v):
    """largest |u| at which the outline still contains height v."""
    best = 0.0
    for u in np.linspace(0.0, w / 2, 241):
        lo, hi = vrange(u)
        if lo - 1e-9 <= v <= hi + 1e-9:
            best = u
    return best


def _rib(S, nrm_at, vrange, w, us, v, back):
    """horizontal back rib following the reflector curvature at height v, `back` metres behind it."""
    um = _u_limit(vrange, w, v)
    uu = [-um] + [u for u in us if abs(u) < um - 0.05] + [um]
    return [S(u, v) - nrm_at(u) * back for u in uu]


def _warren(ctx, top, bot, r, sw=None):
    """zigzag diagonals between two chords given as equal-length point lists."""
    uvs = ctx.paint if sw is None else ctx.sw(sw)
    for k in range(len(top) - 1):
        p, q = (top[k], bot[k + 1]) if k % 2 == 0 else (bot[k], top[k + 1])
        ctx.add(uvs, pipe([p, q], r, seg=4), occ=False)


def _horn(ctx, mouth_at, toward, r0=0.1, r1=0.24, L=0.45, sw='mid'):
    d = normalize(np.asarray(toward, float) - np.asarray(mouth_at, float))
    base = np.asarray(mouth_at, float) - d * L
    ctx.add(ctx.sw(sw), cylinder(r0, L, seg=8, r_top=r1), xf=M(base, axis_frame(d)))


# =============================================================================================
# lattice textures (alpha-tested; drawn in each reflector's own (u, v) image space)
# =============================================================================================
def _lattice_paint(L, rect, layers, col=None):
    """layers: list of (draw_fn(d, s, w, h), shade, relief); everything else is transparent."""
    x0, y0, x1, y1 = rect
    w, h = (x1 - x0) - 2, (y1 - y0) - 2
    col = st.PAL['mesh'] if col is None else col
    L.rect(x0, y0, x1, y1, col=col, alpha=0.0, rough=0.45, metal=0.35)
    for fn, shade, relief in layers:
        mask = L.draw_mask(w, h, lambda d, s: fn(d, s, w, h))
        L.mask_apply(mask, col=col * shade, alpha=1.0, add_height=relief, x0=x0 + 1, y0=y0 + 1)


def _line(d, s, pts, width):
    d.line([(px * s, py * s) for (px, py) in pts], fill=255, width=max(1, int(round(width * s))), joint='curve')


# ---- Top Pair main reflector: elongated octagon, square bays with alternating diagonals
TPM_W, TPM_H, TPM_FLAT, TPM_CH, TPM_SAG = 4.8, 8.3, 1.0, 1.9, 1.2


def tpm_vrange(u):
    a, hh = abs(u), TPM_H / 2
    if a <= TPM_FLAT:
        return (-hh, hh)
    v = hh - (a - TPM_FLAT) / (TPM_W / 2 - TPM_FLAT) * TPM_CH
    return (-v, v)


TPM_US = [-2.4, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.4]


def paint_tp_main(L, rect):
    def X(u, w):
        return (0.5 + u / TPM_W) * w

    def Y(v, h):
        return (0.5 - v / TPM_H) * h

    bays_u = np.linspace(-TPM_W / 2, TPM_W / 2, 9)
    bays_v = np.linspace(-TPM_H / 2, TPM_H / 2, 15)

    def fine(d, s, w, h):
        for v in np.arange(-TPM_H / 2, TPM_H / 2, 0.16):
            _line(d, s, [(0, Y(v, h)), (w, Y(v, h))], 1.1)

    def members(d, s, w, h):
        for u in bays_u:
            _line(d, s, [(X(u, w), 0), (X(u, w), h)], 2.6)
        for v in bays_v:
            _line(d, s, [(0, Y(v, h)), (w, Y(v, h))], 2.4)
        for i in range(len(bays_u) - 1):
            for j in range(len(bays_v) - 1):
                ua, ub = bays_u[i], bays_u[i + 1]
                va, vb = (bays_v[j], bays_v[j + 1]) if (i + j) % 2 == 0 else (bays_v[j + 1], bays_v[j])
                _line(d, s, [(X(ua, w), Y(va, h)), (X(ub, w), Y(vb, h))], 1.8)

    def rim(d, s, w, h):
        us = np.linspace(-TPM_W / 2, TPM_W / 2, 49)
        top = [(X(u, w), Y(tpm_vrange(u)[1], h)) for u in us]
        bot = [(X(u, w), Y(tpm_vrange(u)[0], h)) for u in us[::-1]]
        _line(d, s, top + bot + top[:1], 4.0)

    _lattice_paint(L, rect, [(fine, 0.95, 0.3), (members, 0.85, 0.6), (rim, 0.75, 0.8)])


# ---- Top Pair rear dish: ellipse with a flat lower edge, radial ribs, rings and cross bracing
TPD_A, TPD_B, TPD_DEPTH, TPD_CUT = 2.2, 1.55, 1.1, 0.8


def _ell_pt(rr, th, cut):
    s = math.sin(th)
    k = min(1.0, cut / -s) if s < -1e-6 else 1.0
    return rr * k * math.cos(th), rr * k * s


def paint_tp_dish(L, rect):
    def XY(x, y, w, h):
        return ((0.5 + x / 2) * w, (0.5 - y / 2) * h)          # x, y normalised to the semi-axes

    nrib = 12

    def fine(d, s, w, h):
        for y in np.arange(-1.0, 1.0, 0.1):
            _line(d, s, [XY(-1, y, w, h), XY(1, y, w, h)], 1.1)

    def members(d, s, w, h):
        for j in range(nrib):
            th = 2 * math.pi * j / nrib
            _line(d, s, [XY(0, 0, w, h), XY(*_ell_pt(1.0, th, TPD_CUT), w, h)], 2.5)
        for rr in (0.36, 0.7):
            ring = [XY(*_ell_pt(rr, 2 * math.pi * k / 48, TPD_CUT), w, h) for k in range(49)]
            _line(d, s, ring, 2.2)
        rings = (0.0, 0.36, 0.7, 1.0)
        for j in range(nrib):
            for k in range(3):
                t0, t1 = 2 * math.pi * j / nrib, 2 * math.pi * (j + 1) / nrib
                pa = _ell_pt(rings[k], t0 if k % 2 == 0 else t1, TPD_CUT)
                pb = _ell_pt(rings[k + 1], t1 if k % 2 == 0 else t0, TPD_CUT)
                _line(d, s, [XY(*pa, w, h), XY(*pb, w, h)], 1.6)

    def rim(d, s, w, h):
        ring = [XY(*_ell_pt(1.0, 2 * math.pi * k / 64, TPD_CUT), w, h) for k in range(65)]
        _line(d, s, ring, 4.0)

    _lattice_paint(L, rect, [(fine, 0.95, 0.3), (members, 0.85, 0.6), (rim, 0.75, 0.8)])


# ---- Fregat main reflector: rectangle with chamfered lower corners, square grid
FR_W, FR_H, FR_FLAT, FR_CH, FR_SAG = 5.0, 4.9, 1.6, 0.9, 1.3


def fr_vrange(u):
    a, hh = abs(u), FR_H / 2
    lo = -hh if a <= FR_FLAT else -hh + (a - FR_FLAT) / (FR_W / 2 - FR_FLAT) * FR_CH
    return (lo, hh)


FR_US = [-2.5, -2.1, -1.6, -1.0, -0.5, 0.0, 0.5, 1.0, 1.6, 2.1, 2.5]


def _grid_painter(Wm, Hm, cell_u, cell_v, fine_dv, vrange=None):
    def X(u, w):
        return (0.5 + u / Wm) * w

    def Y(v, h):
        return (0.5 - v / Hm) * h

    gu = np.arange(-Wm / 2, Wm / 2 + 1e-6, cell_u)
    gv = np.arange(-Hm / 2, Hm / 2 + 1e-6, cell_v)

    def fine(d, s, w, h):
        for v in np.arange(-Hm / 2, Hm / 2, fine_dv):
            _line(d, s, [(0, Y(v, h)), (w, Y(v, h))], 1.1)

    def members(d, s, w, h):
        for u in gu:
            _line(d, s, [(X(u, w), 0), (X(u, w), h)], 2.6)
        for v in gv:
            _line(d, s, [(0, Y(v, h)), (w, Y(v, h))], 2.4)
        for i in range(len(gu) - 1):
            for j in range(len(gv) - 1):
                va, vb = (gv[j], gv[j + 1]) if (i + j) % 2 == 0 else (gv[j + 1], gv[j])
                _line(d, s, [(X(gu[i], w), Y(va, h)), (X(gu[i + 1], w), Y(vb, h))], 1.5)

    def rim(d, s, w, h):
        vr = vrange or (lambda u: (-Hm / 2, Hm / 2))
        us = np.linspace(-Wm / 2, Wm / 2, 41)
        top = [(X(u, w), Y(vr(u)[1], h)) for u in us]
        bot = [(X(u, w), Y(vr(u)[0], h)) for u in us[::-1]]
        _line(d, s, top + bot + top[:1], 4.0)

    return lambda L, rect: _lattice_paint(L, rect, [(fine, 0.95, 0.3), (members, 0.85, 0.6), (rim, 0.75, 0.8)])


paint_fregat = _grid_painter(FR_W, FR_H, FR_W / 10, FR_H / 10, 0.12, fr_vrange)
FR2_W, FR2_H, FR2_SAG = 3.2, 1.4, 0.55
paint_fregat2 = _grid_painter(FR2_W, FR2_H, FR2_W / 8, FR2_H / 3, 0.1)


# ---- Front Door reflector: horizontal slats on side frames (the surface reads as a louvred plate)
FD_W = 2.0


def paint_fd_refl(L, rect):
    x0, y0, x1, y1 = rect
    w, h = (x1 - x0) - 2, (y1 - y0) - 2
    col = st.PAL['super'] * 0.97
    L.rect(x0, y0, x1, y1, col=col, alpha=0.0, rough=0.5, metal=0.2)

    def slats(d, s):
        n = 26
        for k in range(n):
            ya = (k + 0.18) / n * h
            yb = (k + 0.82) / n * h
            d.rectangle([0, ya * s, w * s, yb * s], fill=255)

    def frame(d, s):
        for xx in (0, w * 0.5, w):
            d.line([(xx * s, 0), (xx * s, h * s)], fill=255, width=int(5 * s))
        for yy in (0, h):
            d.line([(0, yy * s), (w * s, yy * s)], fill=255, width=int(5 * s))

    L.mask_apply(L.draw_mask(w, h, slats), col=col, alpha=1.0, add_height=0.5, x0=x0 + 1, y0=y0 + 1)
    L.mask_apply(L.draw_mask(w, h, frame), col=col * 0.8, alpha=1.0, add_height=0.9, x0=x0 + 1, y0=y0 + 1)


def register(m):
    m.alloc('tp_main', 256, 512, paint_tp_main)
    m.alloc('fregat', 256, 256, paint_fregat)
    m.alloc('tp_dish', 256, 192, paint_tp_dish)
    m.alloc('fd_refl', 128, 256, paint_fd_refl)
    m.alloc('fregat2', 128, 64, paint_fregat2)


# =============================================================================================
# MR-800 "Top Pair": forward Top Sail-type parabolic cylinder + aft elliptical dish (sheet 1)
# =============================================================================================
def top_pair(ctx, o, name='TopPair_Radar', parent='Superstructure'):
    """o = pivot on the column top (B 101.1, h 25.6)."""
    c, b = ctx, ctx.b
    o = np.asarray(o, float)
    b.push(name, parent=parent, translation=tuple(o))

    def W(dB, dx, dy):                          # offsets from the pivot: aft, port, up
        return o + np.array([dx, dy, -dB])

    pt = c.paint
    # turntable, gearbox and the cap dome
    c.add(pt, lathe([(0.98, 0.0), (0.98, 0.16), (0.9, 0.22), (0.9, 0.45), (0.82, 0.5)], seg=18), xf=M(o))
    c.add(pt, rbox(1.4, 1.3, 1.6, r=0.14, seg=2, bevel=0.06, y0=0.5), xf=M(W(0.3, 0, 0)))
    c.add(pt, lathe([(0.5, 0.0), (0.5, 0.12), (0.38, 0.32), (0.18, 0.42), (0.0, 0.44)], seg=14), xf=M(W(0.3, 0, 1.8)))

    # ---- main reflector, leaning back 20 deg, concave forward
    tl = math.radians(20)
    up = np.array([0, math.cos(tl), -math.sin(tl)])
    nrm = np.array([0, math.sin(tl), math.cos(tl)])
    right = np.array([1.0, 0, 0])
    V = W(-1.2, 0, 0.4)
    geo, outline, S, nat = cyl_reflector(V, right, up, nrm, TPM_W, TPM_H, TPM_SAG, tpm_vrange, TPM_US)
    c.add(c.rect('tp_main'), geo)
    c.add(pt, pipe(outline, 0.06, seg=4, closed=True), occ=False)
    c.add(pt, pipe([V - nrm * 0.12 - up * 4.0, V - nrm * 0.12 + up * 4.0], 0.08, seg=5), occ=False)
    for v in (-3.0, -1.0, 1.0, 3.0):
        c.add(pt, pipe(_rib(S, nat, tpm_vrange, TPM_W, TPM_US, v, 0.1), 0.045, seg=4), occ=False)
    # back truss chord (its top end runs into the gearbox) and side struts from the gearbox
    vs = np.linspace(-3.0, 1.2, 5)
    spine = [V - nrm * 0.12 + up * v for v in vs]
    chord = [V - nrm * 0.55 + up * v for v in vs]
    c.add(pt, pipe(chord, 0.07, seg=5), occ=False)
    _warren(c, spine, chord, 0.04)
    for sx in (1, -1):
        c.add(pt, pipe([W(-0.3, sx * 0.62, 0.62), S(sx * 1.5, -1.0) - nat(sx * 1.5) * 0.1], 0.06, seg=4), occ=False)
        c.add(pt, pipe([W(-0.3, sx * 0.62, 1.7), S(sx * 1.5, 1.0) - nat(sx * 1.5) * 0.1], 0.05, seg=4), occ=False)
    # line feed in front of the reflector, its arms and stays
    f0, f1 = V + nrm * 1.35 - up * 3.6, V + nrm * 1.35 + up * 3.9
    c.add(c.sw('mid'), cylinder(0.13, np.linalg.norm(f1 - f0), seg=8), xf=M(f0, axis_frame(up)))
    c.add(pt, pipe([f1, S(0, TPM_H / 2)], 0.05, seg=4), occ=False)
    c.add(pt, pipe([f0, S(0, -TPM_H / 2)], 0.05, seg=4), occ=False)
    fm = V + nrm * 1.35 - up * 1.4
    for sx in (1, -1):
        c.add(pt, pipe([fm, S(sx * 2.4, -1.0)], 0.035, seg=4), occ=False)
    # waveguide from the gearbox over the top of the reflector to the feed head
    wg = [W(0.5, 0, 1.78), W(0.5, 0, 3.95), W(0.32, 0, 4.62), W(-0.2, 0, 4.92), W(-0.8, 0, 4.85), f1 + up * 0.05]
    c.add(pt, pipe(wg, 0.09, seg=5), occ=False)
    # IFF post with its box antenna and a stay to the feed head
    c.add(pt, pipe([W(0.7, 0, 1.75), W(0.7, 0, 5.0)], 0.08, seg=5), occ=False)
    c.add(pt, rbox(0.45, 0.9, 0.45, r=0.08, seg=1, y0=5.0), xf=M(W(0.7, 0, 0)))
    c.add(c.sw('dark'), pipe([W(0.7, 0, 5.9), W(0.7, 0, 6.45)], 0.03, seg=4), occ=False)
    c.add(pt, pipe([W(0.7, 0, 4.7), f1], 0.035, seg=4), occ=False)

    # ---- aft dish, facing aft and slightly up, on a lattice boom
    e = math.radians(8)
    nrm2 = np.array([0, math.sin(e), -math.cos(e)])
    up2 = np.array([0, math.cos(e), math.sin(e)])
    right2 = np.array([-1.0, 0, 0])
    V2 = W(3.0, 0, 0.0)
    geo, rim, D = ell_dish(V2, right2, up2, nrm2, TPD_A, TPD_B, TPD_DEPTH, cut=TPD_CUT, nr=4, ns=20)
    c.add(c.rect('tp_dish'), geo)
    c.add(pt, pipe(rim, 0.06, seg=4, closed=True), occ=False)
    for j in range(6):
        th = 2 * math.pi * j / 6 + math.pi / 2
        pts = []
        for rr in (0.0, 0.35, 0.7, 1.0):
            x, y = _ell_pt(rr, th, TPD_CUT)
            pts.append(D(x * TPD_A, y * TPD_B) - nrm2 * 0.1)
        c.add(pt, pipe(pts, 0.04, seg=4), occ=False)
    hub2 = V2 - nrm2 * 0.15
    truss(c, W(0.95, 0, 1.25), D(0, 0.95) - nrm2 * 0.25, 0.8, 0.6, rep=4.0)
    c.add(pt, pipe([W(0.95, 0, 0.6), hub2 - up2 * 0.2, D(0, -1.15) - nrm2 * 0.1], 0.08, seg=5), occ=False)
    for sx in (1, -1):
        c.add(pt, pipe([W(0.95, sx * 0.55, 0.6), D(sx * 1.4, 0.2) - nrm2 * 0.1], 0.05, seg=4), occ=False)
    # J-shaped feed arm under the dish with its horn looking back into it
    J = [W(0.75, 0, 0.55), W(1.3, 0, -0.7), W(2.3, 0, -1.75), W(3.6, 0, -2.0), W(4.9, 0, -1.75),
         W(5.35, 0, -1.2), W(5.3, 0, -0.8)]
    c.add(c.sw('mid'), pipe(J, 0.11, seg=6), occ=False)
    _horn(c, J[-1] + normalize(V2 + nrm2 * 0.4 - J[-1]) * 0.45, V2 + nrm2 * 0.4, r0=0.11, r1=0.26)
    b.pop()
    b.node(parent)


# =============================================================================================
# Fregat (foremast top): forward parabolic-cylinder array + aft trough, IFF bar on top (sheet 1)
# =============================================================================================
def fregat(ctx, o, name='Fregat_Radar', parent='Superstructure'):
    """o = pivot on top of the stalk (B 78.15, h 33.6)."""
    c, b = ctx, ctx.b
    o = np.asarray(o, float)
    b.push(name, parent=parent, translation=tuple(o))

    def W(dB, dx, dy):
        return o + np.array([dx, dy, -dB])

    pt = c.paint
    c.add(pt, lathe([(0.62, 0.0), (0.62, 0.14), (0.56, 0.2), (0.56, 0.42), (0.5, 0.48)], seg=16), xf=M(o))
    c.add(pt, rbox(1.2, 1.0, 1.4, r=0.12, seg=2, bevel=0.05, y0=0.45), xf=M(W(0.4, 0, 0)))

    # ---- main reflector, leaning back 22 deg, concave forward
    tl = math.radians(22)
    up = np.array([0, math.cos(tl), -math.sin(tl)])
    nrm = np.array([0, math.sin(tl), math.cos(tl)])
    right = np.array([1.0, 0, 0])
    V = W(-0.55, 0, 1.0)
    geo, outline, S, nat = cyl_reflector(V, right, up, nrm, FR_W, FR_H, FR_SAG, fr_vrange, FR_US)
    c.add(c.rect('fregat'), geo)
    c.add(pt, pipe(outline, 0.06, seg=4, closed=True), occ=False)
    c.add(pt, pipe([V - nrm * 0.1 - up * 2.4, V - nrm * 0.1 + up * 2.4], 0.07, seg=5), occ=False)
    for v in (-1.6, -0.4, 0.8, 2.0):
        c.add(pt, pipe(_rib(S, nat, fr_vrange, FR_W, FR_US, v, 0.08), 0.045, seg=4), occ=False)
    vs = np.linspace(-2.0, 0.4, 4)
    spine = [V - nrm * 0.1 + up * v for v in vs]
    chord = [V - nrm * 0.55 + up * v for v in vs]
    c.add(pt, pipe(chord, 0.06, seg=5), occ=False)
    _warren(c, spine, chord, 0.035)
    for sx in (1, -1):
        c.add(pt, pipe([W(0.0, sx * 0.55, 0.55), S(sx * 1.4, -1.0) - nat(sx * 1.4) * 0.08], 0.05, seg=4), occ=False)
    # line feed and arms
    f0, f1 = V + nrm * 1.9 - up * 2.0, V + nrm * 1.9 + up * 2.2
    c.add(c.sw('mid'), cylinder(0.1, np.linalg.norm(f1 - f0), seg=8), xf=M(f0, axis_frame(up)))
    c.add(pt, pipe([f1, S(0, FR_H / 2)], 0.045, seg=4), occ=False)
    c.add(pt, pipe([f0, S(0, -FR_H / 2)], 0.045, seg=4), occ=False)
    for sx in (1, -1):
        c.add(pt, pipe([V + nrm * 1.9, S(sx * 2.5, 0.0)], 0.03, seg=4), occ=False)
    # IFF bar along the top edge, with dipoles
    R = np.column_stack([right, up, nrm])
    bar = S(0, FR_H / 2) + up * 0.18 + nrm * 0.75
    c.add(pt, rbox(3.6, 0.2, 0.3, r=0.05, seg=1, y0=0.0), xf=M(bar, R))
    for sx in (1, -1):
        c.add(pt, pipe([S(sx * 1.2, FR_H / 2), bar + right * sx * 1.2 + up * 0.02], 0.035, seg=4), occ=False)
    for k in range(9):
        c.add(c.sw('dark'), box(0.05, 0.28, 0.05, center=(0, 0.34, 0)), xf=M(bar + right * (-1.6 + 0.4 * k), R))

    # ---- aft trough antenna, facing aft and slightly up
    e = math.radians(12)
    nrm2 = np.array([0, math.sin(e), -math.cos(e)])
    up2 = np.array([0, math.cos(e), math.sin(e)])
    right2 = np.array([-1.0, 0, 0])
    V2 = W(1.55, 0, 1.3)
    vr2 = lambda u: (-FR2_H / 2, FR2_H / 2)
    us2 = list(np.linspace(-FR2_W / 2, FR2_W / 2, 7))
    geo, outline2, S2, nat2 = cyl_reflector(V2, right2, up2, nrm2, FR2_W, FR2_H, FR2_SAG, vr2, us2, rows=1)
    c.add(c.rect('fregat2'), geo)
    c.add(pt, pipe(outline2, 0.045, seg=4, closed=True), occ=False)
    c.add(pt, pipe([S2(0, -0.7) - nrm2 * 0.08, S2(0, 0.7) - nrm2 * 0.08], 0.05, seg=4), occ=False)
    for sx in (1, -1):
        c.add(pt, pipe([W(1.05, sx * 0.4, 1.1), S2(sx * 0.5, 0.3) - nat2(sx * 0.5) * 0.08], 0.05, seg=4), occ=False)
        c.add(pt, pipe([W(1.05, sx * 0.4, 0.6), S2(sx * 0.5, -0.5) - nat2(sx * 0.5) * 0.08], 0.04, seg=4), occ=False)
    J = [W(0.85, 0, 0.5), W(1.5, 0, -0.25), W(2.6, 0, -0.4), W(3.3, 0, -0.1), W(3.4, 0, 0.45)]
    c.add(c.sw('mid'), pipe(J, 0.08, seg=6), occ=False)
    _horn(c, J[-1] + normalize(V2 + nrm2 * 0.3 - J[-1]) * 0.35, V2 + nrm2 * 0.3, r0=0.08, r1=0.19, L=0.35)
    b.pop()
    b.node(parent)


# =============================================================================================
# Argon-1164 "Front Door": J-shaped louvred reflector over a cantilevered cabin (sheet 1, photo)
# =============================================================================================
def front_door(ctx, tower_front_B, name='FrontDoor', parent='Superstructure'):
    c, b = ctx, ctx.b
    pt = c.paint
    fB = tower_front_B
    b.node(parent)
    # cantilevered platform with railing, knee braces, the equipment cabin and its bracket
    pf = fB(24.6) + 0.3
    poly = [(73.8, 1.45), (pf, 1.45), (pf, -1.45), (73.8, -1.45)]
    slab(c, poly, 24.45, 24.65)
    railing_pts(c, [P3(fB(24.65) - 0.05, 1.38, 24.65), P3(73.88, 1.38, 24.65), P3(73.88, -1.38, 24.65),
                    P3(fB(24.65) - 0.05, -1.38, 24.65)])
    for sx in (1, -1):
        c.add(pt, pipe([P3(fB(23.2) - 0.05, sx * 1.0, 23.2), P3(74.0, sx * 1.0, 24.45)], 0.07, seg=5), occ=False)
    c.add(pt, rbox(1.8, 1.15, 1.7, r=0.12, seg=2, bevel=0.06, y0=25.1), xf=M(P3(74.8, 0, 0)))
    c.add(pt, cylinder(0.3, 0.45, seg=10, y0=24.65), xf=M(P3(74.8, 0, 0)))
    for sx in (1, -1):
        decal(c, 'louvre', P3(74.6, sx * 0.91, 25.65), (sx, 0, 0), 0.7, 0.6)
    decal(c, 'louvre', P3(73.94, 0, 25.65), (0, 0, 1), 0.9, 0.55)

    # rotating head: drive dome, yoke, reflector, feed boom
    o = P3(74.4, 0, 26.25)
    b.push(name, parent=parent, translation=tuple(o))
    c.add(pt, lathe([(0.62, 0.0), (0.62, 0.25), (0.52, 0.48), (0.32, 0.64), (0.0, 0.68)], seg=16), xf=M(o))

    def Pb(B, h, x=0.0):
        return P3(B, x, h)

    def back(t):
        return 73.55 - 1.45 * t ** 2.4, 28.6 - 3.55 * t

    def front(t):
        Bb, hb = back(t)
        dB, dh = -3.48 * t ** 1.4, -3.55
        nB, nh = dh, -dB
        n = math.hypot(nB, nh)
        return Bb + 0.4 * nB / n, hb + 0.4 * nh / n, (nB / n, nh / n)

    ts = np.linspace(0.0, 1.0, 9)
    P, N, UV, I = [], [], [], []
    acc, prev = 0.0, None
    for k, t in enumerate(ts):
        Bf, hf, (nB, nh) = front(t)
        if prev is not None:
            acc += math.hypot(Bf - prev[0], hf - prev[1])
        prev = (Bf, hf)
        for x in (-FD_W / 2, FD_W / 2):
            P.append(Pb(Bf, hf, x)); N.append(np.array([0.0, nh, -nB])); UV.append((0.5 + x / FD_W, acc))
    total = acc
    UV = [(u, v / total) for (u, v) in UV]
    for k in range(len(ts) - 1):
        a = 2 * k
        I += [(a, a + 1, a + 3), (a, a + 3, a + 2)]
    c.add(c.rect('fd_refl'), _orient(np.array(P), np.array(N), np.array(UV), np.array(I)))
    # side trusses (back chord, front chord, zigzag) and the centre back rib
    for x in (-FD_W / 2, 0.0, FD_W / 2):
        bk = [Pb(*back(t), x) for t in ts]
        fr = [Pb(*front(t)[:2], x) for t in ts]
        c.add(pt, pipe(bk, 0.05, seg=4), occ=False)
        if x != 0.0:
            c.add(pt, pipe(fr, 0.05, seg=4), occ=False)
        _warren(c, bk, fr, 0.032)
    for t in (0.0, 1.0):
        c.add(pt, pipe([Pb(*back(t), -FD_W / 2), Pb(*back(t), FD_W / 2)], 0.05, seg=4), occ=False)
    # yoke from the drive to the reflector back
    c.add(pt, beam(Pb(74.25, 26.8), Pb(*back(0.35)), 0.5, 0.4))
    c.add(pt, beam(Pb(74.3, 26.6), Pb(*back(0.7)), 0.35, 0.3))
    # feed boom (lattice girder) forward from the reflector top, feed horn and equipment box
    truss(c, Pb(73.75, 28.75), Pb(71.75, 28.55), 0.45, 0.45, rep=4.0)
    c.add(pt, rbox(0.5, 0.35, 0.45, r=0.06, seg=1, y0=28.78), xf=M(Pb(71.8, 0)))
    tip = Pb(71.85, 28.25)
    aim = Pb(*front(0.55)[:2])
    _horn(c, tip + normalize(aim - tip) * 0.5, aim, r0=0.1, r1=0.25, L=0.5)
    c.add(pt, pipe([Pb(71.8, 28.35), tip], 0.06, seg=4), occ=False)
    c.add(pt, rbox(0.6, 0.35, 0.5, r=0.06, seg=1, y0=29.0), xf=M(Pb(73.0, 0)))
    c.add(pt, pipe([Pb(73.0, 28.8), Pb(73.0, 29.0)], 0.06, seg=4), occ=False)
    b.pop()
    b.node(parent)
