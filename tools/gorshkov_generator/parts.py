"""Generic hard-surface parts shared by the equipment modules (copied from the Slava-class generator's
detail, antennas, weapons and deckgear modules, which are bound to that ship's hull module)."""
import math
import numpy as np

from meshkit import *
from kit import M, orient_outward


def rrect(hx, hz, r, seg=3, cx=0.0, cz=0.0):
    """rounded rectangle outline [(x, z), ...] (counter-clockwise in x/z)."""
    r = max(1e-3, min(r, hx - 1e-3, hz - 1e-3))
    pts = []
    for (sx, sz, a0) in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
        ccx, ccz = cx + sx * (hx - r), cz + sz * (hz - r)
        for i in range(seg + 1):
            a = math.radians(a0 + 90.0 * i / seg)
            pts.append((ccx + r * math.cos(a), ccz + r * math.sin(a)))
    return pts


def band(loop0, y0, loop1, y1, closed=True, smooth=True):
    """surface between two horizontal outlines (same vertex count); normals point away from the axis."""
    a = np.asarray(loop0, float); b = np.asarray(loop1, float)
    n = len(a)
    P = np.vstack([np.column_stack([a[:, 0], np.full(n, y0), a[:, 1]]),
                   np.column_stack([b[:, 0], np.full(n, y1), b[:, 1]])])
    I = []
    m = n if closed else n - 1
    for i in range(m):
        j = (i + 1) % n
        I += [(i, j, n + j), (i, n + j, n + i)]
    I = np.array(I)
    cen = np.array([a[:, 0].mean(), (y0 + y1) / 2, a[:, 1].mean()])
    N = compute_smooth_normals(P, I)
    P, N, I = orient_outward(P, N, I, lambda p: np.array([cen[0], p[1], cen[2]]))
    if not smooth:
        P, N, UV, I = flatten(P, N, np.zeros((len(P), 2)), I)
    # uv: arc length / height in metres
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(np.vstack([a, a[:1]]), axis=0), axis=1))])[:n]
    if smooth:
        UV = np.column_stack([np.concatenate([L, L]), -P[:, 1]])
    else:
        UV = np.column_stack([P[:, 0] + P[:, 2], -P[:, 1]])
    return P, N, UV, I


def rbox(sx, sy, sz, r=0.12, seg=2, bevel=0.0, y0=0.0, bottom=False):
    """box with rounded vertical edges and an optional bevelled top edge; base at y0, centred in x/z."""
    hx, hz = sx / 2.0, sz / 2.0
    loop = rrect(hx, hz, r, seg)
    ytop = y0 + sy
    parts = []
    if bevel > 0:
        inner = rrect(hx - bevel, hz - bevel, max(r - bevel * 0.7, 0.02), seg)
        parts.append(band(loop, y0, loop, ytop - bevel))
        parts.append(band(loop, ytop - bevel, inner, ytop))
        parts.append(cap_polygon(np.array(inner), ytop, up=True))
    else:
        parts.append(band(loop, y0, loop, ytop))
        parts.append(cap_polygon(np.array(loop), ytop, up=True))
    if bottom:
        parts.append(cap_polygon(np.array(loop), y0, up=False))
    return merge(parts)


def dish(r, depth, seg=18, rings=4, rim=0.06):
    """parabolic dish, vertex at the origin, concave side facing +y."""
    prof = [(r * (k / rings), depth * (k / rings) ** 2) for k in range(rings + 1)]
    prof = prof + [(r + rim, depth + rim * 0.3)]
    return lathe(prof, seg=seg)


def axis_frame(direction):
    """rotation taking local +y onto `direction`."""
    d = normalize(np.asarray(direction, float))
    return frame_from_dir(d) @ rot_x(math.pi / 2)


def taper_beam(p0, p1, w0, h0, w1, h1, up=(0, 1, 0), cham=0.0):
    """closed tapered box beam from p0 to p1 (rectangular sections w x h); cham > 0 chamfers the four long
    edges by that fraction of the smaller side of each section."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    d = normalize(p1 - p0)
    upv = normalize(np.asarray(up, float))
    if abs(np.dot(upv, d)) > 0.95:
        upv = np.array([1.0, 0, 0])
    side = normalize(np.cross(d, upv)); upv = np.cross(side, d)
    def ring(p, w, h):
        if cham <= 0:
            return [p + side * sx * w / 2 + upv * sy * h / 2 for (sx, sy) in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        c, hw, hh = cham * min(w, h), w / 2, h / 2
        return [p + side * sx + upv * sy for (sx, sy) in ((c - hw, -hh), (hw - c, -hh), (hw, c - hh), (hw, hh - c),
                                                          (hw - c, hh), (c - hw, hh), (-hw, hh - c), (-hw, c - hh))]
    a = ring(p0, w0, h0); b = ring(p1, w1, h1)
    polys = [a[::-1], b]
    for i in range(len(a)):
        j = (i + 1) % len(a)
        polys.append([a[i], a[j], b[j], b[i]])
    P, N, UV, I = flat_poly_faces(polys)
    cen = (p0 + p1) / 2
    fn = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    fc = P[I].mean(axis=1)
    bad = np.sum(fn * (fc - cen), axis=1) < 0
    I[bad] = I[bad][:, ::-1]
    N = np.zeros_like(P)
    for t in I:
        N[t] = normalize(np.cross(P[t[1]] - P[t[0]], P[t[2]] - P[t[0]]))
    return P, N, UV, I


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


def _xf_pts(P, o, R):
    return (np.asarray(P, float) @ R.T) + o


def disc(r, seg=8, uv_rect=False):
    """flat disc in the xz plane facing +y (triangle fan); uv in 0..1 over the disc if uv_rect."""
    P = [np.zeros(3)] + [np.array([r * math.cos(2 * math.pi * k / seg), 0.0, -r * math.sin(2 * math.pi * k / seg)])
                         for k in range(seg)]
    P = np.array(P)
    UV = np.column_stack([0.5 + P[:, 0] / (2 * r), 0.5 - P[:, 2] / (2 * r)]) if uv_rect else P[:, [0, 2]]
    I = np.array([[0, 1 + k, 1 + (k + 1) % seg] for k in range(seg)])
    N = np.tile([0, 1.0, 0], (len(P), 1))
    return _orient(P, N, UV, I)


def ellipsoid(rx, ry, rz, seg=10, rings=6, hemi=False):
    P, N, UV, I = sphere(1.0, seg=seg, rings=rings, hemi=hemi)
    S = np.array([rx, ry, rz])
    P = P * S
    N = normalize(N / S)
    return P, N, UV, I


def round_vent(ctx, pos, r=0.32, h=0.55):
    """mushroom deck ventilator: red-brown coaming under a grey domed cap."""
    o = np.asarray(pos, float)
    ctx.add(ctx.sw('deck_red'), cylinder(r * 0.72, h * 0.55, seg=12, caps=(False, False)), xf=M(o))
    ctx.add(ctx.paint, lathe([(r * 0.96, h * 0.5), (r, h * 0.66), (r * 0.7, h * 0.95), (0.0, h)], seg=12), xf=M(o))


def ready_locker(ctx, pos, w=1.2, h=1.0, d=0.6, yaw=0.0):
    """ready-use locker: rounded box with a sloped lid, two hinged doors and a drip rail."""
    o = np.asarray(pos, float)
    R = rot_y(yaw)
    ctx.add(ctx.paint, rbox(w, h, d, r=0.05, seg=1, bevel=0.05), xf=M(o, R))
    front = o + R @ np.array([0, 0, d / 2 + 0.005])
    for sx in (-0.25, 0.25):
        ctx.add(ctx.sw('mid'), box(w * 0.44, h * 0.72, 0.02, center=(0, 0, 0)), xf=M(front + R @ np.array([sx * w, h * 0.46, 0]), R))
        ctx.add(ctx.sw('dark'), box(0.03, 0.14, 0.04, center=(0, 0, 0)), xf=M(front + R @ np.array([sx * w * 0.35, h * 0.5, 0.02]), R))
    ctx.add(ctx.paint, box(w + 0.06, 0.05, 0.08, center=(0, 0, 0)), xf=M(front + R @ np.array([0, h * 0.9, 0.02]), R))


def capstan(ctx, pos, r=0.6):
    """warping capstan: base flange, waisted drum with whelps, domed cap (dark grey)."""
    k = r / 0.6
    prof = [(0.66, 0.0), (0.66, 0.1), (0.52, 0.15), (0.43, 0.42), (0.47, 0.68), (0.57, 0.72), (0.57, 0.83),
            (0.34, 0.92), (0.0, 0.96)]
    ctx.add(ctx.sw('dark'), lathe([(a * k, b * k) for (a, b) in prof], seg=12), xf=M(np.asarray(pos, float)))


def windlass(ctx, pos, side):
    """anchor windlass: cable lifter between flanges on a bed plate, brake band, motor housing outboard."""
    o = np.asarray(pos, float)
    dk_ = ctx.sw('dark')
    ctx.add(dk_, rbox(1.5, 0.12, 1.5, r=0.12, seg=1), xf=M(o))
    ctx.add(dk_, lathe([(0.66, 0.12), (0.66, 0.2), (0.5, 0.24), (0.5, 0.52), (0.66, 0.56), (0.66, 0.64),
                        (0.42, 0.7), (0.3, 0.92), (0.0, 0.95)], seg=12), xf=M(o))
    for k in range(6):                                      # whelps on the cable lifter
        a = 2 * math.pi * k / 6
        ctx.add(dk_, box(0.08, 0.26, 0.12, center=(0, 0, 0)), xf=M(o + np.array([0.53 * math.cos(a), 0.38, 0.53 * math.sin(a)]), rot_y(-a)))
    mo = o + np.array([side * 1.05, 0, 0])
    ctx.add(ctx.paint, rbox(0.95, 0.75, 1.3, r=0.12, seg=2, bevel=0.06, y0=0.0), xf=M(mo))
    ctx.add(ctx.paint, lathe([(0.3, 0.0), (0.3, 0.42), (0.22, 0.5), (0.0, 0.52)], seg=10),
            xf=M(mo + np.array([0, 0.52, -0.15]), rot_x(math.pi / 2)))


def chain_stopper(ctx, pos, along):
    """bar-type chain stopper: low saddle with two cheeks and a hinged bar."""
    o = np.asarray(pos, float)
    d = normalize(np.asarray(along, float))
    yaw = math.atan2(d[0], d[2])
    R = rot_y(yaw)
    dk_ = ctx.sw('dark')
    ctx.add(dk_, rbox(0.62, 0.12, 1.0, r=0.06, seg=1), xf=M(o, R))
    for sx in (-1, 1):
        ctx.add(dk_, rbox(0.1, 0.34, 0.8, r=0.04, seg=1, bevel=0.03, y0=0.1), xf=M(o + R @ np.array([sx * 0.24, 0, 0]), R))
    ctx.add(ctx.sw('mid'), beam(o + R @ np.array([-0.3, 0.4, 0.1]), o + R @ np.array([0.3, 0.4, 0.1]), 0.08, 0.08))


def lamp(ctx, pos, colour):
    """navigation / signal lamp: pedestal, coloured lens, cap."""
    o = np.asarray(pos, float)
    ctx.add(ctx.sw('dark'), cylinder(0.09, 0.12, seg=8), xf=M(o))
    ctx.add(ctx.sw(colour), cylinder(0.13, 0.2, seg=10, y0=0.12), xf=M(o))
    ctx.add(ctx.sw('dark'), lathe([(0.16, 0.32), (0.16, 0.36), (0.08, 0.42), (0.0, 0.43)], seg=10), xf=M(o))


def radome(ctx, pos, r=0.5, h=0.7, seg=12):
    """small ECM / communications radome: cylinder with a hemispherical cap."""
    o = np.asarray(pos, float)
    ctx.add(ctx.paint, cylinder(r + 0.05, 0.08, seg=seg, caps=(False, True)), xf=M(o))
    ctx.add(ctx.sw('white'), cylinder(r, h - 0.08, seg=seg, caps=(False, False), y0=0.08), xf=M(o))
    ctx.add(ctx.sw('white'), sphere(r, seg=seg, rings=3, hemi=True), xf=M(o + np.array([0, h, 0])))
