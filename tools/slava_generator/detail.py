"""Detail pass: refined equipment shapes (weapons, directors, radars) and geometry helpers.

Local part frames: +y up, +z = the direction the part faces (bow when facing = 0).
All builders take world positions; rotating parts get their own node pivoted at `pos`.
"""
import math
import numpy as np

from meshkit import *
from meshkit import _quad
import atlas as st
from kit import M, orient_outward


# =============================================================================================
# geometry helpers
# =============================================================================================
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


def revolve(profile, seg=16, axis_xf=None):
    """lathe helper returning geometry (optionally transformed by a 4x4)."""
    g = lathe(profile, seg=seg)
    return g


def axis_frame(direction):
    """rotation taking local +y onto `direction`."""
    d = normalize(np.asarray(direction, float))
    return frame_from_dir(d) @ rot_x(math.pi / 2)


# =============================================================================================
# AK-630M close-in weapon system
# =============================================================================================
def ak630(ctx, pos, facing=0.0, name='AK630', parent=None):
    """AK-630M: ribbed base ring, conical turret with flat top, gun port and the 6-barrel cluster jacket."""
    b = ctx.b
    o = np.asarray(pos, float)
    b.push(name, parent=parent, translation=tuple(o))
    R = rot_y(facing)
    pt = ctx.paint
    # base ring with stiffening ribs
    ctx.add(pt, lathe([(1.32, 0.0), (1.32, 0.26), (1.22, 0.34)], seg=20), xf=M(o, R))
    for k in range(10):
        a = 2 * math.pi * (k + 0.5) / 10
        ctx.add(pt, box(0.08, 0.26, 0.16, center=(0, 0.13, 0)), xf=M(o + R @ np.array([1.36 * math.sin(a), 0, 1.36 * math.cos(a)]), R @ rot_y(a)))
    # turret: conical body with a rounded shoulder and flat roof
    prof = [(1.18, 0.34), (1.16, 0.52), (1.02, 0.9), (0.84, 1.36), (0.76, 1.52), (0.62, 1.6), (0.0, 1.62)]
    ctx.add(pt, lathe(prof, seg=20), xf=M(o, R))
    # gun port block on the front face and the barrel jacket
    ctx.add(ctx.sw('mid'), rbox(0.62, 0.62, 0.45, r=0.1, seg=1, y0=0.68), xf=M(o + R @ np.array([0, 0, 0.92]), R))
    Rg = R @ rot_x(math.pi / 2)
    ctx.add(ctx.sw('dark'), cylinder(0.2, 0.55, seg=10), xf=M(o + R @ np.array([0, 0.99, 1.1]), Rg))
    ctx.add(ctx.sw('dark'), cylinder(0.16, 1.55, seg=10, r_top=0.15), xf=M(o + R @ np.array([0, 0.99, 1.62]), Rg))
    ctx.add(ctx.sw('dark'), cylinder(0.2, 0.3, seg=10), xf=M(o + R @ np.array([0, 0.99, 3.12]), Rg))
    # optical sight and vent on the roof
    ctx.add(pt, rbox(0.34, 0.26, 0.5, r=0.06, seg=1, y0=1.6), xf=M(o + R @ np.array([-0.32, 0, -0.15]), R))
    ctx.add(ctx.sw('dark'), box(0.24, 0.12, 0.04, center=(0, 0, 0)), xf=M(o + R @ np.array([-0.32, 1.74, 0.11]), R))
    ctx.add(pt, cylinder(0.12, 0.18, seg=8), xf=M(o + R @ np.array([0.3, 1.6, -0.3]), R))
    b.pop()


# =============================================================================================
# MR-123 "Bass Tilt" fire-control director (for the AK-630s)
# =============================================================================================
def bass_tilt(ctx, pos, facing=0.0, name='BassTilt', parent=None, pedestal=1.4):
    b = ctx.b
    o = np.asarray(pos, float)
    b.push(name, parent=parent, translation=tuple(o))
    R = rot_y(facing)
    pt = ctx.paint
    # flared pedestal and turntable
    ctx.add(pt, lathe([(0.8, 0.0), (0.8, 0.12), (0.5, 0.32), (0.42, 0.6), (0.42, pedestal)], seg=16), xf=M(o, R))
    ctx.add(pt, cylinder(0.78, 0.16, seg=18, y0=pedestal), xf=M(o, R))
    y1 = pedestal + 0.16
    # electronics housing with a sloped front
    ctx.add(pt, rbox(1.55, 0.95, 1.35, r=0.14, seg=2, bevel=0.08, y0=y1), xf=M(o + R @ np.array([0, 0, -0.15]), R))
    # capsule radome (dome-ended cylinder) lying fore and aft on the housing
    prof = [(0.0, -1.05), (0.5, -1.03), (0.74, -0.92), (0.8, -0.7), (0.8, 0.3), (0.76, 0.6), (0.62, 0.86),
            (0.38, 1.02), (0.0, 1.08)]
    ctx.add(ctx.sw('radome'), lathe(prof, seg=20), xf=M(o + R @ np.array([0, y1 + 1.62, 0.05]), R @ rot_x(math.pi / 2)))
    ctx.add(pt, cylinder(0.84, 0.1, seg=20, caps=(False, False)), xf=M(o + R @ np.array([0, y1 + 1.62, -0.72]), R @ rot_x(math.pi / 2)))
    # side-mounted optical sight
    ctx.add(pt, rbox(0.38, 0.42, 0.62, r=0.08, seg=1, y0=y1 + 0.75), xf=M(o + R @ np.array([1.0, 0, 0.1]), R))
    ctx.add(ctx.sw('glass'), box(0.26, 0.2, 0.02, center=(0, 0, 0)), xf=M(o + R @ np.array([1.0, y1 + 1.0, 0.42]), R))
    b.pop()


# =============================================================================================
# MR-184 "Kite Screech" gun fire-control director (bridge roof)
# =============================================================================================
def kite_screech(ctx, pos, name='KiteScreech', parent='Superstructure'):
    b = ctx.b
    o = np.asarray(pos, float)
    b.push(name, parent=parent, translation=tuple(o))
    pt = ctx.paint
    ctx.add(pt, lathe([(0.95, 0.0), (0.95, 0.14), (0.62, 0.34), (0.5, 0.7), (0.5, 1.9)], seg=16), xf=M(o))
    ctx.add(pt, cylinder(0.8, 0.18, seg=18, y0=1.9), xf=M(o))
    # main housing (long axis fore-aft) with a hood
    ctx.add(pt, rbox(1.5, 1.15, 2.5, r=0.16, seg=2, bevel=0.1, y0=2.08), xf=M(o + np.array([0, 0, -0.1])))
    ctx.add(pt, rbox(1.1, 0.3, 1.4, r=0.1, seg=1, bevel=0.06, y0=3.23), xf=M(o + np.array([0, 0, -0.35])))
    # main dish facing forward, feed rod and struts
    dc = o + np.array([0, 2.75, 1.25])
    ctx.add(ctx.sw('light'), dish(1.2, 0.42, seg=22, rings=4), xf=M(dc, rot_x(math.pi / 2)))
    ctx.add(ctx.sw('light'), cylinder(0.25, 0.25, seg=10), xf=M(dc + np.array([0, 0, -0.2]), rot_x(math.pi / 2)))
    ctx.add(ctx.sw('dark'), tube_path([dc + np.array([0, 0, 0.1]), dc + np.array([0, 0, 1.25])], 0.045, seg=5))
    ctx.add(ctx.sw('dark'), cylinder(0.11, 0.22, seg=8, r_top=0.05), xf=M(dc + np.array([0, 0, 1.2]), rot_x(math.pi / 2)))
    for a in (0, 2.1, 4.2):
        tip = dc + np.array([1.05 * math.cos(a), 1.05 * math.sin(a), 0.4])
        ctx.add(ctx.sw('dark'), tube_path([tip, dc + np.array([0, 0, 1.05])], 0.025, seg=4))
    # TV/laser sight on the left, small horn aft
    ctx.add(pt, rbox(0.42, 0.5, 0.9, r=0.1, seg=1, y0=2.6), xf=M(o + np.array([-0.98, 0, 0.3])))
    ctx.add(ctx.sw('glass'), box(0.3, 0.3, 0.02, center=(0, 0, 0)), xf=M(o + np.array([-0.98, 2.85, 0.76])))
    ctx.add(ctx.sw('white'), lathe([(0.0, 0.0), (0.16, 0.0), (0.34, 0.55), (0.0, 0.56)], seg=12),
            xf=M(o + np.array([0, 2.7, -1.35]), rot_x(-math.pi / 2)))
    b.pop()


# =============================================================================================
# 4R33 "Pop Group" (Osa-M fire control): housing with two dishes + horn, stowed facing aft
# =============================================================================================
def pop_group(ctx, pos, side, name, parent='Superstructure'):
    b = ctx.b
    o = np.asarray(pos, float)
    b.push(name, parent=parent, translation=tuple(o))
    pt = ctx.paint
    R = rot_y(math.pi)          # faces aft
    ctx.add(pt, lathe([(0.7, 0.0), (0.7, 0.12), (0.45, 0.3), (0.42, 0.75)], seg=14), xf=M(o, R))
    ctx.add(pt, cylinder(0.7, 0.15, seg=16, y0=0.75), xf=M(o, R))
    ctx.add(pt, rbox(1.9, 1.25, 1.55, r=0.14, seg=2, bevel=0.08, y0=0.9), xf=M(o, R))
    Rd = R @ rot_x(math.pi / 2)
    ctx.add(ctx.sw('light'), dish(0.72, 0.24, seg=18, rings=3), xf=M(o + R @ np.array([-0.35, 1.5, 0.8]), Rd))
    ctx.add(ctx.sw('dark'), tube_path([o + R @ np.array([-0.35, 1.5, 0.9]), o + R @ np.array([-0.35, 1.5, 1.6])], 0.03, seg=4))
    ctx.add(ctx.sw('light'), dish(0.42, 0.14, seg=14, rings=2), xf=M(o + R @ np.array([0.62, 1.75, 0.8]), Rd))
    ctx.add(ctx.sw('dark'), lathe([(0.0, 0.0), (0.12, 0.0), (0.24, 0.4), (0.0, 0.41)], seg=10),
            xf=M(o + R @ np.array([0.62, 1.12, 0.78]), Rd))
    ctx.add(pt, rbox(0.4, 0.35, 0.5, r=0.06, seg=1, y0=2.15), xf=M(o + R @ np.array([0.2, 0, -0.2]), R))
    b.pop()


# =============================================================================================
# 3R41 "Top Dome" (S-300F guidance): spherical-cap radome + truncated cone front with a flat face
# =============================================================================================
def top_dome(ctx, pos, name='TopDome', parent='Superstructure'):
    b = ctx.b
    o = np.asarray(pos, float)
    b.push(name, parent=parent, translation=tuple(o))
    pt = ctx.paint
    ctx.add(pt, cylinder(1.55, 0.45, seg=24), xf=M(o))
    ctx.add(pt, rbox(2.5, 0.95, 2.6, r=0.25, seg=2, bevel=0.08, y0=0.45), xf=M(o + np.array([0, 0, -0.2])))
    for s in (1, -1):
        ctx.add(pt, rbox(0.4, 1.3, 1.2, r=0.08, seg=1, y0=1.2), xf=M(o + np.array([s * 1.05, 0, 0.0])))
    el = math.radians(8)
    d = np.array([0.0, math.sin(el), math.cos(el)])
    Rax = axis_frame(d)
    c0 = o + np.array([0, 3.15, -0.25])
    Rr, cut = 2.3, 0.75
    prof = []
    for k in range(10):
        th = math.acos(-cut / Rr) * k / 9.0
        prof.append((Rr * math.sin(th), -Rr * math.cos(th)))
    rc = math.sqrt(Rr * Rr - cut * cut)
    prof_cone = [(rc, cut), (rc - 0.03, cut + 0.12), (1.3, cut + 2.05), (1.24, cut + 2.12)]
    ctx.add(pt, lathe(prof, seg=28), xf=M(c0, Rax))
    ctx.add(pt, lathe(prof_cone, seg=28), xf=M(c0, Rax))
    ctx.add(pt, cylinder(rc + 0.07, 0.14, seg=28, caps=(False, False), y0=cut - 0.07), xf=M(c0, Rax))
    # flat front face with the access hatch
    fc = c0 + d * (cut + 2.12)
    ctx.add(ctx.rect('td_face'), cylinder(1.24, 0.001, seg=24, caps=(False, True), cap_uv_rect=(0, 0, 1, 1)), xf=M(fc, Rax))
    # small auxiliary radome on the base, vent mast behind
    ctx.add(ctx.sw('radome'), sphere(0.42, seg=12, rings=6), xf=M(o + np.array([-1.35, 1.75, 1.0])))
    b.pop()


# =============================================================================================
# AK-130 twin 130 mm turret body (called inside the turret node) and guns
# =============================================================================================
def ak130_turret(ctx, Bc, ybase):
    """turret shell centred at (B=Bc, y=ybase): rounded body, cylindrical front shield, hood, cupola."""
    pt = ctx.paint
    zc = 93.2 - Bc
    # body: superellipse sections, flat front, rounded rear, filleted roof
    secs = []
    for h, sc in ((0.0, 1.0), (0.25, 1.03), (1.8, 1.04), (2.55, 1.0), (2.95, 0.93), (3.2, 0.8), (3.3, 0.55), (3.33, 0.1)):
        ring = []
        for k in range(28):
            a = 2 * math.pi * k / 28
            ca, sa = math.cos(a), math.sin(a)
            x = 2.6 * sc * np.sign(ca) * abs(ca) ** 0.55
            zz = 2.9 * sc * np.sign(sa) * abs(sa) ** (0.42 if sa > 0 else 0.85)
            zz = min(zz, 1.9)
            ring.append((x, ybase + h, zc + zz - 0.3))
        secs.append(ring)
    P, N, UV, I = loft(secs, closed=True, uvscale=1.0)
    P, N, I = orient_outward(P, N, I, lambda p: np.array([0, ybase + 1.5, zc - 0.5]))
    ctx.add(pt, (P, N, UV, I))
    # cylindrical front shield (rotates with the guns in reality; kept with the turret)
    ang = np.linspace(-0.82, 0.82, 11)
    Rf = 2.45
    cz = zc - 0.65
    loop_b = [(Rf * math.sin(a), cz + Rf * math.cos(a)) for a in ang]
    P, N, UV, I = band(loop_b, ybase + 0.35, loop_b, ybase + 2.95, closed=False)
    P, N, I = orient_outward(P, N, I, lambda p: np.array([0.0, p[1], cz]))
    ctx.add(pt, (P, N, UV, I))
    # shield top / bottom caps (fans)
    for yy, up in ((ybase + 2.95, True), (ybase + 0.35, False)):
        pts = [(0.0, cz)] + loop_b
        Pc = np.array([(x, yy, z) for (x, z) in pts])
        Ic = np.array([[0, i, i + 1] for i in range(1, len(pts) - 1)])
        Nn = np.tile([0, 1.0 if up else -1.0, 0], (len(Pc), 1))
        if np.cross(Pc[Ic[0, 1]] - Pc[Ic[0, 0]], Pc[Ic[0, 2]] - Pc[Ic[0, 0]])[1] * (1 if up else -1) < 0:
            Ic = Ic[:, ::-1]
        ctx.add(pt, (Pc, Nn, Pc[:, [2, 0]], Ic))
    # gun ports (dark slots) on the shield face
    for s in (1, -1):
        ctx.add(ctx.sw('black'), rbox(0.7, 1.1, 0.12, r=0.15, seg=2, y0=ybase + 1.05), xf=M(np.array([s * 0.78, 0, cz + Rf - 0.02])))
    # roof hood with a sloped front, sight cupola, vents
    ctx.add(pt, frustum_box((-0.95, 0.95, zc - 1.6, zc + 0.7), (-0.8, 0.8, zc - 1.45, zc + 0.35), ybase + 3.25, ybase + 3.75))
    ctx.add(pt, lathe([(0.0, 0.0), (0.58, 0.0), (0.58, 0.18), (0.5, 0.42), (0.3, 0.58), (0.0, 0.62)], seg=16),
            xf=M(np.array([-1.15, ybase + 3.2, zc - 1.1])))
    ctx.add(ctx.sw('glass'), box(0.42, 0.14, 0.04, center=(0, 0, 0)), xf=M(np.array([-1.15, ybase + 3.62, zc - 0.55])))
    for s in (1, -1):
        ctx.add(pt, cylinder(0.16, 0.2, seg=8), xf=M(np.array([s * 1.6, ybase + 3.05, zc - 1.9])))
    # side hatches with handles
    for s in (1, -1):
        ctx.add(ctx.rect('door'), _quad(np.array([s * 2.66, ybase + 0.85, zc - 1.5]), np.array([s * 2.66, ybase + 0.85, zc - 0.7]),
                                        np.array([s * 2.66, ybase + 2.35, zc - 0.7]), np.array([s * 2.66, ybase + 2.35, zc - 1.5]),
                                        uv=[(0, 1), (1, 1), (1, 0), (0, 0)] if s > 0 else [(1, 1), (0, 1), (0, 0), (1, 0)],
                                        n=np.array([s, 0, 0.0])))
        for yy in (1.3, 1.7, 2.1):
            ctx.add(pt, box(0.05, 0.05, 0.3, center=(0, 0, 0)), xf=M(np.array([s * 2.7, ybase + yy, zc - 0.35])))


def ak130_guns(ctx, Bt, xs, y, Bm):
    """two barrels from the shield (B = Bt) to the muzzle (B = Bm) at height y."""
    pt = ctx.paint
    Rz = rot_x(math.pi / 2)          # local +y -> +z (forward)
    z0 = 93.2 - Bt
    L = Bt - Bm
    for s in xs:
        ctx.add(pt, cylinder(0.4, 0.55, seg=16), xf=M(np.array([s, y, z0 - 0.35]), Rz))
        ctx.add(pt, cylinder(0.31, 1.1, seg=14, r_top=0.27), xf=M(np.array([s, y, z0 + 0.2]), Rz))
        ctx.add(pt, cylinder(0.3, 0.12, seg=14), xf=M(np.array([s, y, z0 + 1.3]), Rz))
        ctx.add(pt, cylinder(0.2, L - 1.85, seg=12, r_top=0.165), xf=M(np.array([s, y, z0 + 1.42]), Rz))
        ctx.add(pt, cylinder(0.205, 0.32, seg=12), xf=M(np.array([s, y, z0 + L - 0.43]), Rz))
        ctx.add(ctx.sw('black'), cylinder(0.1, 0.02, seg=10, caps=(False, True)), xf=M(np.array([s, y, z0 + L - 0.1]), Rz))


# =============================================================================================
# atlas painters for the detail pass
# =============================================================================================
def paint_td_face(L, rect):
    """Top Dome front face: grey disc, rim, rectangular access hatch with a red mark."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    PAL = st.PAL
    L.rect(x0, y0, x1, y1, col=PAL['super'] * 1.02, alpha=1.0)

    def rim(d, s):
        d.ellipse([2 * s, 2 * s, (w - 2) * s, (h - 2) * s], outline=255, width=3 * s)

    def hatch(d, s):
        d.rounded_rectangle([0.36 * w * s, 0.18 * h * s, 0.66 * w * s, 0.8 * h * s], radius=int(0.05 * w * s),
                            outline=255, width=2 * s)
    L.mask_apply(L.draw_mask(w, h, rim), col=PAL['super'] * 0.7, add_height=0.6, x0=x0, y0=y0)
    L.mask_apply(L.draw_mask(w, h, hatch), col=PAL['super'] * 0.6, add_height=0.8, x0=x0, y0=y0)
    L.rect(x0 + 0.47 * w, y0 + 0.28 * h, x0 + 0.53 * w, y0 + 0.33 * h, col=PAL['flagred'])


def register(m):
    m.alloc('td_face', 128, 128, paint_td_face)
    m.alloc('grille', 128, 64, paint_grille)


# =============================================================================================
# side plate with an opening, built as vertical columns (robust, no polygon-with-hole triangulation)
# =============================================================================================
def plate_columns(Bs, ybot, ytop, hole=None, x=0.0, normal_x=1.0):
    """vertical plate at x spanning B samples `Bs` (ascending) between ybot(B) and ytop(B).
    hole = (B0, B1, ylo(B), yhi(B)) cut out for B in [B0, B1]. Returns geometry facing +x*normal_x."""
    P, I, UV = [], [], []

    def quad(Ba, Bb, ya0, ya1, yb0, yb1):
        base = len(P)
        P.extend([(x, ya0, 93.2 - Ba), (x, yb0, 93.2 - Bb), (x, yb1, 93.2 - Bb), (x, ya1, 93.2 - Ba)])
        UV.extend([(Ba, -ya0), (Bb, -yb0), (Bb, -yb1), (Ba, -ya1)])
        I.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])
    for Ba, Bb in zip(Bs[:-1], Bs[1:]):
        if Bb - Ba < 1e-6:
            continue
        inside = hole is not None and Ba >= hole[0] - 1e-6 and Bb <= hole[1] + 1e-6
        if inside:
            quad(Ba, Bb, ybot(Ba), hole[2](Ba), ybot(Bb), hole[2](Bb))
            quad(Ba, Bb, hole[3](Ba), ytop(Ba), hole[3](Bb), ytop(Bb))
        else:
            quad(Ba, Bb, ybot(Ba), ytop(Ba), ybot(Bb), ytop(Bb))
    P = np.array(P); I = np.array(I); UV = np.array(UV)
    n = np.array([normal_x, 0.0, 0.0])
    fn = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    flip = np.sum(fn * n, axis=1) < 0
    I[flip] = I[flip][:, ::-1]
    return P, np.tile(n, (len(P), 1)), UV, I


def rounded_opening(B0, B1, ylo, yhi, rt=0.55, rb=0.18, nseg=4):
    """opening outline functions with rounded corners; ylo/yhi are callables of B (straight edges).
    Returns (Bsamples, lo(B), hi(B))."""
    def lo(B):
        y = ylo(B)
        for (c, sgn) in ((B0 + rb, 1), (B1 - rb, -1)):
            if (B - c) * sgn < 0:
                dx = abs(B - c)
                y += rb - math.sqrt(max(rb * rb - dx * dx, 0.0))
        return y

    def hi(B):
        y = yhi(B)
        for (c, sgn) in ((B0 + rt, 1), (B1 - rt, -1)):
            if (B - c) * sgn < 0:
                dx = abs(B - c)
                y -= rt - math.sqrt(max(rt * rt - dx * dx, 0.0))
        return y
    samples = set([B0, B1])
    for (r, c, sgn) in ((rt, B0 + rt, -1), (rt, B1 - rt, 1), (rb, B0 + rb, -1), (rb, B1 - rb, 1)):
        for k in range(nseg + 1):
            a = math.pi / 2 * k / nseg
            samples.add(c + sgn * r * math.cos(a))
    return sorted(b for b in samples if B0 - 1e-9 <= b <= B1 + 1e-9), lo, hi


# =============================================================================================
# P-500 / P-1000 launcher pair support (one per pair of containers)
# =============================================================================================
def launcher_support(ctx, Bf, side, tube_y, deck_y, x_in=4.95, x_out=9.65, tubes=((8.65, 0.0), (6.0, 0.35)),
                     rail=None, ladder_fn=None):
    """Bf: front of the containers; tube_y(B, dy) = container axis height; deck_y(B) = deck height."""
    s = side
    pt = ctx.paint
    Bfront, Brear = Bf + 0.25, Bf + 11.15

    def ytop(B):
        return tube_y(B, 0.0) - 1.0

    def ybot(B):
        return deck_y(B) - 0.25
    # opening (arched top, parallel to the slope)
    h0, h1 = Bf + 2.4, Bf + 7.9
    Hs, lo, hi = rounded_opening(h0, h1, lambda B: deck_y(B) + 0.3, lambda B: ytop(B) - 0.55, rt=0.6, rb=0.2)
    Bs = sorted(set([Bfront, Brear] + Hs + list(np.linspace(Bfront, Brear, 7))))
    xo = s * x_out
    ctx.add(pt, plate_columns(Bs, ybot, ytop, hole=(h0, h1, lo, hi), x=xo, normal_x=s))
    # inboard plate, top deck, front and rear faces
    xi = s * x_in
    ctx.add(pt, plate_columns(Bs, ybot, ytop, None, x=xi, normal_x=-s))
    P = []
    for B in Bs:
        P.append([(xi, ytop(B), 93.2 - B), (xo, ytop(B), 93.2 - B)])
    P = np.array(P).reshape(-1, 3)
    I = []
    for k in range(len(Bs) - 1):
        a = 2 * k
        I += [(a, a + 1, a + 3), (a, a + 3, a + 2)]
    I = np.array(I)
    N = compute_smooth_normals(P, I)
    if N[:, 1].mean() < 0:
        I = I[:, ::-1]; N = -N
    ctx.add(pt, (P, N, P[:, [2, 0]], I))
    for B, nz in ((Bfront, 1.0), (Brear, -1.0)):
        q = _quad((xi, ybot(B), 93.2 - B), (xo, ybot(B), 93.2 - B), (xo, ytop(B), 93.2 - B), (xi, ytop(B), 93.2 - B),
                  n=np.array([0, 0, nz]))
        Pq, Nq, UVq, Iq = q
        if np.dot(np.cross(Pq[1] - Pq[0], Pq[2] - Pq[0]), [0, 0, nz]) < 0:
            Iq = Iq[:, ::-1]
        ctx.add(pt, (Pq, Nq, Pq[:, [0, 1]], Iq))
    # recess behind the opening: side walls along the outline and a dark back panel
    depth = 1.1
    xb = xo - s * depth
    loop = [(B, lo(B)) for B in Hs] + [(B, hi(B)) for B in reversed(Hs)]
    P, I = [], []
    for (Ba, ya), (Bb, yb) in zip(loop, loop[1:] + loop[:1]):
        base = len(P)
        P.extend([(xo, ya, 93.2 - Ba), (xo, yb, 93.2 - Bb), (xb, yb, 93.2 - Bb), (xb, ya, 93.2 - Ba)])
        I.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])
    P = np.array(P); I = np.array(I)
    cen = np.array([xo, np.mean([p[1] for p in loop]), 93.2 - np.mean([p[0] for p in loop])])
    fn = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    fc = P[I].mean(axis=1)
    bad = np.sum(fn * (cen - fc) * np.array([0, 1, 1]), axis=1) < 0      # walls face into the opening
    I[bad] = I[bad][:, ::-1]
    N = np.zeros_like(P)
    for t in I:
        n = normalize(np.cross(P[t[1]] - P[t[0]], P[t[2]] - P[t[0]]))
        N[t] = n
    ctx.add(ctx.sw('mid'), (P, N, P[:, [2, 1]], I))
    ctx.add(ctx.sw('dark'), plate_columns(Hs, lo, hi, None, x=xb, normal_x=s))
    # walkway grating + rail seen through the opening
    if rail is not None:
        rail([(h0 + 0.3, s * (x_out - 0.6), lo(h0 + 1.0) + 0.02), (h1 - 0.3, s * (x_out - 0.6), lo(h1 - 1.0) + 0.02)])
    # saddles under the container heads and hold-down straps on the bodies
    for (xt, dy) in tubes:
        yb = tube_y(Bf + 0.85, dy) - 1.08
        if yb > ytop(Bf + 0.85) + 0.05:
            ctx.add(pt, rbox(1.7, yb - ytop(Bf + 0.85) + 0.15, 1.0, r=0.12, seg=1, y0=ytop(Bf + 0.85) - 0.1),
                    xf=M(np.array([s * xt, 0, 93.2 - (Bf + 0.85)])))
        for Bsd in (Bf + 3.4, Bf + 8.6):
            yb2 = tube_y(Bsd, dy) - 0.9
            if yb2 > ytop(Bsd) + 0.02:
                ctx.add(pt, box(1.3, yb2 - ytop(Bsd) + 0.1, 0.5, center=(0, 0, 0)),
                        xf=M(np.array([s * xt, (yb2 + ytop(Bsd)) / 2, 93.2 - Bsd])))
    # ladders up the front pedestal
    if ladder_fn is not None:
        ladder_fn(np.array([s * (x_out + 0.02), ybot(Bfront) + 0.3, 93.2 - (Bf + 1.0)]),
                  np.array([s * (x_out + 0.02), ytop(Bf + 1.0) - 0.15, 93.2 - (Bf + 1.0)]), s)


# =============================================================================================
# helpers for beams with taper and tubes
# =============================================================================================
def taper_beam(p0, p1, w0, h0, w1, h1, up=(0, 1, 0)):
    """closed tapered box beam from p0 to p1 (rectangular sections w x h)."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    d = normalize(p1 - p0)
    upv = normalize(np.asarray(up, float))
    if abs(np.dot(upv, d)) > 0.95:
        upv = np.array([1.0, 0, 0])
    side = normalize(np.cross(d, upv)); upv = np.cross(side, d)
    def ring(p, w, h):
        return [p + side * sx * w / 2 + upv * sy * h / 2 for (sx, sy) in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    a = ring(p0, w0, h0); b = ring(p1, w1, h1)
    polys = [a[::-1], b]
    for i in range(4):
        j = (i + 1) % 4
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


def hydraulic_ram(ctx, p0, p1, r=0.12):
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    m = p0 + (p1 - p0) * 0.55
    ctx.add(ctx.paint, tube_path([p0, m], r, seg=8, caps=True))
    ctx.add(ctx.sw('steel'), tube_path([m - (p1 - p0) * 0.05, p1], r * 0.55, seg=6, caps=True))


# =============================================================================================
# funnel exhaust outlets
# =============================================================================================
def funnel_outlets(ctx, B0, B1, x0, x1, ytop):
    """raised gas-turbine uptakes on one stack top (4 athwartships slots aft, 2 round forward)."""
    L = B1 - B0
    xc = (x0 + x1) / 2
    wx = abs(x1 - x0)
    grille = ctx.rect('grille')
    for u in (0.12, 0.24, 0.36, 0.48):
        Bc = B1 - u * L
        ctx.add(ctx.sw('black'), rbox(wx - 1.3, 0.42, 0.62, r=0.22, seg=2, y0=ytop - 0.02), xf=M(np.array([xc, 0, 93.2 - Bc])))
        # grille top as a textured quad
        q = _quad(np.array([xc - (wx - 1.5) / 2, ytop + 0.41, 93.2 - Bc - 0.24]),
                  np.array([xc + (wx - 1.5) / 2, ytop + 0.41, 93.2 - Bc - 0.24]),
                  np.array([xc + (wx - 1.5) / 2, ytop + 0.41, 93.2 - Bc + 0.24]),
                  np.array([xc - (wx - 1.5) / 2, ytop + 0.41, 93.2 - Bc + 0.24]),
                  uv=[(0, 1), (1, 1), (1, 0), (0, 0)], n=np.array([0, 1.0, 0]))
        Pq, Nq, UVq, Iq = q
        if np.cross(Pq[1] - Pq[0], Pq[2] - Pq[0])[1] < 0:
            Iq = Iq[:, ::-1]
        ctx.add(grille, (Pq, Nq, UVq, Iq))
    for (u, dx, r) in ((0.72, -0.45, 0.72), (0.9, 0.75, 0.5)):
        Bc = B1 - u * L
        o = np.array([xc + dx * np.sign(xc), ytop - 0.02, 93.2 - Bc])
        ctx.add(ctx.sw('black'), cylinder(r, 0.55, seg=16, caps=(False, False)), xf=M(o))
        ctx.add(ctx.sw('black'), cylinder(r + 0.05, 0.08, seg=16, caps=(False, False), y0=0.5), xf=M(o))
        ctx.add(grille, cylinder(r, 0.001, seg=16, caps=(False, True), cap_uv_rect=(0, 0, 1, 1), y0=0.5), xf=M(o))


def paint_grille(L, rect):
    x0, y0, x1, y1 = rect
    PAL = st.PAL
    L.rect(x0, y0, x1, y1, col=PAL['black'] * 0.7, alpha=1.0, rough=0.95)
    for xx in range(x0, x1, 5):
        L.rect(xx, y0, xx + 2, y1, col=PAL['dark'] * 0.9, add_height=0.6)
    for yy in range(y0, y1, 9):
        L.rect(x0, yy, x1, yy + 2, col=PAL['dark'] * 0.8, add_height=0.6)


# =============================================================================================
# ship's launch (PK-011.1 type) on its cradle, with two slewing davits
# =============================================================================================
def launch_boat(ctx, origin, side, L=10.2, beam=3.0):
    """origin = (x, y, z) of the keel midpoint; bow toward +z."""
    o = np.asarray(origin, float)
    ts = [0.0, 0.06, 0.15, 0.25, 0.35, 0.45, 0.55, 0.64, 0.72, 0.8, 0.87, 0.93, 1.0]
    sheer, chine, keel = [], [], []
    for t in ts:
        z = (t - 0.5) * L
        hb = 1.5 * (1.0 - max(0.0, (t - 0.5) / 0.5) ** 2.3) if t < 0.999 else 0.02
        hb = max(hb, 0.02)
        hb *= 0.95 if t < 0.05 else 1.0
        ys = 1.55 + 0.42 * t ** 2
        yk = 0.95 * max(0.0, (t - 0.62) / 0.38) ** 2
        hc = hb * (0.9 - 0.35 * max(0.0, (t - 0.5) / 0.5))
        yc = 0.42 + 0.5 * max(0.0, (t - 0.45) / 0.55) ** 1.5
        if t >= 0.999:
            hc, yc = 0.02, (ys + yk) / 2
        sheer.append((hb, ys, z)); chine.append((hc, yc, z)); keel.append((0.0, yk, z))
    def pts(lst, sx):
        return [o + np.array([sx * p[0], p[1], p[2]]) for p in lst]
    for sx in (1, -1):
        top = [[a, b] for a, b in zip(pts(sheer, sx), pts(chine, sx))]
        P, N, UV, I = loft(top, closed=False)
        P, N, I = orient_outward(P, N, I, lambda p: np.array([o[0], p[1], p[2]]))
        ctx.add(ctx.sw('white'), (P, N, UV, I))
        bot = [[a, b] for a, b in zip(pts(chine, sx), pts(keel, sx))]
        P, N, UV, I = loft(bot, closed=False)
        P, N, I = orient_outward(P, N, I, lambda p: np.array([o[0], o[1] + 1.4, p[2]]))
        ctx.add(ctx.sw('green'), (P, N, UV, I))
        ctx.add(ctx.sw('dark'), tube_path(pts(sheer, sx)[:-1], 0.07, seg=5))
    # deck
    dk_ = [[a, o + np.array([0, p[1] + 0.12, p[2]]), b] for a, b, p in zip(pts(sheer, 1), pts(sheer, -1), sheer)]
    P, N, UV, I = loft(dk_, closed=False)
    if N[:, 1].mean() < 0:
        I = I[:, ::-1]; N = -N
    ctx.add(ctx.sw('white'), (P, N, UV, I))
    # transom
    tr = [pts(sheer, 1)[0], pts(chine, 1)[0], pts(keel, 1)[0], pts(chine, -1)[0], pts(sheer, -1)[0]]
    P = np.array(tr); I = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]])
    if np.cross(P[I[0, 1]] - P[I[0, 0]], P[I[0, 2]] - P[I[0, 0]])[2] > 0:
        I = I[:, ::-1]
    ctx.add(ctx.sw('white'), (P, np.tile([0, 0, -1.0], (5, 1)), np.zeros((5, 2)), I))
    # aft cabin (long, low) and the wheelhouse with a raked windscreen
    yd = o[1] + 1.62
    ctx.add(ctx.sw('white'), rbox(2.3, 0.95, 4.8, r=0.35, seg=2, bevel=0.1, y0=0.0), xf=M(o + np.array([0, yd - o[1], -1.45])))
    for sx in (1, -1):
        p_a = o + np.array([sx * 1.16, yd - o[1] + 0.32, -3.6]); p_b = o + np.array([sx * 1.16, yd - o[1] + 0.32, 0.6])
        strip_b(ctx, p_a if sx > 0 else p_b, p_b if sx > 0 else p_a, 0.42, (sx, 0, 0))
    wh_b = [(-1.1, 0.75), (1.1, 0.75), (1.1, 2.75), (-1.1, 2.75)]
    wh_t = [(-1.0, 0.75), (1.0, 0.75), (1.0, 2.15), (-1.0, 2.15)]
    P, N, UV, I = prism(wh_b, yd, yd + 1.45, top=True, top_poly=wh_t)
    ctx.add(ctx.sw('white'), (P, N, UV, I), xf=M(o * np.array([1, 0, 1])))
    nrm = normalize(np.array([0, 0.6 / 1.45, 1.0]))
    q = _quad(o * np.array([1, 0, 1]) + np.array([0.95, yd + 0.75, 2.75 - 0.31]),
              o * np.array([1, 0, 1]) + np.array([-0.95, yd + 0.75, 2.75 - 0.31]),
              o * np.array([1, 0, 1]) + np.array([-0.9, yd + 1.25, 2.75 - 0.52]),
              o * np.array([1, 0, 1]) + np.array([0.9, yd + 1.25, 2.75 - 0.52]), uv=[(0, 1), (1, 1), (1, 0), (0, 0)], n=nrm)
    Pq, Nq, UVq, Iq = q
    Pq = Pq + nrm * 0.03
    if np.dot(np.cross(Pq[1] - Pq[0], Pq[2] - Pq[0]), nrm) < 0:
        Iq = Iq[:, ::-1]
    ctx.add(ctx.sw('glass'), (Pq, Nq, UVq, Iq))
    for sx in (1, -1):
        p_a = o * np.array([1, 0, 1]) + np.array([sx * 1.07, yd + 0.72, 0.95])
        p_b = o * np.array([1, 0, 1]) + np.array([sx * 1.07, yd + 0.72, 2.2])
        strip_b(ctx, p_a if sx > 0 else p_b, p_b if sx > 0 else p_a, 0.5, (sx, 0, 0))
    # mast, light and antenna on the wheelhouse roof
    mt = o * np.array([1, 0, 1]) + np.array([0, yd + 1.45, 1.4])
    ctx.add(ctx.sw('white'), tube_path([mt, mt + np.array([0, 1.1, -0.3])], 0.05, seg=4))
    ctx.add(ctx.sw('white'), box(0.5, 0.18, 0.3, center=(0, 0, 0)), xf=M(mt + np.array([0, 0.75, -0.2])))
    # foredeck rail
    fr = [p + np.array([0, 0.02, 0]) for p in pts(sheer, 1)[8:12]]
    fl = [p + np.array([0, 0.02, 0]) for p in pts(sheer, -1)[8:12]]
    return fr, fl


def strip_b(ctx, a, b, h, normal):
    n = normalize(np.asarray(normal, float))
    a = np.asarray(a, float) + n * 0.03; b = np.asarray(b, float) + n * 0.03
    up = np.array([0, h, 0.0])
    Ln = np.linalg.norm(b - a)
    P, N, UV, I = _quad(a, b, b + up, a + up, uv=[(0.0, 1.0), (Ln, 1.0), (Ln, 0.0), (0.0, 0.0)], n=n)
    if np.dot(np.cross(P[1] - P[0], P[2] - P[0]), n) < 0:
        I = I[:, ::-1]
    ctx.add(ctx.band('WINDOW', 3.0), (P, N, UV, I))


def slewing_davit(ctx, base, tip, side):
    """post from `base` (deck) to its top, arm reaching to `tip`, with a hydraulic ram and the fall."""
    base = np.asarray(base, float); tip = np.asarray(tip, float)
    top = np.array([base[0], tip[1] + 0.3, base[2]])
    ctx.add(ctx.paint, taper_beam(base, top, 0.55, 0.55, 0.4, 0.4, up=(1, 0, 0)))
    ctx.add(ctx.paint, cylinder(0.45, 0.25, seg=12), xf=M(base))
    ctx.add(ctx.paint, taper_beam(top + np.array([0, -0.05, 0]), tip, 0.38, 0.45, 0.24, 0.28, up=(0, 1, 0)))
    hydraulic_ram(ctx, base + (top - base) * 0.45, top + (tip - top) * 0.5 + np.array([0, -0.15, 0]), r=0.1)
    ctx.add(ctx.sw('dark'), tube_path([tip, tip + np.array([0, -1.5, 0])], 0.025, seg=4))
    ctx.add(ctx.sw('dark'), box(0.22, 0.3, 0.22, center=(0, 0, 0)), xf=M(tip + np.array([0, -1.65, 0])))


# =============================================================================================
# missile/boat crane on the crane house (S-300F reloads and boats)
# =============================================================================================
def crane(ctx, o, name='Crane', parent='Superstructure'):
    """o = base on the crane-house roof; the jib stows pointing forward (+z)."""
    b = ctx.b
    o = np.asarray(o, float)
    b.push(name, parent=parent, translation=tuple(o))
    pt = ctx.paint
    ctx.add(pt, cylinder(1.15, 0.55, seg=20), xf=M(o))
    ctx.add(ctx.sw('dark'), cylinder(1.22, 0.12, seg=20, y0=0.55), xf=M(o))
    ctx.add(pt, rbox(2.2, 1.55, 2.9, r=0.18, seg=2, bevel=0.1, y0=0.67), xf=M(o + np.array([0, 0, -0.35])))
    ctx.add(ctx.sw('glass'), box(0.03, 0.55, 1.1, center=(0, 0, 0)), xf=M(o + np.array([1.11, 1.55, 0.35])))
    ctx.add(ctx.sw('glass'), box(0.9, 0.55, 0.03, center=(0, 0, 0)), xf=M(o + np.array([0.55, 1.55, 1.11])))
    # tapered king post and luffing jib
    p0 = o + np.array([0, 2.2, -1.1]); p1 = o + np.array([0, 7.1, -1.4])
    ctx.add(pt, taper_beam(p0, p1, 0.8, 0.9, 0.6, 0.62, up=(0, 0, 1)))
    j0 = o + np.array([0, 7.0, -1.75]); j1 = o + np.array([0, 7.45, 5.1])
    ctx.add(pt, taper_beam(j0, j1, 0.62, 0.72, 0.34, 0.4, up=(0, 1, 0)))
    hydraulic_ram(ctx, o + np.array([0, 2.6, 0.6]), o + np.array([0, 6.85, 2.2]), r=0.16)
    # sheave, wire and hook block at the jib head
    ctx.add(ctx.sw('dark'), cylinder(0.32, 0.12, seg=12), xf=M(j1 + np.array([-0.06, 0, 0]), rot_z(math.pi / 2)))
    ctx.add(ctx.sw('dark'), tube_path([j1 + np.array([0, -0.2, 0]), j1 + np.array([0, -1.6, 0])], 0.03, seg=4))
    ctx.add(ctx.sw('dark'), rbox(0.36, 0.42, 0.24, r=0.08, seg=1, y0=0.0), xf=M(j1 + np.array([0, -2.0, 0])))
    hook = [j1 + np.array([0, -2.05 - 0.25 * math.sin(a), 0.2 * math.cos(a)]) for a in np.linspace(0.2, math.pi * 1.4, 7)]
    ctx.add(ctx.sw('dark'), tube_path(hook, 0.04, seg=4))
    b.pop()
