"""Weapons rebuilt from close-up photographs (Varyag 2011/2017, Moskva, Nastoychivyy, Bystryy,
Admiral Tributs): AK-130 (A-218) twin 130 mm mount, AK-630M CIWS, S-300F (B-204) launcher topside.

Turret-local frames: x = port, y = up from the mount's base plane, z = forward (bow when trained ahead).
"""
import math
import numpy as np
from PIL import ImageFont

from meshkit import *
from meshkit import _quad
import atlas as st
from kit import M, P3, orient_outward
from detail import rbox, taper_beam
from antennas import pipe, _orient


# =============================================================================================
# small helpers
# =============================================================================================
def _xf_pts(P, o, R):
    return (np.asarray(P, float) @ R.T) + o


def _poly_fan(pts3, normal):
    """triangle fan over a convex planar polygon (3D points), facing `normal`."""
    P = np.asarray(pts3, float)
    c = P.mean(axis=0)
    P = np.vstack([c, P])
    n = len(P) - 1
    I = np.array([[0, 1 + k, 1 + (k + 1) % n] for k in range(n)])
    N = np.tile(normalize(np.asarray(normal, float)), (len(P), 1))
    return _orient(P, N, np.zeros((len(P), 2)), I)


def _planar_uv(geo, axes=(2, 1), scale=1.0):
    P, N, UV, I = geo
    return P, N, P[:, list(axes)] * scale, I


def _loop(p, n, s, w=0.18, d=0.1, up=None):
    """U-shaped grab loop standing off a surface point p along normal n, spread along s."""
    p = np.asarray(p, float); n = normalize(np.asarray(n, float)); s = normalize(np.asarray(s, float))
    return [p - s * w / 2, p - s * w / 2 + n * d, p + s * w / 2 + n * d, p + s * w / 2]


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


# =============================================================================================
# AK-130 (A-218): two rounded side pods around a central gun slot
# =============================================================================================
A_YB = 0.25            # underside of the shell above the roller ring
A_ZT, A_YT, A_RF = 0.95, 1.7, 1.85   # front drum circle of the pods (around the trunnions)
A_YR = A_YT + A_RF     # roof height (3.55)
A_ZR0 = -0.45          # where the roof turns into the rear dome
A_ZREAR = -2.8         # rear-most point on the centreline
A_YRV = 0.95           # the rear dome becomes vertical below this height
A_HW = 2.35            # half width (flat side panels)
A_SLOT = 0.85          # half width of the gun slot
A_ZS = 0.0             # back wall of the slot
A_YS = 0.6             # slot sill
A_BX = 0.4             # barrel offset from the centreline


def _a_ytop(ax):
    return A_YR - 0.62 * max(0.0, (ax - 1.0) / (A_HW - 1.0)) ** 1.8


def _a_rf(ax):
    """front drum radius: the drum's outer ends are rounded back."""
    return A_RF - 0.32 * max(0.0, (ax - 1.65) / (A_HW - 1.65)) ** 2


def _a_zrear(ax):
    return A_ZR0 + (A_ZREAR - A_ZR0) * math.sqrt(max(0.0, 1.0 - (ax / (A_HW + 0.15)) ** 2))


def _a_dome_z(ax, y):
    """z of the rear dome surface at half-width ax and height y (y >= A_YRV)."""
    a, b = A_ZR0 - _a_zrear(ax), _a_ytop(ax) - A_YRV
    t = min(1.0, max(0.0, (y - A_YRV) / b))
    return A_ZR0 - a * math.sqrt(1.0 - t * t)


def _a_dome_y(ax, z):
    """roof / rear-dome height at half-width ax and position z."""
    yt = _a_ytop(ax)
    if z >= A_ZR0:
        return yt
    a, b = A_ZR0 - _a_zrear(ax), yt - A_YRV
    t = min(1.0, (A_ZR0 - z) / a)
    return A_YRV + b * math.sqrt(1.0 - t * t)


def _a_profile(ax, front=True, n_arc=10):
    """open side profile [(z, y)] from the front-bottom (or the slot sill) over the roof to the rear-bottom."""
    yt = _a_ytop(ax)
    zr = _a_zrear(ax)
    pts = []
    if front:
        rf = _a_rf(ax)
        th0 = math.asin((A_YB - A_YT) / rf)
        th1 = math.asin(min(1.0, (yt - A_YT) / rf))
        for k in range(n_arc + 1):
            th = th0 + (th1 - th0) * k / n_arc
            pts.append((A_ZT + rf * math.cos(th), A_YT + rf * math.sin(th)))
        zr0 = pts[-1][0]
    else:
        pts.append((A_ZS, yt))
        zr0 = A_ZS
    for t in (0.35, 0.7):
        pts.append((zr0 + (A_ZR0 - zr0) * t, yt))
    a, b = A_ZR0 - zr, yt - A_YRV
    for k in range(7):
        ph = math.radians(15 * k)
        pts.append((A_ZR0 - a * math.sin(ph), A_YRV + b * math.cos(ph)))
    pts.append((zr, A_YB))
    return pts


def _a_section(ax, x, front=True):
    return [np.array([x, y, z]) for (z, y) in _a_profile(ax, front)]


def _loft_x(sections, smooth=True):
    """quads between consecutive point lists (equal length)."""
    P = np.array([p for sec in sections for p in sec])
    m = len(sections[0])
    I = []
    for i in range(len(sections) - 1):
        for j in range(m - 1):
            a = i * m + j; b = a + m
            I += [(a, b, b + 1), (a, b + 1, a + 1)]
    I = np.array(I)
    N = compute_smooth_normals(P, I)
    UV = np.column_stack([P[:, 2] + P[:, 0] * 0.3, -P[:, 1]])
    return P, N, UV, I


def ak130_mount(ctx, Bc, ybase, stars=True):
    """AK-130 turret (node AK130_Turret) and its elevating guns (child node AK130_Guns)."""
    b = ctx.b
    pt, lt, dk = ctx.paint, ctx.sw('light'), ctx.sw('dark')
    o = P3(Bc, 0, ybase)
    b.push('AK130_Turret', parent='Hull', translation=tuple(o))

    def W(p):
        return o + np.asarray(p, float)

    # roller ring and the brackets under the shell
    ctx.add(dk, lathe([(2.45, 0.0), (2.45, 0.08), (2.3, 0.12), (2.3, A_YB + 0.02), (2.1, A_YB + 0.02)], seg=24), xf=M(o))
    for k in range(8):
        a = 2 * math.pi * (k + 0.5) / 8
        ctx.add(dk, box(0.12, A_YB, 0.3, center=(0, A_YB / 2, 0)), xf=M(W((2.34 * math.sin(a), 0, 2.34 * math.cos(a))), rot_y(a)))

    # ---- the two pods: lofted along x from the slot face to the flat side panel
    xs = [A_SLOT, 1.3, 1.75, 2.05, 2.25, A_HW]
    for s in (1, -1):
        secs = [[W(p) for p in _a_section(ax, s * ax)] for ax in xs]
        P, N, UV, I = _loft_x(secs)
        P, N, I = orient_outward(P, N, I, lambda p, s=s: W((s * 1.4, 1.6, -0.2)))
        ctx.add(pt, (P, N, UV, I))
        # flat side panel (outer) and the pod's inner face toward the slot
        prof = _a_profile(A_HW)
        ctx.add(pt, _planar_uv(_poly_fan([W((s * A_HW, y, z)) for (z, y) in prof], (s, 0, 0))))
        inner = [(A_ZS, A_YB)] + [(z, y) for (z, y) in _a_profile(A_SLOT) if z > A_ZS + 1e-6] + [(A_ZS, A_YR)]
        inner = [(z, y) for (z, y) in inner if not (y > A_YR - 1e-6 and z < A_ZS - 1e-6)]
        ctx.add(pt, _planar_uv(_poly_fan([W((s * A_SLOT, y, z)) for (z, y) in inner], (-s, 0, 0))))
    # ---- slot region: roof and rear dome across the slot, back wall, sill
    secs = [[W(p) for p in _a_section(ax, x, front=False)] for (ax, x) in ((A_SLOT, -A_SLOT), (0.0, 0.0), (A_SLOT, A_SLOT))]
    P, N, UV, I = _loft_x(secs)
    P, N, I = orient_outward(P, N, I, lambda p: W((0.0, 1.6, -0.6)))
    ctx.add(pt, (P, N, UV, I))
    ctx.add(ctx.sw('mid'), _quad(W((-A_SLOT, A_YS, A_ZS)), W((A_SLOT, A_YS, A_ZS)), W((A_SLOT, A_YR, A_ZS)),
                                  W((-A_SLOT, A_YR, A_ZS)), n=np.array([0, 0, 1.0])))
    zf = A_ZT + math.sqrt(A_RF ** 2 - (A_YT - A_YS) ** 2) - 0.05
    ctx.add(pt, box(2 * A_SLOT, A_YS - A_YB, zf - A_ZS, center=(0, (A_YS + A_YB) / 2, (zf + A_ZS) / 2)), xf=M(o))
    # rolled canvas cover across the top of the slot
    ctx.add(lt, cylinder(0.3, 2 * A_SLOT, seg=12), xf=M(W((-A_SLOT, 3.2, 1.6)), rot_z(-math.pi / 2)))
    # step and hand rail at the bottom of the slot
    ctx.add(pt, pipe([W((-0.6, A_YS, zf - 0.1)), W((-0.6, A_YS + 0.55, zf - 0.1)), W((-0.2, A_YS + 0.25, zf - 0.1)),
                      W((0.2, A_YS + 0.25, zf - 0.1)), W((0.6, A_YS + 0.55, zf - 0.1)), W((0.6, A_YS, zf - 0.1))], 0.03, seg=4),
            occ=False)

    # ---- canvas-cover loops along the inner edges of both pods
    for s in (1, -1):
        for k in range(10):
            th = math.radians(-40 + 12.5 * k)
            p = np.array([s * (A_SLOT + 0.1), A_YT + A_RF * math.sin(th), A_ZT + A_RF * math.cos(th)])
            n = np.array([0, math.sin(th), math.cos(th)])
            ctx.add(pt, pipe([W(q) for q in _loop(p, n, (s, 0, 0), w=0.16, d=0.09)], 0.022, seg=3), occ=False)
    # ---- side panels: big oval door, rungs, rear door, round hatch, warning notices
    for s in (1, -1):
        nx = np.array([s, 0, 0.0])
        for (name, z, y, w, h) in (('ak_door', 0.45, 2.2, 0.78, 1.2), ('ak_door', -0.55, 1.45, 0.55, 1.0),
                                   ('ak_hatch', 1.35, 1.0, 0.5, 0.5), ('ak_warn', 1.45, 1.65, 0.75, 0.38)):
            c = W((s * (A_HW + 0.012), y, z))
            u = np.array([0, 0, -s * w / 2]); v = np.array([0, h / 2, 0])
            ctx.add(ctx.rect(name), _quad(c - u - v, c + u - v, c + u + v, c - u + v, uv=[(0, 1), (1, 1), (1, 0), (0, 0)], n=nx))
        for y in (0.6, 0.95, 1.3):
            ctx.add(pt, pipe([W(q) for q in _loop((s * A_HW, y, 0.45), nx, (0, 0, 1), w=0.36, d=0.11)], 0.025, seg=3), occ=False)
        # grab rail along the rear dome
        rail = []
        for ax in (1.0, 1.5, 1.95, 2.25):
            zd = _a_dome_z(ax, 2.7)
            rail.append(W((s * (ax + 0.03), 2.7, zd - 0.1)))
        ctx.add(pt, pipe(rail, 0.025, seg=4), occ=False)
    # ---- roof: commander's sight cupola (starboard pod), sight box (port pod), lifting eyes
    ctx.add(pt, lathe([(0.55, 0.0), (0.55, 0.22), (0.46, 0.42), (0.26, 0.54), (0.0, 0.57)], seg=16), xf=M(W((-1.5, A_YR - 0.06, 0.15))))
    ctx.add(ctx.sw('glass'), box(0.36, 0.1, 0.04, center=(0, 0, 0)), xf=M(W((-1.5, A_YR + 0.32, 0.62))))
    ctx.add(pt, rbox(0.55, 0.42, 0.8, r=0.08, seg=1, bevel=0.05, y0=A_YR - 0.05), xf=M(W((1.45, 0, 0.55))))
    ctx.add(ctx.sw('glass'), box(0.36, 0.14, 0.03, center=(0, 0, 0)), xf=M(W((1.45, A_YR + 0.2, 0.96))))
    for (x, z) in ((1.0, -1.6), (-1.0, -1.6), (1.6, 1.4), (-1.6, 1.4)):
        y = _a_dome_y(abs(x), z) - 0.02
        ctx.add(pt, pipe([W((x, y, z - 0.1)), W((x, y + 0.12, z - 0.06)), W((x, y + 0.12, z + 0.06)), W((x, y, z + 0.1))],
                         0.025, seg=3), occ=False)
    # ---- rear: ammunition-handling hatch box with a rail
    zr = _a_zrear(0.0)
    ctx.add(pt, rbox(1.3, 0.75, 0.55, r=0.08, seg=1, bevel=0.04, y0=A_YB), xf=M(W((0, 0, zr - 0.12))))
    ctx.add(pt, pipe([W((-0.7, A_YB + 0.75, zr - 0.3)), W((-0.7, A_YB + 1.1, zr - 0.45)), W((0.7, A_YB + 1.1, zr - 0.45)),
                      W((0.7, A_YB + 0.75, zr - 0.3))], 0.025, seg=4), occ=False)

    # ---- guns (elevating mass), pivot on the trunnion axis
    tp = W((0, A_YT, A_ZT))
    b.push('AK130_Guns', parent='AK130_Turret', translation=tuple(tp))
    Rz = rot_x(math.pi / 2)                                  # lathe axis (+y) -> forward (+z)
    ctx.add(ctx.sw('mid'), rbox(0.42, 0.95, 1.7, r=0.08, seg=1, y0=-0.47), xf=M(tp + np.array([0, 0, 0.2])))
    prof = [(0.33, -0.4), (0.33, 2.4), (0.36, 2.43), (0.36, 2.56), (0.28, 2.6), (0.28, 3.4), (0.21, 3.65), (0.19, 5.0),
            (0.215, 5.02), (0.215, 5.15), (0.185, 5.17), (0.175, 7.45), (0.2, 7.48), (0.2, 7.75), (0.0, 7.75)]
    for s in (1, -1):
        bo = tp + np.array([s * A_BX, 0, 0])
        ctx.add(pt, lathe(prof, seg=12), xf=M(bo, Rz))
        if stars:
            ctx.add(ctx.rect('capstar'), _quad(bo + np.array([-0.19, -0.19, 7.76]), bo + np.array([0.19, -0.19, 7.76]),
                                               bo + np.array([0.19, 0.19, 7.76]), bo + np.array([-0.19, 0.19, 7.76]),
                                               uv=[(1, 1), (0, 1), (0, 0), (1, 0)], n=np.array([0, 0, 1.0])))
        # recuperator under the sleeve, cable conduit along the top, the hanging hose
        ctx.add(pt, cylinder(0.13, 2.4, seg=8), xf=M(bo + np.array([0, -0.42, -0.2]), Rz))
        ctx.add(pt, pipe([bo + np.array([0, 0.33, 2.62]), bo + np.array([0, 0.27, 3.6]), bo + np.array([0, 0.23, 7.3])], 0.035, seg=4),
                occ=False)
        for zc in (3.9, 5.6, 6.9):
            ctx.add(pt, box(0.1, 0.09, 0.1, center=(0, 0, 0)), xf=M(bo + np.array([0, 0.22 - 0.012 * (zc - 3.9), zc])))
        ctx.add(ctx.sw('black'), pipe([bo + np.array([s * 0.12, -0.15, 4.3]), bo + np.array([s * 0.2, -0.6, 3.7]),
                                       bo + np.array([s * 0.22, -0.78, 2.9]), bo + np.array([s * 0.2, -0.55, 2.2])], 0.045, seg=5),
                occ=False)
    b.pop()
    b.pop()


# =============================================================================================
# AK-630M: ribbed conical pedestal (fixed) + low domed turret with a twin-lobed hood (rotating)
# =============================================================================================
def ak630m(ctx, pos, facing=0.0, name='AK630', parent=None, base_h=0.6):
    b = ctx.b
    o = np.asarray(pos, float)
    R = rot_y(facing)
    pt = ctx.paint
    # fixed pedestal: frustum with flanges and tapered stiffener ribs
    ctx.add(pt, lathe([(1.07, 0.0), (1.07, 0.05), (0.99, 0.08), (0.87, base_h - 0.07), (0.93, base_h - 0.05),
                       (0.93, base_h)], seg=16), xf=M(o), node=parent)
    for k in range(10):
        a = 2 * math.pi * (k + 0.5) / 10
        d = np.array([math.sin(a), 0, math.cos(a)])
        p0 = o + d * 1.02 + np.array([0, 0.07, 0])
        p1 = o + d * 0.89 + np.array([0, base_h - 0.08, 0])
        ctx.add(pt, taper_beam(p0, p1, 0.04, 0.11, 0.04, 0.04, up=d), node=parent)
    b.push(name, parent=parent, translation=tuple(o))
    c = o + np.array([0, base_h, 0])
    ctx.add(pt, lathe([(1.0, 0.0), (1.0, 0.07), (0.97, 0.1), (0.97, 0.32), (0.93, 0.48), (0.8, 0.63), (0.5, 0.73),
                       (0.0, 0.76)], seg=16), xf=M(c, R))
    # hood: two long lobes with a groove between them, bridged over the round gun port
    ctx.add(pt, ellipsoid(0.46, 0.27, 0.72, seg=12, rings=5), xf=M(c + R @ np.array([0, 0.54, 0.26]), R))
    gy = 0.33
    Rz = R @ rot_x(math.pi / 2)                               # lathe axis (+y) -> along the gun
    ctx.add(ctx.sw('black'), cylinder(0.21, 0.1, seg=14, caps=(False, True)), xf=M(c + R @ np.array([0, gy, 0.88]), Rz))
    # gun: cooling jacket with a collar, steel muzzle clamp, six barrel ends around the hub
    ctx.add(ctx.sw('dark'), lathe([(0.13, 0.8), (0.13, 1.2), (0.148, 1.22), (0.148, 1.32), (0.13, 1.34), (0.13, 2.24),
                                   (0.0, 2.24)], seg=12), xf=M(c + R @ np.array([0, gy, 0.0]), Rz))
    ctx.add(ctx.sw('steel'), lathe([(0.152, 2.2), (0.152, 2.34), (0.125, 2.36), (0.0, 2.36)], seg=12),
            xf=M(c + R @ np.array([0, gy, 0.0]), Rz))
    for k in range(6):
        ang = 2 * math.pi * k / 6 + math.pi / 6
        off = np.array([0.088 * math.cos(ang), gy + 0.088 * math.sin(ang), 0.0])
        ctx.add(ctx.sw('steel'), cylinder(0.032, 0.12, seg=10, caps=(False, True)), xf=M(c + R @ (off + np.array([0, 0, 2.3])), Rz))
        ctx.add(ctx.sw('black'), disc(0.02, seg=8), xf=M(c + R @ (off + np.array([0, 0, 2.422])), Rz))
    ctx.add(ctx.sw('dark'), cylinder(0.034, 0.1, seg=8), xf=M(c + R @ np.array([0, gy, 2.3]), Rz))
    # grab rails along both sides with stand-offs
    for s in (1, -1):
        arc = [c + R @ np.array([s * 1.05 * math.sin(math.radians(a)), 0.42, 1.05 * math.cos(math.radians(a))])
               for a in np.linspace(35, 140, 5)]
        ctx.add(pt, pipe(arc, 0.022, seg=3), occ=False)
        for a in (50, 125):
            p = c + R @ np.array([s * 0.98 * math.sin(math.radians(a)), 0.42, 0.98 * math.cos(math.radians(a))])
            q = c + R @ np.array([s * 1.06 * math.sin(math.radians(a)), 0.42, 1.06 * math.cos(math.radians(a))])
            ctx.add(pt, pipe([p, q], 0.025, seg=3), occ=False)
    b.pop()


# =============================================================================================
# S-300F (B-204 revolver launchers): hatch plate, lid-drive unit with loading cover, gantry
# =============================================================================================
S3_HATCH = 1.9         # revolving hatch disc radius
S3_RING = 1.30         # ring of the eight cell lids (one per missile in the B-204 drum)
S3_LID = 0.47          # cell lid radius (lids do not touch)
S3_COVER_ANG = 45.0    # the grey loading cover sits on the aft-port lid (degrees from aft toward port)
S3_TRE_R = 0.385       # each lid carries three small rings: ring radius / lid radius ...
S3_TRE_D = 0.475       # ... ring centre offset / lid radius (rings 2.5 cm apart, never overlapping)


def s3_lid(k):
    """(dB, dx) offset of cell lid k from the hatch centre; lid 0 is under the grey loading cover."""
    a = math.radians(S3_COVER_ANG + 45.0 * k)
    return S3_RING * math.cos(a), S3_RING * math.sin(a)


def s3_trefoil(k):
    """(dB, dx) unit directions of the three small rings in cell lid k. They sit in a trefoil with one ring
    pointing away from the hatch centre on even lids and toward it on odd lids (museum-model close-ups);
    the loading cover repeats the pattern of lid 0 under it."""
    a = math.radians(S3_COVER_ANG + 45.0 * k + 180.0 * (k % 2))
    return [(math.cos(a + 2 * math.pi * j / 3), math.sin(a + 2 * math.pi * j / 3)) for j in range(3)]


def s300_hatch(ctx, Bc, cx, ytop):
    """revolving hatch plate, slightly proud of the launcher platform, with a raised rim."""
    o = P3(Bc, cx, ytop)
    ctx.add(ctx.rect('vls_top'), disc(S3_HATCH, seg=32, uv_rect=True), xf=M(o + np.array([0, 0.05, 0])))
    ctx.add(ctx.sw('deck_red'), lathe([(2.0, 0.0), (2.0, 0.06), (1.9, 0.07)], seg=32), xf=M(o))


def s300_drive(ctx, Bc, cx, ytop):
    """lid-drive unit over the centre of the eight-cell drum (same orientation on every launcher, per the
    1984 overhead photo): base frame with rails, two flanged boxes with a bolted seam, upper and side
    boxes, lifting eyes; the grey loading cover lies on cell lid 0 (aft-port), carried by L-shaped arms."""
    pt = ctx.paint
    y0 = ytop + 0.03
    bx, bxx = Bc - 0.45, cx - 0.3                            # box cluster over the drum centre
    ctx.add(pt, rbox(1.7, 0.1, 1.8, r=0.12, seg=1), xf=M(P3(bx, bxx, y0)))
    for dx in (-0.65, 0.65):
        ctx.add(pt, box(0.1, 0.08, 1.8, center=(0, 0, 0)), xf=M(P3(bx, bxx + dx, y0 + 0.14)))
    yb = y0 + 0.1
    for (dx, h, d, dz) in ((0.47, 1.05, 1.4, 0.0), (-0.46, 0.95, 1.3, -0.05)):
        p = P3(bx + dz, bxx + dx, 0)
        ctx.add(pt, rbox(0.88, h, d, r=0.05, seg=1, bevel=0.05, y0=yb), xf=M(p))
        ctx.add(pt, rbox(0.96, 0.06, d + 0.08, r=0.05, seg=1, y0=yb + h - 0.03), xf=M(p))      # flanged lid
        e = P3(bx + dz, bxx + dx, yb + h + 0.03)                                                # lifting eye
        ctx.add(pt, pipe([e + np.array([-0.09, 0, 0]), e + np.array([-0.05, 0.12, 0]), e + np.array([0.05, 0.12, 0]),
                          e + np.array([0.09, 0, 0])], 0.02, seg=3), occ=False)
    ctx.add(pt, box(0.07, 1.02, 1.45, center=(0, 0, 0)), xf=M(P3(bx, bxx + 0.005, yb + 0.55)))       # bolted seam
    ctx.add(pt, rbox(1.15, 0.45, 0.7, r=0.05, seg=1, bevel=0.04, y0=yb + 1.02), xf=M(P3(bx - 0.35, bxx + 0.05, 0)))
    ctx.add(pt, rbox(1.23, 0.06, 0.78, r=0.05, seg=1, y0=yb + 1.44), xf=M(P3(bx - 0.35, bxx + 0.05, 0)))
    ctx.add(pt, rbox(0.34, 0.8, 1.0, r=0.05, seg=1, bevel=0.03, y0=yb), xf=M(P3(bx - 0.05, bxx - 1.08, 0)))
    # grey loading cover on cell lid 0, with two pins
    dB, dX = s3_lid(0)
    cB, cX = Bc + dB, cx + dX
    ctx.add(pt, lathe([(0.6, 0.0), (0.6, 0.1), (0.55, 0.15), (0.0, 0.15)], seg=24), xf=M(P3(cB, cX, ytop)))
    cc = P3(cB, cX, ytop + 0.152)
    ctx.add(ctx.rect('vls_lid'), _quad(cc + np.array([-0.55, 0, 0.55]), cc + np.array([0.55, 0, 0.55]),
                                       cc + np.array([0.55, 0, -0.55]), cc + np.array([-0.55, 0, -0.55]),
                                       uv=[(0, 1), (1, 1), (1, 0), (0, 0)], n=np.array([0, 1.0, 0])))
    psi = math.atan2(bxx - cX, bx - cB)                      # direction cover -> drive in (B, x)
    for off in (-1.0, 1.0):
        pp = P3(cB + 0.47 * math.cos(psi + off * 1.9), cX + 0.47 * math.sin(psi + off * 1.9), ytop + 0.15)
        ctx.add(pt, lathe([(0.075, 0.0), (0.075, 0.17), (0.055, 0.2), (0.0, 0.21)], seg=8), xf=M(pp))
    # L-shaped arms: a post on the frame (kept inside its outline) and a raking arm down to a pivot on the
    # cover rim
    for off in (-1.0, 1.0):
        ang = psi + off * math.radians(62)
        fB, fX = cB + 0.6 * math.cos(ang), cX + 0.6 * math.sin(ang)
        aB = min(fB + 0.55 * math.cos(psi), bx + 0.78)
        aX = min(fX + 0.55 * math.sin(psi), bxx + 0.74)
        f = P3(fB, fX, ytop + 0.18)
        a = P3(aB, aX, y0 + 0.1)
        e = P3(aB, aX, y0 + 0.5)
        ctx.add(pt, taper_beam(a, e, 0.14, 0.18, 0.12, 0.16))
        ctx.add(pt, taper_beam(e, f, 0.12, 0.16, 0.1, 0.12))
        ctx.add(pt, cylinder(0.09, 0.18, seg=8), xf=M(f + np.array([0, -0.07, 0])))
    # junction box at the hatch rim, between two cell lids
    ja = math.radians(S3_COVER_ANG - 112.5)
    ctx.add(ctx.sw('red'), rbox(0.3, 0.42, 0.3, r=0.04, seg=1, bevel=0.03, y0=ytop),
            xf=M(P3(Bc + 1.72 * math.cos(ja), cx + 1.72 * math.sin(ja), 0)))


def s300_gantry(ctx, B0, B1, yd, yb, legs):
    """loading gantry on the centreline: box girder on perforated plate legs, a vertical leg forward and a
    raking knee aft on red jack posts, guide rod and hand wheels."""
    lt = ctx.paint
    Bm = (B0 + B1) / 2
    ctx.add(lt, rbox(0.95, 0.8, B1 - B0, r=0.08, seg=1, bevel=0.04, y0=yb - 0.4), xf=M(P3(Bm, 0, 0)))
    for sx in (1, -1):                                       # access panels on the girder sides
        for Bp in (B0 + 2.2, Bm + 1.5):
            ctx.add(lt, box(0.04, 0.5, 1.1, center=(0, 0, 0)), xf=M(P3(Bp, sx * 0.49, yb)))
    for Bl in legs:
        for sx in (1, -1):
            x = sx * 0.24
            h = yb - 0.4 - yd
            q = [P3(Bl - 0.8, x, yd), P3(Bl + 0.8, x, yd), P3(Bl + 0.22, x, yd + h), P3(Bl - 0.22, x, yd + h)]
            ctx.add(ctx.rect('perf'), _orient(*_quad(q[0], q[1], q[2], q[3], uv=[(0, 1), (1, 1), (1, 0), (0, 0)],
                                                     n=np.array([sx, 0, 0.0]))))
            for (p, r_) in ((q[0], q[3]), (q[1], q[2])):
                ctx.add(lt, pipe([p, r_], 0.04, seg=4), occ=False)
        ctx.add(lt, rbox(0.75, 0.08, 1.8, r=0.06, seg=1, y0=yd), xf=M(P3(Bl, 0, 0)))
    # forward vertical leg and aft raking knee, each on a red jack post
    for (Bt, Bf_, rake) in ((B0 + 0.35, B0 + 0.35, 0.0), (B1 - 0.35, B1 + 0.6, 1.0)):
        for sx in (1, -1):
            top = P3(Bt, sx * 0.62, yb - 0.3)
            foot = P3(Bf_, sx * 0.62, yd + 0.85)
            ctx.add(lt, taper_beam(top, foot, 0.22, 0.3, 0.2, 0.24))
            ctx.add(ctx.sw('red'), cylinder(0.1, 0.85, seg=8, y0=yd), xf=M(P3(Bf_, sx * 0.62, 0)))
            ctx.add(ctx.sw('red'), cylinder(0.16, 0.06, seg=8, y0=yd), xf=M(P3(Bf_, sx * 0.62, 0)))
        ctx.add(lt, rbox(1.5, 0.35, 0.4, r=0.05, seg=1, y0=yb - 0.5), xf=M(P3(Bt, 0, 0)))
    ctx.add(lt, rbox(1.05, 0.95, 1.2, r=0.08, seg=1, bevel=0.05, y0=yb - 1.3), xf=M(P3(B0 + 1.1, 0, 0)))
    ctx.add(lt, pipe([P3(B0 + 0.4, 0.62, yd + 1.25), P3(B1 + 0.5, 0.62, yd + 1.25)], 0.045, seg=5), occ=False)
    for (Bw, sx) in ((B0 + 0.35, 1), (B1 - 0.35, -1)):        # hand wheels
        ctx.add(ctx.sw('dark'), cylinder(0.2, 0.04, seg=12, caps=(True, True)),
                xf=M(P3(Bw, sx * 0.55, yb - 0.1), rot_z(-sx * math.pi / 2)))


# =============================================================================================
# RBU-6000 "Smerch-2" (12 barrels in a horseshoe) and PK-2 twin decoy launcher
# =============================================================================================
def rbu6000m(ctx, pos, facing=0.0, name='RBU6000', parent=None, el_deg=25.0):
    """rotating base and column, yoke arms to the trunnions, twelve barrels on a horseshoe (open at the
    bottom) with dark breech caps and muzzle rings, front and rear clamp frames, centre tube."""
    b = ctx.b
    o = np.asarray(pos, float)
    b.push(name, parent=parent, translation=tuple(o))
    R = rot_y(facing)
    pt, dk = ctx.paint, ctx.sw('dark')
    ctx.add(pt, lathe([(0.85, 0.0), (0.85, 0.12), (0.72, 0.18), (0.72, 0.32), (0.6, 0.36), (0.0, 0.36)], seg=12), xf=M(o, R))
    ctx.add(pt, rbox(0.6, 0.7, 0.7, r=0.08, seg=1, bevel=0.04, y0=0.36), xf=M(o + R @ np.array([0, 0, -0.15]), R))
    T = o + R @ np.array([0, 1.3, -0.1])                     # trunnion point on the cluster axis
    E = R @ rot_x(-math.radians(el_deg))                     # cluster frame, local +z along the barrels
    for s in (1, -1):
        ctx.add(pt, taper_beam(o + R @ np.array([s * 0.3, 0.95, -0.15]), T + R @ np.array([s * 0.74, 0, 0]),
                               0.12, 0.3, 0.1, 0.22))
        ctx.add(pt, cylinder(0.12, 0.12, seg=8), xf=M(T + R @ np.array([s * 0.68, 0, 0]), R @ rot_z(-s * math.pi / 2)))
    Rz = E @ rot_x(math.pi / 2)
    for k in range(12):
        a = math.radians(-150 + 300 * k / 11)
        p0 = T + E @ np.array([0.56 * math.sin(a), 0.56 * math.cos(a), -0.75])
        d = E @ np.array([0, 0, 1.0])
        ctx.add(dk, cylinder(0.112, 0.12, seg=8, caps=(True, False)), xf=M(p0, Rz))
        ctx.add(pt, cylinder(0.1, 1.36, seg=8, caps=(False, False)), xf=M(p0 + d * 0.12, Rz))
        ctx.add(pt, cylinder(0.118, 0.11, seg=8, caps=(False, False)), xf=M(p0 + d * 1.47, Rz))
        ctx.add(ctx.sw('black'), disc(0.098, seg=8), xf=M(p0 + d * 1.5, Rz))
    for zf in (-0.6, 0.4):
        arc = [T + E @ np.array([0.56 * math.sin(math.radians(a)), 0.56 * math.cos(math.radians(a)), zf])
               for a in np.linspace(-150, 150, 12)]
        ctx.add(pt, pipe(arc, 0.05, seg=3))
    ctx.add(pt, cylinder(0.16, 1.2, seg=10), xf=M(T + E @ np.array([0, 0, -0.65]), Rz))
    b.pop()


def pk2(ctx, pos, facing=0.0, node=None):
    """PK-2 (ZIF-121) twin 140 mm decoy launcher: pedestal, yoke, two tubes at 45 deg with breech blocks."""
    o = np.asarray(pos, float)
    R = rot_y(facing)
    pt = ctx.paint
    ctx.add(pt, cylinder(0.3, 0.55, seg=10), xf=M(o), node=node)
    ctx.add(pt, cylinder(0.42, 0.08, seg=10), xf=M(o), node=node)
    ctx.add(pt, box(0.82, 0.22, 0.36, center=(0, 0.66, 0)), xf=M(o, R), node=node)
    E = R @ rot_x(-math.radians(45))
    Rz = E @ rot_x(math.pi / 2)
    for s in (1, -1):
        p0 = o + R @ np.array([s * 0.22, 0.8, 0]) + E @ np.array([0, 0, -0.55])
        d = E @ np.array([0, 0, 1.0])
        ctx.add(ctx.sw('mid'), cylinder(0.09, 1.35, seg=10, caps=(False, False)), xf=M(p0, Rz), node=node)
        ctx.add(ctx.sw('black'), disc(0.085, seg=10), xf=M(p0 + d * 1.33, Rz), node=node)
        ctx.add(pt, box(0.22, 0.22, 0.3, center=(0, 0, 0)), xf=M(p0 + d * 0.1, E), node=node)


# =============================================================================================
# decal painters
# =============================================================================================
def paint_ak_door(L, rect):
    """oval turret door: raised rim, hinge strap, two dog handles, red marker."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    PAL = st.PAL
    L.rect(x0, y0, x1, y1, col=PAL['super'], alpha=0.0)

    def body(d, s):
        d.rounded_rectangle([3 * s, 3 * s, (w - 3) * s, (h - 3) * s], radius=int(w * 0.45 * s), fill=255)

    def rim(d, s):
        d.rounded_rectangle([3 * s, 3 * s, (w - 3) * s, (h - 3) * s], radius=int(w * 0.45 * s), outline=255, width=4 * s)
        d.line([(w * 0.2 * s, h * 0.18 * s), (w * 0.2 * s, h * 0.82 * s)], fill=255, width=4 * s)
        for yy in (0.32, 0.68):
            d.rectangle([w * 0.68 * s, h * yy * s, w * 0.82 * s, (h * yy + 5) * s], fill=255)
    L.mask_apply(L.draw_mask(w, h, body), col=PAL['super'] * 0.98, alpha=1.0, x0=x0, y0=y0)
    L.mask_apply(L.draw_mask(w, h, rim), col=PAL['super'] * 0.62, add_height=0.9, x0=x0, y0=y0)
    L.rect(x0 + w * 0.44, y0 + h * 0.2, x0 + w * 0.58, y0 + h * 0.26, col=PAL['flagred'])
    a = L.alpha[y0:y1, x0:x1]
    L.alpha[y0:y1, x0:x1] = np.where(a > 0.5, 1.0, 0.0)


def paint_ak_hatch(L, rect):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    PAL = st.PAL
    L.rect(x0, y0, x1, y1, col=PAL['super'], alpha=0.0)

    def body(d, s):
        d.ellipse([3 * s, 3 * s, (w - 3) * s, (h - 3) * s], fill=255)

    def rim(d, s):
        d.ellipse([3 * s, 3 * s, (w - 3) * s, (h - 3) * s], outline=255, width=4 * s)
        d.line([(w * 0.3 * s, h * 0.5 * s), (w * 0.7 * s, h * 0.5 * s)], fill=255, width=4 * s)
    L.mask_apply(L.draw_mask(w, h, body), col=PAL['super'] * 0.97, alpha=1.0, x0=x0, y0=y0)
    L.mask_apply(L.draw_mask(w, h, rim), col=PAL['super'] * 0.62, add_height=0.9, x0=x0, y0=y0)
    a = L.alpha[y0:y1, x0:x1]
    L.alpha[y0:y1, x0:x1] = np.where(a > 0.5, 1.0, 0.0)


def paint_ak_warn(L, rect):
    """'ПРИ ВРАЩЕНИИ НЕ ХОДИТЬ / ОПАСНО' stencil (blue and red, as on the mounts)."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    PAL = st.PAL
    L.rect(x0, y0, x1, y1, col=PAL['super'], alpha=0.0)
    f1 = ImageFont.truetype(st.FONT_SANS, 13 * 4)
    f2 = ImageFont.truetype(st.FONT_SANS, 20 * 4)

    def line(text, font, y):
        def fn(d, s):
            tw = d.textlength(text, font=font)
            d.text(((w * s - tw) / 2, y * s), text, fill=255, font=font)
        return fn
    blue = np.array([0.13, 0.33, 0.72])
    L.mask_apply(L.draw_mask(w, h, line('ПРИ ВРАЩЕНИИ', f1, 4)), col=blue, alpha=1.0, x0=x0, y0=y0)
    L.mask_apply(L.draw_mask(w, h, line('НЕ ХОДИТЬ', f1, 21)), col=blue, alpha=1.0, x0=x0, y0=y0)
    L.mask_apply(L.draw_mask(w, h, line('ОПАСНО', f2, 38)), col=PAL['flagred'], alpha=1.0, x0=x0, y0=y0)
    a = L.alpha[y0:y1, x0:x1]
    L.alpha[y0:y1, x0:x1] = np.where(a > 0.45, 1.0, 0.0)


def _circ(d, s, cx, cy, r, wd=None):
    """filled circle (wd None) or ring of width wd px, in decal pixels, on a draw_mask canvas of scale s."""
    bb = [(cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s]
    if wd is None:
        d.ellipse(bb, fill=255)
    else:
        d.ellipse(bb, outline=255, width=max(1, int(round(wd * s))))


def paint_vls_lid(L, rect):
    """grey loading cover (top face, v = forward): raised rim and the same trefoil of three small rings as
    the cell lid under it (rings close but not touching), centre boss and rim bolts."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    PAL = st.PAL
    base = PAL['super'] * 1.05
    L.rect(x0, y0, x1, y1, col=base, alpha=0.0)
    cx, cy, R = w / 2, h / 2, (w - 4) / 2
    tre = [(cx + ux * 0.44 * R, cy - ub * 0.44 * R) for (ub, ux) in s3_trefoil(0)]   # px = port, py = forward
    bolts = []
    for (ub, ux) in s3_trefoil(0):                           # bolt pairs in the gaps between the rings
        for da in (-0.32, 0.32):
            a = math.atan2(-ux, -ub) + da
            bolts.append((cx + 0.85 * R * math.sin(a), cy - 0.85 * R * math.cos(a)))

    def disc_(d, s):
        _circ(d, s, cx, cy, R)

    def lines(d, s):
        _circ(d, s, cx, cy, R * 0.93, wd=3)
        for (px, py) in tre:
            _circ(d, s, px, py, 0.36 * R, wd=2.2)
        _circ(d, s, cx, cy, 0.05 * R)

    def studs(d, s):
        for (px, py) in bolts:
            _circ(d, s, px, py, 0.035 * R)
    L.mask_apply(L.draw_mask(w, h, disc_), col=base, alpha=1.0, x0=x0, y0=y0)
    L.mask_apply(L.draw_mask(w, h, lines), col=base * 0.7, add_height=0.8, x0=x0, y0=y0)
    L.mask_apply(L.draw_mask(w, h, studs), col=base * 0.8, add_height=0.8, x0=x0, y0=y0)
    a = L.alpha[y0:y1, x0:x1]
    L.alpha[y0:y1, x0:x1] = np.where(a > 0.5, 1.0, 0.0)


def paint_perf(L, rect):
    """lightening holes in a gantry leg plate (alpha cut-outs) with a flanged border."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    PAL = st.PAL
    L.rect(x0, y0, x1, y1, col=PAL['super'] * 1.05, alpha=1.0)

    def holes(d, s):
        for k in range(6):
            cy = h * (0.16 + 0.135 * k)
            r = w * (0.07 + 0.012 * k)
            d.ellipse([(w / 2 - r) * s, (cy - r) * s, (w / 2 + r) * s, (cy + r) * s], fill=255)
    m = L.draw_mask(w, h, holes)
    L.mask_apply(m, alpha=0.0, x0=x0, y0=y0)
    a = L.alpha[y0:y1, x0:x1]
    L.alpha[y0:y1, x0:x1] = np.where(a > 0.5, 1.0, 0.0)


def paint_vls_top(L, rect):
    """B-204 revolving hatch (red-brown, u = port, v = aft): stepped edge and the ring of eight separate
    cell lids. Each lid is a raised plate with a stepped rim carrying three small rings in a trefoil (close
    but never overlapping), a boss where the rings meet and lugs on the rim between them. Lid 0 lies under
    the grey loading cover."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    PAL = st.PAL
    red = PAL['deck_red']
    L.rect(x0, y0, x1, y1, col=red, alpha=1.0, rough=0.75, metal=0.03)
    k_px = w / (2 * S3_HATCH)                                 # pixels per metre
    R = S3_LID * k_px

    def at(k, f=0.0, dirn=(0.0, 0.0)):
        """pixel position of lid k's centre, moved f lid radii along the (dB, dx) direction dirn."""
        dB, dx = s3_lid(k)
        return (w / 2 + (dx + dirn[1] * f * S3_LID) * k_px, h / 2 + (dB + dirn[0] * f * S3_LID) * k_px)

    def plates(d, s):
        for k in range(8):
            _circ(d, s, *at(k), R)

    def rims(d, s):
        for rr, wd in ((S3_HATCH - 0.03, 3.0), (S3_HATCH - 0.08, 2.0)):
            _circ(d, s, w / 2, h / 2, rr * k_px, wd=wd)
        for k in range(8):
            _circ(d, s, *at(k), R, wd=3.0)
            _circ(d, s, *at(k), R * 0.92, wd=1.6)

    def rings(d, s):
        for k in range(8):
            for u in s3_trefoil(k):
                _circ(d, s, *at(k, S3_TRE_D, u), S3_TRE_R * R, wd=2.6)

    def studs(d, s):
        for k in range(8):
            _circ(d, s, *at(k), 0.05 * R)                      # boss where the three rings meet
            for (ub, ux) in s3_trefoil(k):                     # rim lugs in the gaps between the rings
                _circ(d, s, *at(k, 0.9, (-ub, -ux)), 0.045 * R)
    L.mask_apply(L.draw_mask(w, h, plates), col=red * 1.07, add_height=0.5, x0=x0, y0=y0)
    L.mask_apply(L.draw_mask(w, h, rims), col=red * 0.68, add_height=1.0, x0=x0, y0=y0)
    L.mask_apply(L.draw_mask(w, h, rings), col=red * 0.74, add_height=0.9, x0=x0, y0=y0)
    L.mask_apply(L.draw_mask(w, h, studs), col=red * 0.8, add_height=1.0, x0=x0, y0=y0)


def register(m):
    m.alloc('ak_door', 48, 96, paint_ak_door)
    m.alloc('ak_hatch', 48, 48, paint_ak_hatch)
    m.alloc('ak_warn', 128, 64, paint_ak_warn)
    m.alloc('vls_lid', 128, 128, paint_vls_lid)
    m.alloc('perf', 64, 128, paint_perf)
