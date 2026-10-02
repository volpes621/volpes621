"""Detail pass, deck equipment: AK-130 splash shield, S-300F lid-drive units and loading gantry,
deck vents and ready-use lockers (shapes follow the Varyag / Slava close-up photos)."""
import math
import numpy as np

from meshkit import *
import atlas as st
from kit import M, P3
from detail import rbox, taper_beam
from antennas import pipe, _orient


def splash_shield(ctx, Bc, R, a0, a1, ybot, ytop, n=30, stiff_every=3):
    """smooth curved shield wall around a gun barbette (angles from dead ahead, + to port), with a
    rolled top edge and external stiffeners. ybot(B, x) follows the deck."""
    pts = []
    for k in range(n + 1):
        a = math.radians(a0 + (a1 - a0) * k / n)
        pts.append((Bc - R * math.cos(a), R * math.sin(a), a))
    P, N, UV, I = [], [], [], []
    acc = 0.0
    for k, (B, x, a) in enumerate(pts):
        if k > 0:
            acc += R * math.radians(abs(a1 - a0) / n)
        nrm = np.array([math.sin(a), 0.0, math.cos(a)])         # outward (world: +x port, +z forward)
        yb = ybot(B, x)
        for y in (yb, ytop):
            P.append(P3(B, x, y)); N.append(nrm); UV.append((acc, (ytop - y) / (ytop - yb)))
    for k in range(n):
        i = 2 * k
        I += [(i, i + 2, i + 3), (i, i + 3, i + 1)]
    # perforated plating (see-through), rolled top edge and a horizontal stiffening rib outside
    ctx.add(ctx.band('PERF', st.PERF_REPEAT_M), _orient(np.array(P), np.array(N), np.array(UV), np.array(I)))
    ctx.add(ctx.paint, pipe([P3(B, x, ytop) for (B, x, _) in pts], 0.06, seg=5))
    mid = []
    for (B, x, a) in pts:
        out = np.array([math.sin(a), 0.0, math.cos(a)])
        ym = 0.49 * (ytop + ybot(B, x))
        mid.append(P3(B, x, ym) + out * 0.03)
    ctx.add(ctx.paint, pipe(mid, 0.035, seg=3), occ=False)
    for k in range(stiff_every // 2, n + 1, stiff_every):
        B, x, a = pts[k]
        out = np.array([math.sin(a), 0.0, math.cos(a)])
        p = P3(B, x, 0.0) + out * 0.06
        y0, y1 = ybot(B, x), ytop - 0.08
        ctx.add(ctx.paint, taper_beam(np.array([p[0], y0, p[2]]) + out * 0.08, np.array([p[0], y1, p[2]]),
                                      0.08, 0.2, 0.08, 0.06, up=out))


def round_vent(ctx, pos, r=0.32, h=0.55):
    """mushroom deck ventilator: red-brown coaming under a grey domed cap."""
    o = np.asarray(pos, float)
    ctx.add(ctx.sw('deck_red'), cylinder(r * 0.72, h * 0.55, seg=8, caps=(False, False)), xf=M(o))
    ctx.add(ctx.paint, lathe([(r * 0.96, h * 0.5), (r, h * 0.66), (r * 0.7, h * 0.95), (0.0, h)], seg=8), xf=M(o))


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
