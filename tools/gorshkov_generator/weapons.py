"""Project 22350 weapons: A-192M 130 mm gun, Redut and UKSK 3S14 vertical launchers, Palash CIWS,
KT-216 / KT-308 decoy launchers and the 14.5 mm MTPU pedestal mounts.
Shapes follow the 2018 Russian MoD and 2023 Mehr News photographs of Admiral Gorshkov."""
import math
import numpy as np

from meshkit import *
from meshkit import _quad
import atlas as st
from kit import M, P3
from parts import rbox, pipe, disc, axis_frame

PAL = st.PAL


def _faces(c, uvs, polys, node=None):
    """closed (convex) polyhedron from its faces; every face is turned to point away from the centroid."""
    P, N, UV, I = flat_poly_faces([[np.asarray(p, float) for p in f] for f in polys])
    cen = np.mean([np.mean(f, axis=0) for f in polys], axis=0)
    fn = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    fc = P[I].mean(axis=1)
    bad = np.sum(fn * (fc - cen), axis=1) < 0
    I[bad] = I[bad][:, ::-1]
    N = np.zeros_like(P)
    for t in I:
        N[t] = normalize(np.cross(P[t[1]] - P[t[0]], P[t[2]] - P[t[0]]))
    c.add(uvs, (P, N, UV, I), node=node)


def loft_solid(c, uvs, rings, node=None):
    """closed solid through a list of same-size 3D rings (end caps as fans); faces point outward."""
    polys = [list(rings[0])[::-1], list(rings[-1])]
    for a, b in zip(rings[:-1], rings[1:]):
        n = len(a)
        for i in range(n):
            j = (i + 1) % n
            polys.append([a[i], a[j], b[j], b[i]])
    _faces(c, uvs, polys, node=node)


# =============================================================================================
# A-192M "Armat" 130 mm gun
# =============================================================================================
# turret outline (plan, local: x port, z forward) and profile (from the drawing and photos):
# chamfered nose, sides leaning in, flat top, vertical back; base ring under it.
GUN_L = (-2.35, 2.05)      # turret rear / front z (local, about the training axis)
GUN_HW = 1.85              # half-width at the base of the housing
GUN_H0, GUN_H1 = 0.6, 3.25    # housing bottom / top above the deck
GUN_AXIS_Y = 2.2           # trunnion height above the deck
GUN_BARREL = 6.9           # barrel length ahead of the mantlet


def a192m(c, Bc, ydeck, train=0.0, node='A192M'):
    o = P3(Bc, 0.0, ydeck)
    R = rot_y(train)

    def W(p):
        return o + R @ np.asarray(p, float)
    # deck shield: low cylindrical wall round the mount, open over the aft quarter
    rr, hh = 2.95, 0.78
    n = 28
    a0, a1 = math.radians(-58), math.radians(238)          # opening astern (angle from +x toward +z)
    pts = [(rr * math.cos(a0 + (a1 - a0) * k / n), rr * math.sin(a0 + (a1 - a0) * k / n)) for k in range(n + 1)]
    for t, s in ((0.0, 1), (0.08, -1)):
        r_ = rr - t
        P, Nn = [], []
        for (px, pz) in pts:
            d = np.array([px, 0, pz]) / rr
            for yy in (0.0, hh):
                P.append(o + d * r_ + np.array([0, yy, 0]))
                Nn.append(d * s)
        P = np.array(P); Nn = np.array(Nn)
        I = []
        for k in range(n):
            a = 2 * k
            I += [(a, a + 2, a + 3), (a, a + 3, a + 1)]
        I = np.array(I)
        fn = np.cross(P[I[0, 1]] - P[I[0, 0]], P[I[0, 2]] - P[I[0, 0]])
        if np.dot(fn, Nn[I[0, 0]]) < 0:
            I = I[:, ::-1]
        c.add(c.paint, (P, Nn, np.stack([np.arange(len(P)) * 0.2, P[:, 1]], axis=1), I))
    rim = [o + np.array([px, hh, pz]) for (px, pz) in pts]
    c.add(c.paint, pipe(rim, 0.06, seg=4))
    # rotating base and the turret housing (node pivot on the training axis)
    c.b.node(node, parent='Weapons', translation=o)
    c.add(c.sw('mid'), cylinder(1.75, GUN_H0, seg=24, caps=(False, True)), xf=M(o))
    zr, zf = GUN_L
    hw0, hw1 = GUN_HW, GUN_HW - 0.42
    nose = 0.95                                            # nose chamfer (plan)
    y0, y1 = GUN_H0, GUN_H1

    def ring(y, hw, zf_, zr_, ch):
        return [W((hw, y, zr_ + 0.25)), W((hw - 0.25, y, zr_)), W((-hw + 0.25, y, zr_)), W((-hw, y, zr_ + 0.25)),
                W((-hw, y, zf_ - ch)), W((-hw + ch * 0.75, y, zf_)), W((hw - ch * 0.75, y, zf_)), W((hw, y, zf_ - ch))]
    lo = ring(y0, hw0, zf, zr, nose)
    mid = ring(y0 + 1.6, hw0 - 0.18, zf - 0.35, zr, nose)
    top = ring(y1, hw1, zf - 1.55, zr + 0.1, 0.55)
    loft_solid(c, c.paint, [lo, mid, top])
    # mantlet (embrasure) and the barrel with its thicker breech end and a thin muzzle collar
    elev = math.radians(3.0)
    Rb = R @ rot_x(-elev)
    pz = zf - 0.55
    piv = W((0.0, GUN_AXIS_Y, pz))
    dirv = R @ np.array([0.0, math.sin(elev), math.cos(elev)])
    # sighting port and access hatch on the roof, ladder rungs at the back
    c.add(c.sw('dark'), box(0.5, 0.06, 0.7, center=(0, 0, 0)), xf=M(W((0.75, y1 + 0.02, zr + 1.0)), R))
    c.add(c.sw('glass'), box(0.42, 0.22, 0.05, center=(0, 0, 0)), xf=M(W((-1.05, y1 - 0.25, zf - 1.45)), R @ rot_x(-0.5)))
    for k in range(4):
        c.add(c.sw('dark'), box(0.45, 0.04, 0.05, center=(0, 0, 0)), xf=M(W((0.8, y0 + 0.35 + k * 0.4, zr - 0.04)), R))
    # elevating mass: mantlet and barrel (child node, pivot on the trunnions)
    c.b.node(node + '_Gun', parent=node, translation=piv)
    c.add(c.paint, rbox(0.95, 0.85, 0.9, r=0.18, seg=2, y0=-0.42), xf=M(piv, Rb @ rot_x(math.pi / 2)))
    prof = [(0.0, 0.0), (0.27, 0.0), (0.27, 0.6), (0.2, 0.9), (0.17, 2.2), (0.135, 4.4), (0.12, GUN_BARREL - 0.35),
            (0.15, GUN_BARREL - 0.3), (0.15, GUN_BARREL - 0.05), (0.11, GUN_BARREL), (0.0, GUN_BARREL)]
    c.add(c.paint, lathe(prof, seg=12), xf=M(piv + dirv * 0.4, axis_frame(dirv)))


# =============================================================================================
# vertical launchers
# =============================================================================================
REDUT_MOD = (2.7, 4.2, 0.42)     # module footprint across, along, height above the deck


def redut_module(c, Bc, xc, ydeck):
    """one 8-cell Redut launcher module: low raised box with bevelled edges, 2 x 4 square lids on top."""
    w, l, h = REDUT_MOD
    o = P3(Bc, xc, ydeck)
    c.add(c.paint, rbox(w, h, l, r=0.08, seg=1, bevel=0.12, bottom=False), xf=M(o))
    top = o + np.array([0, h + 0.005, 0])
    hw, hl = w / 2 - 0.13, l / 2 - 0.13
    c.add(c.rect('redut_top'), _quad(top + np.array([-hw, 0, -hl]), top + np.array([hw, 0, -hl]),
                                      top + np.array([hw, 0, hl]), top + np.array([-hw, 0, hl]),
                                      uv=[(0, 1), (1, 1), (1, 0), (0, 0)], n=(0, 1, 0)))


def paint_redut_top(L, rect):
    """2 x 4 square cell lids with hinge bars and drain slots (u across, v along the ship)."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['super'] * 0.8, alpha=1.0, rough=0.55)
    for i in range(2):
        for j in range(4):
            a = x0 + w * (0.04 + 0.48 * i); b = a + w * 0.44
            cc = y0 + h * (0.03 + 0.2425 * j); d = cc + h * 0.215
            L.rect(a - 1, cc - 1, b + 1, d + 1, col=PAL['super'] * 0.55, add_height=-0.8)
            L.rect(a + 1, cc + 1, b - 1, d - 1, col=PAL['super'] * 0.93, add_height=0.5)
            L.rect(a + 3, cc + 2, a + 6, d - 2, col=PAL['super'] * 0.75, add_height=0.8)      # hinge bar
            L.rect((a + b) / 2 - 2, (cc + d) / 2 - 2, (a + b) / 2 + 2, (cc + d) / 2 + 2, col=PAL['dark'])


def uksk_field(c, B0, B1, hw, ytop):
    """UKSK 3S14: two 8-cell modules side by side, flush lids slightly proud of the deck."""
    yl = ytop + 0.055
    a = P3(B0, hw, yl); b = P3(B0, -hw, yl); cc = P3(B1, -hw, yl); d = P3(B1, hw, yl)
    c.add(c.paint, box(2 * hw + 0.2, 0.06, B1 - B0 + 0.2, center=(0, 0, 0)), xf=M(P3((B0 + B1) / 2, 0.0, ytop + 0.01)))
    c.add(c.rect('uksk_top'), _quad(d, a, b, cc, uv=[(0, 0), (0, 1), (1, 1), (1, 0)], n=(0, 1, 0)))


def paint_uksk_top(L, rect):
    """16 large square lids in 4 columns x 4 rows (u along the ship, v across), module seam on the centreline."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['deck_red'] * 0.95, alpha=1.0, rough=0.6)
    for i in range(4):
        for j in range(4):
            a = x0 + w * (0.02 + 0.245 * i); b = a + w * 0.225
            cc = y0 + h * (0.025 + 0.24 * j + (0.02 if j >= 2 else 0.0)); d = cc + h * 0.215
            L.rect(a - 1, cc - 1, b + 1, d + 1, col=PAL['super'] * 0.5, add_height=-0.8)
            L.rect(a + 1, cc + 1, b - 1, d - 1, col=PAL['deck_red'] * 1.12, add_height=0.4)
            L.rect(a + 2, cc + 2, a + 5, d - 2, col=PAL['super'] * 0.72, add_height=0.8)
            for k in (0.3, 0.7):
                yy = cc + (d - cc) * k
                L.rect(b - 7, yy - 1, b - 3, yy + 1, col=PAL['dark'])
    L.rect(x0, y0 + h * 0.5 - 1, x1, y0 + h * 0.5 + 1, col=PAL['super'] * 0.45, add_height=-0.6)


# =============================================================================================
# Palash CIWS
# =============================================================================================
def palash(c, pos, facing=0.0, node='Palash'):
    """Palash: drum pedestal, low turret body, two six-barrel 30 mm guns on the sides of a central
    cradle, radar / optronic head on top."""
    o = np.asarray(pos, float)
    c.b.node(node, parent='Weapons', translation=o)
    R = rot_y(facing) * 0.92            # whole mount scaled to the photographed height (about 2.2 m)

    def W(p):
        return o + R @ np.asarray(p, float)
    c.add(c.paint, cylinder(0.97, 0.5, seg=20, caps=(False, True)), xf=M(o))
    c.add(c.paint, rbox(2.1, 0.75, 2.2, r=0.35, seg=2, bevel=0.1, y0=0.5), xf=M(o, R))
    # sensor head: radar box with an optronic window
    hb = W((0.0, 2.05, -0.35))
    c.add(c.paint, rbox(0.9, 0.55, 0.75, r=0.15, seg=2, bevel=0.06), xf=M(hb, R))
    c.add(c.sw('glass'), box(0.4, 0.22, 0.05, center=(0, 0, 0)), xf=M(W((0.18, 2.32, 0.03)), R))
    c.add(c.sw('dark'), box(0.55, 0.3, 0.04, center=(0, 0, 0)), xf=M(W((-0.15, 2.3, 0.03)), R))
    # elevating cradle with the two six-barrel guns (child node, pivot on the elevation axis)
    piv = W((0.0, 1.55, 0.15))
    c.b.node(node + '_Guns', parent=node, translation=piv)
    el = math.radians(8.0)
    Re = R @ rot_x(-el)
    c.add(c.paint, rbox(1.2, 0.9, 1.6, r=0.22, seg=2, bevel=0.08, y0=-0.45), xf=M(piv, Re))
    d = Re @ np.array([0, 0, 1.0])
    for sx in (-1, 1):
        g0 = piv + Re @ np.array([sx * 0.85, 0.0, -0.6])
        c.add(c.sw('mid'), rbox(0.5, 0.5, 1.5, r=0.1, seg=1, y0=-0.25), xf=M(g0, Re))
        for k in range(6):
            a = 2 * math.pi * k / 6
            off = Re @ np.array([0.11 * math.cos(a), 0.11 * math.sin(a), 0])
            b0 = g0 + off + d * 0.7
            c.add(c.sw('dark'), tube_path([b0, b0 + d * 1.75], 0.035, seg=5))
        c.add(c.sw('dark'), tube_path([g0 + d * 1.6, g0 + d * 1.75], 0.16, seg=10))


# =============================================================================================
# decoy launchers and small arms
# =============================================================================================
def kt216(c, pos, facing=0.0, el=45.0, rows=2, cols=5, r=0.07, length=1.35, node=None):
    """KT-216 (PK-10) decoy launcher: base plate, yoke and a pack of tubes fixed at `el` degrees,
    pointing along `facing` (yaw about +y, 0 = +z)."""
    o = np.asarray(pos, float)
    R = rot_y(facing)
    c.add(c.paint, rbox(0.9, 0.3, 0.8, r=0.08, seg=1, bevel=0.04), xf=M(o, R))
    Re = R @ rot_x(-math.radians(el))
    piv = o + R @ np.array([0, 0.45, -0.1])
    d = Re @ np.array([0, 0, 1.0])
    for s in (-1, 1):
        c.add(c.paint, box(0.06, 0.35, 0.3, center=(0, 0, 0)), xf=M(o + R @ np.array([s * 0.42, 0.42, -0.1]), R))
    for i in range(cols):
        for j in range(rows):
            off = Re @ np.array([(i - (cols - 1) / 2) * (2 * r + 0.02), (j - (rows - 1) / 2) * (2 * r + 0.02), 0])
            a = piv + off - d * length * 0.35
            c.add(c.sw('mid'), cylinder(r, length, seg=8, caps=(True, True)), xf=M(a, axis_frame(d)))
            c.add(c.sw('black'), disc(r * 0.75, seg=8), xf=M(a + d * (length + 0.005), axis_frame(d)))
    c.add(c.paint, box(cols * (2 * r + 0.02) + 0.1, rows * (2 * r + 0.02) + 0.1, 0.08, center=(0, 0, 0)),
          xf=M(piv + d * 0.15, Re))


def mtpu(c, pos, facing=0.0):
    """14.5 mm MTPU pedestal machine gun with a small shield."""
    o = np.asarray(pos, float)
    R = rot_y(facing)
    c.add(c.sw('dark'), cylinder(0.12, 0.95, seg=8), xf=M(o))
    piv = o + np.array([0, 1.05, 0])
    c.add(c.sw('dark'), rbox(0.3, 0.25, 0.9, r=0.05, seg=1, y0=-0.12), xf=M(piv, R))
    d = R @ np.array([0, 0.05, 1.0])
    c.add(c.sw('dark'), tube_path([piv, piv + normalize(d) * 1.4], 0.03, seg=5))
    c.add(c.paint, box(0.7, 0.55, 0.03, center=(0, 0, 0)), xf=M(piv + R @ np.array([0, 0.1, 0.3]), R))


def register(m):
    m.alloc('redut_top', 96, 160, paint_redut_top)
    m.alloc('uksk_top', 256, 128, paint_uksk_top)
