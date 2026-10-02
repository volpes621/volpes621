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
# turret (local: z forward of the training axis, x port, y above the deck), from the 1:500 plan and profile
# and the 2018 aerial photograph: wedge nose sloping up to a flat roof, slab sides leaning in, a vertical
# back with chamfered corners; the deck shield is a perforated ring open astern.
GUN_RINGS = (            # (y, nose z, nose half-width, front corner z, half-width, rear corner z, rear z, rear hw)
    (0.62, 2.55, 0.55, 1.45, 1.85, -2.3, -3.0, 1.1),
    (1.5, 2.15, 0.6, 1.15, 1.84, -2.3, -3.0, 1.1),
    (2.25, 1.7, 0.68, 0.85, 1.78, -2.3, -3.0, 1.09),
    (2.65, 1.52, 0.72, 0.62, 1.72, -2.28, -2.98, 1.08),
    (3.42, 0.2, 0.92, -0.18, 1.52, -2.22, -2.85, 1.0),
    (3.7, -0.35, 1.0, -0.6, 1.34, -1.85, -2.42, 0.88),
)
GUN_AXIS = (0.3, 2.45)     # trunnions: z, height above the deck
GUN_BARREL = 7.15          # trunnion to muzzle
SHIELD = (3.55, 1.35, 35.0)   # deck shield ring: radius, height, half-angle of the opening astern


def a192m(c, Bc, ydeck, train=0.0, node='A192M'):
    o = P3(Bc, 0.0, ydeck)
    R = rot_y(train)

    def W(p):
        return o + R @ np.asarray(p, float)
    # deck shield: perforated double wall round the mount, open astern, rim tube on top, end posts
    rr, hh, op = SHIELD
    n = 40
    a0, a1 = math.radians(-90 + op), math.radians(270 - op)   # angle from +x toward +z (forward)
    ang = [a0 + (a1 - a0) * k / n for k in range(n + 1)]
    for t, s in ((0.0, 1), (0.07, -1)):
        r_ = rr - t
        P, Nn, UV = [], [], []
        for k, aa in enumerate(ang):
            d = np.array([math.cos(aa), 0, math.sin(aa)])
            for j, yy in enumerate((0.0, hh)):
                P.append(o + d * r_ + np.array([0, yy, 0]))
                Nn.append(d * s)
                UV.append((rr * (aa - a0), 1.0 - j))
        P = np.array(P); Nn = np.array(Nn)
        I = []
        for k in range(n):
            q = 2 * k
            I += [(q, q + 2, q + 3), (q, q + 3, q + 1)]
        I = np.array(I)
        fn = np.cross(P[I[0, 1]] - P[I[0, 0]], P[I[0, 2]] - P[I[0, 0]])
        if np.dot(fn, Nn[I[0, 0]]) < 0:
            I = I[:, ::-1]
        c.add(c.band('PERF', st.PERF_REPEAT_M), (P, Nn, np.array(UV), I))
    rim = [o + np.array([math.cos(aa) * (rr - 0.035), hh, math.sin(aa) * (rr - 0.035)]) for aa in ang]
    c.add(c.paint, pipe(rim, 0.055, seg=6))
    for aa in (a0, a1):
        d = np.array([math.cos(aa), 0, math.sin(aa)])
        c.add(c.paint, box(0.1, hh, 0.14, center=(0, hh / 2, 0)), xf=M(o + d * (rr - 0.035), rot_y(math.atan2(d[0], d[2]))))
    for k in range(4, n, 6):                               # stiffeners inside the ring
        d = np.array([math.cos(ang[k]), 0, math.sin(ang[k])])
        c.add(c.paint, box(0.05, hh * 0.9, 0.22, center=(0, hh * 0.45, -0.11)), xf=M(o + d * (rr - 0.07),
                                                                                       rot_y(math.atan2(d[0], d[2]))))
    # rotating base and the turret housing (node pivot on the training axis)
    c.b.node(node, parent='Weapons', translation=o)
    c.add(c.sw('mid'), cylinder(1.95, 0.62, seg=32, caps=(False, True)), xf=M(o))
    c.add(c.sw('dark'), cylinder(2.02, 0.08, seg=32, caps=(False, True), y0=0.54), xf=M(o))

    def ring(y, zn, hn, zc, hc, zrc, zr, hr):
        return [W((hn, y, zn)), W((hc, y, zc)), W((hc, y, zrc)), W((hr, y, zr)),
                W((-hr, y, zr)), W((-hc, y, zrc)), W((-hc, y, zc)), W((-hn, y, zn))]
    loft_solid(c, c.paint, [ring(*g) for g in GUN_RINGS])
    # embrasure: dark recess on the steep face round the barrel (the cloth boot covers its middle)
    (ya, za), (yb, zb) = (GUN_RINGS[2][0], GUN_RINGS[2][1]), (GUN_RINGS[3][0], GUN_RINGS[3][1])
    phi = math.atan2(za - zb, yb - ya)
    nf = R @ np.array([0.0, math.sin(phi), math.cos(phi)])
    emb = W((0.0, (ya + yb) / 2, (za + zb) / 2)) + nf * 0.012
    c.add(c.sw('black'), box(1.1, (yb - ya) / math.cos(phi) * 0.95, 0.02, center=(0, 0, 0)), xf=M(emb, R @ rot_x(-phi)))
    # roof: hatch, sight, vents; rear: ladder rungs and a door outline
    y1 = GUN_RINGS[-1][0]
    c.add(c.sw('mid'), rbox(0.7, 0.07, 0.8, r=0.12, seg=1), xf=M(W((0.62, y1, -1.3)), R))
    c.add(c.sw('mid'), rbox(0.42, 0.25, 0.5, r=0.06, seg=1, bevel=0.03), xf=M(W((-0.65, y1, -0.9)), R))
    c.add(c.sw('glass'), box(0.3, 0.14, 0.03, center=(0, 0, 0)), xf=M(W((-0.65, y1 + 0.15, -0.64)), R))
    for x in (0.0, -0.6):
        c.add(c.sw('mid'), cylinder(0.11, 0.12, seg=8), xf=M(W((x + 0.3, y1, -2.05))))
    for k in range(5):
        c.add(c.sw('dark'), box(0.45, 0.04, 0.05, center=(0, 0, 0)), xf=M(W((0.9, 0.95 + k * 0.42, -3.03)), R))
    c.add(c.sw('mid'), box(0.8, 1.5, 0.02, center=(0, 0, 0)), xf=M(W((-0.55, 1.55, -3.005)), R))
    for sx in (1, -1):                                         # side hatches
        c.add(c.paint, cylinder(0.42, 0.03, seg=16, caps=(True, False)),
              xf=M(W((sx * 1.86, 1.55, -1.2)), R @ rot_z(-sx * math.pi / 2)))
        c.add(c.sw('dark'), cylinder(0.43, 0.035, seg=16, caps=(False, False)),
              xf=M(W((sx * 1.855, 1.55, -1.2)), R @ rot_z(-sx * math.pi / 2)))
    # elevating mass: boot and barrel (child node, pivot on the trunnions)
    elev = math.radians(3.0)
    piv = W((0.0, GUN_AXIS[1], GUN_AXIS[0]))
    c.b.node(node + '_Gun', parent=node, translation=piv)
    dirv = R @ np.array([0.0, math.sin(elev), math.cos(elev)])
    Rg = R @ rot_x(-elev)
    c.add(c.sw('tan'), rbox(0.78, 0.62, 0.5, r=0.12, seg=2, y0=-0.31), xf=M(piv + dirv * 1.2, Rg))
    c.add(c.sw('tan'), lathe([(0.0, 1.4), (0.3, 1.4), (0.27, 1.6), (0.21, 1.78), (0.0, 1.8)], seg=14),
          xf=M(piv, axis_frame(dirv)))
    L = GUN_BARREL
    prof = [(0.0, 1.5), (0.21, 1.5), (0.2, 2.6), (0.17, 3.6), (0.2, 3.75), (0.2, 4.55), (0.165, 4.7), (0.15, L - 0.32),
            (0.165, L - 0.28), (0.165, L - 0.04), (0.12, L), (0.0, L)]
    c.add(c.paint, lathe(prof, seg=14), xf=M(piv, axis_frame(dirv)))


# =============================================================================================
# vertical launchers
# =============================================================================================
REDUT_MOD = (2.75, 4.2, 0.38)    # module footprint across, along, height above the deck (1:500 plan)
REDUT_LID = (1.22, 0.96, 0.07)   # lid across, along, thickness


def redut_module(c, Bc, xc, ydeck):
    """one 8-cell Redut module: low box with bevelled edges, deck plate with seams, 2 rows x 4 raised lids,
    each row hinged along the module's long outer edge with a line of hinge knuckles (2018 photograph)."""
    w, l, h = REDUT_MOD
    o = P3(Bc, xc, ydeck)
    c.add(c.paint, rbox(w, h, l, r=0.08, seg=1, bevel=0.1, bottom=False), xf=M(o))
    top = o + np.array([0, h + 0.004, 0])
    hw, hl = w / 2 - 0.1, l / 2 - 0.1
    c.add(c.rect('redut_top'), _quad(top + np.array([-hw, 0, -hl]), top + np.array([hw, 0, -hl]),
                                      top + np.array([hw, 0, hl]), top + np.array([-hw, 0, hl]),
                                      uv=[(0, 1), (1, 1), (1, 0), (0, 0)], n=(0, 1, 0)))
    lw, ll, lt = REDUT_LID
    for sx in (-1, 1):
        xl = sx * (lw / 2 + 0.04)
        for j in range(4):
            zl = (j - 1.5) * (ll + 0.06)
            c.add(c.paint, rbox(lw, lt, ll, r=0.05, seg=1, bevel=0.025), xf=M(top + np.array([xl, 0.0, zl])))
            for k in (-0.3, 0.0, 0.3):                         # hinge knuckles on the outer edge
                p = top + np.array([sx * (lw + 0.07), 0.045, zl + k * ll])
                c.add(c.sw('mid'), cylinder(0.045, 0.16, seg=8, caps=(True, True), y0=-0.08), xf=M(p, rot_x(math.pi / 2)))


def paint_redut_top(L, rect):
    """deck plate of a Redut module under the lids: seams, drain slots and the lid footprints in shadow."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['super'] * 0.72, alpha=1.0, rough=0.55)
    for i in range(2):
        for j in range(4):
            a = x0 + w * (0.03 + 0.49 * i); b = a + w * 0.45
            cc = y0 + h * (0.025 + 0.2425 * j); d = cc + h * 0.22
            L.rect(a - 1, cc - 1, b + 1, d + 1, col=PAL['super'] * 0.45, add_height=-0.8)
    L.rect(x0 + w * 0.495, y0, x0 + w * 0.505, y1, col=PAL['dark'], add_height=-0.6)


def uksk_field(c, B0, B1, hw, ytop):
    """UKSK 3S14: two 8-cell modules in line (seam across the ship), flush lids slightly proud of the
    deck; each module's forward row is hinged at its forward edge, the aft row at its aft edge (plan)."""
    yl = ytop + 0.055
    a = P3(B0, hw, yl); b = P3(B0, -hw, yl); cc = P3(B1, -hw, yl); d = P3(B1, hw, yl)
    c.add(c.paint, box(2 * hw + 0.2, 0.06, B1 - B0 + 0.2, center=(0, 0, 0)), xf=M(P3((B0 + B1) / 2, 0.0, ytop + 0.01)))
    c.add(c.rect('uksk_top'), _quad(d, a, b, cc, uv=[(0, 0), (0, 1), (1, 1), (1, 0)], n=(0, 1, 0)))
    Bm = (B0 + B1) / 2
    for Bh in (B0 + 0.12, Bm - 0.17, Bm + 0.17, B1 - 0.12):     # hinge bars with knuckles
        c.add(c.sw('mid'), cylinder(0.035, 2 * hw - 0.3, seg=8, caps=(True, True), y0=-(hw - 0.15)),
              xf=M(P3(Bh, 0.0, yl + 0.035), rot_z(math.pi / 2)))
        for k in range(8):
            xk = -hw + 0.3 + k * (2 * hw - 0.6) / 7
            c.add(c.sw('mid'), cylinder(0.055, 0.14, seg=8, caps=(True, True), y0=-0.07),
                  xf=M(P3(Bh, xk, yl + 0.035), rot_z(math.pi / 2)))


def paint_uksk_top(L, rect):
    """16 square lids: 4 rows along the ship (u) x 4 across (v); the two modules meet at u = 0.5."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['super'] * 0.8, alpha=1.0, rough=0.6)
    for i in range(4):
        for j in range(4):
            a = x0 + w * (0.03 + 0.235 * i + (0.03 if i >= 2 else 0.0)); b = a + w * 0.205
            cc = y0 + h * (0.025 + 0.24 * j); d = cc + h * 0.215
            L.rect(a - 1, cc - 1, b + 1, d + 1, col=PAL['super'] * 0.5, add_height=-0.8)
            L.rect(a + 1, cc + 1, b - 1, d - 1, col=PAL['super'] * 0.93, add_height=0.4)
            hx = a + 2 if i in (0, 2) else b - 5                 # hinge side of the lid
            L.rect(hx, cc + 2, hx + 3, d - 2, col=PAL['super'] * 0.72, add_height=0.8)
            for k in (0.3, 0.7):                                 # dogs on the free edge
                yy = cc + (d - cc) * k
                fx = b - 7 if i in (0, 2) else a + 3
                L.rect(fx, yy - 1, fx + 4, yy + 1, col=PAL['dark'])
    L.rect(x0 + w * 0.5 - 2, y0, x0 + w * 0.5 + 2, y1, col=PAL['super'] * 0.45, add_height=-0.6)


# =============================================================================================
# Palash CIWS
# =============================================================================================
def palash(c, pos, facing=0.0, node='Palash'):
    """Palash: drum pedestal, rounded turret body, sensor head on a neck (radar face and optronic
    windows), elevating cradle with a six-barrel 30 mm gun in a round housing on each side
    (1:500 profile, museum model photographs)."""
    o = np.asarray(pos, float)
    c.b.node(node, parent='Weapons', translation=o)
    R = rot_y(facing) * 0.92            # whole mount scaled to the photographed height (about 2.2 m)

    def W(p):
        return o + R @ np.asarray(p, float)
    c.add(c.paint, lathe([(0.97, 0.0), (0.97, 0.38), (0.9, 0.45), (0.0, 0.47)], seg=24), xf=M(o))
    c.add(c.sw('dark'), cylinder(0.99, 0.05, seg=24, caps=(False, False), y0=0.4), xf=M(o))
    c.add(c.paint, rbox(2.0, 0.8, 2.1, r=0.4, seg=3, bevel=0.12, y0=0.46), xf=M(o, R))
    for sx in (-1, 1):                                     # access panels on the body sides
        c.add(c.sw('mid'), box(0.02, 0.4, 0.9, center=(0, 0, 0)), xf=M(W((sx * 1.0, 0.86, -0.3)), R))
    # sensor head: neck, rounded box, radar face and two optronic windows, small antenna on top
    c.add(c.paint, cylinder(0.25, 0.4, seg=12, y0=1.26), xf=M(W((0.0, 0.0, -0.4))))
    hb = W((0.0, 1.62, -0.4))
    c.add(c.paint, rbox(0.95, 0.72, 0.85, r=0.14, seg=2, bevel=0.06), xf=M(hb, R))
    c.add(c.sw('dark'), box(0.62, 0.5, 0.03, center=(0, 0, 0)), xf=M(W((-0.08, 1.98, 0.03)), R))
    for k, x in enumerate((0.3, 0.3)):
        c.add(c.sw('glass'), cylinder(0.08, 0.04, seg=10, caps=(True, False)),
              xf=M(W((x, 1.86 + 0.2 * k, 0.03)), R @ rot_x(math.pi / 2)))
    c.add(c.paint, cylinder(0.03, 0.35, seg=5), xf=M(W((0.3, 2.34, -0.65))))
    # elevating cradle with the two guns (child node, pivot on the elevation axis)
    piv = W((0.0, 1.5, 0.15))
    c.b.node(node + '_Guns', parent=node, translation=piv)
    el = math.radians(8.0)
    Re = R @ rot_x(-el)
    c.add(c.paint, rbox(1.15, 0.8, 1.5, r=0.22, seg=2, bevel=0.08, y0=-0.4), xf=M(piv, Re))
    d = Re @ np.array([0, 0, 1.0])
    for sx in (-1, 1):
        g0 = piv + Re @ np.array([sx * 0.82, 0.0, -0.75])
        c.add(c.sw('mid'), lathe([(0.0, 0.0), (0.26, 0.0), (0.28, 0.1), (0.28, 1.25), (0.22, 1.42), (0.0, 1.43)], seg=14),
              xf=M(g0, axis_frame(d)))
        c.add(c.paint, box(0.3, 0.35, 0.8, center=(0, 0, 0)), xf=M(piv + Re @ np.array([sx * 0.62, -0.05, -0.25]), Re))
        for k in range(6):
            a = 2 * math.pi * k / 6
            off = Re @ np.array([0.1 * math.cos(a), 0.1 * math.sin(a), 0])
            b0 = g0 + off + d * 1.35
            c.add(c.sw('dark'), tube_path([b0, b0 + d * 1.15], 0.03, seg=5))
        for z in (1.75, 2.42):                             # barrel clamps
            c.add(c.sw('dark'), cylinder(0.15, 0.06, seg=12, caps=(True, True), y0=-0.03), xf=M(g0 + d * z, axis_frame(d)))


# =============================================================================================
# decoy launchers and small arms
# =============================================================================================
def kt216(c, pos, facing=0.0, el=30.0, rows=2, cols=5, r=0.075, length=0.8, ped=1.0):
    """KT-216 decoy launcher (bow photograph, 2018): box pedestal with two slots and a round port in its
    front face, a turntable and a framed pack of short tubes trained along `facing` (yaw about +y,
    0 = +z) and fixed at `el` degrees."""
    o = np.asarray(pos, float)
    R = rot_y(facing)
    c.add(c.paint, rbox(0.85, ped, 0.95, r=0.06, seg=1, bevel=0.04), xf=M(o, R))
    for s in (-1, 1):                                          # slots and the round port in the pedestal
        c.add(c.sw('black'), box(0.11, ped * 0.5, 0.02, center=(0, 0, 0)),
              xf=M(o + R @ np.array([s * 0.2, ped * 0.5, 0.48]), R))
    c.add(c.sw('dark'), cylinder(0.12, 0.02, seg=12, caps=(True, False)), xf=M(o + R @ np.array([0.0, ped * 0.32, 0.47]),
                                                                             R @ rot_x(math.pi / 2)))
    c.add(c.sw('mid'), cylinder(0.42, 0.1, seg=16, caps=(True, True), y0=ped), xf=M(o))
    Re = R @ rot_x(-math.radians(el))
    piv = o + np.array([0, ped + 0.45, 0])
    d = Re @ np.array([0, 0, 1.0])
    pw, ph = cols * (2 * r + 0.03) + 0.12, rows * (2 * r + 0.03) + 0.12
    c.add(c.paint, rbox(pw, ph, length, r=0.04, seg=1, y0=-ph / 2), xf=M(piv, Re))
    for i in range(cols):
        for j in range(rows):
            off = Re @ np.array([(i - (cols - 1) / 2) * (2 * r + 0.03), (j - (rows - 1) / 2) * (2 * r + 0.03), 0])
            c.add(c.sw('black'), disc(r * 0.8, seg=8), xf=M(piv + off + d * (length / 2 + 0.006), axis_frame(d)))
    for s in (-1, 1):                                          # yoke
        c.add(c.paint, box(0.06, 0.5, 0.3, center=(0, 0, 0)), xf=M(o + R @ np.array([s * (pw / 2 + 0.05), ped + 0.33, 0]), R))


def kt308(c, pos, facing=0.0, el=45.0, r=0.12, L=1.7):
    """twin long-tube decoy launcher: two packs of 2 x 3 tubes on a common cradle, trained along `facing`."""
    o = np.asarray(pos, float)
    R = rot_y(facing)
    c.add(c.paint, rbox(1.3, 0.5, 0.9, r=0.06, seg=1, bevel=0.04), xf=M(o, R))
    Re = R @ rot_x(-math.radians(el))
    d = Re @ np.array([0, 0, 1.0])
    sp = 2 * r + 0.015
    for px in (-0.42, 0.42):
        piv = o + R @ np.array([px, 0.95, -0.15])
        for i in range(3):
            for j in range(2):
                a = piv + Re @ np.array([(i - 1) * sp * 0.98, (j - 0.5) * sp, 0]) - d * L * 0.35
                c.add(c.sw('mid'), cylinder(r, L, seg=10, caps=(True, True)), xf=M(a, axis_frame(d)))
                c.add(c.sw('black'), disc(r * 0.8, seg=10), xf=M(a + d * (L + 0.005), axis_frame(d)))
        c.add(c.paint, box(3 * sp + 0.06, 2 * sp + 0.06, 0.1, center=(0, 0, 0)), xf=M(piv + d * 0.2, Re))
    for s in (-1, 1):
        c.add(c.paint, box(0.08, 0.7, 0.45, center=(0, 0, 0)), xf=M(o + R @ np.array([s * 0.86, 0.75, -0.1]), R))


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
