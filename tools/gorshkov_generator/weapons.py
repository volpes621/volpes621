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
# Turret, local frame on the training axis (z forward, x port, y above the deck). Shapes follow the Arsenal
# plant photographs of a finished mount (2021) and the photographs of Admiral Gorshkov (2018) and Admiral
# Kasatonov (2021); sizes follow the 1:500 plan and profile:
# - an octagonal skirt with vertical sides;
# - above it two sloped cheek blocks with the gun slot between them, open at the front and through the
#   forward roof;
# - a rear block under the flat roof;
# - in the slot, the bronze-coloured cradle drum and the long barrel with a thick muzzle crown.
GUN_SKIRT = (0.5, 1.6, ((1.0, 2.55), (1.85, 1.75), (1.85, -2.4), (1.2, -3.0)))     # y0, y1, half outline (x, z)
GUN_ROOF = 3.7
GUN_UPPER_HW = (1.82, 1.4)    # half-width of the upper body at the skirt top and at the roof (sides lean in)
GUN_SLOT = (0.78, -0.95)      # half-width of the gun slot, z of its back wall
GUN_CHEEK = (2.45, 1.75, 0.75)  # cheek front: inner foot z, outer foot z, inner top z
GUN_AXIS = (-0.05, 2.5)       # trunnions near the training axis: z, height above the deck
GUN_BARREL = 7.5              # trunnion to muzzle (muzzle at B 19.0 on the profile)
SHIELD = (3.55, 1.35, 35.0)   # deck shield ring: radius, height, half-angle of the opening astern


def _cheek(c, uvs, sx, W):
    """one cheek block (sx = +1 port, -1 starboard): planar sloping front, inner slot wall, outer side
    leaning in, flat top at the roof."""
    y0, y1 = GUN_SKIRT[1], GUN_ROOF
    hs, zs = GUN_SLOT
    xo0, xo1 = GUN_UPPER_HW
    za, zb, ze = GUN_CHEEK
    A, B, E = np.array([hs, y0, za]), np.array([xo0, y0, zb]), np.array([hs, y1, ze])
    n = np.cross(B - A, E - A)
    zf = A[2] - (n[0] * (xo1 - A[0]) + n[1] * (y1 - A[1])) / n[2]   # outer top corner on the front plane
    pts = {'A': A, 'B': B, 'C': (xo0, y0, zs), 'D': (hs, y0, zs), 'E': E, 'F': (xo1, y1, zf), 'G': (xo1, y1, zs),
           'H': (hs, y1, zs)}
    P = {k: W(np.array([sx * v[0], v[1], v[2]])) for k, v in pts.items()}
    faces = ['ABCD', 'EFGH', 'ABFE', 'BCGF', 'DCGH', 'ADHE']
    _faces(c, uvs, [[P[k] for k in f] for f in faces])


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
    # rotating base ring (node pivot on the training axis)
    c.b.node(node, parent='Weapons', translation=o)
    y0, y1, half = GUN_SKIRT
    c.add(c.sw('mid'), cylinder(1.95, y0, seg=32, caps=(False, True)), xf=M(o))
    c.add(c.sw('dark'), cylinder(2.0, 0.07, seg=32, caps=(False, True), y0=y0 - 0.07), xf=M(o))
    # skirt: octagonal band with vertical sides
    ring = [(x, z) for (x, z) in half] + [(-x, z) for (x, z) in half[::-1]]
    loft_solid(c, c.paint, [[W((x, y, z)) for (x, z) in ring] for y in (y0, y1)])
    # rear block under the roof: sides leaning in, rear corners chamfered, the rear top edge rounded off
    hs, zs = GUN_SLOT
    xo0, xo1 = GUN_UPPER_HW

    def rear(y, hw, zrc, zr, hr):
        return [W((hw, y, zs)), W((hw, y, zrc)), W((hr, y, zr)), W((-hr, y, zr)), W((-hw, y, zrc)), W((-hw, y, zs))]
    k = lambda y: xo0 + (xo1 - xo0) * (y - y1) / (GUN_ROOF - y1)
    loft_solid(c, c.paint, [rear(y1, xo0, -2.38, -2.96, 1.18), rear(3.42, k(3.42), -2.3, -2.85, 0.98),
                            rear(GUN_ROOF, xo1, -1.9, -2.42, 0.85)])
    # cheeks either side of the gun slot, and the slot floor sloping down to the skirt front
    for sx in (1, -1):
        _cheek(c, c.paint, sx, W)
    chin = [(2.45, y1), (2.45, y1 + 0.05), (1.15, 1.88), (zs, 1.88), (zs, y1)]
    _faces(c, c.paint, [[W((hs, y, z)) for (z, y) in chin], [W((-hs, y, z)) for (z, y) in chin]] +
           [[W((hs, chin[i][1], chin[i][0])), W((hs, chin[(i + 1) % 5][1], chin[(i + 1) % 5][0])),
             W((-hs, chin[(i + 1) % 5][1], chin[(i + 1) % 5][0])), W((-hs, chin[i][1], chin[i][0]))] for i in range(5)])
    # details: round hatch in the skirt front, roof hatch and sight, vents, rear door and rungs, side hatches
    c.add(c.paint, cylinder(0.24, 0.03, seg=16, caps=(True, False)), xf=M(W((0.0, 1.0, 2.55)), R @ rot_x(math.pi / 2)))
    c.add(c.sw('dark'), cylinder(0.25, 0.025, seg=16, caps=(False, False)), xf=M(W((0.0, 1.0, 2.55)), R @ rot_x(math.pi / 2)))
    c.add(c.sw('mid'), rbox(0.72, 0.06, 0.85, r=0.12, seg=1), xf=M(W((0.62, GUN_ROOF, -1.55)), R))
    c.add(c.sw('mid'), rbox(0.42, 0.26, 0.5, r=0.06, seg=1, bevel=0.03), xf=M(W((-0.7, GUN_ROOF, -1.1)), R))
    c.add(c.sw('glass'), box(0.3, 0.14, 0.03, center=(0, 0, 0)), xf=M(W((-0.7, GUN_ROOF + 0.15, -0.84)), R))
    for x in (0.35, -0.3):
        c.add(c.sw('mid'), cylinder(0.11, 0.12, seg=8), xf=M(W((x, GUN_ROOF, -2.05))))
    for kk in range(5):
        c.add(c.sw('dark'), box(0.45, 0.04, 0.05, center=(0, 0, 0)), xf=M(W((0.9, 0.95 + kk * 0.42, -3.02)), R))
    c.add(c.sw('mid'), box(0.8, 1.5, 0.02, center=(0, 0, 0)), xf=M(W((-0.45, 1.3, -3.0)), R))
    for sx in (1, -1):
        c.add(c.paint, rbox(0.03, 0.9, 1.1, r=0.1, seg=1, y0=-0.45), xf=M(W((sx * 1.6, 2.5, -1.4)), R))
    # elevating mass: cradle drum across the slot and the barrel (child node, pivot on the trunnions)
    elev = math.radians(3.0)
    piv = W((0.0, GUN_AXIS[1], GUN_AXIS[0]))
    c.b.node(node + '_Gun', parent=node, translation=piv)
    dirv = R @ np.array([0.0, math.sin(elev), math.cos(elev)])
    c.add(c.sw('tan'), cylinder(0.6, 2 * hs - 0.06, seg=24, caps=(True, True), y0=-(hs - 0.03)),
          xf=M(piv, R @ rot_z(math.pi / 2)))
    L = GUN_BARREL
    prof = [(0.0, 0.45), (0.32, 0.45), (0.32, 0.9), (0.25, 1.0), (0.24, 2.2), (0.28, 2.25), (0.28, 2.4), (0.17, 2.5),
            (0.155, 4.5), (0.175, 4.55), (0.175, 4.7), (0.15, 4.75), (0.14, L - 0.62), (0.19, L - 0.58),
            (0.19, L - 0.06), (0.16, L), (0.0, L)]
    c.add(c.paint, lathe(prof, seg=16), xf=M(piv, axis_frame(dirv)))
    c.add(c.sw('black'), disc(0.07, seg=12), xf=M(piv + dirv * (L + 0.002), axis_frame(dirv)))


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
PALASH_BODY = ((0.55, -0.48), (0.55, 0.4), (0.0, 0.58))   # half plan of the central housing (x, z): prow forward
PALASH_GUN_X = 1.04            # gun axes either side of the body
PALASH_TRUNNION = (1.12, 0.02)  # elevation axis: height, z


def palash(c, pos, facing=0.0, node='Palash'):
    """3M89 Palash combat module in the gun-only fit of Project 22350, after the KBP display model (Army-2016)
    and the Palma-SU photograph: bolted deck ring, pedestal with a buffer cylinder, central housing with a
    V prow and chamfered top, ribbed cradle drums on both sides, an AO-18KD six-barrel gun outboard of each
    drum (receiver, black barrel casing, clamp ring and struts), optronic fire-control head on a turntable
    pedestal. About 2.3 m tall and 2.4 m wide across the guns."""
    o = np.asarray(pos, float)
    R = rot_y(facing)

    def W(p):
        return o + R @ np.asarray(p, float)
    # fixed deck ring with its bolt circle (in the static Weapons node)
    c.add(c.paint, lathe([(1.0, 0.0), (1.0, 0.05), (0.95, 0.07), (0.86, 0.07), (0.86, 0.12), (0.0, 0.12)], seg=28), xf=M(o),
          node='Weapons')
    for k in range(16):
        aa = 2 * math.pi * (k + 0.5) / 16
        c.add(c.sw('mid'), cylinder(0.03, 0.035, seg=6), xf=M(o + np.array([0.93 * math.cos(aa), 0.06, 0.93 * math.sin(aa)])),
              node='Weapons')
    # rotating part (node pivot on the training axis)
    c.b.node(node, parent='Weapons', translation=o)
    c.add(c.sw('mid'), cylinder(0.8, 0.08, seg=24, caps=(False, True), y0=0.12), xf=M(o))
    c.add(c.paint, rbox(1.0, 0.42, 0.9, r=0.14, seg=2, bevel=0.03, y0=0.2), xf=M(o, R))
    c.add(c.paint, cylinder(0.065, 0.62, seg=10, caps=(True, True), y0=-0.31), xf=M(W((0.0, 0.33, 0.47)), R @ rot_z(math.pi / 2)))
    for sx in (-0.31, 0.31):                                     # buffer end blocks
        c.add(c.paint, box(0.1, 0.14, 0.12, center=(0, 0, 0)), xf=M(W((sx, 0.33, 0.45)), R))
    # central housing: V prow, the top front edge chamfered back
    (xh, zr), (_, zc), (_, zp) = PALASH_BODY

    def ring(y, cut):
        return [W(p) for p in ((xh, y, zr), (xh, y, zc - cut), (0.0, y, zp - 1.6 * cut), (-xh, y, zc - cut), (-xh, y, zr))]
    loft_solid(c, c.paint, [ring(0.6, 0.0), ring(1.36, 0.0), ring(1.56, 0.17)])
    for sx in (-1, 1):                                           # handles on the two prow facets
        for y in (0.95, 1.25):
            zf = zp - (zp - zc) * 0.28 / xh + 0.02
            c.add(c.sw('mid'), box(0.18, 0.03, 0.03, center=(0, 0, 0)),
                  xf=M(W((sx * 0.28, y, zf)), R @ rot_y(sx * math.atan2(zp - zc, xh))))
    # optronic fire-control head on a turntable pedestal
    c.add(c.paint, lathe([(0.42, 1.56), (0.42, 1.6), (0.36, 1.62), (0.36, 1.78), (0.3, 1.8), (0.0, 1.8)], seg=24),
          xf=M(o))
    for sx in (-1, 1):                                           # yoke
        c.add(c.paint, box(0.05, 0.4, 0.34, center=(0, 0, 0)), xf=M(W((sx * 0.29, 1.98, 0.0)), R))
    hb = W((0.0, 1.82, 0.02))
    c.add(c.paint, rbox(0.52, 0.5, 0.46, r=0.05, seg=1, bevel=0.03), xf=M(hb, R))
    fz = 0.02 + 0.235
    for (x, y, r_) in ((-0.12, 2.18, 0.075), (0.12, 2.18, 0.075)):
        c.add(c.sw('glass'), cylinder(r_, 0.02, seg=12, caps=(True, False)), xf=M(W((x, y, fz)), R @ rot_x(math.pi / 2)))
    c.add(c.sw('glass'), box(0.15, 0.06, 0.02, center=(0, 0, 0)), xf=M(W((-0.1, 2.02, fz + 0.005)), R))
    for (x, y) in ((0.04, 1.95), (0.15, 1.95), (-0.15, 1.92)):
        c.add(c.sw('black'), cylinder(0.035, 0.02, seg=8, caps=(True, False)), xf=M(W((x, y, fz)), R @ rot_x(math.pi / 2)))
    c.add(c.paint, box(0.56, 0.03, 0.12, center=(0, 0, 0)), xf=M(W((0.0, 2.33, 0.2)), R))   # sun visor
    # elevating mass: cradle drums and the two guns (child node, pivot on the elevation axis)
    ty, tz = PALASH_TRUNNION
    piv = W((0.0, ty, tz))
    c.b.node(node + '_Guns', parent=node, translation=piv)
    el = math.radians(6.0)
    Re = R @ rot_x(-el)

    def G(p):
        return piv + Re @ np.asarray(p, float)
    d = Re @ np.array([0, 0, 1.0])
    for sx in (-1, 1):
        # ribbed cradle drum between the housing and the gun
        c.add(c.paint, cylinder(0.38, 0.3, seg=20, caps=(True, True), y0=0.0), xf=M(G((sx * 0.56, 0, 0)), Re @ rot_z(-sx * math.pi / 2)))
        for k in range(3):
            c.add(c.sw('mid'), cylinder(0.395, 0.03, seg=20, caps=(False, False), y0=0.04 + k * 0.1),
                  xf=M(G((sx * 0.56, 0, 0)), Re @ rot_z(-sx * math.pi / 2)))
        gx = sx * PALASH_GUN_X
        # receiver (light grey box), feed box below it, black barrel casing, muzzle with six bores
        c.add(c.paint, rbox(0.34, 0.4, 0.95, r=0.07, seg=1, bevel=0.03, y0=-0.2), xf=M(G((gx, 0.0, -0.15)), Re))
        c.add(c.paint, rbox(0.3, 0.28, 0.5, r=0.05, seg=1, y0=-0.48), xf=M(G((gx, 0.0, -0.25)), Re))
        c.add(c.sw('black'), lathe([(0.0, 0.3), (0.15, 0.3), (0.15, 1.85), (0.135, 1.9), (0.0, 1.9)], seg=18),
              xf=M(G((gx, 0.0, 0.0)), axis_frame(d)))
        c.add(c.paint, cylinder(0.19, 0.14, seg=18, caps=(True, True), y0=0.0), xf=M(G((gx, 0.0, 0.3)), axis_frame(d)))
        for k in range(6):
            aa = 2 * math.pi * k / 6
            c.add(c.sw('dark'), cylinder(0.022, 0.02, seg=6, caps=(True, False)),
                  xf=M(G((gx + 0.075 * math.cos(aa), 0.075 * math.sin(aa), 1.9)), axis_frame(d)))
        cl = G((gx, 0.0, 1.3))                                      # clamp ring and the struts to the cradle
        c.add(c.paint, cylinder(0.18, 0.1, seg=18, caps=(True, True), y0=-0.05), xf=M(cl, axis_frame(d)))
        for (dy, dx) in ((0.17, -0.12 * sx), (-0.17, -0.12 * sx)):
            c.add(c.paint, tube_path([cl + Re @ np.array([dx, dy, 0]), G((sx * 0.86, dy * 1.4, 0.3))], 0.025, seg=5))


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
