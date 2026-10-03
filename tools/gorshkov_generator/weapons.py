"""Project 22350 weapons: A-192M 130 mm gun, Redut and UKSK 3S14 vertical launchers, Palash CIWS,
KT-216 decoy launchers and the 14.5 mm MTPU pedestal mounts.
Shapes follow the 2018 Russian MoD and 2023 Mehr News photographs of Admiral Gorshkov."""
import math
import numpy as np

from meshkit import *
from meshkit import _quad
import atlas as st
from kit import M, P3, extrude_x
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


def plane_solid(c, uvs, planes, xf=None, node=None):
    """convex solid bounded by half-spaces n . p <= d (planes = [(n, d), ...]); xf maps local points to world.
    Vertices are the feasible intersections of plane triples; each face polygon is sorted round its centre."""
    import itertools
    Nn = np.array([np.asarray(n, float) for n, _ in planes]); D = np.array([d for _, d in planes], float)
    V = []
    for i, j, k in itertools.combinations(range(len(planes)), 3):
        A = Nn[[i, j, k]]
        if abs(np.linalg.det(A)) < 1e-9:
            continue
        p = np.linalg.solve(A, D[[i, j, k]])
        if np.all(Nn @ p <= D + 1e-7) and not any(np.linalg.norm(p - v) < 1e-6 for v in V):
            V.append(p)
    polys = []
    for n, d in zip(Nn, D):
        on = [v for v in V if abs(n @ v - d) < 1e-6]
        if len(on) < 3:
            continue
        cen = np.mean(on, axis=0)
        nn = normalize(n)
        u = normalize(on[0] - cen)
        w = np.cross(nn, u)
        order = np.argsort([math.atan2((v - cen) @ w, (v - cen) @ u) for v in on])
        polys.append([xf(on[q]) if xf else on[q] for q in order])
    _faces(c, uvs, polys, node=node)


def plane_through(p0, p1, p2, inside):
    """half-space (n, d) of the plane through three points, oriented so that `inside` satisfies n . p <= d."""
    p0, p1, p2 = (np.asarray(p, float) for p in (p0, p1, p2))
    n = np.cross(p1 - p0, p2 - p0)
    if n @ np.asarray(inside, float) > n @ p0:
        n = -n
    return n, float(n @ p0)


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
# Turret, local frame on the training axis (z forward, x port, y above the deck). Plan sizes come from the 1:500
# plan. The front profile is convex as on the drawing: steep low down, flattening into the roof. The height
# (roof 4.1 m above the deck) follows the photographs, which show a taller turret than the drawing.
# Shapes come from the Arsenal plant photograph of a finished mount (2021), the Arsenal design-bureau
# renderings, and photographs of Admiral Gorshkov (2018) and Admiral Kasatonov (2020, 2021).
# Every part is a convex solid cut by planes:
# - Skirt: an octagon with vertical sides. A 2 m wide nose face carries the bolted oval hatch, and 45 deg
#   chamfers widen it to the full beam.
# - Upper body: sides leaning in about 12 deg, then large chamfers along the roof edges.
# - Cheeks: one on each side of the gun slot. Each front has two faces. A steep face rises about 42 deg from
#   the nose almost to the roof. Just below the roof it bends into a short upper slope, about 13 deg, which
#   runs on to the roof at B 26.95. A triangular facet cuts off the outer lower corner.
# - Hood: the roof is carried across the gun slot up to the training axis, its top on the flat upper face.
# - Rear block: behind the slot, with the back of the roof bevelled down to the vertical rear face.
# - Mantlet: under the hood front, a large bronze-coloured cradle drum fills the slot and stands out ahead
#   of it. The barrel leaves the drum through a thick breech casing and ends in a thick muzzle crown.
GUN_SKIRT = (0.5, 1.8, ((1.0, 2.65), (1.85, 1.8), (1.85, -2.3), (1.2, -2.85)))   # y0, y1, half outline (x, z)
GUN_ROOF = 4.1
GUN_SIDE = ((1.85, 1.8), (1.5, 3.6), (1.08, 4.1))       # upper half section: side foot, chamfer knee, roof edge
GUN_CREASE = (0.35, 3.9)      # crease between the steep lower front and the flatter upper front: z, y
GUN_RAMP_TOP = -0.5           # z where the flatter upper front meets the flat roof
GUN_FACET_Y = 2.9             # height where the outer corner facet of a cheek meets its side
GUN_SLOT = (0.6, -1.3)        # half-width of the gun slot, z of its back wall
GUN_HOOD = (0.0, 3.45)        # roof carried over the gun slot: front z, underside height
GUN_MANTLET = 0.8             # radius of the bronze cradle drum that fills the slot
GUN_AXIS = (0.35, 2.6)        # trunnions (cradle drum axis): z, height above the deck
GUN_REAR = (-2.85, (3.4, -2.8), (4.1, -2.15))    # rear face z; the bevel across the back of the roof (y, z) to (y, z)
GUN_BARREL = 7.1              # trunnion to muzzle (muzzle at B 19.0 on the profile)
GUN_ELEV = 3.0                # barrel elevation as built (deg)
SHIELD = (3.55, 1.35, 35.0)   # deck shield ring: radius, height, half-angle of the opening astern


def _side_x(y):
    """half-width of the upper body's leaning side face at height y."""
    (x0, y0), (x1, y1), _ = GUN_SIDE
    return x0 + (x1 - x0) * (y - y0) / (y1 - y0)


def _front_z(y):
    """z of the steep lower front face at height y (through the nose foot line and the crease)."""
    y0, zn = GUN_SKIRT[1], GUN_SKIRT[2][0][1]
    zc, yc = GUN_CREASE
    return zn + (zc - zn) * (y - y0) / (yc - y0)


def _gun_planes():
    """half-spaces of the skirt and of the upper body (local frame)."""
    ys0, y0, half = GUN_SKIRT
    (xn, zn), (xc, zc), (xs, zrc), (xr, zr) = half
    inside = (0.0, (y0 + GUN_ROOF) / 2, 0.0)
    X, Y, Z = np.eye(3)
    plan = [(Z, zn), (X, xs), (-X, xs), (-Z, -zr)]
    for sx in (1, -1):
        plan.append(plane_through((sx * xn, 0, zn), (sx * xc, 0, zc), (sx * xc, 1, zc), (0, 0, 0)))
        plan.append(plane_through((sx * xs, 0, zrc), (sx * xr, 0, zr), (sx * xr, 1, zr), (0, 0, 0)))
    skirt = plan + [(-Y, -ys0), (Y, y0)]
    (x0, _), (xk, yk), (x1, y1) = GUN_SIDE
    zk, yk_ = GUN_CREASE
    zt = GUN_RAMP_TOP
    zrear, (yb0, zb0), (yb1, zb1) = GUN_REAR
    ya = GUN_FACET_Y
    upper = plan + [(-Y, -y0), (Y, y1),
                    plane_through((0, y0, zn), (1, y0, zn), (0, yk_, zk), inside),           # steep lower front
                    plane_through((0, yk_, zk), (1, yk_, zk), (0, y1, zt), inside),           # flatter upper front
                    plane_through((0, yb0, zb0), (1, yb0, zb0), (0, yb1, zb1), inside)]      # bevel across the back
    for sx in (1, -1):
        upper.append(plane_through((sx * x0, y0, 0), (sx * x0, y0, 1), (sx * xk, yk, 0), inside))     # side
        upper.append(plane_through((sx * xk, yk, 0), (sx * xk, yk, 1), (sx * x1, y1, 0), inside))     # roof chamfer
        pa = (sx * _side_x(ya), ya, _front_z(ya))
        upper.append(plane_through((sx * xn, y0, zn), (sx * xc, y0, zc), pa, inside))               # corner facet
    return skirt, upper


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
    # skirt, cheeks either side of the gun slot, rear block, and the slot floor sloping down to the nose
    hs, zs = GUN_SLOT
    X, Y, Z = np.eye(3)
    skirt, upper = _gun_planes()
    plane_solid(c, c.paint, skirt, xf=W)
    plane_solid(c, c.paint, upper + [(-X, -hs), (-Z, -zs)], xf=W)
    plane_solid(c, c.paint, upper + [(X, -hs), (-Z, -zs)], xf=W)
    plane_solid(c, c.paint, upper + [(Z, zs)], xf=W)
    zh, yh = GUN_HOOD                                          # the roof carried over the slot (hood)
    plane_solid(c, c.paint, upper + [(X, hs), (-X, hs), (-Z, -zs), (Z, zh), (-Y, -yh)], xf=W)
    zn = half[0][1]
    chin = [(X, hs), (-X, hs), (-Y, -y1), (Z, zn), (-Z, -zs), (Y, y1 + 0.03)]
    plane_solid(c, c.paint, chin, xf=W)
    # details: bolted oval hatch in the nose, roof hatch and sight, vents, rear door and rungs, side hatches
    c.add(c.paint, rbox(0.42, 0.52, 0.03, r=0.18, seg=2, y0=-0.26), xf=M(W((0.0, 1.12, zn + 0.005)), R))
    c.add(c.sw('dark'), rbox(0.46, 0.56, 0.02, r=0.2, seg=2, y0=-0.28), xf=M(W((0.0, 1.12, zn - 0.002)), R))
    for k in range(10):                                        # bolts round the hatch
        aa = 2 * math.pi * k / 10
        c.add(c.sw('mid'), box(0.03, 0.03, 0.02, center=(0, 0, 0)),
              xf=M(W((0.25 * math.cos(aa), 1.12 + 0.31 * math.sin(aa), zn + 0.02)), R))
    c.add(c.sw('mid'), rbox(0.5, 0.06, 0.45, r=0.08, seg=1), xf=M(W((0.42, GUN_ROOF, -1.75)), R))
    c.add(c.sw('mid'), rbox(0.36, 0.26, 0.42, r=0.06, seg=1, bevel=0.03), xf=M(W((-0.45, GUN_ROOF, -1.7)), R))
    c.add(c.sw('glass'), box(0.26, 0.14, 0.03, center=(0, 0, 0)), xf=M(W((-0.45, GUN_ROOF + 0.15, -1.48)), R))
    for (x, z) in ((0.0, -2.0), (0.75, -1.0)):
        c.add(c.sw('mid'), cylinder(0.09, 0.1, seg=8), xf=M(W((x, GUN_ROOF, z))))
    zr = GUN_REAR[0]
    for kk in range(5):
        c.add(c.sw('dark'), box(0.45, 0.04, 0.05, center=(0, 0, 0)), xf=M(W((0.75, 0.95 + kk * 0.42, zr - 0.02)), R))
    c.add(c.sw('mid'), box(0.8, 1.5, 0.02, center=(0, 0, 0)), xf=M(W((-0.45, 1.3, zr - 0.005)), R))
    lean = math.atan((GUN_SIDE[0][0] - GUN_SIDE[1][0]) / (GUN_SIDE[1][1] - GUN_SIDE[0][1]))
    for sx in (1, -1):
        c.add(c.paint, rbox(0.03, 0.9, 0.8, r=0.1, seg=1, y0=-0.45), xf=M(W((sx * (_side_x(2.5) + 0.015), 2.5, -1.7)),
                                                                          R @ rot_z(sx * lean)))
    # elevating mass: cradle drum across the slot and the barrel (child node, pivot on the trunnions)
    elev = math.radians(GUN_ELEV)
    piv = W((0.0, GUN_AXIS[1], GUN_AXIS[0]))
    c.b.node(node + '_Gun', parent=node, translation=piv)
    dirv = R @ np.array([0.0, math.sin(elev), math.cos(elev)])
    rm = GUN_MANTLET
    Rg = R @ rot_x(-elev)
    c.add(c.sw('tan'), cylinder(rm, 2 * hs - 0.04, seg=28, caps=(True, True), y0=-(hs - 0.02)),
          xf=M(piv, Rg @ rot_z(math.pi / 2)))
    for sx in (-1, 1):                                         # trunnion bosses against the cheeks
        c.add(c.sw('mid'), cylinder(0.3, 0.06, seg=16, caps=(True, True), y0=0.0),
              xf=M(piv + Rg @ np.array([sx * (hs - 0.08), 0, 0]), Rg @ rot_z(-sx * math.pi / 2)))
    L = GUN_BARREL
    r0 = rm - 0.06
    prof = [(0.0, r0 - 0.1), (0.36, r0 - 0.1), (0.36, r0 + 0.3), (0.3, r0 + 0.38), (0.26, r0 + 0.42), (0.25, 2.35),
            (0.28, 2.4), (0.28, 2.55), (0.17, 2.65), (0.155, 4.55), (0.175, 4.6), (0.175, 4.75), (0.15, 4.8),
            (0.14, L - 0.62), (0.19, L - 0.58), (0.19, L - 0.06), (0.16, L), (0.0, L)]
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
# Combat module as fitted in the hangar-roof wells of Admiral Kasatonov (photographs at the 2019 naval show and
# Navy Day): a box pedestal on a bolted deck ring, a central housing with a ribbed front door, and the optronic
# head on a narrow neck, with a small sensor on a bracket at its rear right. A gun pod on each side holds the
# AO-18KD gun. Each pod is a long rounded block that
# pivots on trunnions behind the training axis, with a round bearing cover on its outer face. The black barrel
# casing leaves the front of the pod beside the housing, through a brass collar, and is held by a clamp ring on
# struts. Local frame: z along the guns, x to the module's left, y up from the deck ring.
PALASH_HOUSING = (0.42, -1.15, 0.35, 0.7, 1.5)      # central housing: half-width, rear z, front z, bottom and top height
PALASH_TRUNNION = (1.25, -0.45)                      # elevation axis through the gun pods: height, z
PALASH_POD = (0.43, 1.03, -0.6, 0.62, -0.6, 0.28)   # pod about the trunnions: inner/outer x, rear/front z, bottom/top y
PALASH_GUN = (0.9, -0.02, 1.6, 0.13)                 # barrel axis x and height (trunnion frame), casing length, radius
PALASH_HEAD = (-0.1, 1.68, 0.56, 0.6, 0.5)           # optronic head: centre z, base height, width, height, depth
PALASH_ELEV = 6.0                                    # gun elevation as built (deg)


def _rounded_zy(z0, z1, y0, y1, radii, seg=4):
    """convex rounded rectangle in the (z, y) plane, corner radii (front-bottom, front-top, rear-top, rear-bottom)."""
    corners = (((z1, y0), (-1, 1), -math.pi / 2), ((z1, y1), (-1, -1), 0.0), ((z0, y1), (1, -1), math.pi / 2),
               ((z0, y0), (1, 1), math.pi))
    pts = []
    for ((zc, yc), (dz, dy), a0), r in zip(corners, radii):
        cz, cy = zc + dz * r, yc + dy * r
        for k in range(seg + 1):
            a = a0 + (math.pi / 2) * k / seg
            pts.append((cz + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _x_prism(c, uvs, prof_zy, x0, x1, xf, node=None):
    """convex prism: a (z, y) profile extruded along x from x0 to x1, mapped to the world by xf."""
    a = [xf((x0, y, z)) for (z, y) in prof_zy]
    b = [xf((x1, y, z)) for (z, y) in prof_zy]
    polys = [a, b] + [[a[i], a[(i + 1) % len(a)], b[(i + 1) % len(a)], b[i]] for i in range(len(a))]
    _faces(c, uvs, polys, node=node)


def palash(c, pos, facing=0.0, node='Palash'):
    """3M89 Palash combat module in the gun-only fit of Project 22350 (see the notes above PALASH_HOUSING).
    About 2.3 m tall and 2.2 m wide across the gun pods."""
    o = np.asarray(pos, float)
    R = rot_y(facing)
    X, Y, Z = np.eye(3)

    def W(p):
        return o + R @ np.asarray(p, float)
    # fixed deck ring with its bolt circle (in the static Weapons node)
    c.add(c.paint, lathe([(1.0, 0.0), (1.0, 0.05), (0.95, 0.07), (0.86, 0.07), (0.86, 0.12), (0.0, 0.12)], seg=28), xf=M(o),
          node='Weapons')
    for k in range(16):
        aa = 2 * math.pi * (k + 0.5) / 16
        c.add(c.sw('mid'), cylinder(0.03, 0.035, seg=6), xf=M(o + np.array([0.93 * math.cos(aa), 0.06, 0.93 * math.sin(aa)])),
              node='Weapons')
    # rotating part (node pivot on the training axis): turntable, box pedestal with the buffer cylinder in front
    c.b.node(node, parent='Weapons', translation=o)
    c.add(c.sw('mid'), cylinder(0.8, 0.08, seg=24, caps=(False, True), y0=0.12), xf=M(o))
    c.add(c.paint, rbox(1.0, 0.5, 1.3, r=0.1, seg=2, bevel=0.03, y0=0.2), xf=M(W((0.0, 0.0, -0.45)), R))
    c.add(c.paint, cylinder(0.065, 0.62, seg=10, caps=(True, True), y0=-0.31), xf=M(W((0.0, 0.34, 0.27)), R @ rot_z(math.pi / 2)))
    for sx in (-0.31, 0.31):                                     # buffer end blocks
        c.add(c.paint, box(0.1, 0.14, 0.12, center=(0, 0, 0)), xf=M(W((sx, 0.34, 0.25)), R))
    # central housing: chamfered along the top front and top rear edges and down the front corners
    hx, zr, zf, yb, yt = PALASH_HOUSING
    housing = [(X, hx), (-X, hx), (Y, yt), (-Y, -yb), (Z, zf), (-Z, -zr),
               plane_through((0, yt, zf - 0.26), (1, yt, zf - 0.26), (0, yt - 0.18, zf), (0, 1.0, 0.0)),
               plane_through((0, yt, zr + 0.3), (1, yt, zr + 0.3), (0, yt - 0.3, zr), (0, 1.0, 0.0))]
    for sx in (1, -1):
        housing.append(plane_through((sx * (hx - 0.07), 0, zf), (sx * (hx - 0.07), 1, zf), (sx * hx, 0, zf - 0.07), (0, 1.0, 0.0)))
    plane_solid(c, c.paint, housing, xf=W)
    # front door: upper panel with four diagonal stiffeners, plain lower panel, two handles below the chamfer
    c.add(c.paint, box(0.5, 0.26, 0.025, center=(0, 0, 0)), xf=M(W((0.0, 1.17, zf + 0.012)), R))
    for x in (-0.18, -0.06, 0.06, 0.18):
        c.add(c.paint, box(0.025, 0.22, 0.025, center=(0, 0, 0)), xf=M(W((x, 1.17, zf + 0.035)), R @ rot_z(math.pi / 4)))
    c.add(c.paint, box(0.5, 0.24, 0.02, center=(0, 0, 0)), xf=M(W((0.0, 0.88, zf + 0.01)), R))
    for x in (-0.16, 0.16):
        c.add(c.sw('mid'), box(0.14, 0.03, 0.03, center=(0, 0, 0)), xf=M(W((x, 1.28, zf + 0.02)), R))
    # optronic head on a turntable: box with rounded corners, a shutter over the sensor windows and a sun visor,
    # and a small sensor on a bracket at its rear right corner
    hz, hy, hw, hh, hd = PALASH_HEAD
    c.add(c.paint, lathe([(0.3, yt), (0.3, yt + 0.035), (0.21, yt + 0.05), (0.21, hy - 0.03), (0.26, hy - 0.02), (0.26, hy),
                          (0.0, hy)], seg=24), xf=M(W((0.0, 0.0, hz))))
    c.add(c.paint, rbox(hw, hh, hd, r=0.14, seg=2, bevel=0.03, y0=hy), xf=M(W((0.0, 0.0, hz)), R))
    fz = hz + hd / 2
    c.add(c.sw('dark'), box(0.44, 0.42, 0.012, center=(0, 0, 0)), xf=M(W((0.0, hy + 0.29, fz + 0.004)), R))
    c.add(c.sw('mid'), box(0.4, 0.38, 0.02, center=(0, 0, 0)), xf=M(W((0.0, hy + 0.29, fz + 0.012)), R))
    for x in (-0.07, 0.07):
        c.add(c.sw('mid'), cylinder(0.015, 0.012, seg=6, caps=(True, False)), xf=M(W((x, hy + 0.32, fz + 0.022)), R @ rot_x(math.pi / 2)))
    c.add(c.paint, box(hw + 0.04, 0.03, 0.2, center=(0, 0, 0)), xf=M(W((0.0, hy + hh + 0.015, fz - 0.06)), R))
    bx, bz = -(hw / 2 + 0.06), hz - hd / 2 + 0.02               # bracket: a curved plate rising from the turntable
    c.add(c.paint, extrude_x([(bz - 0.2, -0.04), (bz + 0.06, -0.04), (bz + 0.06, 0.42), (bz - 0.06, 0.42), (bz - 0.2, 0.14)],
                             -0.02, 0.02), xf=M(W((bx, hy, 0.0)), R))
    c.add(c.paint, cylinder(0.075, 0.16, seg=12, caps=(True, True), y0=-0.08), xf=M(W((bx, hy + 0.4, bz)), R @ rot_x(math.pi / 2)))
    c.add(c.sw('glass'), cylinder(0.05, 0.01, seg=10, caps=(True, False), y0=0.08), xf=M(W((bx, hy + 0.4, bz)), R @ rot_x(math.pi / 2)))
    # elevating mass: the two gun pods and their barrels (child node, pivot on the trunnion axis)
    ty, tz = PALASH_TRUNNION
    piv = W((0.0, ty, tz))
    c.b.node(node + '_Guns', parent=node, translation=piv)
    el = math.radians(PALASH_ELEV)
    Re = R @ rot_x(-el)

    def G(p):
        return piv + Re @ np.asarray(p, float)
    d = Re @ np.array([0, 0, 1.0])
    xi, xo, pz0, pz1, py0, py1 = PALASH_POD
    gx, gy, gl, gr = PALASH_GUN
    prof = _rounded_zy(pz0, pz1, py0, py1, (0.1, 0.1, 0.44, 0.36), seg=5)
    for sx in (-1, 1):
        _x_prism(c, c.paint, prof, sx * xi, sx * xo, G)
        # bearing cover on the outer face with its bolt ring, trunnion boss against the housing
        c.add(c.paint, cylinder(0.3, 0.035, seg=24, caps=(False, True), y0=0.0), xf=M(G((sx * xo, 0, 0)), Re @ rot_z(-sx * math.pi / 2)))
        c.add(c.paint, cylinder(0.1, 0.07, seg=12, caps=(False, True), y0=0.0), xf=M(G((sx * xo, 0, 0)), Re @ rot_z(-sx * math.pi / 2)))
        for k in range(12):
            aa = 2 * math.pi * k / 12
            c.add(c.sw('mid'), cylinder(0.018, 0.05, seg=6, caps=(False, True), y0=0.0),
                  xf=M(G((sx * xo, 0.25 * math.sin(aa), 0.25 * math.cos(aa))), Re @ rot_z(-sx * math.pi / 2)))
        c.add(c.sw('mid'), cylinder(0.2, 0.02, seg=16, caps=(False, False), y0=0.0), xf=M(G((sx * hx, 0, 0)), Re @ rot_z(-sx * math.pi / 2)))
        # cooling cylinder at the rear outer corner, feed cover on top
        c.add(c.paint, cylinder(0.055, 0.36, seg=10, caps=(True, True), y0=-0.18),
              xf=M(G((sx * (xo + 0.05), 0.0, pz0 + 0.18)), Re @ rot_x(-0.35)))
        c.add(c.paint, rbox(0.3, 0.06, 0.4, r=0.04, seg=1), xf=M(G((sx * (xi + xo) / 2, py1, 0.1)), Re))
        c.add(c.paint, box(0.3, 0.26, 0.12, center=(0, 0, 0)), xf=M(G((sx * (gx - 0.1), py0 + 0.2, pz1 + 0.05)), Re))
        c.add(c.paint, cylinder(0.06, 0.3, seg=10, caps=(True, True), y0=0.0), xf=M(G((sx * (xi + 0.12), py0 + 0.32, pz1)), axis_frame(d)))
        # barrel: brass collar at the pod front, black casing, clamp ring, muzzle with six bores
        b0 = G((sx * gx, gy, pz1))
        c.add(c.sw('tan'), cylinder(0.165, 0.12, seg=18, caps=(True, True), y0=-0.02), xf=M(b0, axis_frame(d)))
        c.add(c.sw('black'), lathe([(0.0, 0.1), (gr, 0.1), (gr, gl - 0.05), (gr - 0.015, gl), (0.0, gl)], seg=18),
              xf=M(b0, axis_frame(d)))
        for k in range(6):
            aa = 2 * math.pi * k / 6
            c.add(c.sw('dark'), cylinder(0.022, 0.02, seg=6, caps=(True, False)),
                  xf=M(G((sx * gx + 0.065 * math.cos(aa), gy + 0.065 * math.sin(aa), pz1 + gl)), axis_frame(d)))
        zc = pz1 + 1.0
        c.add(c.paint, cylinder(gr + 0.035, 0.12, seg=18, caps=(True, True), y0=-0.06), xf=M(G((sx * gx, gy, zc)), axis_frame(d)))
        for p0, p1 in (((sx * gx, gy + gr + 0.03, zc), (sx * (xo - 0.2), py1, pz1 - 0.1)),
                       ((sx * gx, gy + gr + 0.03, zc), (sx * (xo - 0.15), py1, pz0 + 0.5)),
                       ((sx * (gx - gr - 0.03), gy - 0.05, zc), (sx * (xi + 0.1), py0 + 0.25, pz1))):
            c.add(c.paint, tube_path([G(p0), G(p1)], 0.025, seg=5))


# =============================================================================================
# decoy launchers and small arms
# =============================================================================================
def _launcher_pedestal(c, o, R, ped):
    """box pedestal narrowing upward, with a round access plate on its front face and a turntable on top."""
    bot = [(-0.45, -0.42), (0.45, -0.42), (0.45, 0.42), (-0.45, 0.42)]
    top = [(-0.36, -0.32), (0.36, -0.32), (0.36, 0.32), (-0.36, 0.32)]
    c.add(c.paint, prism(bot, 0.0, ped, top=True, top_poly=top), xf=M(o, R))
    zf = 0.42 - 0.1 * 0.45
    c.add(c.paint, cylinder(0.2, 0.02, seg=12, caps=(True, False)), xf=M(o + R @ np.array([0.0, ped * 0.45, zf]),
                                                                         R @ rot_x(math.pi / 2 - 0.1)))
    c.add(c.sw('mid'), cylinder(0.38, 0.08, seg=16, caps=(True, True), y0=ped), xf=M(o))


def kt216(c, pos, facing=0.0, el=30.0, rows=2, cols=5, r=0.065, length=1.6, ped=1.0):
    """KT-216 decoy launcher (2018-2019 photographs): a box pedestal narrowing upward with a round access plate,
    a yoke and a bolted cradle carrying 2 x 5 long tubes that pass right through it, white caps on the
    muzzles; trained along `facing` (yaw about +y, 0 = +z) and fixed at `el` degrees."""
    o = np.asarray(pos, float)
    R = rot_y(facing)
    _launcher_pedestal(c, o, R, ped)
    for s in (-1, 1):                                          # yoke
        c.add(c.paint, box(0.06, 0.5, 0.36, center=(0, 0, 0)), xf=M(o + R @ np.array([s * 0.36, ped + 0.33, 0]), R))
    Re = R @ rot_x(-math.radians(el))
    piv = o + np.array([0, ped + 0.45, 0])
    d = Re @ np.array([0, 0, 1.0])
    c.add(c.paint, box(0.66, 0.42, 0.62, center=(0, 0, 0)), xf=M(piv, Re))
    sp = 2 * r + 0.008
    for i in range(cols):
        for j in range(rows):
            off = Re @ np.array([(i - (cols - 1) / 2) * sp, (j - (rows - 1) / 2) * sp, 0])
            a = piv + off - d * 0.55
            c.add(c.sw('mid'), cylinder(r, length, seg=8, caps=(True, False)), xf=M(a, axis_frame(d)))
            c.add(c.sw('white'), disc(r * 0.92, seg=8), xf=M(a + d * (length + 0.004), axis_frame(d)))


def covered_mount(c, pos, facing=0.0, ped=1.0):
    """launcher kept under a rounded hood on the same box pedestal (front of the Redut field, 2019 photograph)."""
    o = np.asarray(pos, float)
    R = rot_y(facing)
    _launcher_pedestal(c, o, R, ped)
    c.add(c.paint, rbox(0.8, 0.55, 1.2, r=0.3, seg=2, bevel=0.14, y0=ped + 0.08), xf=M(o, R))


def mtpu(c, pos, facing=0.0):
    """14.5 mm MTPU pedestal machine gun (Navy Day 2019 photograph): light grey post and cradle with an
    ammunition box on each side, black KPVT barrel in its perforated jacket."""
    o = np.asarray(pos, float)
    R = rot_y(facing)
    c.add(c.sw('light'), cylinder(0.2, 0.06, seg=10), xf=M(o))
    c.add(c.sw('light'), cylinder(0.07, 0.95, seg=8), xf=M(o))
    piv = o + np.array([0, 1.0, 0])
    Re = R @ rot_x(-math.radians(20.0))
    c.add(c.sw('light'), box(0.16, 0.18, 0.95, center=(0, 0.02, 0.1)), xf=M(piv, Re))
    for s in (-1, 1):
        c.add(c.sw('light'), box(0.22, 0.26, 0.36, center=(0, -0.1, 0)), xf=M(piv + Re @ np.array([s * 0.22, 0, -0.05]), Re))
    d = Re @ np.array([0, 0, 1.0])
    c.add(c.sw('black'), cylinder(0.045, 0.5, seg=8), xf=M(piv + d * 0.55, axis_frame(d)))
    c.add(c.sw('black'), cylinder(0.022, 0.75, seg=6), xf=M(piv + d * 1.05, axis_frame(d)))


def register(m):
    m.alloc('redut_top', 96, 160, paint_redut_top)
    m.alloc('uksk_top', 256, 128, paint_uksk_top)
