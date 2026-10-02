"""Admiral Gorshkov-class (Project 22350) -- superstructure, weapons, sensors and markings.

Positions were measured on a 1:500 general-arrangement drawing (registered to the 135.0 m overall length)
and checked against photographs of Admiral Gorshkov (2018-2023) and Admiral Kasatonov.
B = metres aft of the stem head, x = metres to port, y = metres above the design waterline.
"""
import math
import numpy as np

from meshkit import *
from meshkit import _quad
import atlas as st
from texkit import srgb, value_noise
from hull import (zB, KNUCKLE, DECK_HB, TUMBLE, upper_hb, deck_y, section, HELIDECK_B, TRANSOM_TOP, BULWARK_TOP,
                  PALASH_WELL)
from kit import *
import model as S2
import markings as MK
import parts as PT
from parts import rbox
import weapons as WP
import sensors as SN

PAL = st.PAL

PENNANT = '454'
SHIP_NAME = 'АДМИРАЛ ФЛОТА СОВЕТСКОГО СОЮЗА ГОРШКОВ'
AO = dict(max_len=4.0, min_area=2.5, rays=64, max_dist=6.0, floor=0.25, gamma=2.0)

# principal levels (m above the DWL)
UKSK_Y = 8.1          # top of the raised UKSK block
DECK01_Y = 10.0       # 01 deck (bridge wings, funnel deck)
HANGAR_Y = 12.2       # hangar roof
SLOPE_Y = 11.3        # top of the sloped wheelhouse front = foot of the window band
WIN_Y = 12.6          # top of the window band
ROOF_Y = 12.9         # wheelhouse / mast-house roof
PLATFORM_Y = 22.6     # radar platform at the top of the tower


def dk(B, x=0.0):
    return float(deck_y(B, x))


def hw_at(B, y):
    """half-breadth of the hull side plane (upper hull) at B, height y."""
    return upper_hb(B, y)


def lean_poly(B0, B1, y, inset=0.0, nB=2):
    """symmetric outline following the hull side plane at height y between B0 and B1 (inset metres)."""
    half = [(B, hw_at(B, y) - inset) for B in np.linspace(B0, B1, nB)]
    return sym_poly(half)


def quad_faces(c, uvs, pts, facing):
    """single quad (4 points) turned to face `facing`."""
    P = np.array([np.asarray(p, float) for p in pts])
    I = S2._outward(P, np.array([(0, 1, 2), (0, 2, 3)]), facing)
    n = normalize(np.cross(P[I[0, 1]] - P[I[0, 0]], P[I[0, 2]] - P[I[0, 0]]))
    c.add(uvs, (P, np.tile(n, (4, 1)), np.stack([P[:, 2] + P[:, 0], P[:, 1]], axis=1), I))


# =============================================================================================
# atlas decals
# =============================================================================================
NAME_H = 0.75         # letter height of the ship's name on the quarter (m)
PENNANT_B = 34.5      # hull number centre (photos and drawing: 417 / 461 centred near B 34-36)
PENNANT_H = 3.0


def paint_helideck(L, rect):
    """22350 helideck (u = across, port at u=0; v = from the hangar aft): white touchdown circle with the
    deck-lock grid, transverse line ahead of it, lead-in line on the centreline, side lines."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['heli'], alpha=1.0, rough=0.82, metal=0.0)
    n = value_noise(h, w, 24, seed=31, octaves=4)
    L.col[y0:y1, x0:x1] *= (0.9 + 0.16 * n)[..., None]
    Lb = TRANSOM_TOP[0] - HELIDECK_B
    W_ = 16.2

    def px(xm, Bm):
        return ((xm + W_ / 2) / W_ * w, (Bm - HELIDECK_B) / Lb * h)

    def fn(d, s):
        lw = max(2, int(0.28 / W_ * w * s))
        cx, cy = px(0.0, 126.3)
        r = 4.3 / W_ * w
        d.ellipse([(cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s], outline=255, width=lw)
        r2 = 1.6 / W_ * w
        d.ellipse([(cx - r2) * s, (cy - r2) * s, (cx + r2) * s, (cy + r2) * s], outline=255, width=lw)
        ya = px(0, 121.6)[1]
        d.line([(px(-5.8, 0)[0] * s, ya * s), (px(5.8, 0)[0] * s, ya * s)], fill=255, width=lw)
        for k in range(5):
            yb0 = px(0, 114.6 + k * 1.3)[1]
            d.line([(cx * s, yb0 * s), (cx * s, (yb0 + 0.7 / Lb * h) * s)], fill=255, width=lw)
        for sx in (-1, 1):
            xx = px(sx * 7.6, 0)[0]
            d.line([(xx * s, px(0, 114.2)[1] * s), (xx * s, px(0, 134.4)[1] * s)], fill=255, width=max(1, lw // 2))
    L.mask_apply(L.draw_mask(w, h, fn), col=PAL['white'] * 0.95, x0=x0, y0=y0, add_height=0.2)

    def grid(d, s):
        cx, cy = px(0.0, 126.3)
        r = 1.25 / W_ * w
        d.ellipse([(cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s], fill=255)
    L.mask_apply(L.draw_mask(w, h, grid), col=PAL['dark'] * 0.9, x0=x0, y0=y0, add_height=-0.3)
    cx, cy = px(0.0, 126.3)
    for k in range(-6, 7):
        xx = int(x0 + cx + k * 2.2)
        L.rect(xx, y0 + cy - 1.2 / Lb * h, xx + 1, y0 + cy + 1.2 / Lb * h, col=PAL['steel'] * 0.8, add_height=0.4)


def paint_name(L, rect):
    MK.paint_text(L, rect, SHIP_NAME, PAL['white'] * 0.95, size_frac=0.78, spacing=8)


def paint_side_door(L, rect):
    """hinged weathertight door flush with a stealth wall: recessed outline, two hinges, handle."""
    x0, y0, x1, y1 = rect
    h = y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['super'], alpha=1.0)
    L.rect(x0 + 2, y0 + 2, x1 - 2, y1 - 2, col=PAL['super'] * 0.6, add_height=-0.7)
    L.rect(x0 + 4, y0 + 4, x1 - 4, y1 - 4, col=PAL['super'] * 1.03, add_height=0.3)
    for f in (0.2, 0.8):
        L.rect(x0 + 4, y0 + h * f - 3, x0 + 9, y0 + h * f + 3, col=PAL['super'] * 0.7, add_height=0.6)
    L.rect(x1 - 12, y0 + h * 0.5 - 2, x1 - 6, y0 + h * 0.5 + 2, col=PAL['dark'], add_height=0.6)


def paint_grille(L, rect):
    """air intake grille: framed panel of fine horizontal louvres."""
    x0, y0, x1, y1 = rect
    L.rect(x0, y0, x1, y1, col=PAL['super'] * 0.7, alpha=1.0)
    for yy in range(y0 + 4, y1 - 4, 3):
        L.rect(x0 + 4, yy, x1 - 4, yy + 1, col=PAL['super'] * 0.35, add_height=-0.8)
    for e in ((x0, y0, x1, y0 + 3), (x0, y1 - 3, x1, y1), (x0, y0, x0 + 3, y1), (x1 - 3, y0, x1, y1)):
        L.rect(*e, col=PAL['super'] * 0.85, add_height=0.6)


def paint_pk_door(L, rect):
    """Paket-NK torpedo-tube shutter on the hangar side: large flush panel in a recessed seam."""
    x0, y0, x1, y1 = rect
    w = x1 - x0
    L.rect(x0, y0, x1, y1, col=PAL['super'], alpha=0.0)
    for e in ((x0, y0, x1, y0 + 2), (x0, y1 - 2, x1, y1), (x0, y0, x0 + 2, y1), (x1 - 2, y0, x1, y1)):
        L.rect(*e, col=PAL['super'] * 0.55, alpha=1.0, add_height=-0.8)
    for f in (0.25, 0.5, 0.75):
        L.rect(x0 + w * f, y0, x0 + w * f + 2, y1, col=PAL['super'] * 0.7, alpha=1.0, add_height=-0.5)
    a = L.alpha[y0:y1, x0:x1]
    L.alpha[y0:y1, x0:x1] = np.where(a > 0.5, 1.0, 0.0)


def paint_hawse(L, rect):
    """anchor pocket seen from outside: dark recess with a rounded top and rust run-off below."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['hull'], alpha=0.0)

    def fn(d, s):
        d.rounded_rectangle([0.06 * w * s, 0.04 * h * s, 0.94 * w * s, 0.62 * h * s], radius=int(0.25 * h * s), fill=255)
    m = L.draw_mask(w, h, fn)
    L.mask_apply(m, col=PAL['black'] * 1.6, alpha=1.0, add_height=-1.0, x0=x0, y0=y0)
    for k in range(int(w * 0.3), int(w * 0.7)):
        ln = int(h * 0.35)
        strk = np.linspace(0.35, 0.0, ln)[:, None]
        L.mask_apply(strk * (1 - abs(k - w / 2) / (w * 0.2)), col=srgb('6a4a35'), alpha=1.0, x0=x0 + k, y0=y0 + int(h * 0.62))
    a = L.alpha[y0:y1, x0:x1]
    L.alpha[y0:y1, x0:x1] = np.where(a > 0.3, 1.0, 0.0)


def register_decals(m):
    m.alloc('helideck', 384, 512, paint_helideck)
    m.alloc('pennant', 384, 160, lambda L, r: MK.paint_pennant(L, r, PENNANT, PAL['white'], outline=PAL['black'] * 1.2))
    name_px = MK.text_width_px(SHIP_NAME, 64, 0.78, 8) + 16
    m.name_w = NAME_H * name_px / 64.0
    m.alloc('name', name_px, 64, paint_name)
    m.alloc('vds_door', 96, 96, MK.paint_vds_door)
    m.alloc('door', 48, 96, MK.paint_door)
    m.alloc('sdoor', 48, 96, paint_side_door)
    m.alloc('vent', 96, 48, MK.paint_vent)
    m.alloc('louvre', 128, 96, MK.paint_louvre_panel)
    m.alloc('grille', 160, 72, paint_grille)
    m.alloc('pk_door', 320, 64, paint_pk_door)
    m.alloc('hawse', 64, 64, paint_hawse)
    m.alloc('raft', 64, 128, MK.paint_raft)
    m.alloc('ensign', 96, 64, MK.paint_ensign)
    m.alloc('hangar_door', 192, 160, MK.paint_hangar_door)
    m.alloc('win14', 672, 40, lambda L, r: MK.paint_window_row(L, r, 14, frac=0.78))
    m.alloc('win2', 96, 40, lambda L, r: MK.paint_window_row(L, r, 2, frac=0.78))
    m.alloc('win1', 48, 40, lambda L, r: MK.paint_window_row(L, r, 1, frac=0.78))
    WP.register(m)
    SN.register(m)


def searchlight(c, pos, facing=0.0):
    """searchlight on a short post: drum with a glass front on a yoke."""
    o = np.asarray(pos, float)
    R = rot_y(facing)
    c.add(c.paint, cylinder(0.08, 0.55, seg=8), xf=M(o))
    piv = o + np.array([0, 0.75, 0])
    c.add(c.paint, cylinder(0.24, 0.5, seg=12, caps=(True, True), y0=-0.25), xf=M(piv, R @ rot_x(math.pi / 2)))
    c.add(c.sw('glass'), cylinder(0.2, 0.02, seg=12, caps=(True, False), y0=0.25), xf=M(piv, R @ rot_x(math.pi / 2)))


# =============================================================================================
# hull markings and fittings
# =============================================================================================
def hull_decal(c, name, Bc, yc, w, h, side, nseg=6, offset=0.04):
    """decal strip following the main hull side (constant height band), readable from outside."""
    P, UV, I = [], [], []
    for k in range(nseg + 1):
        t = k / nseg
        B = Bc + (t - 0.5) * w if side > 0 else Bc - (t - 0.5) * w
        for j, yy in enumerate((yc - h / 2, yc + h / 2)):
            xs, ys = section(B, 64)
            xh = float(np.interp(yy, ys, xs)) + offset
            P.append((side * xh, yy, zB(B)))
            UV.append((t, 1.0 - j))
    for k in range(nseg):
        a = 2 * k
        I += [(a, a + 2, a + 3), (a, a + 3, a + 1)]
    P = np.array(P); UV = np.array(UV); I = np.array(I)
    N = compute_smooth_normals(P, I)
    if np.dot(N[0], np.array([side, 0.0, 0.0])) < 0:
        I = I[:, ::-1]; N = -N
    c.add(c.rect(name), (P, N, UV, I))


def wall_decal(c, name, Bc, yc, w, h, side, offset=0.03):
    """decal on the inward-leaning upper hull plane (above the knuckle)."""
    P = []
    for (dB, dy) in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
        B = Bc + dB * w * (-1 if side > 0 else 1)
        y = yc + dy * h
        P.append((side * (hw_at(B, y) + offset), y, zB(B)))
    c.add(c.rect(name), _quad(*[np.array(p) for p in P], uv=[(0, 1), (1, 1), (1, 0), (0, 0)],
                              n=normalize(np.array([side, TUMBLE, 0.0]))))


def hull_fittings(m):
    c = m.ctx
    m.b.node('Hull')
    for s in (1, -1):
        hull_decal(c, 'pennant', PENNANT_B, 2.55, PENNANT_H * 384 / 160, PENNANT_H, s)
        hull_decal(c, 'name', 123.5, 3.5, m.name_w, NAME_H, s)
        hull_decal(c, 'hawse', 3.3, 5.75, 1.7, 1.7, s, nseg=2, offset=0.03)
        # anchor stowed in the pocket: shank along the hull, crown and flukes at the bottom
        xs, ys = section(3.3, 48)
        p0 = np.array([s * (float(np.interp(5.6, ys, xs)) + 0.04), 5.95, zB(3.3)])
        c.add(c.sw('dark'), box(0.16, 0.75, 0.2, center=(0, -0.3, 0)), xf=M(p0))
        c.add(c.sw('dark'), box(0.22, 0.18, 0.95, center=(0, -0.72, 0)), xf=M(p0))
        # grilles, doors and the Paket-NK shutters on the upper hull
        wall_decal(c, 'grille', 85.6, 6.4, 3.4, 1.5, s)
        wall_decal(c, 'grille', 69.6, 6.1, 1.6, 0.9, s)
        wall_decal(c, 'grille', 108.5, 6.0, 1.6, 0.9, s)
        wall_decal(c, 'pk_door', 103.2, 8.85, 10.4, 1.9, s)
        for B in (64.2, 79.6, 94.6):
            wall_decal(c, 'sdoor', B, (DECK01_Y if B > 61 else UKSK_Y) - 1.05, 0.8, 1.8, s)
        wall_decal(c, 'sdoor', 47.5, 7.0, 0.8, 1.6, s)


# =============================================================================================
# forecastle: ground tackle, breakwater, A-192M, Redut, decoys in the bulwark cut-outs
# =============================================================================================
GUN_B = 27.4
REDUT = ((37.3, 1.45), (37.3, -1.45), (41.95, 1.45), (41.95, -1.45))


def forecastle(m):
    c = m.ctx
    m.b.node('Forecastle')
    # windlasses, chain stoppers and the cables running to the hawse pipes
    for s in (1, -1):
        PT.windlass(c, P3(12.6, s * 1.55, dk(12.6, 1.55)), s)
        PT.chain_stopper(c, P3(9.6, s * 1.75, dk(9.6, 1.75)), (0, 0, 1))
        a = P3(12.0, s * 1.6, dk(12.0, 1.6) + 0.12)
        b = P3(5.2, s * 2.35, dk(5.2, 2.35) + 0.08)
        c.add(c.sw('dark'), beam(a, b, 0.22, 0.12))
        c.add(c.sw('dark'), cylinder(0.38, 0.12, seg=12), xf=M(P3(4.9, s * 2.35, dk(4.9, 2.35))))
        for B in (6.8, 16.0, 33.5):
            xb = hw_at(B, float(KNUCKLE(B))) - 0.75
            bollard(c, P3(B, s * xb, dk(B, xb)))
    # breakwater: V-plate across the deck with stiffeners on its aft face
    apex, endB, endx, hb = 18.7, 17.3, 4.0, 0.9
    for s in (1, -1):
        a = P3(apex, 0.0, dk(apex)); b = P3(endB, s * endx, dk(endB, endx))
        quad_faces(c, c.paint, [a, b, b + np.array([0, hb, 0]), a + np.array([0, hb, 0])], (0, 0, 1))
        quad_faces(c, c.paint, [a, b, b + np.array([0, hb, 0]), a + np.array([0, hb, 0])], (0, 0, -1))
        for t in (0.25, 0.5, 0.75):
            p = a + (b - a) * t
            c.add(c.paint, beam(p - np.array([0, 0, 0.02]), p - np.array([0, 0, 0.45]) + np.array([0, -0.0, 0]), 0.04, hb * 1.9))
    # vertical stiffeners on the inside of the bulwark
    t = S2.BULWARK_T
    for B in np.arange(3.0, S2.OPENING[0] - 0.3, 1.25):
        yb, yt = dk(B, hw_at(B, float(KNUCKLE(B))) - t), float(BULWARK_TOP(B)) - 0.12
        for s in (1, -1):
            pb = P3(B, s * (hw_at(B, yb) - t - 0.1), yb)
            pt_ = P3(B, s * (hw_at(B, yt) - t - 0.06), yt)
            c.add(c.paint, beam(pb, pt_, 0.07, 0.2, up=(0, 0, 1)))
    # liferaft canisters on brackets inside the bulwark, life rings on its top edge
    for s in (1, -1):
        for B in (6.8, 8.4, 10.0):
            xb = hw_at(B, 7.1) - S2.BULWARK_T - 0.36
            c.add(c.rect('raft'), lathe(RAFT_PROF, seg=10, uv_rect=(0.0, 0.0, 1.0, 1.0)),
                  xf=M(P3(B, s * xb, 7.1), rot_x(math.pi / 2)))
        for B in (17.5, 22.0):
            yt = float(BULWARK_TOP(B))
            c.add(c.sw('orange'), lathe([(0.24, -0.06), (0.36, -0.06), (0.36, 0.06), (0.24, 0.06)], seg=12, close_top=False),
                  xf=M(P3(B, s * (hw_at(B, yt - 0.45) - S2.BULWARK_T - 0.08), yt - 0.45), rot_z(math.pi / 2)))
    # jackstaff on the stem
    c.add(c.paint, tube_path([P3(0.9, 0, float(BULWARK_TOP(0.9))), P3(0.9, 0, float(BULWARK_TOP(0.9)) + 2.4)], 0.05, seg=5))
    # gun, Redut modules, decoy launchers in the bulwark cut-outs
    m.b.node('Weapons')
    WP.a192m(c, GUN_B, dk(GUN_B))
    m.b.node('Weapons')
    for (B, x) in REDUT:
        WP.redut_module(c, B, x, dk(B, abs(x)))
    for s in (1, -1):
        xb = hw_at(41.4, 6.2) - 0.9
        WP.kt216(c, P3(41.4, s * xb, dk(41.4, xb)), facing=s * math.pi / 2, el=40.0, rows=2, cols=3, r=0.11, length=1.5)
    # deck vents and lockers by the Redut field
    m.b.node('Forecastle')
    for s in (1, -1):
        PT.round_vent(c, P3(34.2, s * 3.4, dk(34.2, 3.4)), r=0.3, h=0.5)
        PT.ready_locker(c, P3(44.0, s * 5.0, dk(44.0, 5.0)), w=1.1, h=0.9, d=0.55, yaw=math.pi)


# =============================================================================================
# UKSK block, wheelhouse
# =============================================================================================
UKSK_B0, UKSK_B1 = S2.UKSK_FRONT, 61.0
BR_FRONT = 56.2       # foot of the sloped wheelhouse front on the UKSK block
BR_SLOPE_B = 58.2     # top of the slope
BR_AFT = 61.5         # aft end of the full-width wheelhouse
CF_LO, CF_HI = 1.1, 0.9


def wh_half(y, Bf, cf, aft=BR_AFT, out=0.0):
    """half outline of the wheelhouse at height y with front at Bf and a corner chamfer cf."""
    return [(Bf, hw_at(Bf, y) - cf + out), (Bf + cf, hw_at(Bf + cf, y) + out), (aft, hw_at(aft, y) + out)]


def uksk_block(m):
    c = m.ctx
    m.b.node('Superstructure')
    yk = float(KNUCKLE(UKSK_B0))
    xb = hw_at(UKSK_B0, yk) - S2.BULWARK_T
    quad_faces(c, c.paint, [P3(UKSK_B0, -xb, yk), P3(UKSK_B0, xb, yk), P3(UKSK_B0, hw_at(UKSK_B0, UKSK_Y), UKSK_Y),
                            P3(UKSK_B0, -hw_at(UKSK_B0, UKSK_Y), UKSK_Y)], (0, 0, 1))
    for s in (1, -1):
        door(c, UKSK_B0, s * 3.2, yk + 0.05, (0, 0, 1))
        ladder(c, P3(UKSK_B0 - 0.05, s * 5.2, yk), P3(UKSK_B0 - 0.05, s * 5.2, UKSK_Y), width=0.55, normal=(0, 0, 1))
    top = lean_poly(UKSK_B0, BR_FRONT + 0.4, UKSK_Y, nB=4)
    P, N, UV, I = cap_polygon(np.asarray(poly_Bx_to_xz(top)), UKSK_Y, up=True)
    c.add(c.deck, (P, N, np.stack([P[:, 2], P[:, 0]], axis=1), I))
    m.b.node('Weapons')
    WP.uksk_field(c, 47.0, 51.6, 2.35, UKSK_Y)
    for s in (1, -1):
        for B in (52.9, 54.7):
            WP.kt216(c, P3(B, s * 6.3, UKSK_Y), facing=s * math.pi / 2, el=45.0)
    m.b.node('Superstructure')
    # railings along the block edges and across its front
    for s in (1, -1):
        railing_pts(c, [P3(UKSK_B0 + 0.1, s * (hw_at(UKSK_B0, UKSK_Y) - 0.08), UKSK_Y),
                        P3(BR_FRONT + 0.6, s * (hw_at(BR_FRONT, UKSK_Y) - 0.08), UKSK_Y)])
    railing_pts(c, [P3(UKSK_B0 + 0.08, hw_at(UKSK_B0, UKSK_Y) - 0.1, UKSK_Y), P3(UKSK_B0 + 0.08, 1.0, UKSK_Y)])
    railing_pts(c, [P3(UKSK_B0 + 0.08, -1.0, UKSK_Y), P3(UKSK_B0 + 0.08, -hw_at(UKSK_B0, UKSK_Y) + 0.1, UKSK_Y)])


def wheelhouse(m):
    c = m.ctx
    m.b.node('Superstructure')
    lo = sym_poly(wh_half(UKSK_Y, BR_FRONT, CF_LO))
    hi = sym_poly(wh_half(SLOPE_Y, BR_SLOPE_B, CF_HI))
    house(c, lo, UKSK_Y, SLOPE_Y, top_poly_Bx=hi, top=None)
    wlo = sym_poly(wh_half(SLOPE_Y, BR_SLOPE_B, CF_HI))
    whi = sym_poly(wh_half(WIN_Y, BR_SLOPE_B - 0.25, CF_HI))
    house(c, wlo, SLOPE_Y, WIN_Y, top_poly_Bx=whi, top=None)
    roof = sym_poly(wh_half(WIN_Y, BR_SLOPE_B - 0.45, CF_HI + 0.1, out=0.12))
    house(c, roof, WIN_Y, ROOF_Y, top='paint', bottom=True)
    # window rows: front (14), corner chamfers (1 each), sides (2 each)
    yb, yt = SLOPE_Y + 0.25, WIN_Y - 0.15

    hb = wh_half(SLOPE_Y, BR_SLOPE_B, CF_HI)
    ht = wh_half(WIN_Y, BR_SLOPE_B - 0.25, CF_HI)

    def at(i, f, y):
        # point on the band's outline vertex i of the half outline, at height y (linear in y)
        t = (y - SLOPE_Y) / (WIN_Y - SLOPE_Y)
        B = hb[i][0] + (ht[i][0] - hb[i][0]) * t
        x = hb[i][1] + (ht[i][1] - hb[i][1]) * t
        return np.array([x, y, zB(B)])
    fr = [at(0, 0, yb), at(0, 0, yt)]
    SN.face_quad(c, c.rect('win14'), [fr[0] * [-1, 1, 1], fr[0], fr[1], fr[1] * [-1, 1, 1]], off=0.025)
    for s in (1, -1):
        mir = np.array([s, 1, 1])
        a0, a1 = at(0, 0, yb) * mir, at(1, 0, yb) * mir
        b0, b1 = at(0, 0, yt) * mir, at(1, 0, yt) * mir
        corners = [a0, a1, b1, b0] if s > 0 else [a1, a0, b0, b1]
        SN.face_quad(c, c.rect('win1'), corners, off=0.025)
        a0, a1 = at(1, 0, yb) * mir, at(2, 0, yb) * mir
        b0, b1 = at(1, 0, yt) * mir, at(2, 0, yt) * mir
        g = lambda p, q, f: p + (q - p) * f
        corners = [g(a0, a1, 0.1), g(a0, a1, 0.85), g(b0, b1, 0.85), g(b0, b1, 0.1)]
        if s < 0:
            corners = [corners[1], corners[0], corners[3], corners[2]]
        SN.face_quad(c, c.rect('win2'), corners, off=0.025)
    # roof fittings: Puma radome, navigation radars, whips, searchlights, railing
    m.b.node('Sensors')
    SN.radome(c, P3(61.2, 0.0, ROOF_Y), 1.8, ped_h=0.32, ped_r=1.15)
    for s in (1, -1):
        SN.pal_n(c, P3(58.7, s * 4.6, ROOF_Y), yaw=s * 0.4)
        whip(c, P3(58.4, s * 6.4, ROOF_Y), 6.0, r=0.05)
        whip(c, P3(62.5, s * 2.2, ROOF_Y), 6.3, r=0.05)
        PT.lamp(c, P3(58.3, s * 7.0, ROOF_Y), 'white')
        searchlight(c, P3(60.4, s * 6.6, ROOF_Y), facing=s * 0.6)
    m.b.node('Superstructure')
    rp = wh_half(ROOF_Y, BR_SLOPE_B - 0.45, CF_HI + 0.1, out=0.05)
    pts = [P3(B, x, ROOF_Y) for (B, x) in rp[::-1]] + [P3(B, -x, ROOF_Y) for (B, x) in rp]
    railing_pts(c, pts)


# =============================================================================================
# mast house, bridge wings, tower with the Poliment faces, radar platform
# =============================================================================================
MH = (61.5, 75.6, 4.2)                    # mast house B0, B1, half-width at the 01 deck
LOWER = (66.3, 74.4, 3.25, 17.0)          # lower tower block: B0, B1, half-width, top
DIAMOND_B = 70.6                          # upper tower: square pyramid turned 45 deg (faces at +-45 deg)
DIAMOND_HD = (3.0, 1.85)                  # half-diagonal at the foot (LOWER top) and at the platform
ARRAY_Y, ARRAY_W, ARRAY_H = 19.7, 2.6, 3.6


def diamond(y):
    t = (y - LOWER[3]) / (PLATFORM_Y - LOWER[3])
    hd = DIAMOND_HD[0] + (DIAMOND_HD[1] - DIAMOND_HD[0]) * t
    return [(DIAMOND_B - hd, 0.0), (DIAMOND_B, hd), (DIAMOND_B + hd, 0.0), (DIAMOND_B, -hd)]


def diamond_pt(k, y):
    B, x = diamond(y)[k]
    return np.array([x, y, zB(B)])


def mast(m):
    c = m.ctx
    m.b.node('Superstructure')
    B0, B1, hwm = MH
    # bridge wings: wooden decking on the 01 deck either side of the mast house
    for s in (1, -1):
        P = [P3(BR_AFT, s * hwm, DECK01_Y + 0.03), P3(B1 - 0.5, s * hwm, DECK01_Y + 0.03),
             P3(B1 - 0.5, s * (hw_at(B1, DECK01_Y) - 0.15), DECK01_Y + 0.03),
             P3(BR_AFT, s * (hw_at(BR_AFT, DECK01_Y) - 0.15), DECK01_Y + 0.03)]
        quad_faces(c, c.sw('wood'), P, (0, 1, 0))
        railing_pts(c, [P3(BR_AFT + 0.1, s * (hw_at(BR_AFT, DECK01_Y) - 0.08), DECK01_Y),
                        P3(98.8, s * (hw_at(98.8, DECK01_Y) - 0.08), DECK01_Y)])
    house(c, chamfer_rect(B0, B1, hwm, cf=0.0, ca=0.8), DECK01_Y, ROOF_Y, top='paint',
          top_poly_Bx=chamfer_rect(B0, B1 - 0.25, hwm - 0.28, cf=0.0, ca=0.8))
    for s in (1, -1):
        door(c, 66.5, s * (hwm - 0.15), DECK01_Y, (s, 0, 0))
        door(c, 72.5, s * (hwm - 0.18), DECK01_Y, (s, 0, 0))
    # lower tower block (vertical walls) and the turned pyramid above it
    lb0, lb1, lhw, ltop = LOWER
    house(c, chamfer_rect(lb0, lb1, lhw, cf=0.7, ca=0.7), ROOF_Y, ltop, top='paint')
    rail_poly(c, chamfer_rect(lb0, lb1, lhw, cf=0.7, ca=0.7), ltop, inset=0.1)
    house(c, diamond(ltop), ltop, PLATFORM_Y, top='paint', top_poly_Bx=diamond(PLATFORM_Y))
    for s in (1, -1):
        door(c, 69.0, s * (lhw + 0.01), ROOF_Y, (s, 0, 0))
        ladder(c, P3(lb1 + 0.05, s * 1.2, ROOF_Y), P3(lb1 + 0.05, s * 1.2, ltop), normal=(0, 0, -1))
    a, b_ = diamond_pt(0, LOWER[3]), diamond_pt(0, PLATFORM_Y - 0.1)
    ladder(c, a + np.array([0, 0.05, 0.12]), b_ + np.array([0, 0, 0.12]), width=0.45, normal=normalize(np.array([0, 0.25, 1.0])))
    # Poliment faces on the four faces of the pyramid
    m.b.node('Sensors')
    for k in range(4):
        a0, a1 = diamond_pt(k, ARRAY_Y), diamond_pt((k + 1) % 4, ARRAY_Y)
        b0, b1 = diamond_pt(k, ARRAY_Y + 1.0), diamond_pt((k + 1) % 4, ARRAY_Y + 1.0)
        n = normalize(np.cross(a1 - a0, b0 - a0))
        cen = (a0 + a1) / 2
        if np.dot(n, cen - np.array([0.0, cen[1], zB(DIAMOND_B)])) < 0:
            n = -n
        up = normalize((b0 + b1) / 2 - cen)
        SN.array_panel(c, cen, n, up, ARRAY_W, ARRAY_H, depth=0.14)
    # ESM boxes on the four edges of the pyramid, optronic drums on the lower block
    for k, nrm in ((1, (1, 0, 0)), (2, (0, 0, -1)), (3, (-1, 0, 0))):
        SN.esm_box(c, diamond_pt(k, 18.9), nrm, w=1.15, h=1.6, d=0.55)
    # the forward box stands off the pyramid on a bracket, ahead of the tower (drawing: B 64.5-66.5)
    fv = diamond_pt(0, 18.9)
    SN.esm_box(c, fv + np.array([0, 0, 1.3]), (0, 0, 1), w=1.3, h=1.8, d=0.6)
    c.add(c.paint, box(0.3, 0.3, 1.4, center=(0, 0, 0.7)), xf=M(fv))
    SN.eo_drum(c, P3(lb0 - 0.05, 0.0, 15.2) + np.array([0, 0, 0.35]), yaw=0.0)
    for s in (1, -1):
        SN.eo_drum(c, P3(69.4, s * (lhw + 0.35), 15.0), yaw=s * math.pi / 2)
    # radar platform with railing, small domes on brackets at the corners, Furke-4 on top
    m.b.node('Superstructure')
    plat = chamfer_rect(67.2, 77.6, 2.7, cf=0.7, ca=0.5)
    slab(c, plat, PLATFORM_Y - 0.25, PLATFORM_Y, top='deck')
    rail_poly(c, plat, PLATFORM_Y, inset=0.1)
    m.b.node('Sensors')
    for (B, x) in ((67.5, 2.15), (67.5, -2.15), (77.2, 2.2), (77.2, -2.2)):
        SN.small_dome(c, P3(B, x, PLATFORM_Y), r=0.32)
    SN.furke4(c, P3(DIAMOND_B - 0.4, 0.0, PLATFORM_Y), yaw=0.0)
    # aft balconies stacked on the pyramid's back, signal mast, flag gaff
    m.b.node('Superstructure')
    for yb in (14.6, 17.0, 19.4):
        Bb = DIAMOND_B + (DIAMOND_HD[0] if yb >= LOWER[3] else 4.6) - 0.6
        Bb = max(Bb, lb1 - 0.3)
        bal = chamfer_rect(Bb, Bb + 2.2, 1.6, cf=0.0, ca=0.5)
        slab(c, bal, yb - 0.18, yb, top='deck')
        rail_poly(c, bal, yb, inset=0.08, sides=(1, 2, 3, 4, 5))
    for (ya, yb_, x) in ((ROOF_Y, 14.6, 1.0), (14.6, 17.0, -1.0), (17.0, 19.4, 1.0), (19.4, PLATFORM_Y, -1.0)):
        ladder(c, P3(77.2, x, ya), P3(77.2, x, yb_), normal=(0, 0, -1))
    m.b.node('Sensors')
    SN.small_dome(c, P3(76.6, 1.0, 14.6), r=0.45, post=0.2)
    SN.small_dome(c, P3(76.9, -0.8, 17.0), r=0.38, post=0.2)
    m.b.node('Superstructure')
    c.add(c.paint, rbox(0.9, 0.9, 0.7, r=0.08, seg=1), xf=M(P3(77.0, 0.6, 19.4)))
    ped = P3(76.6, 0.0, LOWER[3])
    top = P3(76.6, 0.0, 28.4)
    c.add(c.paint, tube_path([ped, top], 0.13, seg=8))
    for (yy, half) in ((25.4, 1.7), (27.2, 0.9)):
        c.add(c.paint, tube_path([P3(76.6, half, yy), P3(76.6, -half, yy)], 0.05, seg=5))
        for s in (1, -1):
            PT.lamp(c, P3(76.6, s * (half - 0.1), yy + 0.05), 'white')
    c.add(c.paint, tube_path([P3(76.0, 0.0, 18.1), P3(81.2, 0.0, 18.6)], 0.07, seg=5))
    PT.lamp(c, top, 'white')
    for dx in (-0.35, 0.35):
        c.add(c.sw('dark'), tube_path([P3(79.0, dx, 21.0), P3(97.9, dx * 0.4, 16.9)], 0.015, seg=3), occ=False)
    for yy in (23.6, 24.6):
        PT.lamp(c, P3(76.75, 0.0, yy), 'flagred')


# =============================================================================================
# 01 deck: funnel, boats, side radomes, aft decoy launchers
# =============================================================================================
FUNNEL = (82.5, 89.5, 3.4, 13.9)


def boat(c, Bc, x, y, L=7.2, beam_=2.4):
    """rigid-hull inflatable on a cradle: grey hull, dark tube collar, console under a cover."""
    o = P3(Bc, x, y)
    secs = []
    for t in np.linspace(0.0, 1.0, 9):
        zz = (t - 0.5) * L
        bowf = max(0.0, (t - 0.72) / 0.28)
        hw = beam_ / 2 * (1 - 0.92 * bowf ** 1.6)
        hh = 0.55 + 0.35 * bowf
        secs.append((zz, hw, hh))
    rings = []
    for (zz, hw, hh) in secs:
        ring = []
        for k in range(10):
            a = math.pi * k / 9
            ring.append(o + np.array([hw * math.cos(a), 0.55 - hh * math.sin(a) * 0.55, zz]))
        rings.append(ring)
    # outer hull as a lofted open surface (closed at the top by the deck cover)
    P = np.array([p for r in rings for p in r]); nv = 10
    I = []
    for i in range(len(rings) - 1):
        for k in range(nv - 1):
            a = i * nv + k
            I += [(a, a + nv, a + nv + 1), (a, a + nv + 1, a + 1)]
    I = np.array(I)
    cen = o + np.array([0, 0.9, 0])
    fn = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    if np.sum(np.sum(fn * (P[I].mean(axis=1) - cen), axis=1) < 0) > len(I) / 2:
        I = I[:, ::-1]
    c.add(c.sw('light'), (P, compute_smooth_normals(P, I), P[:, [2, 1]], I))
    # inflatable collar along the gunwale
    path = [r[0] for r in rings] + [r[-1] for r in rings[::-1]]
    c.add(c.sw('dark'), tube_path(path, 0.24, seg=8))
    # cover / console and cradle
    c.add(c.sw('canvas'), rbox(beam_ * 0.7, 0.55, L * 0.55, r=0.2, seg=2, bevel=0.08, y0=0.55), xf=M(o))
    for t in (-0.25, 0.25):
        c.add(c.paint, box(beam_ * 0.8, 0.45, 0.25, center=(0, 0.15, t * L)), xf=M(o))


def midships(m):
    c = m.ctx
    m.b.node('Superstructure')
    fb0, fb1, fhw, fy = FUNNEL
    base = chamfer_rect(fb0, fb1, fhw, cf=0.5, ca=0.3)
    top = chamfer_rect(fb0 + 1.6, fb1, fhw - 0.5, cf=0.4, ca=0.3)
    house(c, base, DECK01_Y, fy - 0.9, top_poly_Bx=chamfer_rect(fb0 + 1.25, fb1, fhw - 0.38, cf=0.43, ca=0.3), top=None)
    house(c, chamfer_rect(fb0 + 1.25, fb1, fhw - 0.38, cf=0.43, ca=0.3), fy - 0.9, fy, top_poly_Bx=top, top='black',
          walls='dark')
    for s in (1, -1):
        decal(c, 'louvre', P3(86.7, s * (fhw - 0.16), 11.6), (s, 0.12, 0), 3.4, 1.6)
    for k in range(3):
        bb = fb0 + 2.4 + k * 2.1
        c.add(c.sw('black'), box(2.0 * (fhw - 0.75), 0.05, 1.5, center=(0, 0, 0)), xf=M(P3(bb, 0, fy + 0.02)))
    # boats beside the funnel on cradles, slewing davits
    m.b.node('Boats')
    for s in (1, -1):
        boat(c, 86.6, s * 5.55, DECK01_Y, L=7.0, beam_=2.3)
        for B in (83.6, 89.6):
            p0 = P3(B, s * 6.9, DECK01_Y)
            c.add(c.paint, cylinder(0.14, 2.6, seg=8), xf=M(p0))
            c.add(c.paint, tube_path([p0 + np.array([0, 2.6, 0]), P3(B, s * 5.6, DECK01_Y + 3.2)], 0.1, seg=6))
    # side radomes, decoy launchers, searchlights, lockers aft of the funnel
    m.b.node('Sensors')
    for s in (1, -1):
        SN.capsule(c, P3(97.0, s * 4.0, DECK01_Y), r=0.75, cyl=0.95, ped_h=0.65)
    m.b.node('Weapons')
    for s in (1, -1):
        for B in (91.8, 93.6):
            WP.kt216(c, P3(B, s * 6.6, DECK01_Y), facing=s * math.pi / 2, el=45.0)
        WP.mtpu(c, P3(73.8, s * 6.9, DECK01_Y), facing=s * math.pi / 2)
    m.b.node('Superstructure')
    for s in (1, -1):
        PT.ready_locker(c, P3(80.4, s * 6.4, DECK01_Y), yaw=s * math.pi / 2)
        PT.round_vent(c, P3(91.0, s * 2.0, DECK01_Y))
        c.add(c.paint, cylinder(0.22, 0.9, seg=8), xf=M(P3(94.6, s * 5.2, DECK01_Y)))
        c.add(c.sw('mid'), cylinder(0.35, 0.5, seg=12, y0=0.9), xf=M(P3(94.6, s * 5.2, DECK01_Y)))


# =============================================================================================
# hangar and its roof, Palash, aft mast
# =============================================================================================
HANGAR_B0, HANGAR_B1 = 99.0, HELIDECK_B
WELL_B, WELL_Y, WELL_X = PALASH_WELL


def hangar(m):
    c = m.ctx
    m.b.node('Superstructure')
    quad_faces(c, c.paint, [P3(HANGAR_B0, -hw_at(HANGAR_B0, DECK01_Y), DECK01_Y), P3(HANGAR_B0, hw_at(HANGAR_B0, DECK01_Y), DECK01_Y),
                            P3(HANGAR_B0, hw_at(HANGAR_B0, HANGAR_Y), HANGAR_Y), P3(HANGAR_B0, -hw_at(HANGAR_B0, HANGAR_Y), HANGAR_Y)],
               (0, 0, 1))
    for s in (1, -1):
        door(c, HANGAR_B0, s * 2.8, DECK01_Y, (0, 0, 1))
    # roof: full width forward of the Palash wells, the centre block between them
    roof = sym_poly([(HANGAR_B0, hw_at(HANGAR_B0, HANGAR_Y)), (WELL_B, hw_at(WELL_B, HANGAR_Y)), (WELL_B, WELL_X),
                     (HANGAR_B1, WELL_X)])
    P, N, UV, I = cap_polygon(np.asarray(poly_Bx_to_xz(roof)), HANGAR_Y, up=True)
    c.add(c.deck, (P, N, np.stack([P[:, 2], P[:, 0]], axis=1), I))
    house(c, chamfer_rect(HANGAR_B0 + 0.4, WELL_B - 0.4, 2.2, cf=0.4, ca=0.0), HANGAR_Y - 0.05, HANGAR_Y + 0.35, top='paint')
    # Palash wells: floor, inboard wall, forward wall, railing round the open sides
    for s in (1, -1):
        xo0, xo1 = hw_at(WELL_B, WELL_Y), hw_at(HANGAR_B1, WELL_Y)
        quad_faces(c, c.deck, [P3(WELL_B, s * WELL_X, WELL_Y), P3(WELL_B, s * xo0, WELL_Y),
                               P3(HANGAR_B1, s * xo1, WELL_Y), P3(HANGAR_B1, s * WELL_X, WELL_Y)], (0, 1, 0))
        quad_faces(c, c.paint, [P3(WELL_B, s * WELL_X, WELL_Y), P3(HANGAR_B1, s * WELL_X, WELL_Y),
                                P3(HANGAR_B1, s * WELL_X, HANGAR_Y), P3(WELL_B, s * WELL_X, HANGAR_Y)], (s, 0, 0))
        quad_faces(c, c.paint, [P3(WELL_B, s * WELL_X, WELL_Y), P3(WELL_B, s * xo0, WELL_Y),
                                P3(WELL_B, s * hw_at(WELL_B, HANGAR_Y), HANGAR_Y), P3(WELL_B, s * WELL_X, HANGAR_Y)], (0, 0, -1))
        railing_pts(c, [P3(WELL_B + 0.1, s * (xo0 - 0.1), WELL_Y), P3(HANGAR_B1 - 0.1, s * (xo1 - 0.1), WELL_Y),
                        P3(HANGAR_B1 - 0.1, s * (WELL_X + 0.1), WELL_Y)])
        ladder(c, P3(WELL_B + 0.05, s * (WELL_X + 0.8), WELL_Y), P3(WELL_B + 0.05, s * (WELL_X + 0.8), HANGAR_Y),
               normal=(0, 0, -1))
    # aft wall: full width up to the well floor, the centre block above it
    yk = float(KNUCKLE(HANGAR_B1))
    quad_faces(c, c.paint, [P3(HANGAR_B1, -hw_at(HANGAR_B1, yk), yk), P3(HANGAR_B1, hw_at(HANGAR_B1, yk), yk),
                            P3(HANGAR_B1, hw_at(HANGAR_B1, WELL_Y), WELL_Y), P3(HANGAR_B1, -hw_at(HANGAR_B1, WELL_Y), WELL_Y)],
               (0, 0, -1))
    quad_faces(c, c.paint, [P3(HANGAR_B1, -WELL_X, WELL_Y), P3(HANGAR_B1, WELL_X, WELL_Y),
                            P3(HANGAR_B1, WELL_X, HANGAR_Y), P3(HANGAR_B1, -WELL_X, HANGAR_Y)], (0, 0, -1))
    decal(c, 'hangar_door', (0.0, yk + 2.6, zB(HANGAR_B1)), (0, 0, -1), 6.2, 5.2)
    gpi = P3(HANGAR_B1 + 0.25, 0.0, HANGAR_Y - 0.9)
    c.add(c.sw('dark'), rbox(1.2, 0.5, 0.45, r=0.06, seg=1, y0=-0.25), xf=M(gpi))
    for k in range(5):
        c.add(c.sw(('flagred', 'orange', 'white', 'green', 'green')[k]), box(0.16, 0.16, 0.04, center=(0, 0, 0)),
              xf=M(gpi + np.array([-0.4 + 0.2 * k, 0.0, -0.24])))
    for s in (1, -1):
        searchlight(c, P3(HANGAR_B1 + 0.3, s * 3.6, HANGAR_Y - 0.6), facing=math.pi)
        door(c, HANGAR_B1, s * 5.4, yk, (0, 0, -1))
    # aft mast: pole with yards, lamps and a navigation radar
    BM = HANGAR_B0 + 0.6
    pm = P3(BM, 0.0, HANGAR_Y + 0.35)
    c.add(c.paint, tube_path([pm, pm + np.array([0, 5.9, 0])], 0.12, seg=8))
    for (yy, half) in ((15.6, 1.2), (17.3, 0.7)):
        c.add(c.paint, tube_path([P3(BM, half, yy), P3(BM, -half, yy)], 0.045, seg=5))
    PT.lamp(c, pm + np.array([0, 5.9, 0]), 'white')
    m.b.node('Sensors')
    SN.pal_n(c, P3(BM, 0.0, 14.4) + np.array([0, 0, -0.5]), yaw=0.3, L=1.8)
    SN.radome(c, P3(100.9, 0.0, HANGAR_Y + 0.35), 0.58, ped_h=0.45)
    SN.capsule(c, P3(102.9, 0.0, HANGAR_Y + 0.35), r=1.1, cyl=1.05, ped_h=0.25)
    m.b.node('Superstructure')
    whip(c, P3(104.4, -1.4, HANGAR_Y + 0.35), 8.2, r=0.05)
    whip(c, P3(107.6, 1.4, HANGAR_Y + 0.35), 12.5, r=0.06)
    for s in (1, -1):
        raft_rack(c, P3(101.4, s * (hw_at(101.4, HANGAR_Y) - 0.7), HANGAR_Y), n=3, along=(0, 0, -1), spacing=0.85)
        raft_rack(c, P3(101.4, s * (hw_at(101.4, HANGAR_Y) - 1.45), HANGAR_Y), n=3, along=(0, 0, -1), spacing=0.85)
    m.b.node('Weapons')
    for s in (1, -1):
        WP.palash(c, P3(111.5, s * 5.9, WELL_Y), facing=s * math.radians(100), node='Palash_' + ('P' if s > 0 else 'S'))


# =============================================================================================
# helideck fittings
# =============================================================================================
def helideck_fittings(m):
    c = m.ctx
    m.b.node('Hull')
    for s in (1, -1):
        pts = [P3(B, s * (float(DECK_HB(B)) - 0.12), dk(B, 8.0)) for B in (HELIDECK_B + 0.4, 124.0, 134.6)]
        railing_pts(c, pts)
    railing_pts(c, [P3(134.75, 7.9, dk(134.7, 7.9)), P3(134.75, 1.0, dk(134.7, 1.0))])
    railing_pts(c, [P3(134.75, -1.0, dk(134.7, 1.0)), P3(134.75, -7.9, dk(134.7, 7.9))])
    # ensign staff and flag
    p0 = P3(134.5, 0.0, dk(134.5)); p1 = p0 + np.array([0, 3.6, -1.1])
    c.add(c.paint, tube_path([p0, p1], 0.06, seg=5))
    fa = p1 + np.array([0, -0.12, 0]); fb = p1 + np.array([0, -1.2, 0.15])
    c.add(c.rect('ensign'), _quad(fb, fb + np.array([0.0, 0, -1.8]), fa + np.array([0.0, 0, -1.8]), fa,
                                  uv=[(0, 1), (1, 1), (1, 0), (0, 0)]))
    for s in (1, -1):
        for B in (117.0, 125.0, 132.5):
            bollard(c, P3(B, s * 7.0, dk(B, 7.0)))
        for B in np.arange(115.5, 134.6, 3.2):
            c.add(c.sw('green'), sphere(0.09, seg=6, rings=3, hemi=True), xf=M(P3(B, s * 7.75, dk(B, 7.75))))


# =============================================================================================
# build
# =============================================================================================
def build_all(path='admiral_gorshkov_class_frigate.glb'):
    m = S2.Model()
    register_decals(m)
    S2.build_hull(m)
    hull_fittings(m)
    forecastle(m)
    uksk_block(m)
    wheelhouse(m)
    mast(m)
    midships(m)
    hangar(m)
    helideck_fittings(m)
    print('triangles:', m.b.tri_count())
    for k, v in m.b.tri_count_by_node().items():
        print(f'  {k:22s} {v}')
    S2.export(m, path, ao=AO)
    return m


if __name__ == '__main__':
    import sys
    build_all(sys.argv[1] if len(sys.argv) > 1 else 'admiral_gorshkov_class_frigate.glb')
