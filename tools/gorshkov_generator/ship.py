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
                  PALASH_WELL, STEM_KN_B)
from kit import *
import model as S2
import markings as MK
import parts as PT
from parts import rbox
import weapons as WP
import sensors as SN
import fittings as FT

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
PLATFORM_Y = 22.3     # radar platform at the top of the tower (profile drawing)


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
NAME_H = 0.42         # name on the quarter: red letters, B 112-123.3 at 3.1 m (2018 side photograph)
PENNANT_B = 34.5      # hull number centre (photos and drawing: 417 / 461 centred near B 34-36)
PENNANT_H = 3.0


# helideck markings (2018 aerial photographs of the stern, 1:500 plan)
LAND = (120.6, 132.7, 5.6, 1.6, 1.0)      # landing area: B0, B1, half-width, front and aft corner chamfers
TD_B, TD_R, TD_R2 = 126.0, 4.7, 2.0       # touchdown circle: centre, radius, inner circle
TRACK = (114.1, 120.6, 0.55, 0.95)        # traverse track from the hangar: B0, B1, inner and outer white lines


def paint_helideck(L, rect):
    """22350 helideck (u = across, starboard at u=0; v = from the hangar aft): landing area of dark
    anti-slip mesh with a white border, touchdown circle, inner circle and centreline mark, the white-edged
    traverse track running out of the hangar, white deck-edge lines."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['heli'], alpha=1.0, rough=0.82, metal=0.0)
    n = value_noise(h, w, 24, seed=31, octaves=4)
    L.col[y0:y1, x0:x1] *= (0.9 + 0.16 * n)[..., None]
    Lb = TRANSOM_TOP[0] - HELIDECK_B
    W_ = 16.2
    kx, ky = w / W_, h / Lb

    def px(xm, Bm):
        return ((xm + W_ / 2) * kx, (Bm - HELIDECK_B) * ky)
    B0, B1, hw, cf, ca = LAND
    octa = [px(-hw + cf, B0), px(hw - cf, B0), px(hw, B0 + cf), px(hw, B1 - ca), px(hw - ca, B1), px(-hw + ca, B1),
            px(-hw, B1 - ca), px(-hw, B0 + cf)]

    def area(d, s):
        d.polygon([(u * s, v * s) for (u, v) in octa], fill=255)
    m_area = L.draw_mask(w, h, area)
    L.mask_apply(m_area, col=PAL['heli'] * 0.62, x0=x0, y0=y0, rough=0.9, add_height=-0.2)
    # anti-slip mesh: fine light grid inside the landing area
    gm = np.zeros((h, w))
    step = max(3, int(round(0.3 * kx)))
    gm[:, ::step] = 1.0
    gm[::step, :] = 1.0
    L.mask_apply(gm * m_area * 0.45, col=PAL['heli'] * 1.25, x0=x0, y0=y0, add_height=0.3)
    lw = max(2, int(round(0.2 * kx)))

    def lines(d, s):
        d.polygon([(u * s, v * s) for (u, v) in octa], outline=255, width=lw * s)
        for r, wd in ((TD_R, lw * 1.3), (TD_R2, lw)):
            cx, cy = px(0.0, TD_B)
            rx, ry = r * kx, r * ky
            d.ellipse([(cx - rx) * s, (cy - ry) * s, (cx + rx) * s, (cy + ry) * s], outline=255, width=int(wd * s))
        cx = px(0.0, 0)[0]
        d.line([(cx * s, px(0, TD_B - TD_R)[1] * s), (cx * s, px(0, TD_B - TD_R2)[1] * s)], fill=255, width=lw * s)
        tb0, tb1, xi, xo = TRACK
        for xx in (-xo, -xi, xi, xo):
            u = px(xx, 0)[0]
            d.line([(u * s, px(0, tb0)[1] * s), (u * s, px(0, tb1)[1] * s)], fill=255, width=lw * s)
        for sx in (-1, 1):
            u = px(sx * 7.55, 0)[0]
            d.line([(u * s, px(0, HELIDECK_B + 0.4)[1] * s), (u * s, px(0, TRANSOM_TOP[0] - 0.3)[1] * s)], fill=255,
                   width=max(1, lw // 2) * s)
    L.mask_apply(L.draw_mask(w, h, lines), col=PAL['white'] * 0.92, x0=x0, y0=y0, add_height=0.25, rough=0.6)

    def rail(d, s):
        tb0, tb1, xi, xo = TRACK
        u0, u1 = px(-0.16, 0)[0], px(0.16, 0)[0]
        d.rectangle([u0 * s, px(0, tb0)[1] * s, u1 * s, px(0, tb1)[1] * s], fill=255)
    L.mask_apply(L.draw_mask(w, h, rail), col=PAL['steel'] * 0.55, x0=x0, y0=y0, add_height=0.5, metal=0.6, rough=0.4)


def paint_name(L, rect):
    MK.paint_text(L, rect, SHIP_NAME, srgb('a8372b'), size_frac=0.78, spacing=8)


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


POCKET = (3.3, 5.9, 4.85, 6.68)       # anchor-pocket decal: B forward, B aft, y bottom, y top (profile drawing)


def paint_pocket(L, rect):
    """anchor pocket seen from port (bow to the left): D-shaped recess under the knuckle with a straight
    top, a forward edge bulging round to the bottom and a raked aft edge, the stowed Hall anchor inside
    (Admiral Golovko photograph, 2021)."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    B0, B1, yb, yt = POCKET
    kx, ky = w / (B1 - B0), h / (yt - yb)
    L.rect(x0, y0, x1, y1, col=PAL['hull'], alpha=0.0)

    def p(u, v):
        return (u * kx, v * ky)
    cu, cv, ru, rv = 1.2, 0.9, 1.06, 0.84
    outline = [p(cu + ru * math.cos(a), cv - rv * math.sin(a)) for a in np.linspace(math.pi / 2, 1.5 * math.pi, 25)]
    outline += [p(2.0, 1.62), p(2.32, cv - rv)]


    def hole(d, s):
        d.polygon([(x * s, y * s) for (x, y) in outline], fill=255)
    m = L.draw_mask(w, h, hole)
    L.mask_apply(m, col=PAL['black'] * 1.3, alpha=1.0, add_height=-1.0, x0=x0, y0=y0)

    def anchor(d, s):
        q = lambda pts: [(x * s, y * s) for (x, y) in (p(*t) for t in pts)]
        d.polygon(q([(1.27, 0.05), (1.45, 0.05), (1.37, 1.32), (1.19, 1.32)]), fill=255)           # shank
        d.polygon(q([(0.78, 1.3), (1.85, 1.28), (1.8, 1.55), (1.2, 1.62), (0.82, 1.52)]), fill=255)  # crown
        d.polygon(q([(0.8, 1.36), (0.86, 0.72), (1.02, 0.52), (1.16, 0.7), (1.12, 1.3)]), fill=255)  # fluke
    L.mask_apply(L.draw_mask(w, h, anchor), col=PAL['dark'] * 0.95, x0=x0, y0=y0, add_height=0.6)

    def lit(d, s):
        q = lambda pts: [(x * s, y * s) for (x, y) in (p(*t) for t in pts)]
        d.line(q([(1.27, 0.08), (1.19, 1.3)]), fill=255, width=s)
        d.line(q([(0.86, 0.74), (1.02, 0.54)]), fill=255, width=s)
        d.line(q([(0.8, 1.31), (1.85, 1.29)]), fill=255, width=s)
        d.line(q([(1.2, 0.07), (2.3, 0.07)]), fill=255, width=2 * s)                                # lit top edge
    L.mask_apply(L.draw_mask(w, h, lit), col=PAL['mid'] * 1.3, x0=x0, y0=y0)
    a = L.alpha[y0:y1, x0:x1]
    L.alpha[y0:y1, x0:x1] = np.where(a > 0.3, 1.0, 0.0)


def paint_star(L, rect):
    """red star with a gold border on the bow (Admiral Golovko photograph)."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['hull'], alpha=0.0)

    def star(r_out, d, s):
        cx, cy = w / 2, h / 2 + h * 0.04
        pts = []
        for k in range(10):
            r = r_out if k % 2 == 0 else r_out * 0.4
            a = -math.pi / 2 + k * math.pi / 5
            pts.append(((cx + r * math.cos(a)) * s, (cy + r * math.sin(a)) * s))
        d.polygon(pts, fill=255)
    L.mask_apply(L.draw_mask(w, h, lambda d, s: star(w * 0.49, d, s)), col=srgb('c9a23a'), alpha=1.0, x0=x0, y0=y0)
    L.mask_apply(L.draw_mask(w, h, lambda d, s: star(w * 0.4, d, s)), col=srgb('b8282a'), alpha=1.0, x0=x0, y0=y0)
    a = L.alpha[y0:y1, x0:x1]
    L.alpha[y0:y1, x0:x1] = np.where(a > 0.4, 1.0, 0.0)


def paint_opening(L, rect, rim=0.12, inner=None, rr=None, glass=False):
    """opening with a raised rim: oval (rr=None) or rounded rectangle (corner radius rr as a fraction of
    the height); a dark through-hole, a dark port glass or a recessed panel (inner colour)."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['hull'], alpha=0.0)
    e = max(2, int(rim * min(w, h)))

    def shape(m, d, s):
        box_ = [m * s, m * s, (w - m) * s, (h - m) * s]
        if rr is None:
            d.ellipse(box_, fill=255)
        else:
            d.rounded_rectangle(box_, radius=int(rr * (h - 2 * m) * s), fill=255)
    L.mask_apply(L.draw_mask(w, h, lambda d, s: shape(1, d, s)), col=PAL['hull'] * 1.12, alpha=1.0, x0=x0, y0=y0,
                 add_height=0.8)
    col = inner if inner is not None else (PAL['glass'] if glass else PAL['black'] * 1.2)
    L.mask_apply(L.draw_mask(w, h, lambda d, s: shape(1 + e, d, s)), col=col, alpha=1.0, x0=x0, y0=y0,
                 add_height=-1.0, rough=0.15 if glass else None)
    a = L.alpha[y0:y1, x0:x1]
    L.alpha[y0:y1, x0:x1] = np.where(a > 0.4, 1.0, 0.0)


def paint_hatch(L, rect):
    """flush hatch in the hull side: rounded seam, hinges and dogs."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['hull'], alpha=0.0)

    def fn(d, s):
        d.rounded_rectangle([2 * s, 2 * s, (w - 2) * s, (h - 2) * s], radius=int(0.18 * h * s), outline=255, width=2 * s)
        for f in (0.25, 0.75):
            d.rectangle([(w * f - 3) * s, 4 * s, (w * f + 3) * s, 9 * s], fill=255)
            d.rectangle([(w * f - 3) * s, (h - 9) * s, (w * f + 3) * s, (h - 4) * s], fill=255)
    L.mask_apply(L.draw_mask(w, h, fn), col=PAL['hull'] * 0.6, alpha=1.0, x0=x0, y0=y0, add_height=-0.6)
    a = L.alpha[y0:y1, x0:x1]
    L.alpha[y0:y1, x0:x1] = np.where(a > 0.4, 1.0, 0.0)


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
    m.alloc('pocket', 192, 136, paint_pocket)
    m.alloc('star', 48, 48, paint_star)
    m.alloc('bullnose', 96, 48, lambda L, r: paint_opening(L, r, rim=0.16))
    m.alloc('fairlead', 80, 36, lambda L, r: paint_opening(L, r, rim=0.22))
    m.alloc('port', 32, 32, lambda L, r: paint_opening(L, r, rim=0.2, glass=True))
    m.alloc('mooring', 128, 48, lambda L, r: paint_opening(L, r, rim=0.2, rr=0.45, inner=PAL['hull'] * 0.55))
    m.alloc('hatch', 64, 64, paint_hatch)
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
def upper_x(B, y):
    """half-breadth of the upper hull (above the knuckle, or above the raked stem line forward of it)."""
    if B >= STEM_KN_B:
        return upper_hb(B, y)
    (xb, yb), (xt, yt) = S2.upper_section(B, float(BULWARK_TOP(B)))
    return 0.0 if y <= yb else xb + (xt - xb) * (y - yb) / (yt - yb)


def hull_decal(c, name, Bc, yc, w, h, side, nseg=6, offset=0.04, flip=False, surface='main'):
    """decal strip following the hull side (constant height band), readable from outside. surface: 'main'
    (below the knuckle), 'upper' (the inward-leaning upper hull / bulwark) or 'inner' (bulwark inner face).
    flip mirrors the image along the ship (for asymmetric decals on the starboard side)."""
    P, UV, I = [], [], []
    for k in range(nseg + 1):
        t = k / nseg
        B = Bc + (t - 0.5) * w if side > 0 else Bc - (t - 0.5) * w
        if surface == 'main':
            xs, ys = section(B, 64)
        for j, yy in enumerate((yc - h / 2, yc + h / 2)):
            if surface == 'main':
                xh = float(np.interp(yy, ys, xs)) + offset
            elif surface == 'upper':
                xh = upper_x(B, yy) + offset
            else:
                xh = upper_x(B, yy) - S2.BULWARK_T - offset
            P.append((side * xh, yy, zB(B)))
            UV.append((1.0 - t if flip else t, 1.0 - j))
    for k in range(nseg):
        a = 2 * k
        I += [(a, a + 2, a + 3), (a, a + 3, a + 1)]
    P = np.array(P); UV = np.array(UV); I = np.array(I)
    N = compute_smooth_normals(P, I)
    facing = -side if surface == 'inner' else side
    if np.dot(N[0], np.array([facing, 0.0, 0.0])) < 0:
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


NAME_B, NAME_Y = 117.7, 3.1
FAIRLEADS = ((9.15, 0.75, 0.32), (17.8, 0.6, 0.34))     # bulwark fairleads: B, width, height (centre 7.15 m)


def hull_fittings(m):
    c = m.ctx
    m.b.node('Hull')
    for s in (1, -1):
        hull_decal(c, 'pennant', PENNANT_B, 2.55, PENNANT_H * 384 / 160, PENNANT_H, s)
        hull_decal(c, 'name', NAME_B, NAME_Y, m.name_w, NAME_H, s, nseg=10)
        # bow: anchor pocket with the stowed anchor, red star, bullnose through the stem, bulwark fairleads
        B0, B1, yb, yt = POCKET
        hull_decal(c, 'pocket', (B0 + B1) / 2, (yb + yt) / 2, B1 - B0, yt - yb, s, nseg=8, offset=0.02, flip=s < 0)
        hull_decal(c, 'star', 7.1, 6.0, 0.46, 0.46, s, nseg=2, offset=0.025)
        hull_decal(c, 'bullnose', 1.1, 7.55, 0.85, 0.42, s, nseg=4, offset=0.015, surface='upper')
        for (B, wd, ht) in FAIRLEADS:
            hull_decal(c, 'fairlead', B, 7.15, wd, ht, s, nseg=2, offset=0.015, surface='upper')
            hull_decal(c, 'fairlead', B, 7.15, wd, ht, s, nseg=2, offset=0.015, surface='inner')
        # stern quarter: side ports, mooring fairlead, hatch and door (profile drawing, 2018 side photograph)
        for B in (124.3, 133.0):
            hull_decal(c, 'port', B, 2.95, 0.42, 0.42, s, nseg=1, offset=0.02)
        hull_decal(c, 'mooring', 127.15, 4.25, 1.9, 0.7, s, nseg=2, offset=0.02)
        hull_decal(c, 'hatch', 130.3, 3.38, 1.2, 1.15, s, nseg=1, offset=0.02)
        hull_decal(c, 'sdoor', 131.95, 3.62, 0.9, 2.2, s, nseg=1, offset=0.02)
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
GUN_B = 26.45         # training axis = centre of the deck shield ring (plan)
REDUT = ((37.3, 1.45), (37.3, -1.45), (42.05, 1.45), (42.05, -1.45))     # 4 x 8 cells (plan)


# ground tackle and breakwater (1:500 plan, checked on the 2018 photograph of the forecastle)
HAWSE_B, COMP_B, STOP_B, CAPSTAN_B, GEAR_X = 8.5, 9.3, 13.4, 15.6, 1.38
BREAKWATER = ((16.9, 0.0), (17.35, 1.0), (17.65, 2.0), (18.05, 3.0), (18.6, 3.6))
CANISTERS = (7.5, 12.4, 16.4)            # pairs of upright canisters against the bulwark (ovals on the plan)


def bulwark_in(B, y):
    """half-breadth of the bulwark's inner face at B, height y."""
    return hw_at(B, y) - S2.BULWARK_T


def forecastle(m):
    c = m.ctx
    m.b.node('Forecastle')
    for s in (1, -1):
        x = s * GEAR_X
        FT.pipe_mouth(c, P3(HAWSE_B, x, dk(HAWSE_B, GEAR_X)), r=0.34)
        FT.compressor(c, P3(COMP_B, x, dk(COMP_B, GEAR_X)), (0, 0, -1), (1, 0, 0))
        FT.chain_stopper(c, P3(STOP_B, x, dk(STOP_B, GEAR_X)), (0, 0, 1))
        FT.capstan(c, P3(CAPSTAN_B, x, dk(CAPSTAN_B, GEAR_X)))
        FT.drive_housing(c, P3(CAPSTAN_B - 0.6, s * 0.42, dk(CAPSTAN_B - 0.6, 0.42)), (0, 0, 1))
        # cable: out of the hawse pipe, through the compressor and the stopper onto the chain wheel
        FT.chain(c, [P3(HAWSE_B + 0.15, x, dk(HAWSE_B, GEAR_X) + 0.1), P3(COMP_B + 0.6, x, dk(COMP_B, GEAR_X) + 0.17),
                     P3(STOP_B, x, dk(STOP_B, GEAR_X) + 0.2), P3(CAPSTAN_B - 0.58, x, dk(CAPSTAN_B, GEAR_X) + 0.41)])
        FT.double_bollard(c, P3(5.1, s * 1.62, dk(5.1, 1.62)), (0, 0, 1), gap=0.5)
    FT.breakwater(c, BREAKWATER, 1.2, 0.6, dk)
    for B in (18.45, 19.1):                                   # lockers behind the breakwater apex
        for x in (0.36, -0.36):
            FT.locker(c, P3(B, x, dk(B, abs(x))), 0.52, 0.48, 0.56)
    FT.vent_head(c, P3(11.5, 0.5, dk(11.5, 0.5)), r=0.42, h=0.56)
    FT.vent_head(c, P3(10.9, -0.55, dk(10.9, 0.55)), r=0.26, h=0.46)
    FT.locker(c, P3(9.65, 0.3, dk(9.65, 0.3)), 0.42, 0.34, 0.42)
    mushroom_vent(c, P3(22.6, -4.0, dk(22.6, 4.0)), r=0.42)
    FT.vent_head(c, P3(22.4, 2.3, dk(22.4, 2.3)), r=0.22, h=0.5)
    # vertical stiffeners on the inside of the bulwark (clear of the fairleads)
    t = S2.BULWARK_T
    for B in np.arange(3.0, S2.OPENING[0] - 0.3, 1.25):
        if any(abs(B - fb) < 0.6 for fb, _, _ in FAIRLEADS):
            continue
        yb, yt = dk(B, hw_at(B, float(KNUCKLE(B))) - t), float(BULWARK_TOP(B)) - 0.12
        for s in (1, -1):
            pb = P3(B, s * (hw_at(B, yb) - t - 0.1), yb)
            pt_ = P3(B, s * (hw_at(B, yt) - t - 0.06), yt)
            c.add(c.paint, beam(pb, pt_, 0.07, 0.2, up=(0, 0, 1)))
    # upright canisters in pairs, red fire boxes and life rings on the bulwark
    for s in (1, -1):
        for Bc in CANISTERS:
            for B in (Bc - 0.31, Bc + 0.31):
                xb = bulwark_in(B, 7.0) - 0.5
                FT.upright_canister(c, P3(B, s * xb, dk(B, xb)), (s, 0, 0))
        for B in (19.6, 20.35):
            FT.fire_box(c, P3(B, s * (bulwark_in(B, 7.3) - 0.02), 7.3), (-s, 0, 0))
        FT.life_ring(c, P3(22.4, s * (bulwark_in(22.4, 7.25) - 0.06), 7.25), (-s, 0, 0))
    # jackstaff on the stem
    c.add(c.paint, tube_path([P3(0.9, 0, float(BULWARK_TOP(0.9))), P3(0.9, 0, float(BULWARK_TOP(0.9)) + 2.4)], 0.05, seg=5))
    # gun, Redut modules, decoy launchers in the bulwark cut-outs
    m.b.node('Weapons')
    WP.a192m(c, GUN_B, dk(GUN_B))
    m.b.node('Weapons')
    for (B, x) in REDUT:
        WP.redut_module(c, B, x, dk(B, abs(x)))
    for s in (1, -1):                                         # KT-216 in the bulwark cut-outs
        WP.kt216(c, P3(40.7, s * 5.0, dk(40.7, 5.0)), facing=s * math.pi / 2)
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
        FT.life_ring(c, P3(UKSK_B0 + 0.02, s * 4.4, 7.45), (0, 0, 1))
        ladder(c, P3(UKSK_B0 - 0.05, s * 5.2, yk), P3(UKSK_B0 - 0.05, s * 5.2, UKSK_Y), width=0.55, normal=(0, 0, 1))
    top = lean_poly(UKSK_B0, BR_FRONT + 0.4, UKSK_Y, nB=4)
    P, N, UV, I = cap_polygon(np.asarray(poly_Bx_to_xz(top)), UKSK_Y, up=True)
    c.add(c.deck, (P, N, np.stack([P[:, 2], P[:, 0]], axis=1), I))
    m.b.node('Weapons')
    WP.uksk_field(c, 46.45, 52.1, 2.2, UKSK_Y)
    for s in (1, -1):
        WP.kt308(c, P3(54.6, s * 5.4, UKSK_Y), facing=s * math.pi / 2)
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
    SN.radome(c, P3(60.5, 0.0, ROOF_Y), 1.95, ped_h=0.3, ped_r=1.25)
    for s in (1, -1):
        SN.pal_n(c, P3(60.45, s * 3.05, ROOF_Y), yaw=s * 0.4)
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
LOWER = (64.8, 75.0, 3.25, 17.0)          # lower tower block: nose B, aft B, half-width, top
LOWER_PLAN = ((64.8, 1.6), (67.3, 3.25), (74.4, 3.25), (75.0, 2.6))   # half outline: tapered nose, aft chamfer
DIAMOND_B = 71.35                         # upper tower: square pyramid turned 45 deg (faces at +-45 deg)
DIAMOND_HD = (3.05, 2.33)                 # half-diagonal at the foot (LOWER top) and at the platform
ARRAY_Y, ARRAY_W, ARRAY_H = 19.2, 2.7, 3.8
PLATFORM = (68.7, 77.6, 2.7)              # radar platform: B0, B1, half-width (profile drawing)


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
    # lower tower block (vertical walls, tapered nose) and the turned pyramid above it
    lb0, lb1, lhw, ltop = LOWER
    lower = sym_poly(list(LOWER_PLAN))
    house(c, lower, ROOF_Y, ltop, top='paint')
    rail_poly(c, lower, ltop, inset=0.1)
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
    # ESM box on the aft edge, small radomes stacked on brackets at the side edges
    SN.esm_box(c, diamond_pt(2, 18.9), (0, 0, -1), w=1.15, h=1.6, d=0.55)
    for k, sx in ((1, 1), (3, -1)):
        for yy in (17.9, 19.9):
            p = diamond_pt(k, yy)
            c.add(c.paint, box(0.9, 0.12, 0.25, center=(0, 0, 0)), xf=M(p + np.array([sx * 0.4, 0.0, 0.0])))
            SN.small_dome(c, p + np.array([sx * 0.75, 0.06, 0.0]), r=0.36, post=0.12)
    # optronic director on a tall pedestal at the front of the lower block (port), a small dome to starboard
    pe = P3(66.2, 1.55, ltop)
    c.add(c.paint, rbox(1.0, 2.0, 1.0, r=0.12, seg=1, bevel=0.05), xf=M(pe))
    c.add(c.paint, rbox(1.4, 1.35, 1.5, r=0.3, seg=2, bevel=0.1, y0=2.0), xf=M(pe))
    c.add(c.sw('glass'), box(0.5, 0.36, 0.03, center=(0, 0, 0)), xf=M(pe + np.array([0.0, 2.75, 0.765])))
    c.add(c.sw('dark'), box(0.36, 0.3, 0.03, center=(0, 0, 0)), xf=M(pe + np.array([-0.4, 2.62, 0.765])))
    SN.small_dome(c, P3(66.4, -1.55, ltop), r=0.45, post=0.35)
    for s in (1, -1):                                         # side sensors on the lower block (profile)
        SN.eo_drum(c, P3(73.7, s * (lhw + 0.35), 15.0), yaw=s * math.pi / 2)
        SN.esm_box(c, P3(66.05, s * 2.42, 15.3), (s * 0.83, 0, 0.55), w=0.9, h=1.0, d=0.35)
    # radar platform with railing, small domes on brackets at the corners, Furke-4 on top
    m.b.node('Superstructure')
    p0, p1, phw = PLATFORM
    plat = chamfer_rect(p0, p1, phw, cf=0.7, ca=0.5)
    slab(c, plat, PLATFORM_Y - 0.25, PLATFORM_Y, top='deck')
    rail_poly(c, plat, PLATFORM_Y, inset=0.1)
    m.b.node('Sensors')
    for (B, x) in ((p0 + 0.3, 2.15), (p0 + 0.3, -2.15), (p1 - 0.4, 2.2), (p1 - 0.4, -2.2)):
        SN.small_dome(c, P3(B, x, PLATFORM_Y), r=0.32)
    SN.furke4(c, P3(71.5, 0.0, PLATFORM_Y), yaw=0.0)
    # aft balconies stacked on the pyramid's back, signal mast, flag gaff
    m.b.node('Superstructure')
    for yb in (14.6, 17.0, 19.4):
        Bb = DIAMOND_B + DIAMOND_HD[0] - 0.6 if yb >= LOWER[3] else lb1 - 0.3
        bal = chamfer_rect(Bb, Bb + 2.2, 1.6, cf=0.0, ca=0.5)
        slab(c, bal, yb - 0.18, yb, top='deck')
        rail_poly(c, bal, yb, inset=0.08, sides=(1, 2, 3, 4, 5))
    for (ya, yb_, x) in ((ROOF_Y, 14.6, 1.0), (14.6, 17.0, -1.0), (17.0, 19.4, 1.0), (19.4, PLATFORM_Y, -1.0)):
        ladder(c, P3(77.2, x, ya), P3(77.2, x, yb_), normal=(0, 0, -1))
    # inclined ladder from the 01 deck behind the mast house up to the aft edge of the radar platform
    la, lb = P3(75.9, 0.0, DECK01_Y), P3(78.1, 0.0, PLATFORM_Y - 0.25)
    ln = normalize(np.cross(lb - la, np.array([1.0, 0.0, 0.0])))
    if ln[2] > 0:
        ln = -ln
    ladder(c, la, lb, width=0.7, normal=ln)
    for sx in (-0.38, 0.38):
        c.add(c.paint, tube_path([la + np.array([sx, 0, 0]), lb + np.array([sx, 0, 0])], 0.05, seg=5))
    m.b.node('Sensors')
    SN.small_dome(c, P3(76.6, 1.0, 14.6), r=0.45, post=0.2)
    SN.small_dome(c, P3(76.9, -0.8, 17.0), r=0.38, post=0.2)
    m.b.node('Superstructure')
    c.add(c.paint, rbox(0.9, 0.9, 0.7, r=0.08, seg=1), xf=M(P3(77.0, 0.6, 19.4)))
    ped = P3(76.4, 0.0, PLATFORM_Y)
    top = P3(76.4, 0.0, 28.2)
    c.add(c.paint, tube_path([ped, top], 0.13, seg=8))
    for (yy, half) in ((25.4, 1.7), (27.2, 0.9)):
        c.add(c.paint, tube_path([P3(76.4, half, yy), P3(76.4, -half, yy)], 0.05, seg=5))
        for s in (1, -1):
            PT.lamp(c, P3(76.4, s * (half - 0.1), yy + 0.05), 'white')
    c.add(c.paint, tube_path([P3(76.0, 0.0, 18.1), P3(81.2, 0.0, 18.6)], 0.07, seg=5))
    PT.lamp(c, top, 'white')
    for dx in (-0.35, 0.35):
        c.add(c.sw('dark'), tube_path([P3(79.0, dx, 21.0), P3(97.9, dx * 0.4, 16.9)], 0.015, seg=3), occ=False)
    for yy in (23.6, 24.6):
        PT.lamp(c, P3(76.55, 0.0, yy), 'flagred')


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
    top = lean_poly(UKSK_B1, HANGAR_B0, DECK01_Y, nB=10)
    P, N, UV, I = cap_polygon(np.asarray(poly_Bx_to_xz(top)), DECK01_Y, up=True)
    c.add(c.deck, (P, N, np.stack([P[:, 2], P[:, 0]], axis=1), I))
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
    # boats beside the funnel on cradles, a slewing davit at the forward end of each (plan B 83.8-90.2)
    m.b.node('Boats')
    for s in (1, -1):
        boat(c, 87.0, s * 5.95, DECK01_Y, L=6.4, beam_=2.25)
        p0 = P3(82.0, s * 6.6, DECK01_Y)
        c.add(c.paint, cylinder(0.2, 0.3, seg=10), xf=M(p0))
        c.add(c.paint, cylinder(0.14, 2.9, seg=10), xf=M(p0))
        c.add(c.paint, tube_path([p0 + np.array([0, 2.9, 0]), P3(82.9, s * 6.3, DECK01_Y + 3.6),
                                  P3(84.6, s * 5.95, DECK01_Y + 3.5)], 0.1, seg=6))
        c.add(c.sw('dark'), tube_path([P3(84.6, s * 5.95, DECK01_Y + 3.4), P3(84.6, s * 5.95, DECK01_Y + 1.6)], 0.012,
                                      seg=3), occ=False)
        raft_rack(c, P3(78.3, s * 6.3, DECK01_Y), n=3, along=(0, 0, -1), spacing=0.85, stack=2)
    # side radomes, decoy launchers, searchlights, lockers aft of the funnel
    m.b.node('Sensors')
    for s in (1, -1):
        SN.capsule(c, P3(97.0, s * 4.0, DECK01_Y), r=0.89, cyl=1.26, ped_h=0.4)
    m.b.node('Weapons')
    for s in (1, -1):
        for B in (91.8, 93.6):
            WP.kt216(c, P3(B, s * 6.6, DECK01_Y), facing=s * math.pi / 2, el=40.0, ped=0.45)
        WP.mtpu(c, P3(73.8, s * 6.9, DECK01_Y), facing=s * math.pi / 2)
    m.b.node('Superstructure')
    for s in (1, -1):
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
    house(c, chamfer_rect(HANGAR_B0 + 0.4, HANGAR_B1 - 0.15, 2.2, cf=0.4, ca=0.0), HANGAR_Y - 0.05, HANGAR_Y + 0.35,
          top='paint')
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
    # aft mast: raked fin with a ladder up its face rising from the 01 deck, pole with three yards on top
    yr = HANGAR_Y + 0.35
    fin = []
    for (B, y) in ((97.9, DECK01_Y), (99.6, DECK01_Y), (99.6, 14.5), (98.4, 14.5)):
        fin.append([P3(B, x, y) for x in (0.3, -0.3)])
    FT.solid(c, c.paint, [[fin[0][0], fin[1][0], fin[2][0], fin[3][0]], [fin[0][1], fin[1][1], fin[2][1], fin[3][1]],
                          [fin[0][0], fin[0][1], fin[1][1], fin[1][0]], [fin[1][0], fin[1][1], fin[2][1], fin[2][0]],
                          [fin[2][0], fin[2][1], fin[3][1], fin[3][0]], [fin[3][0], fin[3][1], fin[0][1], fin[0][0]]])
    ladder(c, P3(97.9, 0.0, DECK01_Y) + np.array([0, 0, 0.04]), P3(98.4, 0.0, 14.5) + np.array([0, 0, 0.04]), width=0.42,
           normal=normalize(np.array([0.0, 0.5 / 4.5, 1.0])))
    BM = 99.0
    pm = P3(BM, 0.0, 14.5)
    c.add(c.paint, tube_path([pm, pm + np.array([0, 3.8, 0])], 0.11, seg=8))
    for (yy, half) in ((15.5, 1.1), (16.4, 0.8), (17.5, 0.55)):
        c.add(c.paint, tube_path([P3(BM, half, yy), P3(BM, -half, yy)], 0.04, seg=5))
        for sx in (1, -1):
            PT.lamp(c, P3(BM, sx * (half - 0.05), yy + 0.04), 'white')
    PT.lamp(c, pm + np.array([0, 3.8, 0]), 'white')
    m.b.node('Sensors')
    SN.pal_n(c, P3(98.9, 0.0, 14.5) + np.array([0, 0, -0.55]), yaw=0.3, L=1.8)
    for sx in (1, -1):
        SN.small_dome(c, P3(100.4, sx * 2.0, yr), r=0.55, post=0.3)
    SN.capsule(c, P3(102.6, 0.0, yr), r=1.25, cyl=1.17, ped_h=0.25)
    pd = P3(107.3, 0.0, yr)                                  # disc antenna on a short post
    c.add(c.paint, cylinder(0.12, 0.45, seg=8), xf=M(pd))
    c.add(c.sw('light'), lathe([(0.0, 0.45), (0.62, 0.47), (0.66, 0.52), (0.6, 0.6), (0.0, 0.62)], seg=20), xf=M(pd))
    m.b.node('Superstructure')
    c.add(c.paint, rbox(1.3, 0.8, 0.9, r=0.08, seg=1, bevel=0.04), xf=M(P3(111.0, 0.6, yr)))
    whip(c, P3(104.0, -1.4, yr), 20.7 - yr, r=0.05)
    whip(c, P3(109.3, 1.4, yr), 25.0 - yr, r=0.06)
    for s in (1, -1):
        raft_rack(c, P3(101.4, s * (hw_at(101.4, HANGAR_Y) - 0.7), HANGAR_Y), n=3, along=(0, 0, -1), spacing=0.85)
        raft_rack(c, P3(101.4, s * (hw_at(101.4, HANGAR_Y) - 1.45), HANGAR_Y), n=3, along=(0, 0, -1), spacing=0.85)
    m.b.node('Weapons')
    for s in (1, -1):
        WP.palash(c, P3(111.2, s * 5.0, WELL_Y), facing=s * math.radians(100), node='Palash_' + ('P' if s > 0 else 'S'))


# =============================================================================================
# helideck fittings
# =============================================================================================
LSO_CAB = (113.9, 115.3, 3.0, 5.8, 8.0)     # helicopter control cab on the port side of the hangar face


def helideck_fittings(m):
    c = m.ctx
    m.b.node('Hull')
    # safety nets raised as a fence round the deck edge and the transom (stern photographs, profile drawing)
    for s in (1, -1):
        pts = [P3(B, s * (float(DECK_HB(B)) - 0.1), float(KNUCKLE(B))) for B in (HELIDECK_B + 1.4, 124.0, 134.7)]
        FT.safety_net(c, pts)
    for xa, xb in ((7.95, 1.1), (-1.1, -7.95)):
        FT.safety_net(c, [P3(134.85, xa, float(KNUCKLE(134.85))), P3(134.85, xb, float(KNUCKLE(134.85)))])
    # helicopter control cab
    b0, b1, x0, x1, ytop = LSO_CAB
    FT.cab(c, b0, b1, x0, x1, dk(b1, x0), ytop)
    # ensign staff and flag
    p0 = P3(134.5, 0.0, dk(134.5)); p1 = p0 + np.array([0, 3.6, -1.1])
    c.add(c.paint, tube_path([p0, p1], 0.06, seg=5))
    fa = p1 + np.array([0, -0.12, 0]); fb = p1 + np.array([0, -1.2, 0.15])
    c.add(c.rect('ensign'), _quad(fb, fb + np.array([0.0, 0, -1.8]), fa + np.array([0.0, 0, -1.8]), fa,
                                  uv=[(0, 1), (1, 1), (1, 0), (0, 0)]))
    for s in (1, -1):
        for B in (117.0, 125.0, 132.5):
            bollard(c, P3(B, s * 7.0, dk(B, 7.0)))
        for B in np.arange(116.5, 134.6, 2.25):               # amber deck-edge lights
            c.add(c.sw('orange'), sphere(0.08, seg=6, rings=3, hemi=True), xf=M(P3(B, s * 7.75, dk(B, 7.75))))


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
