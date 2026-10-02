"""Admiral Gorshkov-class frigate (Project 22350) -- hull, upper hull, decks, appendages, texture atlas and GLB.

Frame (same as the DDG-51 and Slava assets): +Y up, +Z bow, +X port, metres, design waterline at y = 0.
B = metres aft of the stem head (drawing convention), z = Z_BOW - B.
"""
import math
from collections import OrderedDict
import numpy as np

from meshkit import *
from meshkit import _quad
import atlas as st
from texkit import Layers, value_noise, srgb, png_bytes
from kit import Ctx, M, P3, propeller
from hull import (Hull, zB, LOA, Z_BOW, Z_STERN, KNUCKLE, DECK_HB, section, deck_y, transom_B, TRANSOM_BOT,
                  TRANSOM_TOP, TUMBLE, upper_hb, UPPER, HELIDECK_B, STEM_KN_B, stem_y, BULWARK_TOP, P1D,
                  DOME_B0, DOME_B1)

PAL = st.PAL
ATL = 'Gorshkov_Atlas'
HULL_YTOP, HULL_YBOT = 12.6, -7.2
DECK_HALF = 8.3
BULWARK_T = 0.1                       # forecastle bulwark plate thickness
OPENING = (38.4, 44.5, 7.7)           # decoy-launcher cut-out in the bulwark: B0, B1, top of the opening
UKSK_FRONT = 44.5                     # forecastle bulwark ends at the raised UKSK block


# ---------------------------------------------------------------------------- uv helpers
def hull_band_uv(P):
    u = (P[:, 2] - Z_STERN) / LOA
    y0, y1 = st.BANDS['HULL']
    v = y0 + (HULL_YTOP - P[:, 1]) / (HULL_YTOP - HULL_YBOT) * (y1 - y0)
    return np.stack([u, v / st.W], axis=1)


def deck_band_uv(P):
    u = (P[:, 2] - Z_STERN) / LOA
    y0, y1 = st.BANDS['DECK']
    v = y0 + (DECK_HALF - P[:, 0]) / (2 * DECK_HALF) * (y1 - y0)
    return np.stack([u, v / st.W], axis=1)


def z_to_px(z):
    return (z - Z_STERN) / LOA * st.W


def B_to_px(B):
    return z_to_px(zB(B))


def hy_to_px(y):
    y0, y1 = st.BANDS['HULL']
    return y0 + (HULL_YTOP - y) / (HULL_YTOP - HULL_YBOT) * (y1 - y0)


def dx_to_px(x):
    y0, y1 = st.BANDS['DECK']
    return y0 + (DECK_HALF - x) / (2 * DECK_HALF) * (y1 - y0)


BAND_UV = UVSpace(ATL, 'rect', rect=(0, 0, 1, 1), clamp=False)              # decks: subdivided for AO
HULL_UV = UVSpace(ATL, 'rect', rect=(0, 0, 1, 1), clamp=False, subdiv=False)


# ---------------------------------------------------------------------------- model container
class Model:
    def __init__(self):
        self.b = Builder()
        self.layout = st.AtlasLayout()
        self.ctx = Ctx(self.b, self.layout)
        self.deck_marks = []
        self.hull_marks = []
        self.decals = OrderedDict()
        self._sky = [[0, 1012, st.W]]         # skyline of the decal area: [x, first free row, width]
        self.hull = None

    def alloc(self, name, w, h, painter):
        """place a w x h decal in the RECT area (skyline packing: lowest free spot, then leftmost)."""
        fw, fh = w + 2, h + 2                 # 2 px gutter against mip bleeding
        best = None
        for i, (sx, _, _) in enumerate(self._sky):
            if sx + fw > st.W:
                break
            y, rem, j = 0, fw, i
            while rem > 0:
                y = max(y, self._sky[j][1]); rem -= self._sky[j][2]; j += 1
            if y + h <= st.SWATCH_Y0 - 2 and (best is None or (y, sx) < (best[1], best[0])):
                best = (sx, y)
        if best is None:
            raise RuntimeError('atlas full at ' + name)
        x, y = best
        sky = []
        for (sx, sy, sw) in self._sky:            # cut the covered span out of the skyline
            if sx + sw <= x or sx >= x + fw:
                sky.append([sx, sy, sw])
                continue
            if sx < x:
                sky.append([sx, sy, x - sx])
            if sx + sw > x + fw:
                sky.append([x + fw, sy, sx + sw - x - fw])
        sky.append([x, y + fh, fw])
        sky.sort()
        self._sky = []
        for s in sky:                              # merge neighbours at the same height
            if self._sky and self._sky[-1][1] == s[1] and self._sky[-1][0] + self._sky[-1][2] == s[0]:
                self._sky[-1][2] += s[2]
            else:
                self._sky.append(s)
        rect = (x, y, x + w, y + h)
        self.layout.add(name, (rect[0] + 1, rect[1] + 1, rect[2] - 1, rect[3] - 1))
        self.decals[name] = (rect, painter)
        return name


# ---------------------------------------------------------------------------- geometry helpers
def _grid(P, nu, nv, flip=False):
    """triangles of an nu x nv vertex grid (row-major: u outer, v inner)."""
    I = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a = i * nv + j; b = (i + 1) * nv + j; c = (i + 1) * nv + j + 1; d = i * nv + j + 1
            I += [(a, b, c), (a, c, d)]
    I = np.array(I)
    return I[:, ::-1] if flip else I


def _outward(P, I, ref):
    """flip the whole patch if most faces point toward `ref` (a direction)."""
    fn = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    if np.sum(fn @ np.asarray(ref, float)) < 0:
        I = I[:, ::-1]
    return I


def upper_section(B, y_top):
    """outer upper-hull point pair at B: bottom (knuckle, or the stem line forward of it) and top."""
    if B < STEM_KN_B:
        yb, xb = stem_y(B), 0.0
    else:
        yb, xb = float(KNUCKLE(B)), float(DECK_HB(B))
    xt = max(0.0, float(DECK_HB(B)) - TUMBLE * (y_top - float(KNUCKLE(B))))
    return (xb, yb), (xt, y_top)


def _stations(b0, b1, step):
    n = max(1, int(math.ceil((b1 - b0) / step - 1e-9)))
    return list(np.linspace(b0, b1, n + 1))


def upper_wall(b, side, Bs, top_fn, bot_fn=None, rows=None):
    """outer face of the inward-leaning upper hull between the knuckle (or bot_fn(B)) and top_fn(B)."""
    pts = []
    hmax = 0.0
    for B in Bs:
        (xb, yb), (xt, yt) = upper_section(B, top_fn(B))
        if bot_fn is not None:
            yb = bot_fn(B)
            xb = upper_hb(B, yb) if B >= STEM_KN_B else xb
        pts.append(((xb, yb), (xt, yt)))
        hmax = max(hmax, yt - yb)
    nv = rows or max(2, int(math.ceil(hmax / 1.3)) + 1)
    P = []
    for B, ((xb, yb), (xt, yt)) in zip(Bs, pts):
        for t in np.linspace(0, 1, nv):
            P.append((side * (xb + (xt - xb) * t), yb + (yt - yb) * t, zB(B)))
    P = np.array(P)
    I = _outward(P, _grid(P, len(Bs), nv), (side, 0, 0))
    N = compute_smooth_normals(P, I)
    b.add(HULL_UV, P, N, hull_band_uv(P), I)


# ---------------------------------------------------------------------------- hull
def build_hull(m, transom_hole=(1.5, 1.2, 4.3)):
    H = Hull(nrows=24)
    m.hull = H
    b = m.b
    b.node('Hull')
    for side in (+1, -1):
        P, I = H.side_mesh(side)
        N = compute_smooth_normals(P, I)
        b.add(HULL_UV, P, N, hull_band_uv(P), I)
    build_transom(m, transom_hole)
    build_upper_hull(m)
    build_forecastle_deck(m)
    build_helideck(m)
    build_dome(m)
    build_appendages(m)
    return H


def build_transom(m, hole):
    b = m.b
    x, y, z = m.hull.transom_outline()
    levels = sorted(set([float(v) for v in np.round(y, 4)] + [hole[1], hole[2]]))
    levels = [l for l in levels if y.min() - 1e-6 <= l <= y.max() + 1e-6]
    P, I = [], []

    def half_at(yy):
        return float(np.interp(yy, y, x))

    def quad(xa0, xa1, ya, xb0, xb1, yb):
        za, zb_ = zB(transom_B(ya)), zB(transom_B(yb))
        base = len(P)
        P.extend([(xa1, ya, za), (xa0, ya, za), (xb0, yb, zb_), (xb1, yb, zb_)])
        I.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])
    for ya, yb in zip(levels[:-1], levels[1:]):
        xa, xb = half_at(ya), half_at(yb)
        if ya >= hole[1] - 1e-6 and yb <= hole[2] + 1e-6:
            quad(hole[0], xa, ya, hole[0], xb, yb)
            quad(-xa, -hole[0], ya, -xb, -hole[0], yb)
        else:
            quad(-xa, xa, ya, -xb, xb, yb)
    P = np.array(P); I = np.array(I)
    (b0, y0), (b1, y1) = TRANSOM_BOT, TRANSOM_TOP
    nrm = normalize(np.array([0.0, (b1 - b0), -(y1 - y0)]))
    tn = np.cross(P[I[0, 1]] - P[I[0, 0]], P[I[0, 2]] - P[I[0, 0]])
    if np.dot(tn, nrm) < 0:
        I = I[:, ::-1]
    UV = np.stack([0.004 + (P[:, 0] + 8.0) / 16.0 * 0.010, hull_band_uv(P)[:, 1]], axis=1)
    b.add(HULL_UV, P, np.tile(nrm, (len(P), 1)), UV, I)
    # recess for the towed-array door, with the door panel set back inside it
    hw, ha, hb_ = hole
    depth = 1.6
    za, zb_ = zB(transom_B(ha)), zB(transom_B(hb_))
    dk = m.ctx.sw('dark')
    m.ctx.add(dk, _quad((-hw, ha, za), (-hw, ha, za + depth), (-hw, hb_, zb_ + depth), (-hw, hb_, zb_)))
    m.ctx.add(dk, _quad((hw, ha, za + depth), (hw, ha, za), (hw, hb_, zb_), (hw, hb_, zb_ + depth)))
    m.ctx.add(dk, _quad((hw, ha, za), (-hw, ha, za), (-hw, ha, za + depth), (hw, ha, za + depth)))
    m.ctx.add(dk, _quad((-hw, hb_, zb_), (hw, hb_, zb_), (hw, hb_, zb_ + depth), (-hw, hb_, zb_ + depth)))
    m.ctx.add(m.ctx.rect('vds_door'), _quad((hw, ha, za + depth * 0.5), (-hw, ha, za + depth * 0.5),
                                            (-hw, hb_, zb_ + depth * 0.5), (hw, hb_, zb_ + depth * 0.5),
                                            uv=[(1, 1), (0, 1), (0, 0), (1, 0)]))


def build_upper_hull(m):
    """the inward-leaning upper hull: forecastle bulwark (with the decoy cut-outs), then the sides of the
    UKSK block, the 01 level and the hangar, all flush with the main hull below the knuckle."""
    b, c = m.b, m.ctx
    ob0, ob1, otop = OPENING
    btop = lambda B: float(BULWARK_TOP(B))
    for side in (1, -1):
        # bulwark ahead of the cut-out, the beam over it, and the blocks aft
        upper_wall(b, side, _stations(0.0, ob0, 1.5), btop)
        upper_wall(b, side, _stations(ob0, ob1, 1.5), btop, bot_fn=lambda B: otop, rows=2)
        for b0, b1, h in UPPER[1:]:
            upper_wall(b, side, _stations(b0, b1, 2.0), lambda B, h=h: h)
    # bulwark inner face, top cap and the cut-out edges
    t = BULWARK_T
    Bi0 = STEM_KN_B + 0.25

    def inner(B, y):
        return upper_hb(B, y) - t
    for side in (1, -1):
        for (Ba, Bb, bot) in ((Bi0, ob0, None), (ob0, ob1, otop)):
            Bs = _stations(Ba, Bb, 1.0)
            P = []
            for B in Bs:
                yb = float(deck_y(B, upper_hb(B, float(KNUCKLE(B))) - t)) if bot is None else bot
                yt = btop(B)
                for y in (yb, yt):
                    P.append((side * inner(B, y), y, zB(B)))
            P = np.array(P)
            I = _outward(P, _grid(P, len(Bs), 2), (-side, 0, 0))
            c.add(c.paint, (P, compute_smooth_normals(P, I), np.stack([P[:, 2], P[:, 1]], axis=1), I))
            if bot is not None:                   # underside of the beam over the cut-out
                P = []
                for B in Bs:
                    P.append((side * inner(B, bot), bot, zB(B)))
                    P.append((side * upper_hb(B, bot), bot, zB(B)))
                P = np.array(P)
                I = _outward(P, _grid(P, len(Bs), 2), (0, -1, 0))
                c.add(c.paint, (P, compute_smooth_normals(P, I), np.stack([P[:, 2], P[:, 0]], axis=1), I))
        # jambs of the cut-out (from the deck to the beam)
        for Bj, facing in ((ob0, -1), (ob1, 1)):
            ydk = float(deck_y(Bj, upper_hb(Bj, float(KNUCKLE(Bj))) - t))
            yk = float(KNUCKLE(Bj))
            pts = [(side * upper_hb(Bj, yk), yk), (side * inner(Bj, ydk), ydk), (side * inner(Bj, otop), otop),
                   (side * upper_hb(Bj, otop), otop)]
            P = np.array([(x, y, zB(Bj)) for (x, y) in pts])
            I = np.array([(0, 1, 2), (0, 2, 3)])
            I = _outward(P, I, (0, 0, facing))
            c.add(c.paint, (P, np.tile([0, 0, facing], (4, 1)).astype(float), P[:, :2], I))
    # top cap: solid over the stem, then a 0.1 m plate on each side
    Bs = _stations(0.0, ob1, 1.0)
    Bstem = [B for B in Bs if B <= Bi0] + [Bi0]
    P = []
    for B in Bstem:
        yt = btop(B)
        xo = upper_hb(B, yt) if B >= STEM_KN_B else upper_section(B, yt)[1][0]
        for x in (xo, -xo):
            P.append((x, yt, zB(B)))
    P = np.array(P)
    I = _outward(P, _grid(P, len(Bstem), 2), (0, 1, 0))
    c.add(c.paint, (P, np.tile([0, 1.0, 0], (len(P), 1)), np.stack([P[:, 2], P[:, 0]], axis=1), I))
    Bside = [Bi0] + [B for B in Bs if B > Bi0]
    for side in (1, -1):
        P = []
        for B in Bside:
            yt = btop(B)
            xo = upper_hb(B, yt)
            for x in (xo, xo - t):
                P.append((side * x, yt, zB(B)))
        P = np.array(P)
        I = _outward(P, _grid(P, len(Bside), 2), (0, 1, 0))
        c.add(c.paint, (P, np.tile([0, 1.0, 0], (len(P), 1)), np.stack([P[:, 2], P[:, 0]], axis=1), I))
    # transverse face closing the solid stem top behind the stem
    yt = btop(Bi0); yd = float(deck_y(Bi0, 0.0))
    xi = upper_hb(Bi0, yt) - t
    P = np.array([(-xi, yd, zB(Bi0)), (xi, yd, zB(Bi0)), (xi, yt, zB(Bi0)), (-xi, yt, zB(Bi0))])
    I = _outward(P, np.array([(0, 1, 2), (0, 2, 3)]), (0, 0, -1))
    c.add(c.paint, (P, np.tile([0, 0, -1.0], (4, 1)), P[:, :2], I))


def build_forecastle_deck(m):
    """forecastle deck inside the bulwark, from the stem to the UKSK block (camber 0.2 m)."""
    b = m.b
    Bs = _stations(STEM_KN_B + 0.25, UKSK_FRONT, 0.75)
    fr = [1.0, 0.66, 0.33, 0.0, -0.33, -0.66, -1.0]
    P = []
    for B in Bs:
        xe = upper_hb(B, float(KNUCKLE(B))) - BULWARK_T
        for f in fr:
            P.append((f * xe, float(deck_y(B, f * xe)), zB(B)))
    P = np.array(P)
    I = _outward(P, _grid(P, len(Bs), len(fr)), (0, 1, 0))
    b.add(BAND_UV, P, compute_smooth_normals(P, I), deck_band_uv(P), I)


def build_helideck(m):
    """helideck on the main deck from the hangar to the transom; mapped onto the 'helideck' decal."""
    c = m.ctx
    Bs = _stations(HELIDECK_B, TRANSOM_TOP[0], 1.0)
    fr = [1.0, 0.66, 0.33, 0.0, -0.33, -0.66, -1.0]
    P, UV = [], []
    hw = 8.1
    for B in Bs:
        xe = float(DECK_HB(B))
        for f in fr:
            P.append((f * xe, float(deck_y(B, f * xe)), zB(B)))
            UV.append(((f * xe + hw) / (2 * hw), (B - HELIDECK_B) / (TRANSOM_TOP[0] - HELIDECK_B)))
    P = np.array(P); UV = np.array(UV)
    I = _outward(P, _grid(P, len(Bs), len(fr)), (0, 1, 0))
    c.add(c.rect('helideck', clamp=False), (P, compute_smooth_normals(P, I), UV, I))


# ---------------------------------------------------------------------------- sonar dome and appendages
DOME_YT = P1D([DOME_B0, 9.9, 10.2, 10.5, 11.0, 13.0, 16.0, DOME_B1], [-4.3, -3.4, -2.6, -1.95, -1.9, -2.3, -2.7, -3.2])
DOME_YB = P1D([DOME_B0, 9.9, 10.2, 10.6, 11.0, 11.5, 15.0, DOME_B1], [-4.3, -5.05, -5.75, -6.4, -6.7, -6.92, -6.92, -6.65])
DOME_W = P1D([DOME_B0, 9.9, 10.2, 10.6, 11.0, 12.0, 14.0, DOME_B1], [0.0, 0.6, 1.0, 1.3, 1.5, 1.7, 1.8, 1.65])


def build_dome(m):
    """bow sonar dome: super-elliptic sections, rounded nose, flat aft face under the keel."""
    b = m.b
    Bs = [DOME_B0 + 0.001] + list(np.linspace(9.85, 11.0, 7)) + list(np.linspace(11.5, DOME_B1, 9))
    na = 24
    secs = []
    for B in Bs:
        yt, yb, w = float(DOME_YT(B)), float(DOME_YB(B)), max(0.01, float(DOME_W(B)))
        yc, ry = (yt + yb) / 2, (yt - yb) / 2
        ring = []
        for k in range(na):
            a = 2 * math.pi * k / na
            ca, sa = math.cos(a), math.sin(a)
            ring.append((w * math.copysign(abs(sa) ** (2 / 2.6), sa), yc - ry * math.copysign(abs(ca) ** (2 / 2.6), ca),
                         zB(B)))
        secs.append(ring)
    P = np.array([p for r in secs for p in r])
    I = []
    for i in range(len(secs) - 1):
        for k in range(na):
            a = i * na + k; bq = i * na + (k + 1) % na
            c2 = (i + 1) * na + (k + 1) % na; d = (i + 1) * na + k
            I += [(a, bq, c2), (a, c2, d)]
    I = np.array(I)
    cen = np.array([0.0, -4.6, zB(15.0)])
    fn = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    if np.sum(np.sum(fn * (P[I].mean(axis=1) - cen), axis=1) < 0) > len(I) / 2:
        I = I[:, ::-1]
    N = compute_smooth_normals(P, I)
    b.add(HULL_UV, P, N, hull_band_uv(P), I)
    # flat aft face
    ring = np.array(secs[-1])
    Pc = np.vstack([ring, [[0.0, ring[:, 1].mean(), ring[0, 2]]]])
    Ic = np.array([(k, (k + 1) % na, na) for k in range(na)])
    Ic = _outward(Pc, Ic, (0, 0, -1))
    b.add(HULL_UV, Pc, np.tile([0, 0, -1.0], (len(Pc), 1)), hull_band_uv(Pc), Ic)


PROP_B, PROP_X, PROP_Y, PROP_R = 126.0, 3.2, -3.1, 1.6
RUDDER = (128.4, 131.4, -0.9, -4.3)        # B0, B1, top, bottom
BILGE_KEEL = (58.0, 102.0, -2.3)


def build_appendages(m):
    c, b = m.ctx, m.b
    b.node('Appendages')
    red = c.sw('red')
    for s in (1, -1):
        # shaft from the hull to the propeller, with a strut bracket (A-frame) ahead of the propeller
        a = P3(98.0, s * 2.4, -3.9); p = P3(PROP_B - 0.6, s * PROP_X, PROP_Y)
        c.add(c.sw('steel'), tube_path([a, p], 0.22, seg=8))
        c.add(red, tube_path([P3(PROP_B - 4.5, s * PROP_X, PROP_Y), P3(PROP_B - 3.0, s * PROP_X, PROP_Y)], 0.42, seg=10))
        for dx in (-0.9, 0.9):
            top = P3(PROP_B - 2.8, s * (PROP_X + dx), -1.2)
            c.add(red, beam(top, P3(PROP_B - 3.6, s * PROP_X, PROP_Y), 0.16, 0.6, up=(0, 0, 1)))
        propeller(c, P3(PROP_B, s * PROP_X, PROP_Y), r=PROP_R, blades=5, hand=s)
        # twin spade rudders behind the propellers
        B0, B1, yt, yb = RUDDER
        poly = [(zB(B0), yt), (zB(B1), yt), (zB(B1 - 0.2), yb + 0.3), (zB(B1 - 0.6), yb), (zB(B0 + 0.3), yb),
                (zB(B0), yb + 0.4)]
        from kit import extrude_x
        c.add(red, extrude_x(poly, s * PROP_X - 0.18, s * PROP_X + 0.18))
        c.add(red, cylinder(0.2, 1.4, seg=8, y0=yt - 0.2), xf=M(P3(B0 + 1.2, s * PROP_X, 0.0)))
        # bilge keel along the turn of the bilge
        B0, B1, y = BILGE_KEEL
        pts = []
        for B in np.linspace(B0, B1, 12):
            xs, ys = section(B, 64)
            x = float(np.interp(y, ys, xs))
            pts.append((B, x))
        P = []
        for B, x in pts:
            P.append((s * x, y, zB(B)))
            P.append((s * (x + 0.55), y - 0.35, zB(B)))
        P = np.array(P)
        I = _grid(P, len(pts), 2)
        P2 = P.copy()
        c.add(red, (P2, compute_smooth_normals(P2, I), P2[:, [2, 1]], I))
        c.add(red, (P2, -compute_smooth_normals(P2, I), P2[:, [2, 1]], I[:, ::-1]))


# ---------------------------------------------------------------------------- textures
def paint_hull_band(L, m):
    y0, y1 = st.BANDS['HULL']
    W = st.W
    hpx = y1 - y0
    yy = HULL_YTOP - (np.arange(hpx) + 0.5) / hpx * (HULL_YTOP - HULL_YBOT)
    Y = np.repeat(yy[:, None], W, axis=1)
    col = np.zeros((hpx, W, 3)); col[:] = PAL['hull']
    rough = np.full((hpx, W), 0.4); metal = np.full((hpx, W), 0.12)
    red_top, boot_top = 0.12, 0.3
    col[Y < boot_top] = PAL['boot']
    col[Y < red_top] = PAL['red']
    rough[Y < boot_top] = 0.6
    n1 = value_noise(hpx, W, 48, seed=11, octaves=5)
    n2 = value_noise(hpx, W, 6, seed=12, octaves=2)
    col *= (0.95 + 0.08 * n1 + 0.03 * n2)[..., None]
    L.col[y0:y1] = col; L.rough[y0:y1] = rough; L.metal[y0:y1] = metal; L.alpha[y0:y1] = 1.0
    # faint plating seams
    for ys in (1.6, 3.2, 6.8, 8.6):
        r = int(hy_to_px(ys))
        L.rect(0, r, W, r + 1, add_height=0.2, col=None)
    for zf in np.arange(Z_STERN + 3.0, Z_BOW, 6.0):
        cc = int(z_to_px(zf))
        L.rect(cc, int(hy_to_px(8.5)), cc + 1, int(hy_to_px(0.6)), add_height=0.15, col=None)
    rng = np.random.default_rng(7)
    # rust / grime streaks from the scuppers along the knuckle
    for B in np.arange(4.0, 133.0, 2.7):
        ytop = float(KNUCKLE(B)) - 0.1
        cx = B_to_px(B) + rng.uniform(-3, 3)
        ln = rng.uniform(0.6, 2.6)
        a_, b_ = int(hy_to_px(ytop)), int(hy_to_px(ytop - ln))
        if b_ <= a_:
            continue
        strk = np.zeros((b_ - a_, 3))
        strk[:, 1] = np.linspace(rng.uniform(0.04, 0.11), 0.0, b_ - a_)
        strk[:, 0] = strk[:, 1] * 0.4; strk[:, 2] = strk[:, 1] * 0.4
        L.mask_apply(strk, col=srgb('42382f'), x0=int(cx) - 1, y0=a_)
    # draft marks
    for B in (12.0, 130.0):
        cx = B_to_px(B)
        for yv in np.arange(-3.0, 0.31, 0.5):
            r = int(hy_to_px(yv))
            L.rect(cx - 2, r - 1, cx + 2, r + 1, col=PAL['white'] * 0.9)
    for f in m.hull_marks:
        f(L)


def paint_deck_band(L, m):
    y0, y1 = st.BANDS['DECK']
    W = st.W
    hpx = y1 - y0
    n1 = value_noise(hpx, W, 14, seed=21, octaves=4)
    grit = np.random.default_rng(22).random((hpx, W))
    col = np.ones((hpx, W, 3)) * PAL['deck']
    col *= (0.93 + 0.1 * n1 + 0.04 * (grit - 0.5))[..., None]
    L.col[y0:y1] = col
    L.rough[y0:y1] = 0.85; L.metal[y0:y1] = 0.03; L.alpha[y0:y1] = 1.0
    L.height[y0:y1] += grit * 0.25
    for zf in np.arange(Z_STERN, Z_BOW, 6.0):
        cc = int(z_to_px(zf))
        L.rect(cc, y0, cc + 1, y1, add_height=-0.4)
    for f in m.deck_marks:
        f(L)


def paint_atlas(m):
    L = Layers(st.W, st.W, base=PAL['super'], rough=0.5, metal=0.1, alpha=1.0)
    paint_hull_band(L, m)
    paint_deck_band(L, m)
    st.paint_rail(L); st.paint_window(L); st.paint_ports(L); st.paint_louver(L)
    st.paint_ladder(L); st.paint_lattice(L); st.paint_net(L); st.paint_perf_band(L); st.paint_swatches(L)
    for name, (rect, painter) in m.decals.items():
        painter(L, rect)
    return L


def build_materials(atlas_L, paint_L, deck_L):
    mats = OrderedDict()
    mats['Gorshkov_Atlas'] = ({"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 1.0,
                                                        "roughnessFactor": 1.0},
                               "alphaMode": "MASK", "alphaCutoff": 0.5, "doubleSided": True, "_normalScale": 1.0},
                              {'baseColor': png_bytes(atlas_L.base_png(True)),
                               'metallicRoughness': png_bytes(atlas_L.mr_png()),
                               'normal': png_bytes(atlas_L.normal_png(strength=1.5, wrap=False))})
    mats['Gorshkov_Paint'] = ({"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 1.0,
                                                        "roughnessFactor": 1.0},
                               "doubleSided": True, "_normalScale": 1.0},
                              {'baseColor': png_bytes(paint_L.base_png(False)),
                               'metallicRoughness': png_bytes(paint_L.mr_png()),
                               'normal': png_bytes(paint_L.normal_png(strength=1.2))})
    mats['Gorshkov_Deck'] = ({"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 1.0,
                                                       "roughnessFactor": 1.0},
                              "doubleSided": True, "_normalScale": 1.0},
                             {'baseColor': png_bytes(deck_L.base_png(False)),
                              'metallicRoughness': png_bytes(deck_L.mr_png()),
                              'normal': png_bytes(deck_L.normal_png(strength=1.0))})
    return mats


def export(m, path, preview=False, ao=None):
    atlas = paint_atlas(m)
    paint = st.make_tile_paint(1024, seed=3, base=PAL['super'])
    paint.rough = 0.36 + 0.12 * (paint.rough - 0.55) / 0.15
    paint.metal[:] = 0.14
    deck = st.make_tile_deck(512, seed=5, base=PAL['deck_red'])
    mats = build_materials(atlas, paint, deck)
    extras = {'title': 'Admiral Gorshkov-class frigate (Project 22350) - low poly',
              'units': 'metres; +Y up, +Z forward (bow), +X port; design waterline at y = 0',
              'references': '1:500 general-arrangement drawing (measured only); Russian MoD, Kremlin and '
                            'Mehr News Agency photographs on Wikimedia Commons (CC BY 4.0)'}
    builder_to_glb(m.b, mats, path, asset_extras=extras, root_name='Admiral_Gorshkov_Class_Frigate', ao=ao)
    if preview:
        atlas.base_png(True).save(path.replace('.glb', '_atlas_preview.png'))
    return atlas
