"""Slava-class cruiser (Project 1164) -- hull, deck, texture atlas and GLB export.

Frame (same as the DDG-51 asset): +Y up, +Z bow, +X port, metres, design waterline at y = 0.
B = metres aft of the stem head (drawing convention), z = Z_BOW - B.
"""
import math
from collections import OrderedDict
import numpy as np

from meshkit import *
from meshkit import _quad
import atlas as st
from texkit import Layers, value_noise, srgb, png_bytes
from kit import Ctx
from hull import (Hull, zB, Bz, LOA, Z_BOW, Z_STERN, SHEER, DECK_HB, WL_HB, KEEL, keel_y, section, deck_y,
                   transom_B, STEP_B0, STEP_B1, QD_Y, MAIN_Y, TRANSOM_BOT, TRANSOM_TOP)

PAL = st.PAL
ATL = 'Slava_Atlas'
HULL_YTOP, HULL_YBOT = 11.5, -8.5
DECK_HALF = 11.5


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


BAND_UV = UVSpace(ATL, 'rect', rect=(0, 0, 1, 1), clamp=False)              # deck: subdivided for AO
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


# ---------------------------------------------------------------------------- hull
def build_hull(m, transom_hole=(2.4, 0.4, 3.4)):
    H = Hull(nrows=22)
    m.hull = H
    b = m.b
    b.node('Hull')
    for side in (+1, -1):
        P, I = H.side_mesh(side)
        N = compute_smooth_normals(P, I)
        b.add(HULL_UV, P, N, hull_band_uv(P), I)
    # ---- transom (horizontal strips, recess hole for the towed-sonar door)
    x, y, z = H.transom_outline()
    hole = transom_hole
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
    # recess for the VDS door
    hw, ha, hb_ = hole
    depth = 2.2
    za, zb_ = zB(transom_B(ha)), zB(transom_B(hb_))
    dk = m.ctx.sw('dark')
    m.ctx.add(dk, _quad((-hw, ha, za), (-hw, ha, za + depth), (-hw, hb_, zb_ + depth), (-hw, hb_, zb_)))
    m.ctx.add(dk, _quad((hw, ha, za + depth), (hw, ha, za), (hw, hb_, zb_), (hw, hb_, zb_ + depth)))
    m.ctx.add(dk, _quad((hw, ha, za), (-hw, ha, za), (-hw, ha, za + depth), (hw, ha, za + depth)))
    m.ctx.add(dk, _quad((-hw, hb_, zb_), (hw, hb_, zb_), (hw, hb_, zb_ + depth), (-hw, hb_, zb_ + depth)))
    # door panel inside the recess, with the crest decal
    m.ctx.add(m.ctx.rect('vds_door'), _quad((hw, ha, za + depth * 0.6), (-hw, ha, za + depth * 0.6),
                                            (-hw, hb_, zb_ + depth * 0.6), (hw, hb_, zb_ + depth * 0.6),
                                            uv=[(1, 1), (0, 1), (0, 0), (1, 0)]))
    # ---- weather deck (5 points across for camber)
    xe, ye, ze = H.deck_edge()
    P = []
    fr = [1.0, 0.5, 0.0, -0.5, -1.0]
    for xx, yy, zz in zip(xe, ye, ze):
        hb = max(xx, 1e-3)
        for f in fr:
            cam = 0.25 * min(1.0, hb / 10.0) * (1.0 - f * f)
            P.append((f * xx, yy + cam, zz))
    P = np.array(P)
    k = len(fr)
    I = []
    for i in range(len(xe) - 1):
        for j in range(k - 1):
            a = i * k + j; c = (i + 1) * k + j
            I += [(a, c, c + 1), (a, c + 1, a + 1)]
    I = np.array(I)
    t0 = I[0]
    if np.cross(P[t0[1]] - P[t0[0]], P[t0[2]] - P[t0[0]])[1] < 0:
        I = I[:, ::-1]
    N = compute_smooth_normals(P, I)
    b.add(BAND_UV, P, N, deck_band_uv(P), I)
    return H


# ---------------------------------------------------------------------------- textures
def paint_hull_band(L, m):
    y0, y1 = st.BANDS['HULL']
    W = st.W
    hpx = y1 - y0
    zz = Z_STERN + (np.arange(W) + 0.5) / W * LOA
    yy = HULL_YTOP - (np.arange(hpx) + 0.5) / hpx * (HULL_YTOP - HULL_YBOT)
    Y, Zg = np.meshgrid(yy, zz, indexing='ij')
    col = np.zeros((hpx, W, 3)); col[:] = PAL['hull']
    rough = np.full((hpx, W), 0.36); metal = np.full((hpx, W), 0.14)
    red_top, white_top = 0.35, 0.52
    col[Y < white_top] = PAL['boot']
    col[Y < red_top] = PAL['red']
    rough[Y < white_top] = 0.6
    n1 = value_noise(hpx, W, 48, seed=11, octaves=5)
    n2 = value_noise(hpx, W, 6, seed=12, octaves=2)
    col *= (0.95 + 0.08 * n1 + 0.03 * n2)[..., None]
    L.col[y0:y1] = col; L.rough[y0:y1] = rough; L.metal[y0:y1] = metal; L.alpha[y0:y1] = 1.0
    # plating seams / frames (faint, above the boot top)
    for ys in (1.8, 3.6, 5.4):
        r = int(hy_to_px(ys))
        L.rect(0, r, W, r + 1, add_height=0.2, col=None)
    # main-deck level line along the side (visible on photos aft of the forecastle step)
    r = int(hy_to_px(4.45))
    L.rect(int(B_to_px(186.4)), r, int(B_to_px(62.0)), r + 1, col=PAL['hull'] * 0.82, add_height=0.5)
    rng = np.random.default_rng(7)

    def porthole(B, y, rr=0.21):
        cx, cy = B_to_px(B), hy_to_px(y)
        rx, ry = rr * W / LOA, rr * hpx / (HULL_YTOP - HULL_YBOT)
        xa, xb = int(cx - rx - 3), int(cx + rx + 4)
        ya, yb = int(cy - ry - 3), int(cy + ry + 4)
        gy, gx = np.mgrid[ya:yb, xa:xb]
        d = np.hypot((gx + 0.5 - cx) / rx, (gy + 0.5 - cy) / ry)
        L.mask_apply(np.clip(1.0 - np.abs(d - 1.2) * 2.5, 0, 1), col=PAL['hull'] * 0.65, add_height=0.6, x0=xa, y0=ya)
        L.mask_apply(np.clip((1.0 - d) * 2.5, 0, 1), col=PAL['glass'], rough=0.1, metal=0.5, x0=xa, y0=ya)
        sl = int(rng.integers(4, 14))
        strk = np.zeros((sl, xb - xa)); strk[:, (xb - xa) // 2] = np.linspace(0.16, 0.0, sl)
        L.mask_apply(strk, col=srgb('4c4038'), x0=xa, y0=yb)
    # two rows of scuttles (from Kuleshov sheet 1)
    for (Ba, Bb, step, y) in [(14, 33, 2.6, 7.6), (17, 33, 2.6, 5.4), (36, 62, 2.9, 5.9), (64, 160, 3.1, 5.95),
                              (62, 160, 4.4, 3.05), (166, 183, 2.6, 3.1)]:
        for B in np.arange(Ba, Bb, step):
            if 63 < B < 72 and y < 5.6:
                continue          # leave room for the pennant number
            if 150.5 < B < 158 and y > 4.5:
                continue          # torpedo-tube door
            porthole(B + rng.uniform(-0.25, 0.25), y)
    # anchor hawse pockets (dark recess with rust run-off)
    for B, y in ((5.0, 8.3),):
        cx, cy = B_to_px(B), hy_to_px(y)
        rx, ry = 1.05 * W / LOA, 1.3 * hpx / (HULL_YTOP - HULL_YBOT)
        gy, gx = np.mgrid[int(cy - ry - 3):int(cy + ry + 4), int(cx - rx - 3):int(cx + rx + 4)]
        d = np.hypot((gx + 0.5 - cx) / rx, (gy + 0.5 - cy) / ry)
        L.mask_apply(np.clip((1.0 - d) * 8, 0, 1), col=PAL['black'] * 2.2, add_height=-0.8,
                     x0=int(cx - rx - 3), y0=int(cy - ry - 3))
        st_len = int(hy_to_px(1.5) - cy)
        strk = np.zeros((st_len, int(2 * rx + 8)))
        for kx in range(strk.shape[1]):
            strk[:, kx] = np.linspace(0.2, 0.0, st_len) * math.exp(-((kx - strk.shape[1] / 2) / (rx * 0.35)) ** 2)
        L.mask_apply(strk, col=srgb('6a4a35'), x0=int(cx - rx - 4), y0=int(cy + ry))
    # torpedo-tube shutters (151-157 m, 4.9-6.3 m)
    xa, xb = int(B_to_px(157.0)), int(B_to_px(151.2))
    ya, yb = int(hy_to_px(6.3)), int(hy_to_px(4.9))
    L.rect(xa - 1, ya - 1, xb + 1, yb + 1, col=PAL['hull'] * 0.6, add_height=0.8)
    L.rect(xa + 1, ya + 1, xb - 1, yb - 1, col=PAL['hull'] * 0.92, add_height=-0.4)
    for k in range(1, 4):
        c = xa + (xb - xa) * k // 4
        L.rect(c, ya + 1, c + 1, yb - 1, col=PAL['hull'] * 0.7)
    # scupper / rust streaks under the deck edge
    for B in np.arange(6.0, 184.0, 3.3):
        ytop = float(SHEER(B)) - 0.15
        cx = B_to_px(B) + rng.uniform(-3, 3)
        ln = rng.uniform(0.8, 3.0)
        a_, b_ = int(hy_to_px(ytop)), int(hy_to_px(ytop - ln))
        if b_ <= a_:
            continue
        strk = np.zeros((b_ - a_, 3))
        strk[:, 1] = np.linspace(rng.uniform(0.05, 0.14), 0.0, b_ - a_)
        strk[:, 0] = strk[:, 1] * 0.4; strk[:, 2] = strk[:, 1] * 0.4
        L.mask_apply(strk, col=srgb('42382f'), x0=int(cx) - 1, y0=a_)
    # draft marks
    for B in (13.0, 178.0):
        cx = B_to_px(B)
        for yv in np.arange(-3.0, 0.31, 0.5):
            r = int(hy_to_px(yv))
            L.rect(cx - 2, r - 1, cx + 2, r + 1, col=PAL['white'] * 0.9)
    # keep the last metres (transom mapping area) plain
    c1 = int(B_to_px(183.6))
    L.col[y0:y1, :c1] = col[:, :c1]


def paint_deck_band(L, m):
    y0, y1 = st.BANDS['DECK']
    W = st.W
    hpx = y1 - y0
    n1 = value_noise(hpx, W, 14, seed=21, octaves=4)
    grit = np.random.default_rng(22).random((hpx, W))
    zz = Z_STERN + (np.arange(W) + 0.5) / W * LOA
    xs = DECK_HALF - (np.arange(hpx) + 0.5) / hpx * 2 * DECK_HALF
    Xg = xs[:, None]
    red = np.ones((hpx, W, 3)) * PAL['deck_red']
    grey = np.ones((hpx, W, 3)) * PAL['deck']
    hb = np.array([float(DECK_HB(b)) for b in (Z_BOW - zz)])[None, :]
    # grey waterways along the edges and around the quarterdeck, red-brown elsewhere
    edge = np.clip((np.abs(Xg) - (hb - 0.9)) / 0.08, 0, 1)
    mask = 1.0 - edge
    col = red * mask[..., None] + grey * (1 - mask[..., None])
    col *= (0.93 + 0.1 * n1 + 0.04 * (grit - 0.5))[..., None]
    L.col[y0:y1] = col
    L.rough[y0:y1] = 0.82; L.metal[y0:y1] = 0.03; L.alpha[y0:y1] = 1.0
    L.height[y0:y1] += grit * 0.25
    for x in np.arange(-DECK_HALF, DECK_HALF, 1.8):
        r = int(dx_to_px(x))
        L.rect(0, r, W, r + 1, add_height=-0.4)
    for zf in np.arange(Z_STERN, Z_BOW, 7.0):
        c = int(z_to_px(zf))
        L.rect(c, y0, c + 1, y1, add_height=-0.4)
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
    mats['Slava_Atlas'] = ({"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 1.0,
                                                     "roughnessFactor": 1.0},
                            "alphaMode": "MASK", "alphaCutoff": 0.5, "doubleSided": True, "_normalScale": 1.0},
                           {'baseColor': png_bytes(atlas_L.base_png(True)),
                            'metallicRoughness': png_bytes(atlas_L.mr_png()),
                            'normal': png_bytes(atlas_L.normal_png(strength=1.5, wrap=False))})
    mats['Slava_Paint'] = ({"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 1.0,
                                                     "roughnessFactor": 1.0},
                            "doubleSided": True, "_normalScale": 1.0},
                           {'baseColor': png_bytes(paint_L.base_png(False)),
                            'metallicRoughness': png_bytes(paint_L.mr_png()),
                            'normal': png_bytes(paint_L.normal_png(strength=1.2))})
    mats['Slava_Deck'] = ({"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 1.0,
                                                    "roughnessFactor": 1.0},
                           "doubleSided": True, "_normalScale": 1.0},
                          {'baseColor': png_bytes(deck_L.base_png(False)),
                           'metallicRoughness': png_bytes(deck_L.mr_png()),
                           'normal': png_bytes(deck_L.normal_png(strength=1.0))})
    return mats


def export(m, path, preview=False, ao=None):
    atlas = paint_atlas(m)
    paint = st.make_tile_paint(1024, seed=3, base=PAL['super'])
    paint.rough = 0.34 + 0.12 * (paint.rough - 0.55) / 0.15
    paint.metal[:] = 0.15
    deck = st.make_tile_deck(512, seed=5, base=PAL['deck_red'])
    mats = build_materials(atlas, paint, deck)
    extras = {'title': 'Slava-class cruiser (Project 1164) - low poly',
              'units': 'metres; +Y up, +Z forward (bow), +X port; design waterline at y = 0',
              'references': 'A. Kuleshov 1:100 plans (1993); US Navy / Wikimedia Commons photographs'}
    builder_to_glb(m.b, mats, path, asset_extras=extras, root_name='Slava_Class_Cruiser', ao=ao)
    if preview:
        atlas.base_png(True).save(path.replace('.glb', '_atlas_preview.png'))
    return atlas
