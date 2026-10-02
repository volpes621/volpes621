"""Slava-class (Project 1164) -- superstructure, weapons, sensors and markings.

All positions were measured on the A. Kuleshov 1:100 plans (sheets 1, 2, 5, 6, 7, registered to each
other by cross-correlation) and checked against photographs of Moskva, Varyag and Marshal Ustinov.
B = metres aft of the stem head, x = metres to port, y = metres above the design waterline.
"""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from meshkit import *
from meshkit import _quad
import atlas as st
from texkit import srgb, value_noise
from hull import zB, Z_BOW, Z_STERN, LOA, SHEER, DECK_HB, deck_y, section, STEP_B0, STEP_B1, QD_Y, MAIN_Y
from kit import *
import model as S2
import detail as D
import antennas as A
import deckgear as G
import weapons as WP

PAL = st.PAL

PENNANT = '121'
STARS_ON_CAPS = True
AO = dict(max_len=4.0, min_area=2.5, rays=64, max_dist=6.0, floor=0.25, gamma=2.0)
SHIP_NAME = 'МОСКВА'


def dk(B, x=0.0):
    return deck_y(B, x)


def dmin(B0, B1, x=0.0):
    return min(dk(b, x) for b in np.linspace(B0, B1, 9))


# =============================================================================================
# atlas decals
# =============================================================================================
def _font(path, size):
    return ImageFont.truetype(path, size)


NAME_H = 1.2          # letter band height of the ship's name on the hull (m)
PENNANT_B = 60.0      # hull number centre, metres aft of the bow (Moskva 2009-12 photos)


def text_width_px(text, h, size_frac, spacing):
    f = _font(st.FONT_SANS, int(h * size_frac))
    d = ImageDraw.Draw(Image.new('L', (8, 8)))
    widths = [d.textbbox((0, 0), ch, font=f)[2] for ch in text]
    return int(sum(widths) + spacing * (len(text) - 1)) + 1


def hull_frame(B, y, side):
    """point on the hull surface and its local frame: T (aft along the hull), U (up the surface), N (outward)."""
    xs, ys = section(B, 64)
    x = float(np.interp(y, ys, xs))
    dxdy = (float(np.interp(y + 0.3, ys, xs)) - float(np.interp(y - 0.3, ys, xs))) / 0.6
    xs2, ys2 = section(B + 0.3, 64)
    dxdB = (float(np.interp(y, ys2, xs2)) - x) / 0.3
    p = np.array([side * x, y, zB(B)])
    T = normalize(np.array([side * dxdB, 0.0, -1.0]))
    U = normalize(np.array([side * dxdy, 1.0, 0.0]))
    N = normalize(np.cross(T, U))
    if N[0] * side < 0:
        N = -N
    U = normalize(np.cross(N, T)) if np.dot(np.cross(N, T), U) > 0 else -normalize(np.cross(N, T))
    return p, T, U, N


def hull_decal(c, name, Bc, yc, w, h, side, nseg=8, offset=0.05):
    """decal strip that follows the hull surface along the ship (constant height band)."""
    P, UV, I = [], [], []
    for k in range(nseg + 1):
        t = k / nseg
        B = Bc + (t - 0.5) * w if side > 0 else Bc - (t - 0.5) * w      # left->right as seen from outside
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


def paint_text(L, rect, text, fill, outline=None, stroke=0, size_frac=0.82,
               font_path=None, shadow=None, spacing=0):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    ss = 4
    f = _font(font_path or st.FONT_SANS, int(h * ss * size_frac))

    def render(with_stroke):
        im = Image.new('L', (w * ss, h * ss), 0)
        d = ImageDraw.Draw(im)
        if spacing:
            widths = [d.textbbox((0, 0), ch, font=f)[2] for ch in text]
            total = sum(widths) + spacing * ss * (len(text) - 1)
            bb = d.textbbox((0, 0), text, font=f)
            x = (w * ss - total) / 2
            y = (h * ss - (bb[3] - bb[1])) / 2 - bb[1]
            for ch, cw in zip(text, widths):
                d.text((x, y), ch, fill=255, font=f, stroke_width=(stroke * ss if with_stroke else 0), stroke_fill=255)
                x += cw + spacing * ss
        else:
            bb = d.textbbox((0, 0), text, font=f, stroke_width=stroke * ss)
            x = (w * ss - (bb[2] - bb[0])) / 2 - bb[0]
            y = (h * ss - (bb[3] - bb[1])) / 2 - bb[1]
            d.text((x, y), text, fill=255, font=f, stroke_width=(stroke * ss if with_stroke else 0), stroke_fill=255)
        return np.asarray(im.resize((w, h), Image.LANCZOS)).astype(float) / 255.0
    m_in = render(False)
    m_out = render(True) if (outline is not None or stroke) else m_in
    L.rect(x0, y0, x1, y1, alpha=0.0, col=PAL['hull'])
    if shadow is not None:
        sh = np.roll(np.roll(m_out, 3, axis=0), 3, axis=1)
        L.mask_apply(sh, col=shadow, alpha=1.0, x0=x0, y0=y0)
    if outline is not None:
        L.mask_apply(m_out, col=outline, alpha=1.0, x0=x0, y0=y0)
    L.mask_apply(m_in, col=fill, alpha=1.0, add_height=0.3, x0=x0, y0=y0)
    a = L.alpha[y0:y1, x0:x1]
    L.alpha[y0:y1, x0:x1] = np.where(a > 0.45, 1.0, 0.0)


# hull-number digits as strokes on a unit em (x right, y up from the baseline, height 1), in the style of the
# Russian Navy pennant numbers on the 2009-12 photos: tall, light, a flag on the 1 and no foot
def _arc(cx, cy, rx, ry, a0, a1, n=10):
    return [(cx + rx * math.cos(math.radians(a0 + (a1 - a0) * k / n)), cy + ry * math.sin(math.radians(a0 + (a1 - a0) * k / n)))
            for k in range(n + 1)]


PENNANT_GLYPHS = {
    '0': (0.62, [_arc(0.31, 0.72, 0.24, 0.24, 180, 0) + _arc(0.31, 0.28, 0.24, 0.24, 0, -180) + [(0.07, 0.72)]]),
    '1': (0.42, [[(0.06, 0.8), (0.32, 1.0), (0.32, 0.0)]]),
    '2': (0.62, [_arc(0.31, 0.72, 0.24, 0.24, 165, -15) + [(0.07, 0.0), (0.58, 0.0)]]),
    '3': (0.62, [_arc(0.31, 0.75, 0.23, 0.21, 160, -90), _arc(0.31, 0.28, 0.25, 0.26, 90, -160)]),
    '4': (0.62, [[(0.44, 0.0), (0.44, 1.0), (0.04, 0.32), (0.6, 0.32)]]),
    '5': (0.62, [[(0.56, 1.0), (0.11, 1.0), (0.08, 0.56)] + _arc(0.31, 0.33, 0.26, 0.29, 135, -150)]),
    '6': (0.62, [[(0.5, 0.95)] + _arc(0.31, 0.6, 0.24, 0.38, 70, 180) + _arc(0.31, 0.3, 0.24, 0.28, 180, 540)]),
    '7': (0.6, [[(0.04, 1.0), (0.56, 1.0), (0.2, 0.0)]]),
    '8': (0.62, [_arc(0.31, 0.76, 0.21, 0.22, -90, 270), _arc(0.31, 0.28, 0.25, 0.27, 90, 450)]),
    '9': (0.62, [[(0.12, 0.05)] + _arc(0.31, 0.4, 0.24, 0.38, -110, 0) + _arc(0.31, 0.7, 0.24, 0.28, 0, 360)]),
}
PENNANT_STROKE, PENNANT_GAP = 0.17, 0.32      # stroke width and letter gap, in em


def paint_pennant(L, rect, text, fill, outline=None):
    """hull number filling the decal rect: digits drawn as strokes (any other character falls back to the
    big font), the whole string scaled to the full height and centred."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    if not all(ch in PENNANT_GLYPHS for ch in text):
        return paint_text(L, rect, text, fill, outline=PAL['black'] * 1.3 if outline is None else outline, stroke=2,
                          font_path=st.FONT_BIG, size_frac=0.98, spacing=6)
    sw = PENNANT_STROKE
    adv = sum(PENNANT_GLYPHS[ch][0] for ch in text) + PENNANT_GAP * (len(text) - 1)
    em = min((h - 4) / (1.0 + sw), (w - 4) / (adv + sw))   # pixels per em, strokes included
    ox = (w - adv * em) / 2
    oy = (h + em) / 2

    def draw(d, s, extra):
        x = ox
        lw = (sw + extra) * em * s
        for ch in text:
            gw, strokes = PENNANT_GLYPHS[ch]
            for pl in strokes:
                pts = [((x + px * em) * s, (oy - py * em) * s) for (px, py) in pl]
                d.line(pts, fill=255, width=max(1, int(round(lw))), joint='curve')
                for (qx, qy) in (pts[0], pts[-1]):
                    d.rectangle([qx - lw / 2, qy - lw / 2, qx + lw / 2, qy + lw / 2], fill=255)
            x += (gw + PENNANT_GAP) * em
    L.rect(x0, y0, x1, y1, alpha=0.0, col=PAL['hull'])
    if outline is not None:
        L.mask_apply(L.draw_mask(w, h, lambda d, s: draw(d, s, 0.05)), col=outline, alpha=1.0, x0=x0, y0=y0)
    L.mask_apply(L.draw_mask(w, h, lambda d, s: draw(d, s, 0.0)), col=fill, alpha=1.0, add_height=0.3, x0=x0, y0=y0)
    a = L.alpha[y0:y1, x0:x1]
    L.alpha[y0:y1, x0:x1] = np.where(a > 0.45, 1.0, 0.0)


def paint_helideck(L, rect):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['heli'], alpha=1.0, rough=0.8, metal=0.0)
    n = value_noise(h, w, 24, seed=31, octaves=4)
    L.col[y0:y1, x0:x1] *= (0.9 + 0.16 * n)[..., None]

    def fn(d, s):
        cx, cy = w * 0.5 * s, h * 0.5 * s
        lw = max(2, int(0.022 * w * s))
        r = 0.38 * w * s
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=255, width=lw)
        r2 = 0.12 * w * s
        d.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], outline=255, width=lw)
        # dashed lead-in line toward the hangar (+u = forward)
        for k in range(4):
            xa = cx + r2 + (0.03 + k * 0.07) * w * s
            for yy in (cy - r2 * 0.6, cy + r2 * 0.6):
                d.line([(xa, yy), (xa + 0.04 * w * s, yy)], fill=255, width=lw)
        d.line([(cx + r, cy), (w * s, cy)], fill=255, width=lw)
    mask = L.draw_mask(w, h, fn)
    L.mask_apply(mask, col=PAL['white'] * 0.95, x0=x0, y0=y0, add_height=0.2)


def paint_door(L, rect):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['super'] * 1.02, alpha=1.0)

    def fn(d, s):
        d.rounded_rectangle([2 * s, 2 * s, (w - 2) * s, (h - 2) * s], radius=9 * s, outline=255, width=3 * s)
        cx, cy, r = w * 0.5 * s, h * 0.42 * s, w * 0.22 * s
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=255, width=2 * s)
        d.line([(cx - r, cy), (cx + r, cy)], fill=255, width=2 * s)
        d.line([(cx, cy - r), (cx, cy + r)], fill=255, width=2 * s)
    mask = L.draw_mask(w, h, fn)
    L.mask_apply(mask, col=PAL['super'] * 0.55, add_height=0.8, x0=x0, y0=y0)


def paint_vent(L, rect):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['super'] * 0.85, alpha=1.0)

    def fn(d, s):
        d.rounded_rectangle([6 * s, 6 * s, (w - 6) * s, (h - 6) * s], radius=int(h * 0.3 * s), fill=255)
    m = L.draw_mask(w, h, fn)
    L.mask_apply(m, col=PAL['black'] * 1.3, add_height=-1.0, rough=0.95, x0=x0, y0=y0)
    g = np.linspace(0.0, 0.25, h)[:, None] * m
    L.col[y0:y1, x0:x1] += g[..., None] * np.array([0.45, 0.5, 0.55])
    for k in range(3):
        yy = y0 + h * (0.42 + 0.16 * k)
        L.rect(x0 + w * 0.15, yy, x0 + w * 0.85, yy + 2, col=PAL['mid'])


def paint_raft(L, rect):
    """liferaft canister, v along the canister (lathe profile length): white shell, two dark straps and the
    flange seam between the two halves."""
    x0, y0, x1, y1 = rect
    h = y1 - y0
    pr = np.array(RAFT_PROF)
    Lp = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pr, axis=0), axis=1))])
    L.rect(x0, y0, x1, y1, col=PAL['white'], alpha=1.0, rough=0.45)
    for v in (np.interp(-0.27, pr[2:4, 1], Lp[2:4]) / Lp[-1], np.interp(0.27, pr[2:4, 1], Lp[2:4]) / Lp[-1]):
        L.rect(x0, y0 + h * (v - 0.014), x1, y0 + h * (v + 0.014), col=PAL['dark'] * 1.2, add_height=0.6)
    L.rect(x0, y0 + h * 0.495, x1, y0 + h * 0.505, col=PAL['white'] * 0.8, add_height=0.8)


def paint_funnel_top(L, rect):
    """top of one exhaust stack: 4 elongated uptakes + 2 round ones (sheet 6)."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['dark'] * 0.9, alpha=1.0, rough=0.9)

    def fn(d, s):
        # u = along ship (left = aft), v = across
        for k in range(4):
            cx = (0.12 + 0.12 * k) * w * s
            d.rounded_rectangle([cx - 0.045 * w * s, 0.15 * h * s, cx + 0.045 * w * s, 0.85 * h * s],
                                radius=int(0.045 * w * s), fill=255)
        for (cx, cy, r) in ((0.72, 0.62, 0.17), (0.72, 0.22, 0.09), (0.9, 0.55, 0.12)):
            d.ellipse([(cx * w - r * h) * s, (cy * h - r * h) * s, (cx * w + r * h) * s, (cy * h + r * h) * s], fill=255)
    m = L.draw_mask(w, h, fn)
    L.mask_apply(m, col=PAL['black'] * 0.8, add_height=-1.2, x0=x0, y0=y0)
    for k in range(0, h, 5):
        L.rect(x0, y0 + k, x1, y0 + k + 1, col=None, add_height=0.15)


def paint_ensign(L, rect):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['white'], alpha=1.0, rough=0.9, metal=0.0)

    def fn(d, s):
        t = int(h * 0.17 * s)
        d.line([(0, 0), (w * s, h * s)], fill=255, width=t)
        d.line([(0, h * s), (w * s, 0)], fill=255, width=t)
    L.mask_apply(L.draw_mask(w, h, fn), col=PAL['blue'], x0=x0, y0=y0)


def paint_star(L, rect):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['hull'], alpha=0.0)

    def fn(d, s):
        cx, cy = w * s / 2, h * s / 2
        R, r = w * s * 0.48, w * s * 0.19
        pts = []
        for k in range(10):
            a = -math.pi / 2 + k * math.pi / 5
            rr = R if k % 2 == 0 else r
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        d.polygon(pts, fill=255)
    L.mask_apply(L.draw_mask(w, h, fn), col=PAL['flagred'], alpha=1.0, x0=x0, y0=y0)
    a = L.alpha[y0:y1, x0:x1]
    L.alpha[y0:y1, x0:x1] = np.where(a > 0.5, 1.0, 0.0)


def paint_capstar(L, rect):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['super'], alpha=0.0)

    def star(scale):
        def fn(d, s):
            cx, cy = w * s / 2, h * s / 2 + h * s * 0.03
            R, r = w * s * 0.47 * scale, w * s * 0.19 * scale
            pts = []
            for k in range(10):
                a = -math.pi / 2 + k * math.pi / 5
                rr = R if k % 2 == 0 else r
                pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
            d.polygon(pts, fill=255)
        return fn
    L.mask_apply(L.draw_mask(w, h, star(1.0)), col=PAL['white'], alpha=1.0, x0=x0, y0=y0)
    L.mask_apply(L.draw_mask(w, h, star(0.8)), col=PAL['flagred'], alpha=1.0, x0=x0, y0=y0)
    a = L.alpha[y0:y1, x0:x1]
    L.alpha[y0:y1, x0:x1] = np.where(a > 0.5, 1.0, 0.0)


def paint_crest(L, rect):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['hull'], alpha=0.0)

    def disc(d, s):
        d.ellipse([2 * s, 2 * s, (w - 2) * s, (h - 2) * s], fill=255)

    def ring(d, s):
        d.ellipse([2 * s, 2 * s, (w - 2) * s, (h - 2) * s], outline=255, width=int(w * 0.08 * s))

    def eagle(d, s):
        cx, cy = w * s / 2, h * s / 2
        d.polygon([(cx, cy - h * 0.3 * s), (cx + w * 0.27 * s, cy - h * 0.08 * s), (cx + w * 0.13 * s, cy + h * 0.26 * s),
                   (cx - w * 0.13 * s, cy + h * 0.26 * s), (cx - w * 0.27 * s, cy - h * 0.08 * s)], fill=255)
    L.mask_apply(L.draw_mask(w, h, disc), col=srgb('9c1c1c'), alpha=1.0, x0=x0, y0=y0)
    L.mask_apply(L.draw_mask(w, h, ring), col=PAL['brass'], x0=x0, y0=y0, metal=0.8, rough=0.35)
    L.mask_apply(L.draw_mask(w, h, eagle), col=PAL['brass'], x0=x0, y0=y0, metal=0.8, rough=0.35)
    a = L.alpha[y0:y1, x0:x1]
    L.alpha[y0:y1, x0:x1] = np.where(a > 0.5, 1.0, 0.0)


def paint_panel(L, rect, base, lines=4, horizontal=True):
    x0, y0, x1, y1 = rect
    L.rect(x0, y0, x1, y1, col=base, alpha=1.0)
    w, h = x1 - x0, y1 - y0
    for k in range(1, lines + 1):
        if horizontal:
            yy = y0 + h * k / (lines + 1)
            L.rect(x0 + 2, yy, x1 - 2, yy + 2, col=base * 0.6, add_height=-0.5)
        else:
            xx = x0 + w * k / (lines + 1)
            L.rect(xx, y0 + 2, xx + 2, y1 - 2, col=base * 0.6, add_height=-0.5)
    L.rect(x0, y0, x1, y0 + 2, col=base * 0.7); L.rect(x0, y1 - 2, x1, y1, col=base * 0.7)


def paint_hangar_door(L, rect):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['light'] * 0.95, alpha=1.0)
    for k in range(1, 7):
        yy = y0 + h * k / 7
        L.rect(x0 + 3, yy, x1 - 3, yy + 2, col=PAL['light'] * 0.7, add_height=-0.6)
    L.rect(x0 + w * 0.46, y0, x0 + w * 0.54, y1, col=PAL['white'] * 1.05, add_height=0.2)
    L.rect(x0, y0, x0 + 3, y1, col=PAL['dark']); L.rect(x1 - 3, y0, x1, y1, col=PAL['dark'])


def paint_vds_door(L, rect):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['hull'] * 0.8, alpha=1.0)
    L.rect(x0 + 3, y0 + 3, x1 - 3, y1 - 3, col=PAL['hull'] * 1.02, add_height=-0.4)
    for k in range(1, 4):
        yy = y0 + 3 + (h - 6) * k // 4
        L.rect(x0 + 3, yy, x1 - 3, yy + 1, col=PAL['hull'] * 0.75)
    # crest
    cs = int(min(w, h) * 0.13)
    cx, cy = x0 + w // 2, y0 + int(h * 0.36)

    def disc(d, s):
        d.ellipse([(cx - x0 - cs) * s, (cy - y0 - cs) * s, (cx - x0 + cs) * s, (cy - y0 + cs) * s], fill=255)
    L.mask_apply(L.draw_mask(w, h, disc), col=srgb('9c1c1c'), x0=x0, y0=y0)

    def ring(d, s):
        d.ellipse([(cx - x0 - cs) * s, (cy - y0 - cs) * s, (cx - x0 + cs) * s, (cy - y0 + cs) * s], outline=255, width=2 * s)
    L.mask_apply(L.draw_mask(w, h, ring), col=PAL['brass'], metal=0.8, x0=x0, y0=y0)


def paint_window_row(L, rect, n, frac=0.82):
    """row of n bridge windows filling the decal height: rounded frames, dark glass with a sky gradient,
    wall-coloured mullions between them (transparent elsewhere)."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['super'], alpha=0.0)
    pitch = w / n

    def frames(d, s):
        for k in range(n):
            a = (k + 0.5 - frac / 2) * pitch
            b = (k + 0.5 + frac / 2) * pitch
            d.rounded_rectangle([a * s, 0.04 * h * s, b * s, 0.96 * h * s], radius=int(0.18 * h * s), fill=255)

    def glass(d, s):
        for k in range(n):
            a = (k + 0.5 - frac / 2) * pitch + 0.09 * h
            b = (k + 0.5 + frac / 2) * pitch - 0.09 * h
            d.rounded_rectangle([a * s, 0.15 * h * s, b * s, 0.85 * h * s], radius=int(0.12 * h * s), fill=255)
    L.mask_apply(L.draw_mask(w, h, frames), col=PAL['dark'] * 1.1, alpha=1.0, add_height=0.5, x0=x0, y0=y0)
    m = L.draw_mask(w, h, glass)
    L.mask_apply(m, col=PAL['glass'], rough=0.08, metal=0.6, add_height=-0.6, x0=x0, y0=y0)
    g = np.linspace(0.3, 0.0, h)[:, None] * m
    L.col[y0:y1, x0:x1] += g[..., None] * np.array([0.2, 0.24, 0.28])
    a = L.alpha[y0:y1, x0:x1]
    L.alpha[y0:y1, x0:x1] = np.where(a > 0.5, 1.0, 0.0)


def paint_louvre_panel(L, rect):
    x0, y0, x1, y1 = rect
    L.rect(x0, y0, x1, y1, col=PAL['super'] * 0.92, alpha=1.0)
    for yy in range(y0 + 3, y1 - 3, 4):
        L.rect(x0 + 3, yy, x1 - 3, yy + 1, col=PAL['super'] * 0.6, add_height=-0.8)
    L.rect(x0, y0, x1, y0 + 2, col=PAL['super'] * 0.6, add_height=0.8)
    L.rect(x0, y1 - 2, x1, y1, col=PAL['super'] * 0.6, add_height=0.8)
    L.rect(x0, y0, x0 + 2, y1, col=PAL['super'] * 0.6, add_height=0.8)
    L.rect(x1 - 2, y0, x1, y1, col=PAL['super'] * 0.6, add_height=0.8)


def paint_vert_louvre(L, rect):
    """funnel casing side: vertical louvres in a frame"""
    x0, y0, x1, y1 = rect
    L.rect(x0, y0, x1, y1, col=PAL['super'] * 0.86, alpha=1.0)
    for xx in range(x0 + 2, x1 - 2, 6):
        L.rect(xx, y0 + 2, xx + 2, y1 - 2, col=PAL['super'] * 0.5, add_height=-0.7)
    for yy in (y0, y0 + (y1 - y0) // 2, y1 - 2):
        L.rect(x0, yy, x1, yy + 2, col=PAL['super'] * 0.62, add_height=0.8)


def paint_top_dome(L, rect):
    x0, y0, x1, y1 = rect
    L.rect(x0, y0, x1, y1, col=PAL['radome'] * 0.95, alpha=1.0, rough=0.5)
    n = value_noise(y1 - y0, x1 - x0, 12, seed=77, octaves=3)
    L.col[y0:y1, x0:x1] *= (0.94 + 0.1 * n)[..., None]
    for yy in range(y0, y1, 10):
        L.rect(x0, yy, x1, yy + 1, col=PAL['radome'] * 0.8, add_height=0.4)


def register_decals(m):
    D.register(m)
    m.alloc('pennant', 384, 176, lambda L, r: paint_pennant(L, r, PENNANT, PAL['white']))
    name_px = text_width_px(SHIP_NAME, 96, 0.78, 10) + 24
    m.name_w = NAME_H * name_px / 96.0
    m.alloc('name', name_px, 96, lambda L, r: paint_text(L, r, SHIP_NAME, PAL['white'], size_frac=0.78, spacing=10))
    m.alloc('helideck', 320, 320, paint_helideck)
    m.alloc('door', 48, 96, paint_door)
    m.alloc('vent', 256, 112, paint_vent)
    m.alloc('funnel_top', 256, 112, paint_funnel_top)
    m.alloc('vls_top', 512, 512, WP.paint_vls_top)
    m.alloc('ensign', 96, 64, paint_ensign)
    m.alloc('star', 64, 64, paint_star)
    m.alloc('crest', 96, 96, paint_crest)
    m.alloc('capstar', 96, 96, paint_capstar)
    m.alloc('panel_dark', 64, 64, lambda L, r: paint_panel(L, r, PAL['dark'] * 1.1, lines=3))
    m.alloc('hangar_door', 128, 128, paint_hangar_door)
    m.alloc('vds_door', 96, 96, paint_vds_door)
    m.alloc('brwin7', 448, 40, lambda L, r: paint_window_row(L, r, 7))
    m.alloc('brwin4', 256, 40, lambda L, r: paint_window_row(L, r, 4))
    m.alloc('brwin1', 40, 40, lambda L, r: paint_window_row(L, r, 1, frac=0.9))
    m.alloc('louvre', 64, 96, paint_louvre_panel)
    m.alloc('vlouvre', 256, 128, paint_vert_louvre)
    m.alloc('dome', 128, 128, paint_top_dome)
    m.alloc('raft', 8, 128, paint_raft)
    A.register(m)
    WP.register(m)


# =============================================================================================
# deck band (main deck paint pattern) features
# =============================================================================================
def deck_paint(m):
    W = st.W

    def paint(L):
        y0, y1 = st.BANDS['DECK']
        hpx = y1 - y0
        zz = Z_STERN + (np.arange(W) + 0.5) / W * LOA
        xs = S2.DECK_HALF - (np.arange(hpx) + 0.5) / hpx * 2 * S2.DECK_HALF
        Bg = (Z_BOW - zz)[None, :]
        Xg = xs[:, None]
        n1 = value_noise(hpx, W, 12, seed=41, octaves=4)
        grey = np.ones((hpx, W, 3)) * PAL['deck'] * (0.95 + 0.08 * n1)[..., None]
        cur = L.col[y0:y1].copy()
        # quarterdeck (aft of the step, incl. the sloped step itself) dark grey
        g = np.clip((Bg - 161.3) / 0.3, 0, 1)
        # S-shaped walkway boundary around the S-300F field (dark grey outboard)
        edge_x = 9.2 - 2.2 * np.exp(-((Bg - 135.5) / 5.5) ** 2)
        side = ((Bg > 126.0) & (Bg < 145.0) & (np.abs(Xg) > edge_x)).astype(float)
        mask = np.clip(g + side, 0, 1)
        L.col[y0:y1] = cur * (1 - mask[..., None]) + grey * mask[..., None]
        # anchor chains (hawse at B 4.6 -> windlasses at B 18.9, x 1.3)
        for s in (1, -1):
            for t in np.linspace(0, 1, 500):
                B = 4.6 + t * (18.9 - 4.6)
                x = s * (2.6 - t * 1.3)
                c = int(S2.B_to_px(B)); r = int(S2.dx_to_px(x))
                L.rect(c - 1, r - 2, c + 2, r + 2, col=PAL['black'] * 1.4, add_height=0.8)
    m.deck_marks.append(paint)


# =============================================================================================
# hull fittings
# =============================================================================================
def hull_fittings(m):
    c = m.ctx; b = m.b
    b.node('Hull')
    red = c.sw('red')
    # shafts, brackets, propellers, rudder (sheets 1 and 7)
    for s in (1, -1):
        p0 = P3(140.0, s * 2.7, -5.6); p1 = P3(168.3, s * 4.45, -4.6)
        c.add(c.sw('steel'), tube_path([p0, p1], 0.3, seg=8))
        for Bst in (152.0, 165.0):
            t = (Bst - 140.0) / (168.3 - 140.0)
            pc = p0 + t * (p1 - p0)
            for (xo, yo) in ((s * 1.6, -3.6), (s * 6.4, -2.4)):
                top = np.array([xo, yo, pc[2]])
                c.add(red, beam(pc, top, 0.22, 0.85, up=(0, 0, 1)))
        b.push(f'Propeller_{"P" if s > 0 else "S"}', parent='Hull', translation=tuple(P3(169.3, s * 4.45, -4.6)))
        propeller(c, P3(169.3, s * 4.45, -4.6), r=2.1, blades=4, hand=s)
        b.pop()
    # single centreline rudder (B 173-178, 2-7 m below WL)
    poly = [(zB(173.0), -1.9), (zB(178.2), -1.9), (zB(177.6), -7.0), (zB(173.6), -7.0)]
    c.add(red, extrude_x(poly, -0.3, 0.3))
    # bilge keels
    for s in (1, -1):
        pin, pout = [], []
        for B in np.linspace(62, 126, 9):
            xs, ys = section(B, 48)
            j = int(np.argmin(np.abs(ys + 4.9)))
            n2 = normalize(np.array([1.0, -0.9]))
            pin.append((s * xs[j], ys[j], zB(B)))
            pout.append((s * (xs[j] + 0.6 * n2[0]), ys[j] + 0.6 * n2[1], zB(B)))
        P = np.array(pin + pout); k = len(pin)
        I = np.array([(i, i + 1, k + i + 1) for i in range(k - 1)] + [(i, k + i + 1, k + i) for i in range(k - 1)])
        N = compute_smooth_normals(P, I)
        P, N, I = orient_outward(P, N, I, lambda p: np.array([0.0, -2.0, p[2]]))
        c.add(red, (P, N, np.zeros((len(P), 2)), I))
    # hull decals: pennant number, name, bow star
    for s in (1, -1):
        hull_decal(c, 'pennant', PENNANT_B, 3.55, 7.4, 3.4, s, nseg=6)
        hull_decal(c, 'name', 172.0 - m.name_w / 2, 2.85, m.name_w, NAME_H, s, nseg=8)
        for (name, Bc, yc, w, h) in (('star', 8.0, 9.3, 0.85, 0.85),):
            xs, ys = section(Bc, 64)
            xh = float(np.interp(yc, ys, xs))
            xs2, ys2 = section(Bc + 0.5, 64)
            xh2 = float(np.interp(yc, ys2, xs2))
            dxdy = float(np.interp(yc + 0.4, ys, xs) - np.interp(yc - 0.4, ys, xs)) / 0.8
            tang_z = normalize(np.array([s * (xh2 - xh), 0.0, -0.5]))
            nrm = normalize(np.cross(np.array([0, 1.0, 0]) + np.array([s * dxdy, 0, 0]), tang_z))
            if nrm[0] * s < 0:
                nrm = -nrm
            decal(c, name, (s * xh, yc, zB(Bc)), nrm, w, h, offset=0.05, flip=(s < 0 and False))
    # stockless anchors stowed flush in their pockets (B 5.0, y 8.3)
    for s in (1, -1):
        p, T, U, N = hull_frame(5.0, 8.3, s)
        blk = c.sw('black')
        local_prism(c, blk, [(-0.17, -0.2), (0.17, -0.2), (0.17, 1.35), (-0.17, 1.35)], 0.3, p + N * 0.02, T, U, N)
        local_prism(c, blk, [(-0.45, -0.95), (0.45, -0.95), (0.95, 0.1), (-0.95, 0.1)], 0.38, p + N * 0.02, T, U, N)


# =============================================================================================
# forecastle and AK-130
# =============================================================================================
def forecastle(m):
    c = m.ctx; b = m.b
    b.node('Hull')
    whip(c, P3(0.6, 0, 10.85), 4.3, r=0.07, tilt=(0, 1, 0.12))                     # jackstaff
    # windlasses (B 18.9, x 1.3) and chain stoppers
    for s in (1, -1):
        y = dk(18.9, 1.3)
        G.windlass(c, P3(18.9, s * 1.3, y - 0.05), s)
        for Bs_ in (8.5, 12.0):
            t = (Bs_ - 4.6) / (18.9 - 4.6)
            x = s * (2.6 - t * 1.3)
            G.chain_stopper(c, P3(Bs_, x, dk(Bs_, x) - 0.03), (-1.3 * s, 0, -(18.9 - 4.6)))
    # breakwater: curved V from (B 26.0, +-6.9) to the apex (B 20.0, 0); 1.75 m tall at the apex
    pts = [(26.0, 6.9), (24.6, 6.55), (23.2, 5.9), (22.0, 4.7), (21.0, 3.1), (20.3, 1.4), (20.0, 0.0)]
    for s in (1, -1):
        for (Ba, xa), (Bb, xb) in zip(pts[:-1], pts[1:]):
            ha = 0.75 + 1.0 * (1 - xa / 6.9); hb = 0.75 + 1.0 * (1 - xb / 6.9)
            ya, yb = dk(Ba, xa) - 0.1, dk(Bb, xb) - 0.1
            q = _quad(P3(Ba, s * xa, ya), P3(Bb, s * xb, yb), P3(Bb, s * xb, yb + hb + 0.1), P3(Ba, s * xa, ya + ha + 0.1))
            c.add(c.paint, q)
            # back side (double-sided material, but give it a slightly different shade via second quad offset)
        for k in range(1, len(pts) - 1, 2):
            Bk, xk = pts[k]
            yk = dk(Bk, xk)
            c.add(c.paint, beam(P3(Bk, s * xk, yk), P3(Bk + 1.2, s * xk * 0.92, yk), 0.12, 0.12, up=(0, 0, 1)))
    # bollards, fairleads, vents on the forecastle
    for B, x in ((6.5, 3.2), (13.5, 5.3), (20.0, 6.9), (37.5, 9.3)):
        for s in (1, -1):
            bollard(c, P3(B, s * x, dk(B, x) - 0.05), along=(0, 0, 1))
    for B, x in ((34.2, 8.8), (32.3, 8.8), (33.2, 7.5), (34.2, 6.1), (32.2, 6.1), (33.2, 4.8), (23.6, 4.2)):
        for s in (1, -1):
            G.round_vent(c, P3(B, s * x, dk(B, x) - 0.05), r=0.34, h=0.62)


def ak130(m):
    """AK-130 twin 130 mm gun: centre B 30.0, ring at 10.0, roof 13.6, barrel axis 11.6 to B 21.9."""
    c = m.ctx; b = m.b
    Bc = 30.0
    ybase = 10.0
    b.node('Hull')
    # barbette / foundation with perforated splash shield
    yd = dk(Bc)
    c.add(c.paint, cylinder(3.5, ybase - yd + 0.4, seg=28, y0=yd - 0.4), xf=M(P3(Bc, 0, 0)))
    c.add(c.deck, cylinder(3.5, 0.01, seg=28, caps=(False, True), y0=ybase - 0.01), xf=M(P3(Bc, 0, 0)))
    G.splash_shield(c, Bc, 4.4, -130, 130, lambda B, x: dk(B, x) - 0.1, ybase - 0.6)
    # turret (training node) and guns (elevating node)
    WP.ak130_mount(c, Bc, ybase, stars=STARS_ON_CAPS)


# =============================================================================================
# forward deckhouse (B 35.0-58.6, +-3.65, roof 12.95) with AK-630 x2, RBU-6000 x2, MR-123
# =============================================================================================
def forward_deckhouse(m):
    c = m.ctx; b = m.b
    b.node('Superstructure')
    fd = sym_poly([(35.0, 1.6), (36.9, 3.65), (58.7, 3.65)])
    house(c, fd, dmin(35.0, 58.7) - 0.3, 12.95, lip=0.12, bevel=0.3, bevel_where=lambda B, x: B < 58.0)
    # raised gun deck between the AK-630s (ramp 40.9 -> 42.2, flat to 47.5)
    c.add(c.paint, extrude_x([(zB(40.9), 12.9), (zB(42.2), 14.1), (zB(47.5), 14.1), (zB(47.5), 12.9)], -1.6, 1.6))
    rail_poly(c, [(42.2, 1.6), (47.5, 1.6), (47.5, -1.6), (42.2, -1.6)], 14.1, sides=(0, 2))
    side_strip(c, 'PORTS', 37.4, 58.0, 3.67, 11.15, 1.0)
    side_strip(c, 'PORTS', 37.4, 58.0, 3.67, 8.6, 1.0)
    end_strip(c, 'PORTS', 35.0, -1.15, 1.15, 11.0, 1.0, facing=1)
    for s in (1, -1):
        door(c, 44.0, s * 3.67, dk(44.0, 3.6) - 0.05, (s, 0, 0))
        door(c, 55.6, s * 3.67, dk(55.6, 3.6) - 0.05, (s, 0, 0))
    rail_poly(c, [(36.9, 3.65), (58.6, 3.65), (58.6, -3.65), (36.9, -3.65), (35.0, -1.6), (35.0, 1.6)], 12.95,
              sides=(0, 2, 3, 4, 5))
    WP.ak630m(c, P3(39.6, 0.0, 12.95), facing=0.0, name='AK630_1', parent='Superstructure', base_h=0.8)
    WP.ak630m(c, P3(45.2, 0.0, 14.1), facing=0.0, name='AK630_2', parent='Superstructure', base_h=0.8)
    WP.rbu6000m(c, P3(52.3, 2.2, 12.95), facing=0.0, name='RBU6000_P', parent='Superstructure')
    WP.rbu6000m(c, P3(52.3, -2.2, 12.95), facing=0.0, name='RBU6000_S', parent='Superstructure')
    D.bass_tilt(c, P3(57.3, 0.0, 12.95), facing=0.0, name='BassTilt_1', pedestal=2.4, parent='Superstructure')
    b.node('Superstructure')
    for B in (54.2, 50.4):
        for s in (1, -1):
            whip(c, P3(B, s * 3.2, 12.95), 6.5, r=0.05)


# =============================================================================================
# bridge block, gallery, bridge house (sheet 5)
# =============================================================================================
# bridge block after Kuleshov sheet 5 (side, front and plan views registered to sheet 1) and the Moskva
# 2009-17 / Ustinov 2018 photos. Plan outlines are (B, x>=0) halves, front to back.
BR_Y0, BR_YW, BR_Y1 = 17.48, 19.45, 19.95                     # bridge base (BL2 roof), window-band top, roof
BR_BASE = [(61.2, 2.6), (63.55, 5.0), (66.4, 5.0), (68.0, 3.5), (74.0, 3.5)]
BR_WTOP = [(60.85, 2.85), (63.2, 5.38), (66.4, 5.38), (68.0, 3.72), (74.0, 3.72)]
BR_ROOF = [(61.18, 2.72), (63.34, 5.05), (66.26, 5.05), (67.86, 3.39), (74.0, 3.39)]
BR_WIN = (18.88, 19.38)                                      # window row
WALKWAY = [(66.45, 5.05), (66.6, 6.17), (68.7, 6.22), (69.13, 5.55), (69.3, 5.41), (71.33, 5.41), (73.87, 5.92),
           (76.06, 5.8), (82.5, 5.8)]                         # walkway bulwark line (port side)
WALK_TOP = 18.75                                              # bulwark top


def bulwark(c, pts_Bx, y0, y1, t=0.08):
    """thin solid bulwark along an open polyline [(B, x)] on one side of the ship, thickened inboard:
    outer and inner faces, top cap and end caps (flat-shaded, planar UVs in metres)."""
    P2 = np.array([(x, zB(B)) for (B, x) in pts_Bx], float)
    side = 1.0 if np.mean(P2[:, 0]) > 0 else -1.0
    n = len(P2)
    nrm = []
    for i in range(n - 1):
        d = P2[i + 1] - P2[i]
        d = d / np.linalg.norm(d)
        nn = np.array([d[1], -d[0]])
        if nn[0] * side > 0:                                  # make it point inboard
            nn = -nn
        nrm.append(nn)
    inner = []
    for i in range(n):
        a = nrm[max(i - 1, 0)]; b = nrm[min(i, n - 2)]
        m_ = a + b
        m_ = m_ / np.linalg.norm(m_)
        inner.append(P2[i] + m_ * t / max(0.35, float(np.dot(m_, b))))
    inner = np.array(inner)
    polys = []

    def add(poly, want):                                      # want: intended outward direction (x, y, z)
        q = np.array(poly, float)
        nn = np.cross(q[1] - q[0], q[2] - q[0])
        polys.append(q if np.dot(nn, want) >= 0 else q[::-1])
    for i in range(n - 1):
        o0, o1, i0, i1 = P2[i], P2[i + 1], inner[i], inner[i + 1]
        w_in = np.array([nrm[i][0], 0.0, nrm[i][1]])
        add([(o0[0], y0, o0[1]), (o1[0], y0, o1[1]), (o1[0], y1, o1[1]), (o0[0], y1, o0[1])], -w_in)
        add([(i1[0], y0, i1[1]), (i0[0], y0, i0[1]), (i0[0], y1, i0[1]), (i1[0], y1, i1[1])], w_in)
        add([(o0[0], y1, o0[1]), (o1[0], y1, o1[1]), (i1[0], y1, i1[1]), (i0[0], y1, i0[1])], np.array([0, 1.0, 0]))
    for (o, ii, j) in ((P2[0], inner[0], 1), (P2[-1], inner[-1], n - 2)):
        d = P2[j] - o
        add([(o[0], y0, o[1]), (ii[0], y0, ii[1]), (ii[0], y1, ii[1]), (o[0], y1, o[1])], -np.array([d[0], 0.0, d[1]]))
    c.add(c.paint, flat_poly_faces(polys))


def _face_pt(base0, base1, top0, top1, u, h, y0=BR_Y0, y1=BR_YW):
    """point on a raked wall face given by its base edge (at y0) and top edge (at y1), (B, x) pairs."""
    t = (h - y0) / (y1 - y0)
    b = (1 - u) * np.array(base0) + u * np.array(base1)
    tp = (1 - u) * np.array(top0) + u * np.array(top1)
    B, x = (1 - t) * b + t * tp
    return P3(B, x, h)


def face_decal(c, name, base0, base1, top0, top1, u0, u1, h0, h1, off=0.025):
    """decal quad lying on a raked wall face (u along the face from base0 to base1, h height)."""
    q = [_face_pt(base0, base1, top0, top1, u0, h0), _face_pt(base0, base1, top0, top1, u1, h0),
         _face_pt(base0, base1, top0, top1, u1, h1), _face_pt(base0, base1, top0, top1, u0, h1)]
    nrm = normalize(np.cross(q[1] - q[0], q[3] - q[0]))
    cen = P3(*(((np.array(base0) + np.array(base1)) / 2)), h0)
    inside = P3(64.5, 0.0, h0)
    if np.dot(nrm, cen - inside) < 0:
        nrm = -nrm
    q = [p + nrm * off for p in q]
    Pq, Nq, UVq, Iq = _quad(q[0], q[1], q[2], q[3], uv=[(0, 1), (1, 1), (1, 0), (0, 0)], n=nrm)
    if np.dot(np.cross(Pq[1] - Pq[0], Pq[2] - Pq[0]), nrm) < 0:
        Iq = Iq[:, ::-1]
    c.add(c.rect(name), (Pq, Nq, UVq, Iq))


def bridge(m):
    """bridge block (Kuleshov sheet 5, Moskva/Ustinov photos):
    - BL1 (to 15.0) and, under the bridge, a narrow level (15.0-17.48, +-3.6) whose raked front continues the
      bridge front; aft of B 65.8 the block widens to +-4.85 up to the bridge deck;
    - the bridge house rises from 17.48: walls leaning out (front 0.35 m, sides 0.38 m), one row of windows
      (7 on the front, 4 on each chamfer and side), a sloped band to the roof at 19.95; it overhangs the level
      below by up to 1.4 m and narrows aft of B 66.4 to the upper deckhouse running to the tower;
    - an open walkway at 17.48 along both sides from B 66.4 aft, behind a solid bulwark (bulge at B 66.6-68.7
      with the guards' crest), and a small lantern platform at the front."""
    c = m.ctx; b = m.b
    b.node('Superstructure')
    y0 = dmin(58.6, 80.8) - 0.3
    house(c, sym_poly([(58.6, 3.65), (80.75, 3.65)]), y0, 15.0, lip=0.12, bevel=0.3, bevel_where=lambda B, x: B < 60)  # BL1
    house(c, sym_poly([(61.5, 4.85), (65.9, 4.85)]), y0, 15.0, lip=0.12, top='deck')                               # BL2 low
    house(c, sym_poly([(65.8, 4.85), (80.75, 4.85)]), y0, BR_Y0, top='deck')                                     # BL2 wide
    nb = [(61.7, 2.5), (62.6, 3.6), (65.9, 3.6)]
    nt = [(61.2, 2.5), (62.2, 3.6), (65.9, 3.6)]
    house(c, sym_poly(nb), 15.0, BR_Y0, top_poly_Bx=sym_poly(nt), top=None)                                    # BL2 narrow
    # railing round the 15 m deck in front of and beside the narrow level
    for sx in (1, -1):
        railing_pts(c, [P3(65.7, sx * 4.75, 15.02), P3(61.62, sx * 4.75, 15.02), P3(61.62, sx * 3.55, 15.02),
                        P3(58.72, sx * 3.55, 15.02)])
    railing_pts(c, [P3(58.72, 3.55, 15.02), P3(58.72, -3.55, 15.02)])
    # bridge house: raked walls to the window-band top, sloped band to the roof, underside over the overhang
    house(c, sym_poly(BR_BASE), BR_Y0 - 0.02, BR_YW, top_poly_Bx=sym_poly(BR_WTOP), top=None, bottom=True)
    house(c, sym_poly(BR_WTOP), BR_YW, BR_Y1, top_poly_Bx=sym_poly(BR_ROOF), top='deck')
    rail_poly(c, sym_poly(BR_ROOF), BR_Y1, inset=0.1, sides=(0, 1, 2, 3, 5, 6, 7, 8, 9))
    # windows: 7 on the front face, 4 on each chamfer and side; door and a narrow window on the aft corner
    fb0, fb1 = (BR_BASE[0][0], -BR_BASE[0][1]), BR_BASE[0]
    ft0, ft1 = (BR_WTOP[0][0], -BR_WTOP[0][1]), BR_WTOP[0]
    face_decal(c, 'brwin7', fb0, fb1, ft0, ft1, 0.03, 0.97, *BR_WIN)
    for sx in (1, -1):
        bs = [(B, sx * x) for (B, x) in BR_BASE]
        ts = [(B, sx * x) for (B, x) in BR_WTOP]
        face_decal(c, 'brwin4', bs[0], bs[1], ts[0], ts[1], 0.05, 0.95, *BR_WIN)
        face_decal(c, 'brwin4', bs[1], bs[2], ts[1], ts[2], 0.05, 0.95, *BR_WIN)
        face_decal(c, 'brwin1', bs[2], bs[3], ts[2], ts[3], 0.12, 0.3, *BR_WIN)
        face_decal(c, 'door', bs[2], bs[3], ts[2], ts[3], 0.45, 0.78, BR_Y0 + 0.02, BR_Y0 + 1.8)
        side_strip(c, 'PORTS', 68.6, 73.4, 3.70, 18.7, 0.7, sides=(sx,))
    # level below: watertight door and two windows in the raked front, ports on the sides
    rk = normalize(np.array([0.0, 0.5 / (BR_Y0 - 15.0), 1.0]))
    for (xx, nm, w_, h_, hc) in ((0.0, 'door', 0.8, 1.75, 15.95), (1.75, 'brwin1', 0.45, 0.42, 16.85),
                                 (-1.75, 'brwin1', 0.45, 0.42, 16.85)):
        Bc = 61.7 - 0.5 * (hc - 15.0) / (BR_Y0 - 15.0)
        decal(c, nm, P3(Bc, xx, hc), rk, w_, h_)
    side_strip(c, 'PORTS', 62.9, 65.6, 3.62, 16.4, 0.9)
    side_strip(c, 'PORTS', 66.3, 80.2, 4.87, 16.4, 0.9)
    side_strip(c, 'PORTS', 62.5, 80.0, 4.87, 12.6, 1.0)
    end_strip(c, 'PORTS', 58.6, -3.1, 3.1, 13.6, 1.0, facing=1)
    for s in (1, -1):
        door(c, 64.6, s * 3.62, 15.02, (s, 0, 0))
        door(c, 72.2, s * 3.75, BR_Y0, (s, 0, 0))
        door(c, 76.0, s * 4.87, dk(76.0, 4.8) - 0.05, (s, 0, 0))
    # walkway: floor plate over the overhang, solid bulwark, brackets underneath, railing across the aft end
    for sx in (1, -1):
        wl = [(B, sx * x) for (B, x) in WALKWAY]
        fl = [(66.45, sx * 4.8)] + [(B, sx * x) for (B, x) in WALKWAY[:-1]] + [(80.75, sx * 5.8), (80.75, sx * 4.8)]
        slab(c, fl if sx > 0 else fl[::-1], BR_Y0 - 0.14, BR_Y0 + 0.02)
        bulwark(c, wl, BR_Y0 - 0.14, WALK_TOP)
        for Bk in (67.6, 70.3, 72.8, 75.4, 78.0):
            xo = float(np.interp(Bk, [p[0] for p in WALKWAY], [p[1] for p in WALKWAY])) - 0.25
            c.add(c.paint, D.taper_beam(P3(Bk, sx * 4.86, BR_Y0 - 0.95), P3(Bk, sx * xo, BR_Y0 - 0.15),
                                      0.08, 0.2, 0.08, 0.08, up=(0, 0, 1)))
        decal(c, 'crest', P3(67.65, sx * 6.2, 18.05), (sx, 0, 0), 0.85, 0.85, offset=0.03)
    slab(c, [(80.75, 5.8), (82.5, 5.8), (82.5, -5.8), (80.75, -5.8)], BR_Y0 - 0.14, BR_Y0 + 0.02)
    railing_pts(c, [P3(82.42, 5.72, BR_Y0 + 0.02), P3(82.42, -5.72, BR_Y0 + 0.02)])
    # lantern platform at the foot of the bridge front, on V brackets
    lp = sym_poly([(59.1, 1.23), (61.3, 2.05)])
    slab(c, lp, BR_Y0 - 0.12, BR_Y0)
    rail_poly(c, lp, BR_Y0, inset=0.06, sides=(0, 2, 3))
    for sx in (1, -1):
        c.add(c.paint, D.taper_beam(P3(61.62, sx * 1.5, 16.35), P3(59.45, sx * 1.0, BR_Y0 - 0.12), 0.1, 0.16, 0.08, 0.1))
    G.lamp(c, P3(59.55, 0.0, BR_Y0), 'white')
    # roof fittings: signal mast at the front edge (pole, yard, A-frame legs; Moskva 2009-12 photos), Kite Screech
    # (MR-184), radome, sight pedestals, searchlights at the front corners, whips
    c.add(c.paint, tube_path([P3(61.62, 0, BR_Y1), P3(61.62, 0, 24.4)], 0.08, seg=6))
    c.add(c.paint, tube_path([P3(61.62, -1.3, 22.25), P3(61.62, 1.3, 22.25)], 0.045, seg=4))
    for sx in (1, -1):
        c.add(c.paint, tube_path([P3(62.35, sx * 0.6, BR_Y1), P3(61.62, 0, 21.4)], 0.045, seg=4))
    G.lamp(c, P3(61.62, 0, 24.4), 'white')
    D.kite_screech(c, P3(67.7, 0, BR_Y1))
    b.node('Superstructure')
    G.radome(c, P3(71.6, 1.6, BR_Y1), r=0.55, h=1.0)
    for sx in (1, -1):
        c.add(c.paint, box(0.6, 0.55, 0.6, center=(0, 0.275, 0)), xf=M(P3(64.3, sx * 2.7, BR_Y1)))
        c.add(c.sw('dark'), box(0.4, 0.2, 0.45, center=(0, 0.65, 0)), xf=M(P3(64.3, sx * 2.7, BR_Y1)))
        searchlight(c, P3(62.3, sx * 3.85, BR_Y1), facing=sx * math.radians(45))
        whip(c, P3(65.9, sx * 4.1, BR_Y1), 6.0, r=0.05)
        whip(c, P3(67.6, sx * 5.75, BR_Y0), 7.0, r=0.05)
    # low deckhouse aft of the bridge block (B 80.75-85.4, +-3.75, roof 9.6) with intake louvres
    house(c, sym_poly([(80.7, 3.75), (85.4, 3.75)]), dmin(80.7, 85.4) - 0.3, 9.6, lip=0.12, bevel=0.25,
          bevel_where=lambda B, x: B > 85)
    for xx in (-1.6, 1.6):
        decal(c, 'louvre', P3(85.42, xx, 8.2), (0, 0, -1), 1.6, 2.2)
    rail_poly(c, sym_poly([(80.7, 3.75), (85.4, 3.75)]), 9.6, sides=(1, 2, 3))


# =============================================================================================
# foremast pyramid tower, Front Door, Fregat, yards and topmast
# =============================================================================================
def tower_hw(h):
    return 4.06 - (4.06 - 1.35) * (h - 18.65) / (31.2 - 18.65)


def tower_front_B(h):
    return 73.44 + (77.2 - 73.44) * (h - 18.65) / (31.2 - 18.65)


def foremast(m):
    c = m.ctx; b = m.b
    b.node('Superstructure')
    bot = sym_poly([(73.44, 4.06), (80.8, 4.1)])
    top = sym_poly([(77.2, 1.35), (80.3, 1.35)])
    house(c, bot, BR_Y0, 18.62, top=None, bevel=0.3)                    # vertical foot down to the walkway
    house(c, bot, 18.6, 31.2, top_poly_Bx=top, top='deck', bevel=0.3)
    slab(c, sym_poly([(76.6, 1.9), (80.9, 1.9)]), 31.2, 31.45)
    rail_poly(c, sym_poly([(76.6, 1.9), (80.9, 1.9)]), 31.45)
    # side balconies on the pyramid
    for (h, B0, B1) in ((24.5, 76.0, 79.2), (28.0, 77.4, 79.8)):
        hw = tower_hw(h)
        for s in (1, -1):
            xa, xb = (hw - 0.3, hw + 1.25) if s > 0 else (-hw - 1.25, -hw + 0.3)
            poly = [(B0, xb), (B1, xb), (B1, xa), (B0, xa)]
            slab(c, poly, h - 0.2, h)
            rail_poly(c, [(B0, s * (hw + 1.25)), (B1, s * (hw + 1.25))], h, closed=False, inset=0.0)
    # side platforms with ECM radomes at h 21 (B 75.8-80.3, x 4.0-7.0)
    for s in (1, -1):
        poly = [(75.8, s * 7.0), (80.3, s * 7.0), (80.3, s * 4.0), (75.8, s * 4.0)]
        slab(c, poly, 20.8, 21.05)
        rail_poly(c, [(75.8, s * 7.0), (80.3, s * 7.0)], 21.05, closed=False, inset=0.0)
        for B in (76.8, 79.3):
            G.radome(c, P3(B, s * 5.9, 21.05), r=0.45, h=0.8, seg=10)
    # ladders on the faces
    for s in (1, -1):
        ladder(c, P3(80.85, s * 2.4, BR_Y0 + 0.05), P3(80.4, s * 1.0, 31.1), 0.45, normal=(0, 0, -1))
    # tapered Fregat stalk, the rotating Fregat radar and the Front Door director
    c.add(c.paint, lathe([(0.72, 0.0), (0.72, 0.15), (0.6, 0.25), (0.55, 2.0), (0.62, 2.2)], seg=14), xf=M(P3(78.15, 0, 31.4)))
    A.fregat(c, P3(78.15, 0, 33.6))
    A.front_door(c, tower_front_B)
    # cross yard (lattice girder) at B 81.6, h 28.85, x +-7.1 with lattice ECM towers at x +-3.4
    truss(c, P3(81.6, -7.1, 28.85), P3(81.6, 7.1, 28.85), 0.9, 0.9, rep=4.0)
    for s in (1, -1):
        lattice_tower(c, P3(80.4, s * 3.4, 27.2), 0.45, 0.65, 3.6, rep=3.0)
        slab(c, [(79.8, s * 4.0), (81.0, s * 4.0), (81.0, s * 2.8), (79.8, s * 2.8)], 30.8, 30.95)
        G.radome(c, P3(80.4, s * 3.4, 30.95), r=0.5, h=0.7, seg=10)
    # aft lattice girder and the lattice topmast (to 39.5 m)
    truss(c, P3(80.3, 0, 30.6), P3(88.1, 0, 30.6), 1.3, 1.1, rep=4.0)
    lattice_tower(c, P3(84.4, 0, 31.2), 0.25, 0.45, 8.3, rep=3.0)
    c.add(c.paint, beam(P3(84.4, -1.6, 35.6), P3(84.4, 1.6, 35.6), 0.12, 0.12))
    c.add(c.paint, tube_path([P3(84.4, 0, 39.5), P3(84.4, 0, 40.6)], 0.07, seg=4))
    ring = [P3(85.3 + 0.7 * math.cos(2 * math.pi * k / 12), 0, 37.5 + 0.7 * math.sin(2 * math.pi * k / 12)) for k in range(13)]
    c.add(c.paint, tube_path(ring, 0.05, seg=4))


# =============================================================================================
# P-500 / P-1000 launcher batteries: 4 pairs per side
# =============================================================================================
TUBE_FRONTS = (38.95, 50.75, 62.55, 74.35)
TUBE_LEN_B = 11.85
TUBE_EL = math.radians(16.0)
TUBE_YF = 11.95
TUBES_X = ((8.65, 0.0), (6.0, 0.35))      # (|x|, extra height)


def tube_axis_y(Bf, B, dy=0.0):
    return TUBE_YF + dy - math.tan(TUBE_EL) * (B - Bf)


def launcher_tube(m, Bf, x, dy, node=None):
    c = m.ctx
    p_front = P3(Bf, x, TUBE_YF + dy)
    d = normalize(np.array([0.0, math.sin(TUBE_EL), math.cos(TUBE_EL)]))     # forward-up
    L = TUBE_LEN_B / math.cos(TUBE_EL)
    prof = [(0.0, -0.32), (0.55, -0.27), (0.92, -0.13), (1.06, 0.0), (1.08, 0.15), (1.08, 2.0), (1.0, 2.12),
            (0.9, 2.3), (0.86, L - 0.6), (0.72, L - 0.25), (0.45, L - 0.05), (0.0, L)]
    P, N, UV, I = lathe(prof, seg=14, uvscale=1.0)
    R = frame_from_dir(-d)               # local +z points aft along the axis
    Mx = np.eye(4); Mx[:3, :3] = R @ rot_x(math.pi / 2); Mx[:3, 3] = p_front
    c.add(c.paint, (P, N, UV, I), xf=Mx, node=node)
    for sring in (5.6, 6.1):
        c.add(c.paint, cylinder(0.94, 0.12, seg=14, caps=(False, False)), xf=M(p_front - d * sring, R @ rot_x(-math.pi / 2)), node=node)
    # front cover hinge/latches
    c.add(c.sw('mid'), box(0.25, 0.5, 0.35, center=(0, 0, 0)), xf=M(p_front + np.array([0, 1.05, -0.35]), R), node=node)
    # red star painted on the cover (Moskva)
    if STARS_ON_CAPS:
        decal(c, 'capstar', p_front + d * 0.335, d, 0.95, 0.95, offset=0.0, node=node)


def launchers(m):
    c = m.ctx; b = m.b
    b.node('Launchers', parent=None)
    for s in (1, -1):
        xin, xout = (4.95, 9.65)
        for i, Bf in enumerate(TUBE_FRONTS):
            for (xo, dy) in TUBES_X:
                launcher_tube(m, Bf, s * xo, dy)
            D.launcher_support(
                c, Bf, s, lambda B, dy, Bf=Bf: tube_axis_y(Bf, B, dy), lambda B: dk(B, 7.0), x_in=xin, x_out=xout,
                tubes=TUBES_X,
                rail=lambda pts: railing_pts(c, [P3(B, x, y) for (B, x, y) in pts]),
                ladder_fn=lambda a, b_, s_: ladder(c, a, b_, 0.45, normal=(s_, 0, 0)))


# =============================================================================================
# midships 01-level deckhouse, main mast, Top Pair (sheet 6)
# =============================================================================================
def mainmast(m):
    c = m.ctx; b = m.b
    b.node('Superstructure')
    # 01-level deckhouse spanning the beam (B 86.4-103.6)
    md1 = sym_poly([(86.4, 9.9), (102.6, 9.9), (103.6, 8.9)])
    house(c, md1, dmin(86.4, 103.6, 9.0) - 0.3, 9.5, lip=0.12)
    rail_poly(c, md1, 9.5, inset=0.1)
    side_strip(c, 'PORTS', 87.5, 102.0, 9.92, 7.6, 1.0)
    for s in (1, -1):
        door(c, 94.8, s * 9.92, 6.85, (s, 0, 0))
    WP.ak630m(c, P3(95.5, 9.0, 9.5), facing=math.radians(20), name='AK630_3', parent='Superstructure', base_h=0.8)
    WP.ak630m(c, P3(95.5, -9.0, 9.5), facing=math.radians(-20), name='AK630_4', parent='Superstructure', base_h=0.8)
    WP.ak630m(c, P3(101.25, 9.0, 9.5), facing=math.radians(160), name='AK630_5', parent='Superstructure', base_h=0.8)
    WP.ak630m(c, P3(101.25, -9.0, 9.5), facing=math.radians(-160), name='AK630_6', parent='Superstructure', base_h=0.8)
    D.bass_tilt(c, P3(88.4, 8.4, 9.5), facing=math.radians(60), name='BassTilt_2', pedestal=1.3, parent='Superstructure')
    D.bass_tilt(c, P3(88.4, -8.4, 9.5), facing=math.radians(-60), name='BassTilt_3', pedestal=1.3, parent='Superstructure')
    b.node('Superstructure')
    for s in (1, -1):
        raft_rack(c, P3(97.6, s * 9.0, 9.5), n=2, along=(0, 0, -1), spacing=0.8)
        raft_rack(c, P3(91.0, s * 8.8, 9.5), n=2, along=(0, 0, -1), spacing=0.8)
    c.add(c.paint, cylinder(1.6, 0.6, seg=16), xf=M(P3(90.1, -2.6, 9.5)))
    # main mast base block, tower wedge and column
    house(c, sym_poly([(95.75, 4.85), (103.1, 4.85)]), 9.4, 12.4, lip=0.12, bevel=0.25, bevel_where=lambda B, x: B < 103)
    house(c, sym_poly([(96.6, 4.4), (103.1, 4.4)]), 12.3, 14.5, bevel=0.25, bevel_where=lambda B, x: B < 103)
    house(c, sym_poly([(96.6, 4.4), (103.1, 4.4)]), 14.5, 19.5, top_poly_Bx=sym_poly([(97.75, 1.8), (103.0, 1.8)]), bevel=0.25,
          bevel_where=lambda B, x: B < 103)
    rail_poly(c, sym_poly([(97.75, 1.8), (103.0, 1.8)]), 19.5)
    for s in (1, -1):
        decal(c, 'louvre', P3(95.73, s * 2.2, 10.9), (0, 0, 1), 1.4, 2.2)
        decal(c, 'louvre', P3(95.73, s * 0.7, 10.9), (0, 0, 1), 1.0, 2.2)
        side_strip(c, 'PORTS', 97.0, 102.6, 4.87, 10.4, 1.0, sides=(s,))
    # lookout tubs at h 14.5 fore and aft of the radome stack: D-shaped floor, flared bulwark, top rail
    for s in (1, -1):
        for Bp in (97.0, 102.6):
            pts = [(Bp + 1.4 * math.cos(a), s * (4.3 + 1.4 * math.sin(a))) for a in np.linspace(-math.pi / 2, math.pi / 2, 7)]
            slab(c, pts if s > 0 else pts[::-1], 14.3, 14.55)
            R_ = rot_y(math.pi / 2 if s > 0 else -math.pi / 2)
            c.add(c.paint, lathe([(1.4, 0.0), (1.4, 0.15), (1.5, 1.05), (1.44, 1.1)], seg=8, arc=math.pi,
                                 angle0=-math.pi / 2), xf=M(P3(Bp, s * 4.3, 14.3), R_))
            arc = [P3(Bp + 1.52 * math.cos(a), s * (4.3 + 1.52 * math.sin(a)), 15.85) for a in np.linspace(-math.pi / 2, math.pi / 2, 9)]
            c.add(c.paint, A.pipe(arc, 0.03, seg=4), occ=False)
    # ECM radomes ("eggs"): four a side at B 99.8, stacked in two pairs with the long axis athwartships
    # (Kuleshov sheets 1/2; heights and offsets solved from the 2009 and 2012 photos). The lower pair sits on
    # stubs from the vertical wall; the upper pair hangs on a frame off the sloped face, the top egg on an
    # arm from the roof beside the column
    Be = 99.8
    for s in (1, -1):
        for (yc, xc) in ((19.85, 3.55), (17.75, 3.9), (13.8, 5.4), (11.75, 5.6)):
            egg_radome(c, P3(Be, s * xc, yc), (s, 0.0, 0.0))
            wall = 4.85 if yc < 12.3 else (4.4 if yc < 14.5 else max(1.5, 4.4 - (yc - 14.5) / 5.0 * 2.6))
            x_in = xc - EGG_L / 2
            if x_in - wall > 0.02:
                c.add(c.paint, cylinder(0.26, x_in - wall + 0.1, seg=10), xf=M(P3(Be, s * (wall - 0.1), yc), rot_z(-s * math.pi / 2)))
        for Bq in (Be - 1.05, Be + 1.05):          # frame posts for the upper pair, off the sloped face
            x_post = 2.95
            y_base = 14.5 + (4.4 - x_post) / 2.6 * 5.0
            c.add(c.paint, box(0.16, 20.6 - y_base, 0.16, center=(0, (20.6 - y_base) / 2, 0)), xf=M(P3(Bq, s * x_post, y_base)))
            for yc in (19.85, 17.75):
                c.add(c.paint, beam(P3(Bq, s * x_post, yc), P3(Be + (Bq - Be) * 0.55, s * (x_post + 0.45), yc), 0.1, 0.1),
                      occ=False)
    # column tapering from 2.5 m at the roof to 1.6 m under the turntable (sheet 1), flanges at both ends
    # and a mid band, as one lathe (no buried vertices for the AO bake)
    c.add(c.paint, lathe([(1.32, 19.5), (1.32, 19.62), (1.24, 19.62), (1.07, 22.9), (1.12, 22.9), (1.12, 23.0),
                          (1.06, 23.0), (0.82, 26.05), (0.9, 26.05), (0.9, 26.2), (0.0, 26.2)], seg=20),
          xf=M(P3(101.2, 0, 0)))
    ladder(c, P3(101.2 - 1.28, 0, 19.65), P3(101.2 - 0.86, 0, 26.0), 0.45, normal=(0, 0.25, 1))
    # Top Pair (MR-800 Voskhod), rotating on the column top
    A.top_pair(c, P3(101.2, 0, 26.2))


# =============================================================================================
# funnel (sheet 6), boats, crane
# =============================================================================================
def funnel(m):
    c = m.ctx; b = m.b
    b.node('Superstructure')
    ylo = dmin(103.1, 118.0, 6.0) - 0.3
    # intake block between the mast and the funnel
    house(c, sym_poly([(103.1, 5.4), (106.8, 5.4)]), ylo, 14.1, lip=0.1, bevel=0.25)
    for s in (1, -1):
        decal(c, 'vlouvre', P3(104.95, s * 5.42, 11.0), (s, 0, 0), 3.2, 4.6)
    # lower casing: rear face leans forward, chamfered corners, rim at the top
    house(c, sym_poly([(106.75, 5.9), (118.0, 5.9)]), ylo, 15.1, top_poly_Bx=sym_poly([(106.75, 5.9), (116.8, 5.9)]),
          lip=0.12, bevel=0.35)
    # two exhaust stacks: grey lower part, sooty upper part, raised uptakes with grilles
    for s in (1, -1):
        xa, xb = (1.25, 5.75) if s > 0 else (-5.75, -1.25)
        lo_b = [(106.75, xb), (116.85, xb), (116.85, xa), (106.75, xa)]
        lo_t = [(107.65, xb), (116.55, xb), (116.55, xa), (107.65, xa)]
        hi_t = [(108.25, xb), (116.4, xb), (116.4, xa), (108.25, xa)]
        house(c, lo_b, 15.0, 17.3, top_poly_Bx=lo_t, top=None, bevel=0.3)
        house(c, lo_t, 17.3, 18.6, top_poly_Bx=hi_t, walls='dark', top='dark', bevel=0.3)
        D.funnel_outlets(c, 108.25, 116.4, xa, xb, 18.6)
        # louvre panels on the casing sides: 3 rows x 6
        for row_y, hgt in ((7.9, 1.6), (10.0, 1.8), (12.3, 1.0)):
            for k in range(6):
                Bc = 108.3 + k * 1.45
                decal(c, 'louvre', P3(Bc, s * 5.92, row_y + hgt / 2), (s, 0, 0), 1.15, hgt)
        ladder(c, P3(117.2, s * 3.5, 15.1), P3(116.7, s * 3.5, 18.5), 0.45, normal=(0, 0, -1))
    house(c, sym_poly([(108.5, 1.25), (116.5, 1.25)]), 15.0, 16.6, bevel=0.15)
    rail_poly(c, sym_poly([(106.9, 5.75), (116.6, 5.75)]), 15.12, sides=(1, 3))
    decal(c, 'vlouvre', P3(106.73, 0, 12.3), (0, 0, 1), 9.0, 4.0)
    for s in (1, -1):
        door(c, 113.5, s * 5.92, dk(113.5, 5.9) - 0.05, (s, 0, 0))
    # crane deckhouse (B 119.9-125.3) and crane
    cd = sym_poly([(119.9, 3.6), (125.3, 3.6)])
    house(c, cd, dmin(119.9, 125.3) - 0.3, 9.45, lip=0.12, bevel=0.25)
    rail_poly(c, cd, 9.45)
    for s in (1, -1):
        door(c, 122.6, s * 3.62, 6.85, (s, 0, 0))
    D.crane(c, P3(122.9, 0, 9.45))


def boats(m):
    c = m.ctx
    m.b.node('Boats', parent=None)
    Bc = 114.8
    for s in (1, -1):
        xc = s * 8.25
        ydeck = dk(Bc, 8.0)
        o = np.array([xc, ydeck + 0.95, zB(Bc)])
        fr, fl = D.launch_boat(c, o, s)
        railing_pts(c, fr)
        railing_pts(c, fl)
        for dz in (-2.6, 2.4):
            c.add(c.paint, box(2.3, 1.05, 0.4, center=(0, 0, 0)), xf=M(np.array([xc, ydeck + 0.48, zB(Bc) + dz])))
    m.b.node('Superstructure')
    for s in (1, -1):
        ydeck = dk(Bc, 8.0)
        for Bd in (110.4, 119.3):
            D.slewing_davit(c, P3(Bd, s * 6.4, dk(Bd, 6.4) - 0.05), P3(Bd, s * 8.25, ydeck + 4.7), s)
        raft_rack(c, P3(121.0, s * 9.5, dk(121.0, 9.5) - 0.05), n=6, along=(0, 0, -1))
        raft_rack(c, P3(145.0, s * 9.5, dk(145.0, 9.5) - 0.05), n=6, along=(0, 0, -1))


# =============================================================================================
# S-300F Fort launchers
# =============================================================================================
VLS_B = (128.9, 132.95, 137.0, 141.05)
VLS_X = 2.87


def vls(m):
    c = m.ctx; b = m.b
    b.node('Weapons_VLS', parent=None)
    for s in (1, -1):
        cx = s * VLS_X
        r = 2.03
        poly = []
        for k in range(9):
            a = math.pi * k / 8
            poly.append((cx + r * math.cos(a), zB(126.8 + r) + r * math.sin(a)))
        for k in range(9):
            a = math.pi + math.pi * k / 8
            poly.append((cx + r * math.cos(a), zB(143.0 - r) + r * math.sin(a)))
        ytop = dk(135.0, 2.9) + 0.3
        c.add(c.paint, prism(poly, ytop - 0.8, ytop, top=False))
        P, N, UV, I = cap_polygon(np.array(poly), ytop, up=True)
        c.add(S2.BAND_UV, (P, N, S2.deck_band_uv(P), I))
        for Bc in VLS_B:
            WP.s300_hatch(c, Bc, cx, ytop)
            WP.s300_drive(c, Bc, cx, ytop + 0.05)
    # loading gantry on plate legs along the centreline, running on a deck rail
    yd = dk(134.0)
    c.add(c.sw("light"), D.rbox(0.45, 0.22, 14.0, r=0.06, seg=1), xf=M(P3(134.5, 0, yd)))
    WP.s300_gantry(c, 128.2, 137.8, yd, yd + 3.0, legs=(129.4, 136.6))
    for Bv in (131.3, 139.3):
        for s in (1, -1):
            mushroom_vent(c, P3(Bv, s * 6.0, dk(Bv, 6.0) - 0.05), r=0.55)
    for s in (1, -1):
        c.add(c.paint, cylinder(0.75, 0.9, seg=14), xf=M(P3(133.2, s * 8.8, dk(133.2, 8.8) - 0.05)))


# =============================================================================================
# aft superstructure, Top Dome, Pop Group, hangar, Osa-M (sheets 1 and 7)
# =============================================================================================
def aft_superstructure(m):
    c = m.ctx; b = m.b
    b.node('Superstructure')
    y0 = dmin(143.8, 157.5, 7.0) - 0.3
    a1 = sym_poly([(143.8, 6.4), (144.6, 7.25), (153.6, 7.25), (154.6, 6.3), (154.6, 3.0)])
    house(c, a1, y0, 9.7, lip=0.12, bevel=0.25, bevel_where=lambda B, x: B < 154)
    rail_poly(c, a1, 9.7)
    side_strip(c, 'PORTS', 145.1, 153.2, 7.27, 8.0, 1.0)
    end_strip(c, 'PORTS', 143.8, -5.8, 5.8, 8.0, 1.0, facing=1)
    for s in (1, -1):
        door(c, 149.5, s * 7.27, 6.8, (s, 0, 0))
    door(c, 143.8, 2.2, 6.8, (0, 0, 1))
    # Pop Group corner houses (B 154.6-157.4, x 4.7-7.2)
    for s in (1, -1):
        poly = [(154.5, s * 4.7), (154.5, s * 7.0), (156.9, s * 7.0), (157.4, s * 6.5), (157.4, s * 4.7)]
        house(c, poly if s > 0 else poly[::-1], y0, 9.7, bevel=0.15, bevel_where=lambda B, x: abs(x) > 6.0)
    a2 = sym_poly([(144.9, 4.5), (154.6, 4.5)])
    house(c, a2, 9.6, 12.2, lip=0.12, bevel=0.25)
    rail_poly(c, a2, 12.2)
    side_strip(c, 'PORTS', 146.0, 153.8, 4.52, 10.6, 1.0)
    a3 = sym_poly([(149.4, 2.4), (155.1, 2.4)])
    house(c, a3, 12.1, 14.8, lip=0.12, bevel=0.2)
    rail_poly(c, a3, 14.8)
    for s in (1, -1):
        door(c, 151.2, s * 2.42, 12.2, (s, 0, 0))
    # Top Dome (3R41 Volna)
    D.top_dome(c, P3(152.4, 0, 14.8))
    b.node('Superstructure')
    # signal mast beside the Top Dome (port side of the roof, clear of the radome)
    c.add(c.paint, tube_path([P3(154.1, 2.15, 14.8), P3(154.1, 2.15, 22.6)], 0.12, seg=6))
    c.add(c.paint, beam(P3(154.1, 1.1, 21.0), P3(154.1, 3.2, 21.0), 0.1, 0.1))
    # Pop Group directors (4R33), dishes face aft when stowed
    for s_, nm in ((1, 'PopGroup_P'), (-1, 'PopGroup_S')):
        D.pop_group(c, P3(156.1, s_ * 5.9, 9.7), s_, nm)
    b.node('Superstructure')
    # hangar with a steep sloped aft door
    poly = [(zB(154.6), 6.3), (zB(165.6), 6.3), (zB(165.6), 6.95), (zB(164.0), 11.3), (zB(154.6), 11.3)]
    c.add(c.paint, extrude_x(poly, -3.0, 3.0))
    nrm = normalize(np.array([0.0, 1.6, -4.35]))
    q = _quad(P3(165.55, 2.7, 7.05), P3(165.55, -2.7, 7.05), P3(164.05, -2.7, 11.2), P3(164.05, 2.7, 11.2),
              uv=[(0, 1), (1, 1), (1, 0), (0, 0)], n=nrm)
    Pq, Nq, UVq, Iq = q
    Pq = Pq + nrm * 0.04
    if np.dot(np.cross(Pq[1] - Pq[0], Pq[2] - Pq[0]), nrm) < 0:
        Iq = Iq[:, ::-1]
    c.add(c.rect('hangar_door'), (Pq, Nq, UVq, Iq))
    rail_poly(c, sym_poly([(154.6, 3.0), (164.0, 3.0)]), 11.3, sides=(0, 2))
    for s in (1, -1):
        c.add(c.paint, tube_path([P3(161.6, s * 2.75, 7.4), P3(154.8, s * 2.75, 12.6)], 0.16, seg=6))
    # Osa-M launchers (retractable bins) beside the hangar door
    for s, nm in ((1, 'OsaM_P'), (-1, 'OsaM_S')):
        o = P3(164.1, s * 4.9, QD_Y - 0.1)
        b.push(nm, parent='Superstructure', translation=tuple(o))
        c.add(c.paint, cylinder(1.85, 7.0 - QD_Y + 0.1, seg=24), xf=M(o))
        c.add(c.paint, cylinder(1.92, 0.15, seg=24), xf=M(o + np.array([0, 7.0 - QD_Y + 0.1, 0])))
        c.add(c.sw('mid'), lathe([(1.6, 0.0), (1.5, 0.2), (0.9, 0.35), (0.0, 0.38)], seg=18),
              xf=M(o + np.array([0, 7.25 - QD_Y + 0.1, 0])))
        for k in range(5):
            a = math.radians(-60 + 30 * k) + (0 if s > 0 else math.pi)
            p = o + np.array([1.87 * math.cos(a), 1.2, 1.87 * math.sin(a)])
            decal(c, 'panel_dark', p, (math.cos(a), 0, math.sin(a)), 0.6, 0.45, offset=0.03)
        b.pop()


def stern(m):
    c = m.ctx; b = m.b
    b.node('Superstructure')
    # apron deckhouse between the hangar and the helideck (on the quarterdeck)
    ap = sym_poly([(161.6, 3.1), (169.0, 3.1), (169.0, 3.95), (172.6, 3.95), (173.4, 3.1)])
    house(c, ap, QD_Y - 0.2, 6.95, top='heli', lip=0.12)
    for s in (1, -1):
        door(c, 167.6, s * 3.12, QD_Y, (s, 0, 0))
        door(c, 171.0, s * 3.97, QD_Y, (s, 0, 0))
    # helicopter control cab (starboard)
    # low wall, outward-leaning glazing with mullions, overhanging roof
    def rect_Bx(B0, B1, x0, x1, d=0.0):
        return [(B0 - d, x0 + d), (B1 + d, x0 + d), (B1 + d, x1 - d), (B0 - d, x1 - d)]
    cab = rect_Bx(166.6, 169.7, -0.9, -3.1)
    house(c, cab, 6.9, 7.2, top=None, bevel=0.15)
    house(c, cab, 7.2, 7.95, top_poly_Bx=rect_Bx(166.6, 169.7, -0.9, -3.1, 0.15), walls='glass', top=None, bevel=0.15)
    house(c, rect_Bx(166.6, 169.7, -0.9, -3.1, 0.3), 7.95, 8.12, top='paint', bottom=True, bevel=0.2)
    lo, hi = rect_Bx(166.6, 169.7, -0.9, -3.1), rect_Bx(166.6, 169.7, -0.9, -3.1, 0.15)
    for k in range(4):
        a0, a1 = np.array(lo[k]), np.array(lo[(k + 1) % 4])
        b0, b1 = np.array(hi[k]), np.array(hi[(k + 1) % 4])
        n = max(2, int(round(np.linalg.norm(a1 - a0) / 0.75)))
        for t in np.linspace(0.08, 0.92, n):
            pa, pb = a0 + (a1 - a0) * t, b0 + (b1 - b0) * t
            c.add(c.sw('dark'), beam(P3(pa[0], pa[1], 7.2), P3(pb[0], pb[1], 7.95), 0.06, 0.06))
    whip(c, P3(167.2, -2.6, 8.12), 3.0, r=0.04)
    G.lamp(c, P3(169.2, -1.4, 8.12), 'white')
    # helideck: octagon B 173.2-185.2, +-5.3, on a support box and pillars
    Bf, Ba, hw, ch = 173.2, 185.2, 5.3, 1.8
    octo = [(Bf, hw - ch), (Bf + ch, hw), (Ba - ch, hw), (Ba, hw - ch), (Ba, -hw + ch), (Ba - ch, -hw), (Bf + ch, -hw),
            (Bf, -hw + ch)]
    poly_xz = poly_Bx_to_xz(octo)
    P, N, UV, I = prism(poly_xz, 6.2, 6.5, top=False, bottom=True)
    c.add(c.paint, (P, N, UV, I))
    P, N, UV, I = cap_polygon(np.array(poly_xz), 6.5, up=True)
    UV = np.stack([(P[:, 0] + hw) / (2 * hw), (zB(Bf) - P[:, 2]) / (Ba - Bf)], axis=1)
    c.add(c.rect('helideck', clamp=False), (P, N, UV, I))
    house(c, sym_poly([(174.0, 3.4), (184.4, 3.4)]), QD_Y - 0.2, 6.25, top=None)
    for (Bp, xp) in ((174.5, 4.6), (184.0, 4.6)):
        for s in (1, -1):
            c.add(c.paint, cylinder(0.22, 6.25 - QD_Y + 0.2, seg=8, y0=QD_Y - 0.2), xf=M(P3(Bp, s * xp, 0)))
    pts = [P3(B, x, 6.5) for (B, x) in octo]
    railing_pts(c, pts, closed=True, band='NET', repeat=st.NET_REPEAT_M, band_h=1.1)
    # flagstaff + ensign at the stern
    p0 = P3(185.0, 0.0, 6.5); p1 = p0 + np.array([0, 5.0, -0.8])
    c.add(c.paint, tube_path([p0, p1], 0.07, seg=5))
    fa = p1 + np.array([0, -0.15, 0]); fb = p1 + np.array([0, -1.35, 0.2])
    c.add(c.rect('ensign'), _quad(fb, fb + np.array([0.0, 0, -1.8]), fa + np.array([0.0, 0, -1.8]), fa,
                                  uv=[(0, 1), (1, 1), (1, 0), (0, 0)]))
    # quarterdeck fittings
    for B in (167.0, 176.0, 183.5):
        hb = float(DECK_HB(B))
        for s in (1, -1):
            bollard(c, P3(B, s * (hb - 0.8), QD_Y - 0.03))
    for s in (1, -1):
        G.capstan(c, P3(179.5, s * 6.2, QD_Y - 0.03))


def deck_railings(m):
    c = m.ctx
    m.b.node('Hull')
    for s in (1, -1):
        pts = [P3(B, s * (float(DECK_HB(B)) - 0.12), float(SHEER(B))) for B in list(np.arange(1.0, 161.0, 2.5)) + [161.4]]
        railing_pts(c, pts)
        pts = [P3(B, s * (float(DECK_HB(B)) - 0.12), float(SHEER(B))) for B in list(np.arange(165.0, 186.0, 2.5)) + [186.2]]
        railing_pts(c, pts)
    railing_pts(c, [P3(186.25, 7.4, QD_Y), P3(186.25, -7.4, QD_Y)])


# =============================================================================================
# small fittings: searchlights, decoy launchers, vents, lockers, horns
# =============================================================================================
def searchlight(c, pos, facing=0.0):
    o = np.asarray(pos, float)
    R = rot_y(facing)
    c.add(c.paint, cylinder(0.18, 0.7, seg=8), xf=M(o))
    c.add(c.paint, box(0.5, 0.25, 0.25, center=(0, 0.75, 0)), xf=M(o, R))
    c.add(c.paint, cylinder(0.32, 0.6, seg=12, caps=(True, False)), xf=M(o + R @ np.array([0, 1.05, -0.3]), R @ rot_x(math.pi / 2)))
    c.add(c.sw('glass'), cylinder(0.3, 0.02, seg=12, caps=(False, True)), xf=M(o + R @ np.array([0, 1.05, 0.3]), R @ rot_x(math.pi / 2)))


def decoy_launcher(c, pos, facing=0.0, tubes=2):
    """PK-2 twin decoy launcher"""
    WP.pk2(c, pos, facing)


def locker(c, B, x, y, w=1.2, h=1.0, d=0.6, along=False):
    if along:
        c.add(c.paint, box(d, h, w, center=(0, 0, 0)), xf=M(P3(B, x, y + h / 2)))
    else:
        c.add(c.paint, box(w, h, d, center=(0, 0, 0)), xf=M(P3(B, x, y + h / 2)))


def fittings(m):
    c = m.ctx; b = m.b
    b.node('Superstructure')
    # searchlights: bridge wings, tower balconies, aft superstructure
    for s in (1, -1):
        searchlight(c, P3(77.0, s * (tower_hw(28.0) + 0.9), 28.0), facing=s * math.radians(70))
        searchlight(c, P3(146.0, s * 3.6, 12.2), facing=s * math.radians(40))
    # PK-10 / PK-2 decoy launchers on the 01 level by the funnel and aft
    for s in (1, -1):
        for B in (87.6, 103.0):
            decoy_launcher(c, P3(B, s * 7.6, 9.5), facing=s * math.radians(90), tubes=2)
        decoy_launcher(c, P3(144.4, s * 6.2, 9.7), facing=s * math.radians(90), tubes=2)
    # mushroom vents and lockers on the forecastle and main deck walkways
    for (B, x) in ((27.0, 6.2), (36.5, 7.6), (106.0, 7.8), (126.5, 6.6), (143.8, 8.2), (158.5, 7.4)):
        for s in (1, -1):
            G.round_vent(c, P3(B, s * x, dk(B, x) - 0.05), r=0.36, h=0.7)
    for (B, x, w) in ((119.2, 4.6, 1.2), (142.4, 6.8, 1.5)):
        for s in (1, -1):
            G.ready_locker(c, P3(B, s * x, dk(B, x) - 0.05), w=w, h=1.1, d=0.7, yaw=0.0 if s > 0 else math.pi)
    # horns / signal lamps on the foremast yard
    for s in (1, -1):
        c.add(c.sw('mid'), cylinder(0.12, 0.5, seg=6), xf=M(P3(81.6, s * 6.4, 29.3)))
        c.add(c.sw('white'), sphere(0.18, seg=8, rings=4), xf=M(P3(81.6, s * 6.4, 29.9)))
    # navigation lights on the tower top platform
    for s in (1, -1):
        G.lamp(c, P3(77.2, s * 1.85, 31.45), 'flagred' if s > 0 else 'green')
    # stern: towing/mooring capstans on the quarterdeck and liferafts on the hangar sides
    for s in (1, -1):
        raft_rack(c, P3(156.0, s * 3.2, 11.3), n=3, along=(0, 0, -1), spacing=0.8)


# =============================================================================================
# build
# =============================================================================================
def build_all(path='slava_class.glb'):
    m = S2.Model()
    register_decals(m)
    S2.build_hull(m, transom_hole=(2.6, 0.25, 4.0))
    deck_paint(m)
    hull_fittings(m)
    forecastle(m)
    ak130(m)
    forward_deckhouse(m)
    bridge(m)
    foremast(m)
    launchers(m)
    mainmast(m)
    funnel(m)
    boats(m)
    vls(m)
    aft_superstructure(m)
    stern(m)
    fittings(m)
    deck_railings(m)
    print('triangles:', m.b.tri_count())
    for k, v in m.b.tri_count_by_node().items():
        print(f'  {k:22s} {v}')
    S2.export(m, path, ao=AO)
    return m


if __name__ == '__main__':
    import sys
    build_all(sys.argv[1] if len(sys.argv) > 1 else 'slava_class.glb')
