"""Painters for decals and markings (texture atlas), adapted from the Slava-class generator.

Hull numbers are drawn as strokes in the style of the Russian Navy pennant numbers; the rest are plain
painted panels (doors, louvres, window rows, liferafts, ensign, crest)."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import atlas as st
from texkit import srgb, value_noise
from kit import RAFT_PROF

PAL = st.PAL

def _font(path, size):
    return ImageFont.truetype(path, size)


def text_width_px(text, h, size_frac, spacing):
    f = _font(st.FONT_SANS, int(h * size_frac))
    d = ImageDraw.Draw(Image.new('L', (8, 8)))
    widths = [d.textbbox((0, 0), ch, font=f)[2] for ch in text]
    return int(sum(widths) + spacing * (len(text) - 1)) + 1


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


def paint_ensign(L, rect):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['white'], alpha=1.0, rough=0.9, metal=0.0)

    def fn(d, s):
        t = int(h * 0.17 * s)
        d.line([(0, 0), (w * s, h * s)], fill=255, width=t)
        d.line([(0, h * s), (w * s, 0)], fill=255, width=t)
    L.mask_apply(L.draw_mask(w, h, fn), col=PAL['blue'], x0=x0, y0=y0)


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
