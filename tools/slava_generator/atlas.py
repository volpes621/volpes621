"""Texture set for the Slava-class model.

Atlas (2048x2048, RGBA, alphaMode MASK) layout - rows are full-width "bands" so that
u may repeat (wrapS = REPEAT) while v stays inside the band:

  rows    0..255   HULL   : hull side strip, u = (z - zmin)/L, v by height
  rows  256..511   DECK   : main deck plan,  u = (z - zmin)/L, v by x (port at top)
  rows  512..575   RAIL   : guard rail (alpha), 1 repeat = RAIL_REPEAT_M metres
  rows  576..639   WINDOW : bridge window strip, 1 repeat = WIN_REPEAT_M
  rows  640..671   PORTS  : porthole row
  rows  672..703   LOUVER : louvre / vent grille
  rows  704..735   LADDER : vertical ladder (alpha), u runs along the ladder
  rows  736..799   LATTICE: open lattice (alpha) for radar reflectors / mast panels
  rows  800..2047  RECT   : decals, emblems, special faces, colour swatches

Plus tiling materials: PAINT (1024^2) and DECKTILE (512^2).
"""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from texkit import Layers, value_noise, srgb, png_bytes, font, blur
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
FONT_BIG = os.path.join(_HERE, 'fonts', 'BigShoulders-Bold.ttf')     # SIL OFL 1.1
FONT_SANS = os.path.join(_HERE, 'fonts', 'DejaVuSans-Bold.ttf')      # Bitstream Vera licence

W = 2048
BANDS = {
    'HULL': (0, 320), 'DECK': (320, 576), 'RAIL': (576, 640), 'WINDOW': (640, 704),
    'PORTS': (704, 736), 'LOUVER': (736, 768), 'LADDER': (768, 800), 'LATTICE': (800, 896),
    'NET': (896, 960),
}
RAIL_REPEAT_M = 24.0     # metres of railing per full atlas width
WIN_REPEAT_M = 24.0
PORTS_REPEAT_M = 48.0
LOUVER_REPEAT_M = 16.0
LADDER_REPEAT_M = 12.0
LATTICE_REPEAT_M = 25.6      # isotropic: band (96 px) = 1.2 m -> 80 px/m
LAT_BAND_M = 1.2
NET_REPEAT_M = 35.0

# ---------------------------------------------------------------- palette (sRGB)
PAL = {
    'hull': srgb('546167'),        # Russian navy blue-grey (hull)
    'super': srgb('5a676d'),       # superstructure (same paint, slightly lighter)
    'deck': srgb('343536'),        # dark grey walkways
    'deck_red': srgb('4e2a20'),    # red-brown deck paint
    'red': srgb('4a1c16'),         # oxide-red antifouling
    'boot': srgb('d6d2cc'),        # thin white waterline stripe
    'black': srgb('161718'),
    'dark': srgb('2e3337'),
    'mid': srgb('454d52'),
    'light': srgb('7d868c'),
    'white': srgb('dcdcd6'),
    'glass': srgb('1a242c'),
    'rubber': srgb('202020'),
    'orange': srgb('c45a18'),
    'brass': srgb('a8843c'),
    'bronze': srgb('8a6438'),
    'steel': srgb('7c8288'),
    'radome': srgb('b9b9b2'),
    'canvas': srgb('6f7366'),
    'blue': srgb('1f4aa8'),
    'flagred': srgb('b8282a'),
    'green': srgb('3c5a3a'),
    'heli': srgb('4d5f54'),
    'mesh': srgb('5f666b'),
}

# swatch grid in RECT area: 32x32 cells starting at (0, 2016) going right
SWATCH_ORDER = ['hull', 'super', 'deck', 'deck_red', 'red', 'boot', 'black', 'dark', 'mid', 'light', 'white',
                'glass', 'rubber', 'orange', 'brass', 'bronze', 'steel', 'radome', 'canvas', 'blue', 'flagred', 'green',
                'heli', 'mesh']
SWATCH_Y0 = 2016
SWATCH = 32


def swatch_rect(name):
    i = SWATCH_ORDER.index(name)
    x0 = i * SWATCH
    return (x0 + 6, SWATCH_Y0 + 6, x0 + SWATCH - 6, SWATCH_Y0 + SWATCH - 6)


class AtlasLayout:
    """Holds rects (pixel) for decals; converted to uv rects (0..1) for meshkit."""

    def __init__(self):
        self.rects = {}

    def uv(self, name):
        x0, y0, x1, y1 = self.rects[name]
        return (x0 / W, y0 / W, x1 / W, y1 / W)

    def add(self, name, rect):
        self.rects[name] = rect
        return rect


def band_uv(name):
    y0, y1 = BANDS[name]
    return (0.0, (y0 + 0.5) / W, 1.0, (y1 - 0.5) / W)


# ---------------------------------------------------------------- painters

def paint_rail(L):
    """Stanchions every 1.5 m, 3 rails, 1.1 m high; band covers 1.2 m height."""
    y0, y1 = BANDS['RAIL']
    h = y1 - y0
    L.rect(0, y0, W, y1, col=PAL['super'], alpha=0.0, rough=0.5, metal=0.2)
    band_m = 1.2

    def ypx(m):  # height (m) above deck -> pixel row (top of band = 1.2 m)
        return y0 + (band_m - m) / band_m * h
    # rails at 1.05, 0.7, 0.35 m
    for hm, th in ((1.05, 0.045), (0.70, 0.03), (0.35, 0.03)):
        yy = ypx(hm)
        L.rect(0, yy - th / band_m * h / 2 - 0.6, W, yy + th / band_m * h / 2 + 0.6, col=PAL['super'] * 0.95, alpha=1.0)
    n = int(RAIL_REPEAT_M / 1.5)
    for i in range(n):
        x = (i + 0.5) * W / n
        L.rect(x - 2.2, ypx(1.08), x + 2.2, ypx(0.0), col=PAL['super'] * 0.9, alpha=1.0)


def paint_window(L):
    y0, y1 = BANDS['WINDOW']
    L.rect(0, y0, W, y1, col=PAL['super'], alpha=0.0, rough=0.55, metal=0.1)
    n = int(WIN_REPEAT_M / 1.25)
    wpx = W / n
    for i in range(n):
        x0 = i * wpx + wpx * 0.10
        x1 = (i + 1) * wpx - wpx * 0.10
        L.rect(x0 - 2, y0 + 6, x1 + 2, y1 - 6, col=PAL['dark'], alpha=1.0, add_height=0.5)
        L.rect(x0, y0 + 8, x1, y1 - 8, col=PAL['glass'], alpha=1.0, rough=0.08, metal=0.6, height=-0.6)
        # subtle sky reflection gradient
        g = np.linspace(0.30, 0.0, int(y1 - 8 - (y0 + 8)))[:, None]
        L.col[int(y0 + 8):int(y1 - 8), int(x0):int(x1)] += g[..., None] * np.array([0.18, 0.22, 0.26])


def paint_ports(L):
    y0, y1 = BANDS['PORTS']
    h = y1 - y0
    L.rect(0, y0, W, y1, col=PAL['super'], alpha=0.0)
    n = int(PORTS_REPEAT_M / 1.6)
    wpx = W / n
    rr = h * 0.28
    yy, xx = np.mgrid[0:h, 0:int(math.ceil(wpx))]
    for i in range(n):
        cx = i * wpx + wpx / 2
        x0 = int(cx - wpx / 2)
        d = np.hypot(xx + x0 - cx, yy - h / 2)
        rim = np.clip(1.0 - np.abs(d - rr * 1.25) / 1.3, 0, 1)
        glass = np.clip(rr - d + 0.5, 0, 1)
        L.mask_apply(rim, col=PAL['super'] * 0.62, alpha=1.0, add_height=0.6, x0=x0, y0=y0)
        L.mask_apply(glass, col=PAL['glass'], alpha=1.0, rough=0.1, metal=0.5, add_height=-0.4, x0=x0, y0=y0)
    a = L.alpha[y0:y1]
    L.alpha[y0:y1] = np.where(a > 0.4, 1.0, 0.0)


def paint_louver(L):
    y0, y1 = BANDS['LOUVER']
    L.rect(0, y0, W, y1, col=PAL['super'] * 0.82, alpha=1.0)
    for yy in range(y0 + 3, y1 - 3, 4):
        L.rect(0, yy, W, yy + 2, col=PAL['super'] * 0.45, add_height=-0.8)
    L.rect(0, y0, W, y0 + 2, col=PAL['super'] * 0.7, add_height=0.8)
    L.rect(0, y1 - 2, W, y1, col=PAL['super'] * 0.7, add_height=0.8)


def paint_ladder(L):
    y0, y1 = BANDS['LADDER']
    L.rect(0, y0, W, y1, col=PAL['super'], alpha=0.0)
    L.rect(0, y0 + 2, W, y0 + 6, col=PAL['super'] * 0.9, alpha=1.0)
    L.rect(0, y1 - 6, W, y1 - 2, col=PAL['super'] * 0.9, alpha=1.0)
    n = int(LADDER_REPEAT_M / 0.3)
    for i in range(n):
        x = (i + 0.5) * W / n
        L.rect(x - 1.5, y0 + 2, x + 1.5, y1 - 2, col=PAL['super'] * 0.85, alpha=1.0)


def paint_lattice(L):
    """antenna reflector mesh at 80 px/m: rods 0.1 m apart, ribs 0.3 m, truss diagonals (alpha-tested)."""
    y0, y1 = BANDS['LATTICE']
    h = y1 - y0
    c = PAL['mesh']
    L.rect(0, y0, W, y1, col=c, alpha=0.0, rough=0.45, metal=0.35)
    for y in range(y0 + 1, y1 - 1, 8):                     # horizontal rods (3 px)
        L.rect(0, y, W, y + 3, col=c * 0.95, alpha=1.0)
    for x in range(0, W, 24):                               # vertical ribs (4 px)
        L.rect(x, y0, x + 4, y1, col=c * 0.85, alpha=1.0)
    for x0 in range(0, W, 48):                              # diagonals
        for t in range(h):
            xx = x0 + t * 48 / h
            L.rect(xx, y0 + t, xx + 3, y0 + t + 1, col=c * 0.8, alpha=1.0)
            xx2 = x0 + 48 - t * 48 / h
            L.rect(xx2, y0 + t, xx2 + 3, y0 + t + 1, col=c * 0.8, alpha=1.0)
    L.rect(0, y0, W, y0 + 3, alpha=1.0, col=c * 0.75)
    L.rect(0, y1 - 3, W, y1, alpha=1.0, col=c * 0.75)


def paint_net(L):
    """helideck safety net: light diamond mesh + top rail and posts (white, as photographed on Varyag)"""
    y0, y1 = BANDS['NET']
    h = y1 - y0
    c = PAL['white'] * 0.92
    L.rect(0, y0, W, y1, col=c, alpha=0.0, rough=0.6, metal=0.05)
    for x0 in range(-h, W, 7):
        for t in range(h):
            L.rect(x0 + t, y0 + t, x0 + t + 1.6, y0 + t + 1, col=c * 0.95, alpha=1.0)
            L.rect(x0 + h - t, y0 + t, x0 + h - t + 1.6, y0 + t + 1, col=c * 0.95, alpha=1.0)
    L.rect(0, y0, W, y0 + 4, col=c, alpha=1.0)
    for x in range(0, W, 40):
        L.rect(x, y0, x + 4, y1, col=c * 0.97, alpha=1.0)


def paint_swatches(L):
    for i, name in enumerate(SWATCH_ORDER):
        x0 = i * SWATCH
        rough = {'glass': 0.08, 'white': 0.45, 'radome': 0.4, 'rubber': 0.85, 'brass': 0.35,
                 'bronze': 0.3, 'steel': 0.35, 'canvas': 0.9, 'deck': 0.75, 'deck_red': 0.75, 'red': 0.55,
                 'heli': 0.7}.get(name, 0.42)
        metal = {'brass': 0.9, 'bronze': 0.9, 'steel': 0.7, 'glass': 0.5}.get(name, 0.1)
        L.rect(x0, SWATCH_Y0, x0 + SWATCH, SWATCH_Y0 + SWATCH, col=PAL[name], alpha=1.0, rough=rough, metal=metal)


def apply_grime(L, x0, y0, x1, y1, seed, amount=0.07, scale=40):
    w, h = int(x1 - x0), int(y1 - y0)
    n = value_noise(h, w, scale, seed=seed, octaves=5)
    L.col[y0:y1, x0:x1] *= (1.0 - amount + 2 * amount * n)[..., None]


def make_tile_paint(size=1024, seed=3, base=None, repeat_m=12.0):
    """Tileable paint: subtle mottling, faint plate seams (every 3 m x 2 m), weld lines."""
    base = PAL['super'] if base is None else base
    L = Layers(size, size, base=base, rough=0.62, metal=0.12)
    n1 = value_noise(size, size, 128, seed=seed, octaves=5)
    n2 = value_noise(size, size, 16, seed=seed + 7, octaves=3)
    L.col *= (0.93 + 0.10 * n1 + 0.03 * n2)[..., None]
    L.rough = 0.55 + 0.15 * n1
    px_per_m = size / repeat_m
    # plate seams (raised weld beads)
    for i in range(int(repeat_m / 3.0)):
        x = int(i * 3.0 * px_per_m)
        L.rect(x, 0, x + 2, size, add_height=0.5, col=None)
    for j in range(int(repeat_m / 2.0)):
        y = int(j * 2.0 * px_per_m)
        L.rect(0, y, size, y + 2, add_height=0.5)
    # faint vertical run-down streaks
    rng = np.random.default_rng(seed)
    for k in range(40):
        x = rng.integers(0, size)
        y = rng.integers(0, size)
        ln = rng.integers(20, 160)
        w = rng.integers(1, 3)
        a = rng.uniform(0.02, 0.06)
        yy = np.arange(y, y + ln) % size
        L.col[yy, x:x + w] *= (1 - a)
    return L


def make_tile_deck(size=512, seed=5, base=None, repeat_m=6.0):
    base = PAL['deck'] if base is None else base
    L = Layers(size, size, base=base, rough=0.85, metal=0.02)
    n1 = value_noise(size, size, 64, seed=seed, octaves=5)
    rng = np.random.default_rng(seed)
    grit = rng.random((size, size))
    L.col *= (0.88 + 0.16 * n1 + 0.06 * (grit - 0.5))[..., None]
    L.height = grit * 0.6
    px_per_m = size / repeat_m
    for i in range(int(repeat_m / 1.5)):
        x = int(i * 1.5 * px_per_m)
        L.rect(x, 0, x + 1, size, add_height=-0.6)
    return L


def text_mask(text, w, h, fnt, anchor_xy=None, ss=2, stroke=0):
    im = Image.new('L', (w * ss, h * ss), 0)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(fnt.path, int(fnt.size * ss)) if hasattr(fnt, 'path') else fnt
    bbox = d.textbbox((0, 0), text, font=f, stroke_width=stroke * ss)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (w * ss - tw) / 2 - bbox[0]
    y = (h * ss - th) / 2 - bbox[1]
    d.text((x, y), text, fill=255, font=f, stroke_width=stroke * ss, stroke_fill=255)
    im = im.resize((w, h), Image.LANCZOS)
    return np.asarray(im).astype(np.float64) / 255.0
