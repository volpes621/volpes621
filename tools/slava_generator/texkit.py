"""texkit: procedural texture painting helpers (numpy + PIL).

A Layered canvas keeps base colour (linear-ish sRGB floats 0..1), alpha,
height (for normal map generation), roughness and metallic.
"""
import io
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT_DIR = '/usr/share/fonts/truetype/'


def srgb(hexstr):
    hexstr = hexstr.lstrip('#')
    return np.array([int(hexstr[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def value_noise(h, w, cell, seed=0, octaves=4, persistence=0.5):
    """Tileable fractal value noise in [0,1], shape (h, w)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w))
    amp = 1.0
    total = 0.0
    c = cell
    for o in range(octaves):
        gh = max(1, int(round(h / c)))
        gw = max(1, int(round(w / c)))
        grid = rng.random((gh, gw))
        # bicubic-ish upsample with wrap (tileable)
        ys = np.arange(h) * gh / h
        xs = np.arange(w) * gw / w
        y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
        fy = ys - y0; fx = xs - x0
        fy = fy * fy * (3 - 2 * fy); fx = fx * fx * (3 - 2 * fx)
        y1 = (y0 + 1) % gh; x1 = (x0 + 1) % gw
        y0 %= gh; x0 %= gw
        a = grid[np.ix_(y0, x0)]; b = grid[np.ix_(y0, x1)]
        cc = grid[np.ix_(y1, x0)]; d = grid[np.ix_(y1, x1)]
        top = a + (b - a) * fx[None, :]
        bot = cc + (d - cc) * fx[None, :]
        out += amp * (top + (bot - top) * fy[:, None])
        total += amp
        amp *= persistence
        c = max(1, c / 2)
    return out / total


class Layers:
    def __init__(self, w, h, base=(0.5, 0.5, 0.5), rough=0.6, metal=0.0, alpha=1.0):
        self.w, self.h = w, h
        self.col = np.ones((h, w, 3)) * np.asarray(base)
        self.alpha = np.ones((h, w)) * alpha
        self.height = np.zeros((h, w))
        self.rough = np.ones((h, w)) * rough
        self.metal = np.ones((h, w)) * metal

    # --- region helpers (pixel coords: x right, y down) ---
    def rect(self, x0, y0, x1, y1, col=None, alpha=None, height=None, rough=None, metal=None, add_height=None):
        x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
        x0 = max(0, x0); y0 = max(0, y0); x1 = min(self.w, x1); y1 = min(self.h, y1)
        if x1 <= x0 or y1 <= y0:
            return
        if col is not None:
            self.col[y0:y1, x0:x1] = np.asarray(col)
        if alpha is not None:
            self.alpha[y0:y1, x0:x1] = alpha
        if height is not None:
            self.height[y0:y1, x0:x1] = height
        if add_height is not None:
            self.height[y0:y1, x0:x1] += add_height
        if rough is not None:
            self.rough[y0:y1, x0:x1] = rough
        if metal is not None:
            self.metal[y0:y1, x0:x1] = metal

    def mask_apply(self, mask, col=None, alpha=None, height=None, add_height=None, rough=None, metal=None, x0=0, y0=0, blend=1.0):
        """mask: float array (mh, mw) in 0..1 placed at (x0, y0)."""
        mh, mw = mask.shape
        xs0, ys0 = max(0, x0), max(0, y0)
        xs1, ys1 = min(self.w, x0 + mw), min(self.h, y0 + mh)
        if xs1 <= xs0 or ys1 <= ys0:
            return
        m = mask[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] * blend
        sl = (slice(ys0, ys1), slice(xs0, xs1))
        if col is not None:
            c = np.asarray(col)
            if c.ndim == 1:
                self.col[sl] = self.col[sl] * (1 - m[..., None]) + c * m[..., None]
            else:
                cc = c[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
                self.col[sl] = self.col[sl] * (1 - m[..., None]) + cc * m[..., None]
        if alpha is not None:
            self.alpha[sl] = self.alpha[sl] * (1 - m) + alpha * m
        if height is not None:
            self.height[sl] = self.height[sl] * (1 - m) + height * m
        if add_height is not None:
            self.height[sl] += add_height * m
        if rough is not None:
            self.rough[sl] = self.rough[sl] * (1 - m) + rough * m
        if metal is not None:
            self.metal[sl] = self.metal[sl] * (1 - m) + metal * m

    def draw_mask(self, w, h, fn, ss=4):
        """Render a PIL drawing function fn(draw, scale) into an antialiased mask (h, w)."""
        im = Image.new('L', (w * ss, h * ss), 0)
        d = ImageDraw.Draw(im)
        fn(d, ss)
        im = im.resize((w, h), Image.LANCZOS)
        return np.asarray(im).astype(np.float64) / 255.0

    # --- export ---
    def base_png(self, with_alpha=True):
        c = np.clip(self.col, 0, 1)
        if with_alpha:
            a = np.clip(self.alpha, 0, 1)[..., None]
            arr = np.concatenate([c, a], axis=2)
            im = Image.fromarray((arr * 255 + 0.5).astype(np.uint8), 'RGBA')
        else:
            im = Image.fromarray((c * 255 + 0.5).astype(np.uint8), 'RGB')
        return im

    def mr_png(self, occlusion=None):
        # glTF: G = roughness, B = metallic, R = occlusion (if used)
        r = np.ones((self.h, self.w)) if occlusion is None else occlusion
        arr = np.stack([r, np.clip(self.rough, 0, 1), np.clip(self.metal, 0, 1)], axis=2)
        return Image.fromarray((arr * 255 + 0.5).astype(np.uint8), 'RGB')

    def normal_png(self, strength=2.0, wrap=True):
        hgt = self.height
        if wrap:
            dx = (np.roll(hgt, -1, axis=1) - np.roll(hgt, 1, axis=1)) * 0.5
            dy = (np.roll(hgt, -1, axis=0) - np.roll(hgt, 1, axis=0)) * 0.5
        else:
            dx = np.gradient(hgt, axis=1)
            dy = np.gradient(hgt, axis=0)
        nx = -dx * strength
        ny = dy * strength   # OpenGL convention (+Y up in tangent space); image y goes down
        nz = np.ones_like(hgt)
        n = np.stack([nx, ny, nz], axis=2)
        n /= np.linalg.norm(n, axis=2, keepdims=True)
        arr = n * 0.5 + 0.5
        return Image.fromarray((arr * 255 + 0.5).astype(np.uint8), 'RGB')


def png_bytes(im):
    bio = io.BytesIO()
    im.save(bio, format='PNG', optimize=True)
    return bio.getvalue()


def font(name='dejavu/DejaVuSans-Bold.ttf', size=32):
    try:
        return ImageFont.truetype(FONT_DIR + name, size)
    except Exception:
        return ImageFont.load_default()


def blur(a, r):
    if r <= 0:
        return a
    im = Image.fromarray((np.clip(a, 0, 1) * 65535).astype(np.uint16).astype(np.int32), 'I')
    im = im.convert('F')
    im = im.filter(ImageFilter.GaussianBlur(r))
    return np.asarray(im) / 65535.0
