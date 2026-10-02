"""Admiral Gorshkov-class (Project 22350) hull.

Lines measured on a 1:500 general-arrangement drawing (side, plan, bow and stern views; scale set by
the 135.0 m overall length) and checked against photographs of Admiral Gorshkov and Admiral Kasatonov.

B = metres aft of the stem head, z = Z_BOW - B, y = height above the design waterline, +x = port.
The main hull ends at a knuckle (the main-deck edge, KNUCKLE); above it the upper hull leans inward
(TUMBLE) as the forecastle bulwark, the superstructure sides and the hangar sides.
"""
import math
import numpy as np
from scipy.interpolate import PchipInterpolator

LOA = 135.0
Z_BOW = LOA / 2.0
Z_STERN = -LOA / 2.0

STEM_TOP_Y = 7.98      # bulwark top at the stem head (B = 0)
STEM_WL_B = 8.7        # stem meets the DWL
TRANSOM_BOT = (132.6, -0.75)   # (B, y) where the transom meets the bottom
TRANSOM_TOP = (135.0, 4.92)    # (B, y) top of the transom = helideck edge
TUMBLE = math.tan(math.radians(5.5))   # upper hull: inward lean per metre of height
DRAFT = 4.55


def zB(B):
    return Z_BOW - B


def Bz(z):
    return Z_BOW - z


class P1D:
    def __init__(self, xs, ys):
        xs = np.asarray(xs, float); ys = np.asarray(ys, float)
        self.lo, self.hi = xs[0], xs[-1]
        self.p = PchipInterpolator(xs, ys, extrapolate=True)

    def __call__(self, x):
        return self.p(np.clip(x, self.lo, self.hi))


# ---------------------------------------------------------------- tables (functions of B)
# main-deck edge (= knuckle) height
KNUCKLE = P1D([0.0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 40, 45, 48, 51, 54, 57, 60, 63, 69, 72, 75,
               81, 84, 99, 110, 120, LOA],
              [7.05, 6.86, 6.74, 6.66, 6.6, 6.54, 6.5, 6.45, 6.38, 6.27, 6.17, 6.1, 6.04, 5.95, 5.76, 5.63,
               5.52, 5.45, 5.37, 5.32, 5.25, 5.2, 5.13, 5.08, 5.06, 4.98, 4.95, 4.9, 4.92, 4.92])

# knuckle half-breadth (the widest plan outline)
DECK_HB = P1D([0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
               38, 40, 42, 44, 46, 48, 50, 52, 55, 100, 110, 120, LOA],
              [0.0, 0.8, 1.31, 1.72, 2.15, 2.5, 2.78, 3.0, 3.2, 3.38, 3.6, 3.86, 4.05, 4.45, 4.83, 5.2, 5.52,
               5.84, 6.08, 6.33, 6.55, 6.76, 6.92, 7.06, 7.19, 7.31, 7.44, 7.57, 7.72, 7.82, 7.9, 7.97, 8.05,
               8.1, 8.1, 8.05, 8.07, 8.1])

WL_HB = P1D([STEM_WL_B, 9.5, 10.5, 12, 14, 16, 18, 20, 23, 26, 30, 35, 40, 45, 50, 55, 60, 100, 110, 118, 124,
             130, 133.2],
            [0.0, 0.3, 0.75, 1.3, 1.95, 2.55, 3.1, 3.6, 4.3, 4.9, 5.6, 6.3, 6.8, 7.15, 7.35, 7.45, 7.5, 7.5, 7.45,
             7.35, 7.2, 6.95, 6.8])

# centreline bottom of the main hull (the sonar dome under the forefoot is a separate body)
KEEL = P1D([STEM_WL_B, 9.4, 9.95, 10.3, 10.55, 10.75, 11.5, 13.0, 15.0, 17.5, 20.0, 110.0, 113.0, 116.0, 119.0,
            122.0, 126.0, 130.0, TRANSOM_BOT[0]],
           [0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.6, -4.05, -4.4, -DRAFT, -DRAFT, -4.45, -4.0, -3.15,
            -2.55, -1.85, -1.15, TRANSOM_BOT[1]])

# section fullness below the WL (super-ellipse exponent) and flare power between the WL and the knuckle
NEXP = P1D([STEM_WL_B, 12, 20, 30, 45, 60, 100, 112, 120, 128, TRANSOM_BOT[0]],
           [1.5, 1.6, 1.9, 2.3, 2.8, 3.2, 3.2, 3.0, 2.7, 2.4, 2.3])
FLARE = P1D([0, 15, 30, 50, 60, LOA], [1.6, 1.5, 1.3, 1.1, 1.0, 1.0])

# top of the upper hull: forecastle bulwark, UKSK block, 01 deck, hangar, the Palash wells at the hangar's
# aft corners; none over the helideck
BULWARK_TOP = P1D([0.0, 10, 25, 35, 44.5], [STEM_TOP_Y, 7.9, 7.85, 7.88, 8.0])
PALASH_WELL = (108.3, 9.9, 2.5)        # well from B 108.3 aft, floor height, inboard edge |x|
UPPER = ((0.0, 44.5, 'bulwark'), (44.5, 61.0, 8.1), (61.0, 99.0, 10.0), (99.0, PALASH_WELL[0], 12.2),
         (PALASH_WELL[0], 113.9, PALASH_WELL[1]))
HELIDECK_B = 113.9


def stem_y(B):
    """height of the raked stem line above the WL at B (B <= STEM_WL_B)."""
    return STEM_TOP_Y * (1.0 - B / STEM_WL_B)


# where the stem line crosses the knuckle: the first station of the main hull
def _stem_knuckle_B():
    lo, hi = 0.0, 3.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if stem_y(mid) > float(KNUCKLE(mid)):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


STEM_KN_B = _stem_knuckle_B()


def keel_y(B):
    if B <= STEM_WL_B:
        return stem_y(B)
    return float(KEEL(B))


def transom_B(y):
    (b0, y0), (b1, y1) = TRANSOM_BOT, TRANSOM_TOP
    t = (np.clip(y, y0, y1) - y0) / (y1 - y0)
    return b0 + t * (b1 - b0)


def top_y(B):
    """height of the upper-hull top edge at B (the knuckle itself over the helideck)."""
    for b0, b1, h in UPPER:
        if b0 <= B < b1 or (B == b1 == UPPER[-1][1]):
            return float(BULWARK_TOP(B)) if h == 'bulwark' else h
    return float(KNUCKLE(B))


def upper_hb(B, y):
    """half-breadth of the inward-leaning upper hull at height y (y >= knuckle)."""
    return float(DECK_HB(B)) - TUMBLE * (y - float(KNUCKLE(B)))


def section(B, nrows=24):
    """xs, ys of the main-hull half-section at B, from the keel (or stem) to the knuckle."""
    ylo = keel_y(B)
    ydk = float(KNUCKLE(B))
    bdk = float(DECK_HB(B))
    p = float(FLARE(B))
    if ylo >= 0.0:
        # above-water bow section: stem point -> knuckle
        t = np.linspace(0, 1, nrows)
        ys = ylo + t * (ydk - ylo)
        xs = bdk * t ** p
        return xs, ys
    bwl = float(WL_HB(B))
    n = float(NEXP(B))
    depth = -ylo
    frac = min(0.62, max(0.4, depth / (depth + ydk) + 0.05))
    nb = max(4, int(round(nrows * frac)))
    na = nrows - nb + 1
    th = np.linspace(0, math.pi / 2, nb)
    yb = ylo * np.cos(th) ** (2.0 / n)
    xb = bwl * np.sin(th) ** (2.0 / n)
    ta = np.linspace(0, 1, na)[1:]
    ya = ta * ydk
    xa = bwl + (bdk - bwl) * ta ** p
    return np.concatenate([xb, xa]), np.concatenate([yb, ya])


def stations():
    Bs = [STEM_KN_B + 0.02] + list(np.arange(1.5, 12.0, 0.75)) + list(np.arange(12.0, 30.0, 1.5)) + \
        list(np.arange(30.0, 110.0, 4.0)) + list(np.arange(110.0, TRANSOM_BOT[0], 1.5)) + [TRANSOM_BOT[0]]
    return sorted(set(round(b, 3) for b in Bs))


class Hull:
    def __init__(self, nrows=24):
        self.nrows = nrows
        self.Bs = stations()
        X, Y, Z = [], [], []
        for B in self.Bs:
            xs, ys = section(B, nrows)
            zs = np.full(nrows, zB(B))
            if B == self.Bs[-1]:
                # skew the aft-most station onto the raked transom
                bt = transom_B(ys)
                zs = zB(bt)
                hb0 = float(DECK_HB(B))
                for j in range(nrows):
                    if ys[j] > 0:
                        s = float(DECK_HB(bt[j])) / max(hb0, 1e-3)
                        xs[j] *= 1.0 + (s - 1.0) * (ys[j] / ys[-1])
            X.append(xs); Y.append(ys); Z.append(zs)
        # stations are ordered bow -> stern; store stern -> bow
        self.X = np.array(X)[::-1]; self.Y = np.array(Y)[::-1]; self.Z = np.array(Z)[::-1]

    def side_mesh(self, side=+1):
        X, Y, Z = self.X, self.Y, self.Z
        nu, nv = X.shape
        P = np.stack([X * side, Y, Z], axis=2).reshape(-1, 3)
        I = []
        for i in range(nu - 1):
            for j in range(nv - 1):
                a = i * nv + j; b = (i + 1) * nv + j; c = (i + 1) * nv + j + 1; d = i * nv + j + 1
                I += [(a, b, c), (a, c, d)]
        I = np.array(I)
        if side > 0:
            I = I[:, ::-1]
        return P, I

    def knuckle_line(self):
        return self.X[:, -1], self.Y[:, -1], self.Z[:, -1]

    def transom_outline(self):
        return self.X[0], self.Y[0], self.Z[0]

    def half_at(self, B, y):
        """hull half-breadth at station B and height y (upper hull above the knuckle)."""
        if y >= float(KNUCKLE(B)):
            return upper_hb(B, y)
        xs, ys = section(B, 48)
        if y <= ys[0]:
            return 0.0
        return float(np.interp(y, ys, xs))


CAMBER = 0.2


def deck_y(B, x=0.0):
    """main (forecastle / helideck) deck height incl. camber"""
    yd = float(KNUCKLE(B))
    hb = max(1e-3, float(DECK_HB(B)))
    cam = CAMBER * min(1.0, hb / 8.0) * max(0.0, 1.0 - (abs(x) / hb) ** 2)
    return yd + cam


# ---------------------------------------------------------------- sonar dome (Zarya-M) under the forefoot
DOME_B0, DOME_B1 = 9.8, 21.5          # nose, flat aft face
DOME_BOT = P1D([DOME_B0, 10.4, 11.5, 15.0, 21.5], [-4.7, -6.2, -6.95, -6.9, -6.65])
DOME_HW = P1D([DOME_B0, 10.3, 11.2, 12.5, 15.0, 21.5], [0.0, 1.05, 1.5, 1.75, 1.8, 1.65])
DOME_TOP = -2.4
