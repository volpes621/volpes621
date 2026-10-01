"""Slava-class hull, measured from the Kuleshov 1:100 plans (sheets 1, 2 and the lines plan).

B = metres aft of the stem head, z = Z_BOW - B, y = height above the design waterline,
+x = port. All tables are (B, value).
"""
import math
import numpy as np
from scipy.interpolate import PchipInterpolator

LOA = 186.4
Z_BOW = LOA / 2.0
Z_STERN = -LOA / 2.0

STEM_TOP_Y = 10.9      # stem head height
STEM_WL_B = 11.5       # stem meets the DWL
TRANSOM_BOT = (181.9, -1.5)   # (B, y) where the transom meets the bottom
TRANSOM_TOP = (186.4, 4.4)    # (B, y) top of the transom
STEP_B0, STEP_B1 = 161.6, 164.6   # main deck -> quarterdeck cut-down
QD_Y = 4.4             # quarterdeck height
MAIN_Y = 6.7           # main deck height amidships


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
SHEER = P1D([0.0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, STEP_B0, STEP_B0 + 0.8, STEP_B0 + 1.6,
             STEP_B0 + 2.4, STEP_B1, 166.0, 186.4],
            [STEM_TOP_Y, 10.25, 9.6, 9.1, 8.7, 8.3, 7.9, 7.55, 7.25, 7.0, 6.85, 6.75, MAIN_Y, MAIN_Y, 6.45, 5.6,
             4.75, QD_Y + 0.05, QD_Y, QD_Y])

DECK_HB = P1D([0.0, 1, 2, 4, 6, 8, 10, 12, 15, 18, 21, 24, 27, 30, 33, 36, 40, 45, 60, 150, 158, 160.4, 164.4,
               168.4, 172.4, 176.4, 179.4, 182.4, 184.4, 186.4],
              [0.05, 1.0, 1.45, 2.25, 3.0, 3.75, 4.4, 5.05, 5.95, 6.75, 7.45, 8.15, 8.7, 9.15, 9.55, 9.85,
               10.15, 10.3, 10.375, 10.375, 10.36, 10.35, 10.15, 9.9, 9.55, 9.1, 8.75, 8.3, 8.0, 7.6])

WL_HB = P1D([STEM_WL_B, 13, 16, 20, 25, 30, 37, 45, 55, 65, 75, 85, 95, 130, 140, 150, 160, 170, 178, 182.7],
            [0.0, 0.45, 1.25, 2.2, 3.4, 4.4, 5.6, 6.7, 7.8, 8.7, 9.3, 9.6, 9.75, 9.75, 9.6, 9.3, 8.8, 8.0, 7.2,
             6.6])

KEEL = P1D([STEM_WL_B, 13.5, 15.5, 17.5, 19.0, 20.5, 23, 26, 29, 32, 35, 145, 150, 155, 160, 165, 170, 175,
            179, TRANSOM_BOT[0]],
           [0.0, -2.4, -4.6, -6.7, -7.75, -8.1, -7.95, -7.4, -6.7, -6.35, -6.28, -6.28, -6.1, -5.6, -4.8, -3.8,
            -2.9, -2.2, -1.8, TRANSOM_BOT[1]])

# section fullness below the WL (super-ellipse exponent) and flare power above it
NEXP = P1D([STEM_WL_B, 20, 30, 45, 60, 80, 130, 150, 165, 175, 182], [1.6, 1.8, 2.2, 2.8, 3.3, 3.5, 3.5, 3.6, 3.9, 4.3, 4.6])
FLARE = P1D([0, 10, 20, 35, 55, 75, 186.4], [1.7, 1.75, 1.7, 1.45, 1.15, 1.0, 1.0])


def keel_y(B):
    if B <= STEM_WL_B:
        return (STEM_WL_B - B) / STEM_WL_B * STEM_TOP_Y
    return float(KEEL(B))


def transom_B(y):
    (b0, y0), (b1, y1) = TRANSOM_BOT, TRANSOM_TOP
    t = (np.clip(y, y0, y1) - y0) / (y1 - y0)
    return b0 + t * (b1 - b0)


def section(B, nrows=24):
    """returns xs, ys (keel/stem -> deck edge) of the half-section at B."""
    ylo = keel_y(B)
    ydk = float(SHEER(B))
    bdk = float(DECK_HB(B))
    p = float(FLARE(B))
    if ylo >= 0.0:
        # above-water bow section: stem point -> deck edge
        t = np.linspace(0, 1, nrows)
        ys = ylo + t * (ydk - ylo)
        xs = bdk * t ** p
        return xs, ys
    bwl = float(WL_HB(B))
    n = float(NEXP(B))
    depth = -ylo
    frac = min(0.68, max(0.42, depth / (depth + ydk) + 0.1))
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
    Bs = list(np.arange(0.35, 12.0, 0.75)) + list(np.arange(12.0, 40.0, 1.6)) + list(np.arange(40.0, 152.0, 5.0)) + \
        list(np.arange(152.0, STEP_B0, 2.4)) + list(np.arange(STEP_B0, STEP_B1 + 0.01, 0.6)) + \
        list(np.arange(STEP_B1 + 1.2, TRANSOM_BOT[0], 2.0)) + [TRANSOM_BOT[0]]
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
                # skew the aft-most station onto the raked transom; narrow its top toward the stern corners
                bt = transom_B(ys)
                zs = zB(bt)
                hb0 = float(DECK_HB(B))
                for j in range(nrows):
                    if ys[j] > 0:
                        s = float(DECK_HB(bt[j])) / max(hb0, 1e-3)
                        xs[j] *= 1.0 + (s - 1.0) * (ys[j] / ys[-1])
            X.append(xs); Y.append(ys); Z.append(zs)
        # the stations are ordered bow -> stern; store stern -> bow
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

    def deck_edge(self):
        return self.X[:, -1], self.Y[:, -1], self.Z[:, -1]

    def transom_outline(self):
        return self.X[0], self.Y[0], self.Z[0]

    def half_at(self, B, y):
        """hull half-breadth at station B and height y"""
        xs, ys = section(B, 48)
        if y <= ys[0]:
            return 0.0
        return float(np.interp(y, ys, xs))


def deck_y(B, x=0.0):
    """weather deck height incl. 0.25 m camber"""
    yd = float(SHEER(B))
    hb = max(1e-3, float(DECK_HB(B)))
    cam = 0.25 * min(1.0, hb / 10.0) * max(0.0, 1.0 - (abs(x) / hb) ** 2)
    return yd + cam
