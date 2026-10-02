"""Deck and hull fittings of Project 22350 (forecastle ground tackle, breakwater, bulwark fittings,
helideck nets, stern cab). Positions come from the 1:500 general-arrangement plan; shapes follow the
2018 aerial photograph of Admiral Gorshkov's forecastle, the Admiral Golovko fitting-out photograph (2021)
and the 2021 photograph of Admiral Kasatonov."""
import math
import numpy as np

from meshkit import *
from meshkit import _quad
import atlas as st
from kit import M, P3, railing_pts, house, end_strip
from parts import rbox, disc

PAL = st.PAL


def solid(c, uvs, polys, node=None):
    """closed convex solid from its faces; each face is turned away from the solid's centroid."""
    P, N, UV, I = flat_poly_faces([[np.asarray(p, float) for p in f] for f in polys])
    cen = np.mean([np.mean(f, axis=0) for f in polys], axis=0)
    fn = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    bad = np.sum(fn * (P[I].mean(axis=1) - cen), axis=1) < 0
    I[bad] = I[bad][:, ::-1]
    N = np.zeros_like(P)
    for t in I:
        N[t] = normalize(np.cross(P[t[1]] - P[t[0]], P[t[2]] - P[t[0]]))
    c.add(uvs, (P, N, UV, I), node=node)


def prism_between(c, uvs, base, top, node=None):
    """solid between two same-size convex outlines (lists of 3D points)."""
    polys = [list(base)[::-1], list(top)]
    n = len(base)
    for i in range(n):
        j = (i + 1) % n
        polys.append([base[i], base[j], top[j], top[i]])
    solid(c, uvs, polys, node=node)


def _yaw(d):
    d = np.asarray(d, float)
    return rot_y(math.atan2(d[0], d[2]))


# ------------------------------------------------------------------------------------------ ground tackle
def capstan(c, pos):
    """vertical-axis anchor capstan: bed plate, chain wheel with whelps under a dark housing, waisted
    bronze warping head with ribs and a flat cap (2018 forecastle photograph)."""
    o = np.asarray(pos, float)
    dk = c.sw('dark')
    c.add(dk, lathe([(0.66, 0.0), (0.66, 0.07), (0.58, 0.1), (0.5, 0.12)], seg=20), xf=M(o))
    c.add(dk, lathe([(0.5, 0.12), (0.46, 0.3), (0.55, 0.32), (0.55, 0.5), (0.43, 0.54)], seg=20), xf=M(o))
    for k in range(8):                                    # whelps of the chain wheel
        a = 2 * math.pi * k / 8
        c.add(dk, box(0.07, 0.14, 0.12, center=(0, 0, 0)),
              xf=M(o + np.array([0.56 * math.cos(a), 0.41, 0.56 * math.sin(a)]), rot_y(-a)))
    br = c.sw('bronze')
    c.add(br, lathe([(0.43, 0.54), (0.41, 0.58), (0.35, 0.62), (0.33, 0.8), (0.35, 0.98), (0.41, 1.02),
                     (0.41, 1.08), (0.3, 1.1), (0.0, 1.11)], seg=20), xf=M(o))
    for k in range(10):                                   # ribs of the warping head
        a = 2 * math.pi * k / 10
        c.add(br, box(0.05, 0.34, 0.05, center=(0, 0, 0)),
              xf=M(o + np.array([0.34 * math.cos(a), 0.8, 0.34 * math.sin(a)]), rot_y(-a)))


def chain(c, pts, link=0.17, w=0.075):
    """anchor cable as alternating flat links along a polyline."""
    pts = [np.asarray(p, float) for p in pts]
    parts = []
    for a, b in zip(pts[:-1], pts[1:]):
        L = np.linalg.norm(b - a)
        d = (b - a) / L
        n = max(1, int(L / link))
        side = normalize(np.cross(d, [0, 1.0, 0]))
        up = np.cross(side, d)
        for k in range(n):
            ctr = a + d * (k + 0.5) * L / n
            u = up if k % 2 == 0 else side
            v = side if k % 2 == 0 else up
            q = [ctr - d * link * 0.6 - v * w, ctr + d * link * 0.6 - v * w, ctr + d * link * 0.6 + v * w,
                 ctr - d * link * 0.6 + v * w]
            parts.append(_quad(q[0], q[1], q[2], q[3], n=u))
            parts.append(_quad(q[3], q[2], q[1], q[0], n=-u))
    c.add(c.sw('black'), merge(parts), occ=False)


def chain_stopper(c, pos, along):
    """bar-type stopper: saddle with two cheeks and a hinged bar across the cable (plan: '+' shape)."""
    o = np.asarray(pos, float)
    R = _yaw(along)
    dk = c.sw('dark')
    c.add(dk, rbox(0.6, 0.1, 0.9, r=0.06, seg=1), xf=M(o, R))
    for sx in (-1, 1):
        c.add(dk, rbox(0.1, 0.3, 0.7, r=0.04, seg=1, bevel=0.03, y0=0.08), xf=M(o + R @ np.array([sx * 0.22, 0, 0]), R))
    c.add(c.sw('mid'), tube_path([o + R @ np.array([-0.5, 0.36, 0.05]), o + R @ np.array([0.5, 0.36, 0.05])], 0.05, seg=6))
    c.add(c.sw('mid'), tube_path([o + R @ np.array([0.5, 0.36, 0.05]), o + R @ np.array([0.62, 0.62, 0.3])], 0.035, seg=5))


def compressor(c, pos, along, out):
    """chain compressor aft of the hawse pipe: a claw on the cable held by two tie rods diverging aft to
    pad eyes on the deck (the '<' marks on the plan, the dark forked bars in the photograph)."""
    o = np.asarray(pos, float)
    d = normalize(np.asarray(along, float))
    n = normalize(np.asarray(out, float))
    dk = c.sw('dark')
    c.add(dk, rbox(0.34, 0.08, 0.5, r=0.05, seg=1), xf=M(o, _yaw(d)))
    a = o + np.array([0, 0.24, 0])
    c.add(dk, rbox(0.3, 0.16, 0.4, r=0.05, seg=1, y0=0.1), xf=M(o, _yaw(d)))
    for sx in (-1, 1):
        e = o + d * 1.05 + n * sx * 0.8
        c.add(dk, tube_path([a, e + np.array([0, 0.12, 0])], 0.035, seg=6))
        c.add(dk, cylinder(0.1, 0.12, seg=8), xf=M(e))


def pipe_mouth(c, pos, r=0.32):
    """deck end of a hawse or chain pipe: raised ring with a dark throat."""
    o = np.asarray(pos, float)
    c.add(c.sw('dark'), lathe([(r * 0.7, -0.02), (r * 0.7, 0.1), (r, 0.12), (r * 1.15, 0.08), (r * 1.15, 0.0)], seg=16),
          xf=M(o))
    c.add(c.sw('black'), disc(r * 0.7, seg=16), xf=M(o + np.array([0, 0.03, 0])))


def drive_housing(c, pos, along):
    """rounded cover over the capstan drive beside each capstan (light drums in the photograph)."""
    o = np.asarray(pos, float)
    R = _yaw(along)
    c.add(c.paint, rbox(0.62, 0.12, 0.72, r=0.06, seg=1), xf=M(o, R))
    c.add(c.paint, cylinder(0.29, 0.62, seg=14, caps=(True, True), y0=-0.31),
          xf=M(o + np.array([0, 0.4, 0]), R @ rot_x(math.pi / 2)))


def double_bollard(c, pos, along, gap=0.6):
    o = np.asarray(pos, float)
    d = normalize(np.asarray(along, float))
    dk = c.sw('dark')
    c.add(dk, rbox(0.46, 0.07, gap + 0.6, r=0.1, seg=2), xf=M(o, _yaw(d)))
    for t in (-gap / 2, gap / 2):
        c.add(dk, lathe([(0.15, 0.0), (0.15, 0.36), (0.2, 0.39), (0.2, 0.45), (0.0, 0.47)], seg=12),
              xf=M(o + d * t + np.array([0, 0.05, 0])))


def vent_head(c, pos, r=0.4, h=0.6):
    """cylindrical ventilation head with a rounded crown."""
    o = np.asarray(pos, float)
    c.add(c.paint, lathe([(r * 1.08, 0.0), (r * 1.08, 0.05), (r, 0.07), (r, h - r * 0.25), (r * 0.85, h - r * 0.06),
                          (r * 0.5, h), (0.0, h + 0.01)], seg=18), xf=M(o))


def locker(c, pos, w, h, d, yaw=0.0, uvs=None):
    o = np.asarray(pos, float)
    c.add(uvs or c.paint, rbox(w, h, d, r=0.05, seg=1, bevel=0.025), xf=M(o, rot_y(yaw)))


# ------------------------------------------------------------------------------------------ breakwater
def breakwater(c, arm, h, h_end, deck_fn, t=0.08, lean=0.1):
    """V-shaped breakwater, apex forward. `arm` is the plan polyline of one arm, (B, x>=0) from the apex
    outward; the other arm is its mirror. The plate leans forward, carries a top flange and triangular
    stiffeners on its aft face, and its outer end is cut down to `h_end` (plan, 2018 photograph)."""
    for s in (1, -1):
        pts = [np.array([s * x, 0.0, P3(B, 0, 0)[2]]) for (B, x) in arm]
        base = [p + np.array([0, deck_fn(B, x), 0]) for p, (B, x) in zip(pts, arm)]
        n = len(arm)
        hs = [h] * (n - 1) + [h_end]
        # forward normal at each point (mitred between segments)
        fw = []
        for i in range(n):
            d0 = base[min(i + 1, n - 1)] - base[max(i - 1, 0)]
            d0 = normalize(d0 * [1, 0, 1])
            nf = np.array([d0[2], 0, -d0[0]])
            if nf[2] < 0:
                nf = -nf
            fw.append(nf)
        fw[0] = np.array([0.0, 0.0, 1.0])
        front_b = base
        front_t = [b + np.array([0, hh, 0]) + f * lean * hh / h for b, hh, f in zip(base, hs, fw)]
        back_b = [p - f * t for p, f in zip(front_b, fw)]
        back_t = [p - f * t for p, f in zip(front_t, fw)]
        for i in range(n - 1):
            j = i + 1
            prism_between(c, c.paint, [front_b[i], front_b[j], back_b[j], back_b[i]],
                          [front_t[i], front_t[j], back_t[j], back_t[i]])
            # top flange toward aft
            ft = [front_t[i] - fw[i] * t, front_t[j] - fw[j] * t, front_t[j] - fw[j] * 0.17, front_t[i] - fw[i] * 0.17]
            prism_between(c, c.paint, [p - np.array([0, 0.035, 0]) for p in ft], ft)
        # stiffeners on the aft face, roughly every 0.8 m along the arm
        L = [0.0]
        for i in range(n - 1):
            L.append(L[-1] + np.linalg.norm(base[i + 1] - base[i]))
        for sl in np.arange(0.5, L[-1] - 0.4, 0.8):
            i = max(k for k in range(n - 1) if L[k] <= sl)
            f = (sl - L[i]) / (L[i + 1] - L[i])
            p = back_b[i] + (back_b[i + 1] - back_b[i]) * f
            hh = (hs[i] + (hs[i + 1] - hs[i]) * f) * 0.88
            nf = normalize(fw[i] + (fw[i + 1] - fw[i]) * f)
            d = normalize((base[i + 1] - base[i]) * [1, 0, 1])
            tri = [p, p + np.array([0, hh, 0]), p - nf * 0.42]
            q = [x + d * 0.025 for x in tri]
            r = [x - d * 0.025 for x in tri]
            solid(c, c.paint, [q, r[::-1], [q[0], q[1], r[1], r[0]], [q[1], q[2], r[2], r[1]], [q[2], q[0], r[0], r[2]]])


# ------------------------------------------------------------------------------------------ bulwark fittings
def upright_canister(c, pos, outward):
    """white canister standing on a bracket against the bulwark: cylinder with a domed top and two dark
    straps (2018 forecastle photograph; pairs on the plan)."""
    o = np.asarray(pos, float)
    n = normalize(np.asarray(outward, float))
    c.add(c.paint, box(0.5, 0.06, 0.45, center=(0, 0, 0)), xf=M(o + np.array([0, -0.03, 0]), _yaw(n)))
    c.add(c.paint, box(0.08, 0.6, 0.08, center=(0, 0.3, 0)), xf=M(o + n * 0.3, _yaw(n)))
    c.add(c.sw('white'), lathe([(0.0, 0.0), (0.27, 0.0), (0.28, 0.05), (0.28, 0.78), (0.23, 0.89), (0.12, 0.94),
                                (0.0, 0.95)], seg=14), xf=M(o))
    for y in (0.22, 0.62):
        c.add(c.sw('dark'), cylinder(0.288, 0.04, seg=14, caps=(False, False), y0=y), xf=M(o))


def fire_box(c, pos, facing, w=0.5, h=0.7):
    """red fire-hose box hung on a wall at `pos`, its door facing `facing`."""
    o = np.asarray(pos, float)
    n = normalize(np.asarray(facing, float))
    R = _yaw(n)
    c.add(c.sw('flagred'), rbox(w, h, 0.25, r=0.04, seg=1, y0=-h / 2), xf=M(o + n * 0.13, R))
    c.add(c.sw('white'), box(w * 0.6, 0.05, 0.01, center=(0, 0, 0)), xf=M(o + n * 0.257 + np.array([0, h * 0.25, 0]), R))


def life_ring(c, pos, outward):
    o = np.asarray(pos, float)
    n = normalize(np.asarray(outward, float))
    R = frame_from_dir(n) @ rot_x(math.pi / 2)
    ring = lathe([(0.25, -0.05), (0.36, -0.05), (0.36, 0.05), (0.25, 0.05)], seg=16)
    c.add(c.sw('orange'), ring, xf=M(o, R))
    for k in range(4):                                     # white bands
        a = math.pi / 4 + k * math.pi / 2
        c.add(c.sw('white'), box(0.13, 0.115, 0.115, center=(0, 0, 0)),
              xf=M(o + R @ np.array([0.305 * math.cos(a), 0.0, 0.305 * math.sin(a)]), R @ rot_y(-a)))


# ------------------------------------------------------------------------------------------ helideck
def safety_net(c, pts, h=0.95):
    """helideck safety net raised as a fence: net panels between posts along a polyline."""
    railing_pts(c, pts, band='NET', repeat=st.NET_REPEAT_M, band_h=h)


def cab(c, B0, B1, x0, x1, y0, y1):
    """small deckhouse with a window band on its aft face (helicopter control cab)."""
    poly = [(B0, x1), (B1, x1), (B1, x0), (B0, x0)]
    house(c, poly, y0, y1, top='paint')
    hw = 0.62
    end_strip(c, 'WINDOW', B1 + 0.005, min(x0, x1) + 0.2, max(x0, x1) - 0.2, y1 - 0.25 - hw, hw, facing=-1)
