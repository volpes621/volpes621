"""Shared helpers and equipment parts (ship coordinates: B = metres aft of the bow, x = port, y = up)."""
import math
import numpy as np

from meshkit import *
from meshkit import _quad
import atlas as st
from hull import zB, Z_BOW, deck_y, SHEER, DECK_HB

ATL = 'Slava_Atlas'
PAINT = 'Slava_Paint'
DECK = 'Slava_Deck'
ATLAS = ATL


class Ctx:
    def __init__(self, builder, layout):
        self.b = builder
        self.layout = layout
        self.paint = UVSpace(PAINT, 'tile', scale=(1 / 12.0, 1 / 12.0))
        self.deck = UVSpace(DECK, 'tile', scale=(1 / 6.0, 1 / 6.0))
        self._sw = {}

    def sw(self, name):
        if name not in self._sw:
            x0, y0, x1, y1 = st.swatch_rect(name)
            W = st.W
            self._sw[name] = UVSpace(ATLAS, 'rect', rect=(x0 / W, y0 / W, x1 / W, y1 / W))
        return self._sw[name]

    def rect(self, name, clamp=True):
        solid = name in ('helideck', 'dome', 'vls_top')
        return UVSpace(ATLAS, 'rect', rect=self.layout.uv(name), clamp=clamp, occ=solid, subdiv=solid)

    def band(self, name, repeat_m):
        return UVSpace(ATLAS, 'band', rect=st.band_uv(name), scale=(1.0 / repeat_m, 1.0), occ=False, subdiv=False)

    def add(self, uvs, geo, xf=None, node=None, occ=None):
        """occ=False keeps thin members (lattice posts, antenna tubes) out of the AO occluder set."""
        P, N, UV, I = geo
        if len(I) == 0:
            return
        self.b.add(uvs, P, N, UV, I, xf=xf, node=node, occ=occ)




def M(t=(0, 0, 0), R=None, s=1.0):
    return Xf(R=R, T=t, S=s).matrix()


def P3(B, x, y):
    return np.array([x, y, zB(B)], dtype=float)


# ------------------------------------------------------------------------------------------ geometry helpers
def orient_outward(P, N, I, center):
    P = np.asarray(P); N = np.asarray(N).copy(); I = np.asarray(I).copy()
    fn = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    cen = P[I].mean(axis=1)
    cc = np.array([center(p) for p in cen]) if callable(center) else np.asarray(center)[None, :]
    d = np.sum(fn * (cen - cc), axis=1)
    if np.sum(d < 0) > np.sum(d >= 0):
        I = I[:, ::-1]
        N = -N
    return P, N, I


def extrude_x(poly_zy, x0, x1):
    """Closed prism from a polygon in the (z, y) plane, extruded along x from x0 to x1."""
    poly = [(float(z), float(y)) for (z, y) in poly_zy]
    P, N, UV, I = prism(poly, x0, x1, top=True, bottom=True)
    Pw = np.stack([P[:, 1], P[:, 2], P[:, 0]], axis=1)
    Nw = np.stack([N[:, 1], N[:, 2], N[:, 0]], axis=1)
    # the axis permutation (x,y,z)->(y,z,x) is a rotation: winding is preserved
    UVw = np.zeros((len(Pw), 2))
    ax = np.argmax(np.abs(Nw), axis=1)
    UVw[ax == 0] = Pw[ax == 0][:, [2, 1]] * [1, -1]
    UVw[ax == 1] = Pw[ax == 1][:, [2, 0]]
    UVw[ax == 2] = Pw[ax == 2][:, [0, 1]] * [1, -1]
    return Pw, Nw, UVw, I


def poly_Bx_to_xz(poly):
    return [(float(x), float(zB(B))) for (B, x) in poly]


def sym_poly(half):
    """half outline [(B, x>=0) from front to back] -> full CCW-ish outline (port then starboard mirrored)."""
    port = [(B, x) for (B, x) in half]
    stbd = [(B, -x) for (B, x) in reversed(half)]
    pts = port + stbd
    # drop duplicates on the centreline
    out = []
    for p in pts:
        if not out or (abs(out[-1][0] - p[0]) > 1e-6 or abs(out[-1][1] - p[1]) > 1e-6):
            out.append(p)
    if abs(out[0][0] - out[-1][0]) < 1e-6 and abs(out[0][1] - out[-1][1]) < 1e-6:
        out.pop()
    return out


def chamfer_rect(B0, B1, hw, cf=0.0, ca=0.0, hw_front=None):
    """symmetric rectangle B0 (front) .. B1 (aft), half width hw, with front/aft corner chamfers."""
    hf = hw if hw_front is None else hw_front
    half = []
    if cf > 0:
        half += [(B0, max(hf - cf, 0.0)), (B0 + cf, hw)]
    else:
        half += [(B0, hf)]
    if ca > 0:
        half += [(B1 - ca, hw), (B1, hw - ca)]
    else:
        half += [(B1, hw)]
    return sym_poly(half)


def offset_poly_xz(poly, d):
    """offset a simple polygon [(x, z), ...] outward by d (miter, clamped)."""
    P = np.asarray(poly, dtype=float)
    n = len(P)
    area = 0.0
    for i in range(n):
        x0_, z0_ = P[i]; x1_, z1_ = P[(i + 1) % n]
        area += x0_ * (-z1_) - x1_ * (-z0_)
    sgn = 1.0 if area > 0 else -1.0          # CCW seen from +y in (x, -z)
    out = []
    for i in range(n):
        a_, b_, c_ = P[i - 1], P[i], P[(i + 1) % n]
        e1 = normalize(b_ - a_); e2 = normalize(c_ - b_)
        # outward normals of the two edges (in x,z with the (x,-z) orientation convention)
        n1 = np.array([e1[1], -e1[0]]) * -sgn
        n2 = np.array([e2[1], -e2[0]]) * -sgn
        m = n1 + n2
        ml = np.linalg.norm(m)
        if ml < 1e-6:
            out.append(b_ + n1 * d)
            continue
        m = m / ml
        cosh = max(0.35, float(np.dot(m, n1)))
        out.append(b_ + m * d / cosh)
    return [tuple(p) for p in out]


def bevel_poly(poly, d, mask=None):
    """chamfer the corners of a plan polygon [(B, x), ...] by d metres along both edges.
    Returns (new_poly, mask); pass the mask back in to treat a matching polygon identically."""
    pts = [np.asarray(p, float) for p in poly]
    n = len(pts)
    out, used = [], []
    for i in range(n):
        a, b_, c_ = pts[i - 1], pts[i], pts[(i + 1) % n]
        e1, e2 = b_ - a, c_ - b_
        l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
        do = (np.dot(e1, e2) / max(l1 * l2, 1e-9) < 0.985) if mask is None else mask[i]
        used.append(do)
        if not do or l1 < 1e-6 or l2 < 1e-6:
            out.append(tuple(b_))
            continue
        dd = min(d, 0.4 * l1, 0.4 * l2)
        out.append(tuple(b_ - e1 / l1 * dd))
        out.append(tuple(b_ + e2 / l2 * dd))
    return out, used


def house(ctx, poly_Bx, y0, y1, top='deck', walls='paint', top_poly_Bx=None, bottom=False, node=None, lip=0.0,
          bevel=0.0, bevel_where=None):
    """Vertical-walled (or tapered, with top_poly) deckhouse from a plan polygon in (B, x).
    lip > 0 adds a deck-edge plate overhanging the walls by `lip` metres (0.16 m thick);
    bevel > 0 chamfers the vertical corners (only those where bevel_where(B, x) is true, if given)."""
    if bevel > 0:
        mask = None
        if bevel_where is not None:
            auto = bevel_poly(poly_Bx, bevel)[1]
            mask = [a and bool(bevel_where(B, x)) for a, (B, x) in zip(auto, poly_Bx)]
        poly_Bx, mask = bevel_poly(poly_Bx, bevel, mask)
        if top_poly_Bx is not None:
            top_poly_Bx = bevel_poly(top_poly_Bx, bevel, mask)[0]
    poly = poly_Bx_to_xz(poly_Bx)
    tpoly = poly_Bx_to_xz(top_poly_Bx) if top_poly_Bx is not None else None
    wall_uv = ctx.paint if walls == 'paint' else ctx.sw(walls)
    P, N, UV, I = prism(poly, y0, y1, top=False, bottom=bottom, top_poly=tpoly)
    ctx.add(wall_uv, (P, N, UV, I), node=node)
    tp = np.asarray(tpoly if tpoly is not None else poly)
    uvs = ctx.deck if top == 'deck' else (ctx.paint if top == 'paint' else (ctx.sw(top) if top else None))
    if lip > 0:
        lp = offset_poly_xz(tp, lip)
        P, N, UV, I = prism(lp, y1 - 0.16, y1 + 0.02, top=False, bottom=True)
        ctx.add(ctx.paint, (P, N, UV, I), node=node)
        if top:
            P, N, UV, I = cap_polygon(np.asarray(lp), y1 + 0.02, up=True)
            ctx.add(uvs, (P, N, np.stack([P[:, 2], P[:, 0]], axis=1), I), node=node)
        return
    if top:
        P, N, UV, I = cap_polygon(tp, y1, up=True)
        UV = np.stack([P[:, 2], P[:, 0]], axis=1)
        ctx.add(uvs, (P, N, UV, I), node=node)


def slab(ctx, poly_Bx, y0, y1, top='deck', side='paint', node=None):
    house(ctx, poly_Bx, y0, y1, top=top, walls=side, bottom=True, node=node)


def railing_pts(ctx, pts, node=None, closed=False, band='RAIL', repeat=st.RAIL_REPEAT_M, band_h=1.2, white=False):
    pts = [np.asarray(p, dtype=float) for p in pts]
    if closed:
        pts = pts + [pts[0]]
    acc = 0.0
    parts = []
    for a, b in zip(pts[:-1], pts[1:]):
        L = np.linalg.norm(b - a)
        if L < 1e-3:
            continue
        up = np.array([0, band_h, 0])
        d = normalize(b - a)
        n = normalize(np.cross(d, np.array([0, 1.0, 0])))
        P, N, UV, I = _quad(a, b, b + up, a + up, uv=[(acc, 1), (acc + L, 1), (acc + L, 0), (acc, 0)], n=n)
        parts.append((P, N, UV, I))
        acc += L
    if parts:
        ctx.add(ctx.band(band, repeat), merge(parts), node=node)


def rail_poly(ctx, poly_Bx, y, inset=0.12, sides=None, node=None, closed=True):
    """railing around a plan polygon at height y (inset toward the inside)."""
    pts = np.array([(x, zB(B)) for (B, x) in poly_Bx], dtype=float)
    c = pts.mean(axis=0)
    ins = []
    for p in pts:
        d = c - p
        n = np.linalg.norm(d)
        ins.append(p + d / max(n, 1e-6) * inset * 1.2)
    P3s = [np.array([q[0], y, q[1]]) for q in ins]
    if sides is None:
        railing_pts(ctx, P3s, node=node, closed=closed)
    else:
        k = len(P3s)
        for i in sides:
            railing_pts(ctx, [P3s[i % k], P3s[(i + 1) % k]], node=node)


def strip(ctx, band, repeat, a, b, height, normal, offset=0.03, node=None):
    n = normalize(np.asarray(normal, dtype=float))
    a = np.asarray(a, dtype=float) + n * offset
    b = np.asarray(b, dtype=float) + n * offset
    up = np.array([0, height, 0.0])
    L = np.linalg.norm(b - a)
    P, N, UV, I = _quad(a, b, b + up, a + up, uv=[(0.0, 1.0), (L, 1.0), (L, 0.0), (0.0, 0.0)], n=n)
    if np.dot(np.cross(P[1] - P[0], P[2] - P[0]), n) < 0:
        I = I[:, ::-1]
    ctx.add(ctx.band(band, repeat), (P, N, UV, I), node=node)


def side_strip(ctx, band, B0, B1, x, y0, h, repeat=None, node=None, sides=(1, -1)):
    rep = repeat or {'WINDOW': st.WIN_REPEAT_M, 'PORTS': st.PORTS_REPEAT_M, 'LOUVER': st.LOUVER_REPEAT_M}[band]
    for s in sides:
        strip(ctx, band, rep, P3(B0, s * x, y0), P3(B1, s * x, y0), h, (s, 0, 0), node=node)


def end_strip(ctx, band, B, x0, x1, y0, h, facing=1, repeat=None, node=None):
    rep = repeat or {'WINDOW': st.WIN_REPEAT_M, 'PORTS': st.PORTS_REPEAT_M, 'LOUVER': st.LOUVER_REPEAT_M}[band]
    strip(ctx, band, rep, P3(B, x1, y0), P3(B, x0, y0), h, (0, 0, facing), node=node)


def decal(ctx, name, center, normal, w, h, offset=0.03, up=(0, 1, 0), node=None, flip=False):
    n = normalize(np.asarray(normal, dtype=float))
    upv = np.asarray(up, dtype=float)
    right = normalize(np.cross(upv, n))
    upv = np.cross(n, right)
    c = np.asarray(center, dtype=float) + n * offset
    hw, hh = w / 2, h / 2
    p0 = c - right * hw - upv * hh; p1 = c + right * hw - upv * hh
    p2 = c + right * hw + upv * hh; p3 = c - right * hw + upv * hh
    uv = [(0, 1), (1, 1), (1, 0), (0, 0)] if not flip else [(1, 1), (0, 1), (0, 0), (1, 0)]
    P, N, UV, I = _quad(p0, p1, p2, p3, uv=uv, n=n)
    ctx.add(ctx.rect(name), (P, N, UV, I), node=node)


def door(ctx, B, x, y0, normal, w=0.8, h=1.75, node=None):
    decal(ctx, 'door', (x, y0 + h / 2, zB(B)), normal, w, h, node=node)


def ladder(ctx, bottom, top, width=0.5, normal=(0, 0, 1), node=None):
    bottom = np.asarray(bottom, float); top = np.asarray(top, float)
    n = normalize(np.asarray(normal, float))
    d = top - bottom
    L = np.linalg.norm(d)
    side = normalize(np.cross(d, n)) * width / 2
    o = n * 0.06
    P, N, UV, I = _quad(bottom - side + o, bottom + side + o, top + side + o, top - side + o,
                        uv=[(0, 1), (0, 0), (L, 0), (L, 1)], n=n)
    ctx.add(ctx.band('LADDER', st.LADDER_REPEAT_M), (P, N, UV, I), node=node)


def whip(ctx, base, height, r=0.05, tilt=None, node=None):
    base = np.asarray(base, float)
    top = base + (normalize(np.asarray(tilt, float)) * height if tilt is not None else np.array([0, height, 0]))
    ctx.add(ctx.sw('dark'), tube_path([base, top], r, seg=4), node=node)


def lattice_quad(ctx, corners, rep=None, node=None, double=True):
    """alpha-tested mesh panel; split into rows <= LAT_BAND_M so the texture maps isotropically."""
    p = [np.asarray(q, float) for q in corners]
    h = 0.5 * (np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1]))
    nrows = max(1, int(math.ceil(h / st.LAT_BAND_M - 1e-6)))
    parts = []
    for r in range(nrows):
        t0, t1 = r / nrows, (r + 1) / nrows
        a0 = p[0] + (p[3] - p[0]) * t0; b0 = p[1] + (p[2] - p[1]) * t0
        a1 = p[0] + (p[3] - p[0]) * t1; b1 = p[1] + (p[2] - p[1]) * t1
        rowh = h / nrows
        w = 0.5 * (np.linalg.norm(b0 - a0) + np.linalg.norm(b1 - a1))
        vb = min(1.0, rowh / st.LAT_BAND_M)
        parts.append(_quad(a0, b0, b1, a1, uv=[(0, vb), (w, vb), (w, 0), (0, 0)]))
    ctx.add(ctx.band('LATTICE', st.LATTICE_REPEAT_M), merge(parts), node=node)


def truss(ctx, a, b, w, h, up=(0, 1, 0), rep=6.0, node=None):
    """open box girder from a to b (4 lattice faces)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = normalize(b - a)
    upv = normalize(np.asarray(up, float))
    side = normalize(np.cross(d, upv))
    upv = np.cross(side, d)
    hw, hh = w / 2, h / 2
    c = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    pa = [a + side * u + upv * v for (u, v) in c]
    pb = [b + side * u + upv * v for (u, v) in c]
    for i in range(4):
        j = (i + 1) % 4
        lattice_quad(ctx, [pa[i], pb[i], pb[j], pa[j]], rep=rep, node=node)
    for i in range(4):
        ctx.add(ctx.paint, beam(pa[i], pb[i], 0.07, 0.07), node=node, occ=False)


def lattice_tower(ctx, base, top_hw, bot_hw, height, rep=4.0, node=None):
    """4-sided tapered lattice tower centred at base (x,y,z)."""
    base = np.asarray(base, float)
    bc = [(-bot_hw, -bot_hw), (bot_hw, -bot_hw), (bot_hw, bot_hw), (-bot_hw, bot_hw)]
    tc = [(-top_hw, -top_hw), (top_hw, -top_hw), (top_hw, top_hw), (-top_hw, top_hw)]
    pb = [base + np.array([u, 0, v]) for (u, v) in bc]
    pt = [base + np.array([u, height, v]) for (u, v) in tc]
    for i in range(4):
        j = (i + 1) % 4
        lattice_quad(ctx, [pb[i], pb[j], pt[j], pt[i]], rep=rep, node=node)
        ctx.add(ctx.paint, beam(pb[i], pt[i], 0.08, 0.08), node=node, occ=False)


def sphere_at(ctx, sw, c, r, seg=12, rings=6, node=None):
    ctx.add(ctx.sw(sw), sphere(r, seg=seg, rings=rings), xf=M(np.asarray(c, float)), node=node)


# ------------------------------------------------------------------------------------------ equipment


def rbu6000(ctx, pos, facing=0.0, name='RBU6000', parent=None):
    b = ctx.b
    o = np.asarray(pos, float)
    b.push(name, parent=parent, translation=tuple(o))
    R = rot_y(facing)
    ctx.add(ctx.paint, cylinder(0.85, 0.55, seg=14), xf=M(o, R))
    ctx.add(ctx.paint, box(0.8, 1.1, 0.9, center=(0, 1.0, -0.15)), xf=M(o, R))
    el = rot_x(-math.radians(25))
    for k in range(12):
        a = math.radians(-150 + k * (300 / 11.0))
        cx, cy = 0.62 * math.sin(a), 0.62 * math.cos(a)
        Rt = R @ el @ rot_x(math.pi / 2)
        ctx.add(ctx.sw('dark'), cylinder(0.12, 1.75, seg=6), xf=M(o + R @ (el @ np.array([cx, cy, -0.75]) + np.array([0, 1.3, 0])), Rt))
    b.pop()


def raft_rack(ctx, base, n=4, along=(0, 0, -1), node=None, stack=1, spacing=0.8):
    base = np.asarray(base, float)
    d = normalize(np.asarray(along, float))
    for s in range(stack):
        for k in range(n):
            c = base + d * (k * spacing) + np.array([0, 0.33 + s * 0.62, 0])
            cap = lathe([(0.0, -0.66), (0.24, -0.64), (0.3, -0.52), (0.3, 0.52), (0.24, 0.64), (0.0, 0.66)], seg=8)
            Rr = frame_from_dir(np.cross(d, [0, 1, 0])) @ rot_x(math.pi / 2)
            ctx.add(ctx.sw('white'), cap, xf=M(c, Rr), node=node)


def bollard(ctx, pos, along=(0, 0, 1), node=None):
    o = np.asarray(pos, float)
    d = normalize(np.asarray(along, float))
    for t in (-0.35, 0.35):
        ctx.add(ctx.sw('dark'), cylinder(0.16, 0.5, seg=8), xf=M(o + d * t), node=node)
    ctx.add(ctx.sw('dark'), beam(o - d * 0.55 + np.array([0, 0.05, 0]), o + d * 0.55 + np.array([0, 0.05, 0]), 0.5, 0.1), node=node)


def vent(ctx, pos, r=0.3, h=0.8, node=None, mushroom=True):
    o = np.asarray(pos, float)
    if mushroom:
        ctx.add(ctx.paint, cylinder(r * 0.6, h, seg=8), xf=M(o), node=node)
        ctx.add(ctx.paint, cylinder(r, 0.18, seg=10, r_top=r * 0.5), xf=M(o + np.array([0, h, 0])), node=node)
    else:
        ctx.add(ctx.paint, box(2 * r, h, 2 * r, center=(0, h / 2, 0)), xf=M(o), node=node)


def propeller(ctx, pos, r=2.1, blades=4, hand=1, node=None):
    o = np.asarray(pos, float)
    hub = lathe([(0.0, -1.0), (0.45, -0.8), (0.52, 0.3), (0.32, 0.9), (0.0, 1.1)], seg=10)
    ctx.add(ctx.sw('bronze'), hub, xf=M(o, rot_x(math.pi / 2)), node=node)
    for k in range(blades):
        a = 2 * math.pi * k / blades
        secs = []
        for i in range(6):
            t = i / 5.0
            rr = 0.45 + t * (r - 0.45)
            chord = 1.25 * math.sin(math.pi * (0.22 + 0.62 * t)) + 0.3
            pitch = math.radians(58 - 28 * t) * hand
            skew = 0.3 * t * t
            pts = []
            for cc in (-0.5, 0.5):
                xx = cc * chord * math.cos(pitch)
                zz = cc * chord * math.sin(pitch)
                ang = a + skew + xx / max(rr, 0.1)
                pts.append((rr * math.cos(ang), rr * math.sin(ang), zz))
            secs.append(pts)
        P = np.array([p for s in secs for p in s])
        I = []
        for i in range(5):
            a0 = 2 * i
            I += [(a0, a0 + 2, a0 + 3), (a0, a0 + 3, a0 + 1)]
        I = np.array(I)
        N = compute_smooth_normals(P, I)
        ctx.add(ctx.sw('bronze'), (P, N, P[:, :2] * 0.3, I), xf=M(o), node=node)


def local_prism(ctx, uvs, poly_tu, depth, origin, T, U, N, node=None):
    """convex polygon [(t, u), ...] in the plane spanned by T/U at `origin`, extruded along N by `depth`."""
    T = normalize(np.asarray(T, float)); U = normalize(np.asarray(U, float)); N = normalize(np.asarray(N, float))
    o = np.asarray(origin, float)
    base = [o + T * t + U * u for (t, u) in poly_tu]
    top = [p + N * depth for p in base]
    k = len(base)
    polys = [top, base[::-1]]
    for i in range(k):
        j = (i + 1) % k
        polys.append([base[i], base[j], top[j], top[i]])
    P, Nn, UV, I = flat_poly_faces(polys)
    cen = o + N * depth / 2 + T * np.mean([p[0] for p in poly_tu]) + U * np.mean([p[1] for p in poly_tu])
    P, Nn, I = orient_outward(P, Nn, I, cen)
    # orient_outward flips all faces together; make each face individually outward
    fn = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    fc = P[I].mean(axis=1)
    bad = np.sum(fn * (fc - cen), axis=1) < 0
    I[bad] = I[bad][:, ::-1]
    ctx.add(uvs, (P, Nn, UV, I), node=node)


def mushroom_vent(ctx, pos, r=0.5, node=None):
    o = np.asarray(pos, float)
    ctx.add(ctx.sw('deck_red'), cylinder(r * 0.62, 0.38, seg=10), xf=M(o), node=node)
    ctx.add(ctx.sw('light'), lathe([(0.0, 0.62), (r * 0.55, 0.6), (r * 0.9, 0.48), (r, 0.3), (r, 0.22)], seg=12),
            xf=M(o), node=node)
