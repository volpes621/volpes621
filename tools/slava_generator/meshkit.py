"""
meshkit: small procedural hard-surface modelling kit + glTF/GLB writer.

Conventions (match the DDG-51 reference after its node transforms):
  +Y up, +Z = bow (forward), +X = port, units = metres, waterline at y = 0.

Every primitive builds triangles with explicit normals and UVs. UVs are
expressed in "UV spaces" (see Atlas below) so one call can target either a
tiling material or a sub-rectangle of a texture atlas.
"""
import math
import struct
import json
import io
from collections import OrderedDict, defaultdict

import numpy as np

# ----------------------------------------------------------------------------
# small math helpers
# ----------------------------------------------------------------------------

def v3(*a):
    return np.array(a, dtype=np.float64)


def normalize(v, axis=-1):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    n[n < 1e-12] = 1.0
    return v / n


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def rot_axis(axis, a):
    axis = normalize(np.asarray(axis, dtype=np.float64))
    x, y, z = axis
    c, s = math.cos(a), math.sin(a)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]], dtype=np.float64)


def frame_from_dir(d, up_hint=(0, 1, 0)):
    """Rotation whose local +Z axis points along d (local +Y as close to up_hint as possible)."""
    d = normalize(np.asarray(d, dtype=np.float64))
    up = np.asarray(up_hint, dtype=np.float64)
    if abs(np.dot(d, normalize(up))) > 0.98:
        up = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 0, 1.0])
    x = normalize(np.cross(up, d))
    y = np.cross(d, x)
    return np.stack([x, y, d], axis=1)


class Xf:
    """Affine transform: p' = R @ (S * p) + T"""

    def __init__(self, R=None, T=(0, 0, 0), S=(1, 1, 1)):
        self.R = np.eye(3) if R is None else np.asarray(R, dtype=np.float64)
        self.T = np.asarray(T, dtype=np.float64)
        self.S = np.asarray(S if np.ndim(S) else (S, S, S), dtype=np.float64)

    def matrix(self):
        M = np.eye(4)
        M[:3, :3] = self.R @ np.diag(self.S)
        M[:3, 3] = self.T
        return M

    def __matmul__(self, other):
        M = self.matrix() @ other.matrix()
        return XfM(M)


class XfM(Xf):
    def __init__(self, M):
        self.M = np.asarray(M, dtype=np.float64)

    def matrix(self):
        return self.M


def apply_xf(M, P, N):
    A = M[:3, :3]
    P2 = P @ A.T + M[:3, 3]
    Ninv = np.linalg.inv(A).T
    N2 = normalize(N @ Ninv.T)
    if np.linalg.det(A) < 0:
        flip = True
    else:
        flip = False
    return P2, N2, flip


def as_matrix(xf):
    if xf is None:
        return np.eye(4)
    if isinstance(xf, np.ndarray):
        return xf
    return xf.matrix()


def T(x, y, z):
    return Xf(T=(x, y, z))


def TR(t, R):
    return Xf(R=R, T=t)

# ----------------------------------------------------------------------------
# UV spaces
# ----------------------------------------------------------------------------


class UVSpace:
    """Maps "local" uv (in metres or 0..1) into final texture coordinates.

    kind='tile'  : uv * scale (material texture repeats)
    kind='rect'  : uv in 0..1 mapped into atlas rect (u0,v0,u1,v1); clamped
    kind='band'  : u repeats freely (full-width band in atlas), v mapped to band
    Texture-space v follows glTF convention (0 = top of image).
    """

    def __init__(self, material, kind='tile', rect=(0, 0, 1, 1), scale=(1.0, 1.0), clamp=True, occ=True, subdiv=True):
        self.material = material
        self.kind = kind
        self.rect = rect
        self.scale = scale
        self.clamp = clamp
        self.occ = occ          # casts ambient occlusion onto other geometry
        self.subdiv = subdiv    # may be subdivided for per-vertex AO resolution

    def map(self, uv):
        uv = np.asarray(uv, dtype=np.float64).copy()
        if self.kind == 'tile':
            uv[:, 0] *= self.scale[0]
            uv[:, 1] *= self.scale[1]
            return uv
        u0, v0, u1, v1 = self.rect
        if self.kind == 'rect':
            if self.clamp:
                uv = np.clip(uv, 0.0, 1.0)
            out = np.empty_like(uv)
            out[:, 0] = u0 + uv[:, 0] * (u1 - u0)
            out[:, 1] = v0 + uv[:, 1] * (v1 - v0)
            return out
        if self.kind == 'band':
            out = np.empty_like(uv)
            out[:, 0] = uv[:, 0] * self.scale[0]
            vv = np.clip(uv[:, 1], 0.0, 1.0) if self.clamp else uv[:, 1]
            out[:, 1] = v0 + vv * (v1 - v0)
            return out
        raise ValueError(self.kind)

# ----------------------------------------------------------------------------
# Geometry container
# ----------------------------------------------------------------------------


class Geo:
    """Triangle soup with per-vertex attributes for ONE uv-space/material."""

    __slots__ = ('P', 'N', 'UV', 'I', 'occ', 'subdiv')

    def __init__(self, P, N, UV, I, occ=True, subdiv=True):
        self.P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        self.N = np.asarray(N, dtype=np.float64).reshape(-1, 3)
        self.UV = np.asarray(UV, dtype=np.float64).reshape(-1, 2)
        self.I = np.asarray(I, dtype=np.int64).reshape(-1, 3)
        self.occ = occ
        self.subdiv = subdiv


class Builder:
    """Collects geometry per node and per material."""

    def __init__(self):
        # node name -> dict(material name -> list of Geo)
        self.nodes = OrderedDict()
        self.node_info = OrderedDict()  # name -> dict(parent, translation)
        self.current = None
        self.stack = []

    def node(self, name, parent=None, translation=(0, 0, 0)):
        """translation = WORLD position of the node pivot (converted to parent-relative)."""
        if name not in self.nodes:
            poff = self.world_offset(parent) if parent is not None else np.zeros(3)
            rel = np.asarray(translation, dtype=float) - poff
            self.nodes[name] = defaultdict(list)
            self.node_info[name] = {'parent': parent, 'translation': tuple(float(t) for t in rel)}
        self.current = name
        return self

    def push(self, name, parent=None, translation=(0, 0, 0)):
        self.stack.append(self.current)
        self.node(name, parent, translation)

    def pop(self):
        self.current = self.stack.pop()

    def world_offset(self, name):
        """Accumulated translation of node `name` (nodes only carry translations)."""
        off = np.zeros(3)
        while name is not None:
            info = self.node_info[name]
            off += np.asarray(info['translation'])
            name = info['parent']
        return off

    def add(self, uvspace, P, N, UV, I, xf=None, node=None, occ=None, subdiv=None):
        node = node or self.current
        occ = uvspace.occ if occ is None else occ
        subdiv = uvspace.subdiv if subdiv is None else subdiv
        M = as_matrix(xf)
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        N = np.asarray(N, dtype=np.float64).reshape(-1, 3)
        I = np.asarray(I, dtype=np.int64).reshape(-1, 3)
        P2, N2, flip = apply_xf(M, P, N)
        if flip:
            I = I[:, ::-1]
        # geometry is given in world space; convert into node-local space
        P2 = P2 - self.world_offset(node)
        UVm = uvspace.map(UV)
        self.nodes[node][uvspace.material].append(Geo(P2, N2, UVm, I, occ=occ, subdiv=subdiv))

    def tri_count(self):
        n = 0
        for nd in self.nodes.values():
            for lst in nd.values():
                for g in lst:
                    n += len(g.I)
        return n

    def tri_count_by_node(self):
        out = OrderedDict()
        for name, nd in self.nodes.items():
            out[name] = sum(len(g.I) for lst in nd.values() for g in lst)
        return out

# ----------------------------------------------------------------------------
# Primitive generators. All return (P, N, UV, I) in local space.
# UVs: by default planar-in-metres (so tiling materials keep texel density);
# callers can override for atlas rects.
# ----------------------------------------------------------------------------


def _quad(p0, p1, p2, p3, uv=None, n=None):
    """Quad p0-p1-p2-p3 counter-clockwise (seen from the front)."""
    P = np.array([p0, p1, p2, p3], dtype=np.float64)
    if n is None:
        n = normalize(np.cross(P[1] - P[0], P[2] - P[0]))
        if np.linalg.norm(n) < 1e-9:
            n = normalize(np.cross(P[2] - P[0], P[3] - P[0]))
    N = np.tile(n, (4, 1))
    if uv is None:
        uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
    return P, N, np.array(uv, dtype=np.float64), np.array([[0, 1, 2], [0, 2, 3]])


def merge(parts):
    P, N, UV, I = [], [], [], []
    off = 0
    for (p, n, uv, i) in parts:
        P.append(p); N.append(n); UV.append(uv); I.append(np.asarray(i) + off)
        off += len(p)
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int64)
    return np.vstack(P), np.vstack(N), np.vstack(UV), np.vstack(I)


def planar_uv(P, n, scale=1.0):
    """World-metre planar projection along the dominant axis of normal n."""
    n = np.asarray(n)
    a = np.argmax(np.abs(n))
    if a == 0:   # side wall facing +-x: u along z, v along y
        u = P[:, 2] * (1 if n[0] < 0 else -1)
        v = P[:, 1]
    elif a == 1:  # floor/roof: u along z, v along x
        u = P[:, 2]
        v = P[:, 0] * (1 if n[1] > 0 else -1)
    else:        # facing +-z: u along x, v along y
        u = P[:, 0] * (1 if n[2] > 0 else -1)
        v = P[:, 1]
    return np.stack([u * scale, -v * scale], axis=1)


def flat_poly_faces(P_list, scale=1.0):
    """Given list of planar convex polygons (k,3) arrays, build flat-shaded fans with planar UVs."""
    parts = []
    for poly in P_list:
        poly = np.asarray(poly, dtype=np.float64)
        k = len(poly)
        n = np.zeros(3)
        for i in range(k):  # Newell normal
            a, b = poly[i], poly[(i + 1) % k]
            n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])])
        n = normalize(n)
        uv = planar_uv(poly, n, scale)
        I = np.array([[0, i, i + 1] for i in range(1, k - 1)])
        parts.append((poly, np.tile(n, (k, 1)), uv, I))
    return merge(parts)


def box(sx, sy, sz, center=(0, 0, 0), faces='all', uvscale=1.0):
    """Axis-aligned box. faces: 'all' or a string subset of 'xXyYzZ' (lowercase = negative side)."""
    cx, cy, cz = center
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    x0, x1, y0, y1, z0, z1 = cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz
    F = {
        'X': [(x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1)],
        'x': [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)],
        'Y': [(x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0)],
        'y': [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        'Z': [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        'z': [(x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0)],
    }
    use = 'xXyYzZ' if faces == 'all' else faces
    return flat_poly_faces([F[f] for f in use], uvscale)


def frustum_box(bot, top, y0, y1, faces='all', uvscale=1.0):
    """Box with different rectangular footprints at bottom/top.
    bot/top = (xmin, xmax, zmin, zmax). Gives tapered superstructure blocks."""
    bx0, bx1, bz0, bz1 = bot
    tx0, tx1, tz0, tz1 = top
    B = [(bx0, y0, bz0), (bx1, y0, bz0), (bx1, y0, bz1), (bx0, y0, bz1)]
    Tt = [(tx0, y1, tz0), (tx1, y1, tz0), (tx1, y1, tz1), (tx0, y1, tz1)]
    polys = {
        'y': [B[0], B[1], B[2], B[3]],
        'Y': [Tt[3], Tt[2], Tt[1], Tt[0]],
        'z': [B[1], B[0], Tt[0], Tt[1]],
        'Z': [B[3], B[2], Tt[2], Tt[3]],
        'x': [B[0], B[3], Tt[3], Tt[0]],
        'X': [B[2], B[1], Tt[1], Tt[2]],
    }
    use = 'xXyYzZ' if faces == 'all' else faces
    return flat_poly_faces([polys[f] for f in use], uvscale)


def prism(poly_xz, y0, y1, top=True, bottom=False, uvscale=1.0, top_poly=None, smooth=False):
    """Vertical extrusion of a CCW (seen from +y) polygon in the xz plane.
    poly_xz: list of (x, z). top_poly: optional different polygon at top (same vertex count)."""
    poly = np.asarray(poly_xz, dtype=np.float64)
    tpoly = poly if top_poly is None else np.asarray(top_poly, dtype=np.float64)
    # enforce CCW when viewed from +y (screen coords (x, -z))
    area = 0.0
    for i in range(len(poly)):
        x0_, z0_ = poly[i]; x1_, z1_ = poly[(i + 1) % len(poly)]
        area += x0_ * (-z1_) - x1_ * (-z0_)
    if area < 0:
        poly = poly[::-1].copy()
        tpoly = tpoly[::-1].copy()
    k = len(poly)
    parts = []
    if smooth:
        # smooth side normals (for rounded outlines)
        P, N, UV, I = [], [], [], []
        per = 0.0
        lens = [0.0]
        for i in range(k):
            a, b = poly[i], poly[(i + 1) % k]
            per += np.hypot(*(b - a))
            lens.append(per)
        for i in range(k + 1):
            j = i % k
            a = poly[(j - 1) % k]; b = poly[(j + 1) % k]
            t = b - a
            # outward normal for CCW polygon viewed from +y in (x,z): rotate tangent
            nrm = normalize(np.array([-t[1], 0.0, t[0]]))
            P += [(poly[j][0], y0, poly[j][1]), (tpoly[j][0], y1, tpoly[j][1])]
            N += [nrm, nrm]
            UV += [(lens[i] * uvscale, -y0 * uvscale), (lens[i] * uvscale, -y1 * uvscale)]
        for i in range(k):
            a, b = 2 * i, 2 * i + 2
            I += [(a, b, b + 1), (a, b + 1, a + 1)]
        parts.append((np.array(P), np.array(N), np.array(UV), np.array(I)))
    else:
        side = []
        for i in range(k):
            a, b = poly[i], poly[(i + 1) % k]
            ta, tb = tpoly[i], tpoly[(i + 1) % k]
            side.append([(a[0], y0, a[1]), (b[0], y0, b[1]), (tb[0], y1, tb[1]), (ta[0], y1, ta[1])])
        parts.append(flat_poly_faces(side, uvscale))
    if top:
        parts.append(cap_polygon(tpoly, y1, up=True, uvscale=uvscale))
    if bottom:
        parts.append(cap_polygon(poly, y0, up=False, uvscale=uvscale))
    return merge(parts)


def triangulate_polygon(poly):
    """Ear clipping for a simple polygon given as (k,2) array (CCW). Returns index triples."""
    pts = [tuple(p) for p in poly]
    idx = list(range(len(pts)))
    # ensure CCW
    area = 0.0
    for i in range(len(pts)):
        x0, y0 = pts[i]; x1, y1 = pts[(i + 1) % len(pts)]
        area += x0 * y1 - x1 * y0
    if area < 0:
        idx.reverse()
    tris = []

    def is_convex(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 1e-12

    def inside(p, a, b, c):
        d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
        d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1])
        d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])
        neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (neg and pos)

    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        n = len(idx)
        ear_found = False
        for i in range(n):
            ia, ib, ic = idx[(i - 1) % n], idx[i], idx[(i + 1) % n]
            a, b, c = pts[ia], pts[ib], pts[ic]
            if not is_convex(a, b, c):
                continue
            ok = True
            for j in idx:
                if j in (ia, ib, ic):
                    continue
                if inside(pts[j], a, b, c):
                    ok = False
                    break
            if ok:
                tris.append((ia, ib, ic))
                idx.pop(i)
                ear_found = True
                break
        if not ear_found:
            # degenerate; fall back to fan
            for i in range(1, len(idx) - 1):
                tris.append((idx[0], idx[i], idx[i + 1]))
            idx = idx[:2]
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return np.array(tris, dtype=np.int64)


def cap_polygon(poly_xz, y, up=True, uvscale=1.0):
    poly = np.asarray(poly_xz, dtype=np.float64)
    # poly is (x,z); viewed from +y, CCW means... we compute triangulation in (x, -z) so that
    # CCW-from-above corresponds to positive area.
    tri = triangulate_polygon(np.stack([poly[:, 0], -poly[:, 1]], axis=1))
    P = np.stack([poly[:, 0], np.full(len(poly), y), poly[:, 1]], axis=1)
    n = np.array([0, 1.0, 0]) if up else np.array([0, -1.0, 0])
    I = tri if up else tri[:, ::-1]
    # orientation check: ensure triangle normals agree with n
    if len(I):
        a, b, c = P[I[0, 0]], P[I[0, 1]], P[I[0, 2]]
        if np.dot(np.cross(b - a, c - a), n) < 0:
            I = I[:, ::-1]
    uv = np.stack([P[:, 2] * uvscale, P[:, 0] * uvscale * (1 if up else -1)], axis=1)
    return P, np.tile(n, (len(P), 1)), uv, I


def cylinder(r, h, seg=12, caps=(True, True), r_top=None, y0=0.0, uvscale=1.0, smooth=True,
             cap_uv_rect=None, side_uv=None, angle0=0.0):
    """Vertical cylinder/cone frustum along +y from y0 to y0+h."""
    rt = r if r_top is None else r_top
    ang = angle0 + np.linspace(0, 2 * np.pi, seg + 1)
    parts = []
    P, N, UV, I = [], [], [], []
    slope = (r - rt) / h if h != 0 else 0.0
    circ = 2 * np.pi * max(r, rt)
    for i, a in enumerate(ang):
        c, s = math.cos(a), math.sin(a)
        n = normalize(np.array([c, slope, s]))
        if side_uv is None:
            u = (i / seg) * circ * uvscale
            P += [(r * c, y0, r * s), (rt * c, y0 + h, rt * s)]
            UV += [(u, -y0 * uvscale), (u, -(y0 + h) * uvscale)]
        else:
            P += [(r * c, y0, r * s), (rt * c, y0 + h, rt * s)]
            UV += [(side_uv[0] + (side_uv[2] - side_uv[0]) * i / seg, side_uv[3]),
                   (side_uv[0] + (side_uv[2] - side_uv[0]) * i / seg, side_uv[1])]
        N += [n, n]
    for i in range(seg):
        a = 2 * i
        I += [(a, a + 3, a + 2), (a, a + 1, a + 3)]
    P = np.array(P); N = np.array(N); UV = np.array(UV); I = np.array(I)
    if not smooth:
        # split into flat facets
        PP, NN, UU, II = [], [], [], []
        for i in range(seg):
            q = [P[2 * i], P[2 * i + 2], P[2 * i + 3], P[2 * i + 1]]
            n = normalize(np.cross(q[1] - q[0], q[3] - q[0]))
            # outward check
            mid = np.mean(q, axis=0)
            if np.dot(n, np.array([mid[0], 0, mid[2]])) < 0:
                n = -n
            base = len(PP)
            PP += q; NN += [n] * 4
            UU += [UV[2 * i], UV[2 * i + 2], UV[2 * i + 3], UV[2 * i + 1]]
            II += [(base, base + 2, base + 1), (base, base + 3, base + 2)]
        P, N, UV, I = np.array(PP), np.array(NN), np.array(UU), np.array(II)
        # fix winding so that normals point outward
        a, b, c = P[I[0, 0]], P[I[0, 1]], P[I[0, 2]]
        if np.dot(np.cross(b - a, c - a), N[I[0, 0]]) < 0:
            I = I[:, ::-1]
    else:
        a, b, c = P[I[0, 0]], P[I[0, 1]], P[I[0, 2]]
        if np.dot(np.cross(b - a, c - a), N[I[0, 0]]) < 0:
            I = I[:, ::-1]
    parts.append((P, N, UV, I))
    for which, (yy, rr, up) in enumerate([(y0, r, False), (y0 + h, rt, True)]):
        if not caps[which] or rr <= 1e-6:
            continue
        cp = [(rr * math.cos(a), yy, rr * math.sin(a)) for a in ang[:-1]]
        cp = np.array(cp)
        n = np.array([0, 1.0 if up else -1.0, 0])
        if cap_uv_rect is not None:
            u0, v0, u1, v1 = cap_uv_rect
            uv = np.stack([u0 + (u1 - u0) * (0.5 + 0.5 * np.cos(ang[:-1] - angle0)),
                           v0 + (v1 - v0) * (0.5 + 0.5 * np.sin(ang[:-1] - angle0))], axis=1)
        else:
            uv = np.stack([cp[:, 0] * uvscale, cp[:, 2] * uvscale], axis=1)
        I2 = np.array([[0, i, i + 1] for i in range(1, seg - 1)])
        a_, b_, c_ = cp[I2[0, 0]], cp[I2[0, 1]], cp[I2[0, 2]]
        if np.dot(np.cross(b_ - a_, c_ - a_), n) < 0:
            I2 = I2[:, ::-1]
        parts.append((cp, np.tile(n, (seg, 1)), uv, I2))
    return merge(parts)


def lathe(profile, seg=16, uvscale=1.0, smooth=True, close_top=False, close_bottom=False, angle0=0.0,
          arc=2 * np.pi, uv_rect=None):
    """Revolve a profile [(r, y), ...] around the +y axis. Smooth normals along the profile
    are computed per segment with hard edges where the profile turns sharply (>40deg)."""
    prof = np.asarray(profile, dtype=np.float64)
    m = len(prof)
    ang = angle0 + np.linspace(0, arc, seg + 1)
    parts = []
    # split profile into smooth runs at sharp corners
    seg_dirs = normalize(np.diff(prof, axis=0))
    runs = []
    start = 0
    for i in range(1, m - 1):
        d0, d1 = seg_dirs[i - 1], seg_dirs[i]
        if np.dot(d0, d1) < math.cos(math.radians(40)):
            runs.append((start, i))
            start = i
    runs.append((start, m - 1))
    # cumulative length for v
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))])
    Ltot = L[-1] if L[-1] > 0 else 1.0
    for (a0, a1) in runs:
        sub = prof[a0:a1 + 1]
        k = len(sub)
        # 2D normals in (r,y): rotate tangent by -90 => (dy, -dr)
        tang = np.zeros((k, 2))
        for i in range(k):
            if i == 0:
                t = sub[1] - sub[0]
            elif i == k - 1:
                t = sub[-1] - sub[-2]
            else:
                t = sub[i + 1] - sub[i - 1]
            tang[i] = t
        tang = normalize(tang)
        n2 = np.stack([tang[:, 1], -tang[:, 0]], axis=1)  # outward if profile goes upward with r>0
        P, N, UV, I = [], [], [], []
        for j, a in enumerate(ang):
            c, s = math.cos(a), math.sin(a)
            for i in range(k):
                r, y = sub[i]
                P.append((r * c, y, r * s))
                N.append((n2[i, 0] * c, n2[i, 1], n2[i, 0] * s))
                if uv_rect is None:
                    UV.append((j / seg * 2 * np.pi * max(1e-3, prof[:, 0].max()) * uvscale, -L[a0 + i] * uvscale))
                else:
                    u0, v0, u1, v1 = uv_rect
                    UV.append((u0 + (u1 - u0) * j / seg, v0 + (v1 - v0) * L[a0 + i] / Ltot))
        for j in range(seg):
            for i in range(k - 1):
                p00 = j * k + i
                p01 = j * k + i + 1
                p10 = (j + 1) * k + i
                p11 = (j + 1) * k + i + 1
                I += [(p00, p01, p11), (p00, p11, p10)]
        P = np.array(P); N = normalize(np.array(N)); UV = np.array(UV); I = np.array(I)
        if len(I):
            # orient outward
            score = 0.0
            for t in I[:min(len(I), 20)]:
                a_, b_, c_ = P[t[0]], P[t[1]], P[t[2]]
                score += np.dot(np.cross(b_ - a_, c_ - a_), N[t[0]] + N[t[1]] + N[t[2]])
            if score < 0:
                I = I[:, ::-1]
        if not smooth:
            P, N, UV, I = flatten(P, N, UV, I)
        parts.append((P, N, UV, I))
    out = merge(parts)
    return out


def flatten(P, N, UV, I):
    """Convert indexed mesh to flat-shaded (unshared vertices, face normals)."""
    P2 = P[I].reshape(-1, 3)
    UV2 = UV[I].reshape(-1, 2)
    fn = normalize(np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]]))
    # keep original orientation preference
    N2 = np.repeat(fn, 3, axis=0)
    I2 = np.arange(len(P2)).reshape(-1, 3)
    return P2, N2, UV2, I2


def sphere(r, seg=12, rings=8, hemi=False, uvscale=1.0, uv_rect=None):
    a_max = np.pi / 2 if hemi else np.pi
    prof = []
    for i in range(rings + 1):
        t = a_max * i / rings
        # from top (t=0) downward
        prof.append((r * math.sin(t), r * math.cos(t)))
    prof = prof[::-1]  # bottom -> top so the outward normal convention holds
    return lathe(prof, seg=seg, uvscale=uvscale, uv_rect=uv_rect)


def tube_path(points, r, seg=6, uvscale=1.0, caps=False):
    """Sweep a circle of radius r along a polyline (for pipes, rails, booms)."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    P, N, UV, I = [], [], [], []
    acc = 0.0
    for i in range(n):
        if i == 0:
            t = pts[1] - pts[0]
        elif i == n - 1:
            t = pts[-1] - pts[-2]
        else:
            t = normalize(pts[i + 1] - pts[i]) + normalize(pts[i] - pts[i - 1])
        t = normalize(t)
        R = frame_from_dir(t)
        if i > 0:
            acc += np.linalg.norm(pts[i] - pts[i - 1])
        for j in range(seg + 1):
            a = 2 * np.pi * j / seg
            d = R @ np.array([math.cos(a), math.sin(a), 0])
            P.append(pts[i] + r * d)
            N.append(d)
            UV.append((j / seg * 2 * np.pi * r * uvscale, acc * uvscale))
    for i in range(n - 1):
        for j in range(seg):
            a = i * (seg + 1) + j
            b = a + seg + 1
            I += [(a, a + 1, b + 1), (a, b + 1, b)]
    P = np.array(P); N = np.array(N); UV = np.array(UV); I = np.array(I)
    a_, b_, c_ = P[I[0, 0]], P[I[0, 1]], P[I[0, 2]]
    if np.dot(np.cross(b_ - a_, c_ - a_), N[I[0, 0]]) < 0:
        I = I[:, ::-1]
    parts = [(P, N, UV, I)]
    if caps:
        for end, sgn in ((0, -1), (n - 1, 1)):
            t = normalize(pts[min(end + 1, n - 1)] - pts[max(end - 1, 0)]) * sgn
            R = frame_from_dir(t)
            cp = np.array([pts[end] + r * (R @ np.array([math.cos(2 * np.pi * j / seg), math.sin(2 * np.pi * j / seg), 0])) for j in range(seg)])
            I2 = np.array([[0, j, j + 1] for j in range(1, seg - 1)])
            a_, b_, c_ = cp[I2[0, 0]], cp[I2[0, 1]], cp[I2[0, 2]]
            if np.dot(np.cross(b_ - a_, c_ - a_), t) < 0:
                I2 = I2[:, ::-1]
            parts.append((cp, np.tile(t, (seg, 1)), np.zeros((seg, 2)), I2))
    return merge(parts)


def beam(p0, p1, w, h=None, uvscale=1.0, up=(0, 1, 0)):
    """Rectangular bar between two points (lattice members, booms)."""
    h = w if h is None else h
    p0 = np.asarray(p0, dtype=np.float64); p1 = np.asarray(p1, dtype=np.float64)
    d = p1 - p0
    L = np.linalg.norm(d)
    R = frame_from_dir(d, up)
    P, N, UV, I = box(w, h, L, center=(0, 0, L / 2), faces='xXyY', uvscale=uvscale)
    P = P @ R.T + p0
    N = N @ R.T
    return P, N, UV, I


def loft(sections, closed=False, uvscale=1.0, smooth_u=True, smooth_v=True, cap_start=False, cap_end=False):
    """Loft through a list of sections, each an (m,3) array with equal m.
    Normals are averaged (smooth) across both directions; returns indexed mesh."""
    S = np.asarray(sections, dtype=np.float64)  # (n, m, 3)
    n, m, _ = S.shape
    P = S.reshape(-1, 3)
    I = []
    mm = m if closed else m - 1
    for i in range(n - 1):
        for j in range(mm):
            a = i * m + j
            b = i * m + (j + 1) % m
            c = (i + 1) * m + (j + 1) % m
            d = (i + 1) * m + j
            I += [(a, b, c), (a, c, d)]
    I = np.array(I)
    # area-weighted vertex normals
    fn = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    N = np.zeros_like(P)
    for k in range(3):
        np.add.at(N, I[:, k], fn)
    N = normalize(N)
    # uv: u along sections (arc length avg), v along section
    du = np.linalg.norm(np.diff(S, axis=0), axis=2).mean(axis=1)
    U = np.concatenate([[0], np.cumsum(du)])
    dv = np.linalg.norm(np.diff(S, axis=1), axis=2).mean(axis=0)
    V = np.concatenate([[0], np.cumsum(dv)])
    UV = np.stack(np.meshgrid(U, V, indexing='ij'), axis=-1).reshape(-1, 2) * uvscale
    parts = [(P, N, UV, I)]
    for flag, idx in ((cap_start, 0), (cap_end, n - 1)):
        if not flag:
            continue
        ring = S[idx]
        c = ring.mean(axis=0)
        cp = np.vstack([c, ring])
        I2 = np.array([[0, j + 1, (j + 1) % m + 1] for j in range(m)])
        a_, b_, c_ = cp[I2[0, 0]], cp[I2[0, 1]], cp[I2[0, 2]]
        dirv = (S[idx] - S[1 if idx == 0 else n - 2]).mean(axis=0)
        if np.dot(np.cross(b_ - a_, c_ - a_), dirv) < 0:
            I2 = I2[:, ::-1]
        cn = normalize(dirv)
        parts.append((cp, np.tile(cn, (m + 1, 1)), np.zeros((m + 1, 2)), I2))
    return merge(parts)


def compute_smooth_normals(P, I):
    fn = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    N = np.zeros_like(P)
    for k in range(3):
        np.add.at(N, I[:, k], fn)
    return normalize(N)

# ----------------------------------------------------------------------------
# Tangents
# ----------------------------------------------------------------------------


def compute_tangents(P, N, UV, I):
    t = np.zeros((len(P), 3))
    b = np.zeros((len(P), 3))
    p0, p1, p2 = P[I[:, 0]], P[I[:, 1]], P[I[:, 2]]
    w0, w1, w2 = UV[I[:, 0]], UV[I[:, 1]], UV[I[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = w1 - w0, w2 - w0
    r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(r) < 1e-12, 1e-12, r)
    sdir = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / r[:, None]
    tdir = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / r[:, None]
    for k in range(3):
        np.add.at(t, I[:, k], sdir)
        np.add.at(b, I[:, k], tdir)
    # Gram-Schmidt
    tt = t - N * np.sum(N * t, axis=1, keepdims=True)
    bad = np.linalg.norm(tt, axis=1) < 1e-9
    if np.any(bad):
        alt = np.cross(N[bad], np.array([0, 1.0, 0]))
        alt2 = np.cross(N[bad], np.array([1.0, 0, 0]))
        use = np.linalg.norm(alt, axis=1, keepdims=True) > 1e-6
        tt[bad] = np.where(use, alt, alt2)
    tt = normalize(tt)
    w = np.where(np.sum(np.cross(N, tt) * b, axis=1) < 0.0, -1.0, 1.0)
    # glTF convention: bitangent = cross(normal, tangent.xyz) * w ; uv v flipped (top-left origin)
    return np.concatenate([tt, w[:, None]], axis=1)

# ----------------------------------------------------------------------------
# GLB writer
# ----------------------------------------------------------------------------


class GLBWriter:
    def __init__(self):
        self.bin = bytearray()
        self.bufferViews = []
        self.accessors = []
        self.meshes = []
        self.nodes = []
        self.materials = []
        self.textures = []
        self.images = []
        self.samplers = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]

    def _view(self, data, target=None):
        while len(self.bin) % 4:
            self.bin += b'\x00'
        off = len(self.bin)
        self.bin += data
        bv = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            bv["target"] = target
        self.bufferViews.append(bv)
        return len(self.bufferViews) - 1

    def accessor(self, arr, ctype, typ, target=None, minmax=False, normalized=False):
        arr = np.ascontiguousarray(arr)
        bv = self._view(arr.tobytes(), target)
        acc = {"bufferView": bv, "componentType": ctype, "count": int(arr.shape[0]), "type": typ}
        if normalized:
            acc["normalized"] = True
        if minmax:
            acc["min"] = [float(x) for x in arr.min(axis=0)]
            acc["max"] = [float(x) for x in arr.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def add_image_png(self, png_bytes, name=None):
        bv = self._view(png_bytes)
        im = {"bufferView": bv, "mimeType": "image/png"}
        if name:
            im["name"] = name
        self.images.append(im)
        self.textures.append({"sampler": 0, "source": len(self.images) - 1})
        return len(self.textures) - 1

    def add_material(self, mat):
        self.materials.append(mat)
        return len(self.materials) - 1

    def add_mesh(self, name, prims):
        """prims: list of (material_index, P, N, UV, I)"""
        gl_prims = []
        for prim in prims:
            mi, P, N, UV, I = prim[:5]
            C = prim[5] if len(prim) > 5 else None
            Tg = compute_tangents(P, N, UV, I)
            pa = self.accessor(P.astype(np.float32), 5126, "VEC3", 34962, minmax=True)
            na = self.accessor(N.astype(np.float32), 5126, "VEC3", 34962)
            ta = self.accessor(Tg.astype(np.float32), 5126, "VEC4", 34962)
            ua = self.accessor(UV.astype(np.float32), 5126, "VEC2", 34962)
            if len(P) < 65536:
                ia = self.accessor(I.reshape(-1).astype(np.uint16), 5123, "SCALAR", 34963)
            else:
                ia = self.accessor(I.reshape(-1).astype(np.uint32), 5125, "SCALAR", 34963)
            attrs = {"POSITION": pa, "NORMAL": na, "TANGENT": ta, "TEXCOORD_0": ua}
            if C is not None:
                c8 = np.clip(np.round(np.asarray(C) * 255.0), 0, 255).astype(np.uint8)
                if c8.shape[1] == 3:
                    c8 = np.concatenate([c8, np.full((len(c8), 1), 255, np.uint8)], axis=1)
                attrs["COLOR_0"] = self.accessor(c8, 5121, "VEC4", 34962, normalized=True)
            gl_prims.append({"attributes": attrs, "indices": ia, "material": mi, "mode": 4})
        self.meshes.append({"name": name, "primitives": gl_prims})
        return len(self.meshes) - 1

    def write(self, path, scene_nodes, asset_extras=None):
        while len(self.bin) % 4:
            self.bin += b'\x00'
        gltf = {
            "asset": {"version": "2.0", "generator": "meshkit (procedural)"},
            "scene": 0,
            "scenes": [{"name": "Scene", "nodes": scene_nodes}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "materials": self.materials,
            "textures": self.textures,
            "images": self.images,
            "samplers": self.samplers,
            "accessors": self.accessors,
            "bufferViews": self.bufferViews,
            "buffers": [{"byteLength": len(self.bin)}],
        }
        if asset_extras:
            gltf["asset"]["extras"] = asset_extras
        js = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
        while len(js) % 4:
            js += b' '
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        with open(path, 'wb') as f:
            f.write(struct.pack('<4sII', b'glTF', 2, total))
            f.write(struct.pack('<I4s', len(js), b'JSON'))
            f.write(js)
            f.write(struct.pack('<I4s', len(self.bin), b'BIN\x00'))
            f.write(bytes(self.bin))


def weld(P, N, UV, I, eps=1e-5):
    """Merge identical vertices (same position, normal, uv)."""
    key = np.concatenate([np.round(P / eps), np.round(N * 1000), np.round(UV / 1e-5)], axis=1).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    return P[first], N[first], UV[first], inv[I]


def builder_to_glb(builder, materials, path, asset_extras=None, root_name='Root', ao=None):
    """materials: OrderedDict name -> (gltf material dict, textures{slot: png bytes})"""
    w = GLBWriter()
    mat_index = {}
    tex_cache = {}
    used_mats = {mname for nd in builder.nodes.values() for mname, geos in nd.items() if geos}
    for name, (mdef, texs) in materials.items():
        if name not in used_mats:
            continue
        m = json.loads(json.dumps(mdef))
        for slot, png in texs.items():
            key = id(png)
            if key not in tex_cache:
                tex_cache[key] = w.add_image_png(png, name=f"{name}_{slot}")
            ti = tex_cache[key]
            if slot == 'baseColor':
                m.setdefault('pbrMetallicRoughness', {})['baseColorTexture'] = {"index": ti}
            elif slot == 'metallicRoughness':
                m.setdefault('pbrMetallicRoughness', {})['metallicRoughnessTexture'] = {"index": ti}
            elif slot == 'normal':
                m['normalTexture'] = {"index": ti, "scale": m.pop('_normalScale', 1.0)}
            elif slot == 'occlusion':
                m['occlusionTexture'] = {"index": ti}
        m.pop('_normalScale', None)
        m['name'] = name
        mat_index[name] = w.add_material(m)
    # ---- gather primitives (optionally subdivided), occluders in world space
    order = list(builder.nodes.keys())
    prim_data = OrderedDict()
    occ_P, occ_I, occ_n = [], [], 0
    for name in order:
        nd = builder.nodes[name]
        off = builder.world_offset(name)
        for mname, geos in nd.items():
            parts = []
            for g in geos:
                P, N, UV, I = g.P, g.N, g.UV, g.I
                if len(I) == 0:
                    continue
                area = np.linalg.norm(np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]]), axis=1)
                I = I[area > 1e-9]
                if len(I) == 0:
                    continue
                if ao is not None and g.occ:
                    occ_P.append(P + off); occ_I.append(I + occ_n); occ_n += len(P)
                if ao is not None and g.subdiv and ao.get('max_len'):
                    from ao import subdivide_soup
                    sp, sn, su = subdivide_soup(P[I], N[I], UV[I], ao['max_len'], min_area=ao.get('min_area', 0.0))
                    k = len(sp)
                    P, N, UV = sp.reshape(-1, 3), sn.reshape(-1, 3), su.reshape(-1, 2)
                    I = np.arange(k * 3).reshape(-1, 3)
                parts.append((P, N, UV, I))
            if not parts:
                continue
            P, N, UV, I = merge(parts)
            nl = np.linalg.norm(N, axis=1)
            bad = ~(nl > 1e-6)
            if np.any(bad):
                N = N.copy(); N[bad] = np.array([0.0, 1.0, 0.0])
            N = normalize(N)
            used = np.unique(I.reshape(-1))
            remap = -np.ones(len(P), dtype=np.int64)
            remap[used] = np.arange(len(used))
            P, N, UV, I = P[used], N[used], UV[used], remap[I]
            P, N, UV, I = weld(P, N, UV, I)
            prim_data[(name, mname)] = [P, N, UV, I, None, off]
    # ---- ambient occlusion -> COLOR_0
    if ao is not None and occ_P:
        from ao import compute_ao
        OP = np.vstack(occ_P); OI = np.vstack(occ_I)
        keys = list(prim_data.keys())
        pts = np.vstack([prim_data[k][0] + prim_data[k][5] for k in keys])
        nrm = np.vstack([prim_data[k][1] for k in keys])
        vis = compute_ao(OP, OI, pts, nrm, rays=ao.get('rays', 48), max_dist=ao.get('max_dist', 5.0))
        floor = ao.get('floor', 0.35)
        gamma = ao.get('gamma', 1.0)
        shade = floor + (1.0 - floor) * np.clip(vis, 0, 1) ** gamma
        o = 0
        for k in keys:
            n = len(prim_data[k][0])
            sh = shade[o:o + n]; o += n
            prim_data[k][4] = np.repeat(sh[:, None], 3, axis=1)
    # ---- nodes
    node_ids = {}
    w.nodes.append({"name": root_name, "children": []})
    root_id = 0
    for name in order:
        prims = []
        for mname in builder.nodes[name].keys():
            if (name, mname) not in prim_data:
                continue
            P, N, UV, I, C, _ = prim_data[(name, mname)]
            if len(P) > 65535:
                start = 0
                while start < len(I):
                    sub = I[start:start + 20000]
                    used, inv = np.unique(sub.reshape(-1), return_inverse=True)
                    prims.append((mat_index[mname], P[used], N[used], UV[used], inv.reshape(-1, 3),
                                  None if C is None else C[used]))
                    start += 20000
            else:
                prims.append((mat_index[mname], P, N, UV, I, C))
        node = {"name": name}
        tr = builder.node_info[name]['translation']
        if any(abs(t) > 0 for t in tr):
            node["translation"] = list(tr)
        if prims:
            node["mesh"] = w.add_mesh(name, prims)
        w.nodes.append(node)
        node_ids[name] = len(w.nodes) - 1
    for name in order:
        parent = builder.node_info[name]['parent']
        pid = root_id if parent is None else node_ids[parent]
        w.nodes[pid].setdefault("children", []).append(node_ids[name])
    w.write(path, [root_id], asset_extras)
    return w
