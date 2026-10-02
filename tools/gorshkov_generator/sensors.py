"""Project 22350 sensors: 5P-20K Poliment array faces, 5P-27 Furke-4 search radar, radomes, Pal-N
navigation radars, optronic directors, ESM boxes. Shapes follow the 2018-2023 photographs."""
import math
import numpy as np

from meshkit import *
from meshkit import _quad
import atlas as st
from kit import M
from texkit import value_noise
from parts import rbox, disc, axis_frame

PAL = st.PAL


def face_quad(c, uvs, corners, off=0.0, uv=((0, 1), (1, 1), (1, 0), (0, 0))):
    """quad on a face given its corners (bottom-left, bottom-right, top-right, top-left as seen from
    outside), pushed `off` metres along the face normal."""
    p = [np.asarray(q, float) for q in corners]
    n = normalize(np.cross(p[1] - p[0], p[3] - p[0]))
    p = [q + n * off for q in p]
    c.add(uvs, _quad(p[0], p[1], p[2], p[3], uv=list(uv), n=n))
    return n


def array_panel(c, centre, normal, up, w, h, depth=0.16, name='array'):
    """flat phased-array face: raised frame box with the element-grid decal on its front."""
    n = normalize(np.asarray(normal, float))
    u = np.asarray(up, float)
    u = normalize(u - np.dot(u, n) * n)
    r = np.cross(u, n)
    o = np.asarray(centre, float)
    R = np.column_stack([r, u, n])
    c.add(c.paint, box(w + 0.16, h + 0.16, depth, center=(0, 0, depth / 2)), xf=M(o, R))
    f = o + n * (depth + 0.006)
    corners = [f - r * w / 2 - u * h / 2, f + r * w / 2 - u * h / 2, f + r * w / 2 + u * h / 2, f - r * w / 2 + u * h / 2]
    c.add(c.rect(name), _quad(*corners, uv=[(0, 1), (1, 1), (1, 0), (0, 0)], n=n))


def paint_array(L, rect):
    """Poliment face: dark grey radome skin with a fine element grid and a lighter border."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    L.rect(x0, y0, x1, y1, col=PAL['mid'] * 0.85, alpha=1.0, rough=0.7, metal=0.0)
    n = value_noise(h, w, 16, seed=41, octaves=3)
    L.col[y0:y1, x0:x1] *= (0.95 + 0.08 * n)[..., None]
    for k in range(0, w, 6):
        L.rect(x0 + k, y0 + 3, x0 + k + 1, y1 - 3, col=PAL['mid'] * 0.97, add_height=-0.15)
    for k in range(0, h, 6):
        L.rect(x0 + 3, y0 + k, x1 - 3, y0 + k + 1, col=PAL['mid'] * 0.97, add_height=-0.15)
    for e in ((x0, y0, x1, y0 + 3), (x0, y1 - 3, x1, y1), (x0, y0, x0 + 3, y1), (x1 - 3, y0, x1, y1)):
        L.rect(*e, col=PAL['super'] * 0.9, add_height=0.5)


def furke4(c, base, yaw=0.0, node='Furke4'):
    """5P-27 Furke-4: turntable on a drum pedestal carrying a deep, back-tilted array housing with a
    flat face and a rounded back (3.3 m wide, 3.0 m tall, 1.5 m deep)."""
    o = np.asarray(base, float)
    c.add(c.paint, cylinder(0.6, 0.7, seg=16, caps=(False, True)), xf=M(o))
    c.b.node(node, parent='Sensors', translation=o)        # rotating part, pivot on the turntable axis
    R = rot_y(yaw)
    c.add(c.sw('mid'), cylinder(0.95, 0.2, seg=20, caps=(True, True), y0=0.7), xf=M(o))
    tilt = math.radians(14.0)
    Rt = R @ rot_x(-tilt)
    w, h, d = 3.3, 3.0, 1.5
    piv = o + np.array([0, 0.9, 0])
    # plan section of the housing: flat front, rounded back
    sec = []
    for k in range(9):
        a = math.pi * k / 8
        sec.append((w / 2 * math.cos(a), -0.35 - (d - 0.35) * math.sin(a) ** 0.7))
    sec = [(w / 2, 0.0)] + sec + [(-w / 2, 0.0)]
    rings = []
    for yy in (0.0, 0.12, h - 0.12, h):
        inset = 0.08 if yy in (0.0, h) else 0.0
        rings.append([piv + Rt @ np.array([x * (1 - inset / (w / 2)), yy, z * (1 - inset / d) + 0.45]) for (x, z) in sec])
    import weapons as _w
    _w.loft_solid(c, c.sw('light'), rings)
    fc = piv + Rt @ np.array([0, h / 2, 0.46])
    for k in (-1, 0, 1):
        c.add(c.sw('mid'), box(0.03, h * 0.86, 0.02, center=(0, 0, 0)), xf=M(fc + Rt @ np.array([k * w * 0.3, 0, 0]), Rt))
    c.add(c.sw('mid'), box(w * 0.82, 0.3, 0.3, center=(0, 0, 0)), xf=M(piv + Rt @ np.array([0, h + 0.15, -0.2]), Rt))


def radome(c, pos, r, ped_h=0.35, ped_r=None, band_h=0.0, node=None):
    """spherical radome on a cylindrical pedestal (the sphere sits just above its equator)."""
    o = np.asarray(pos, float)
    pr = ped_r if ped_r is not None else r * 0.55
    c.add(c.paint, cylinder(pr, ped_h, seg=14, caps=(False, True)), xf=M(o), node=node)
    cy = ped_h + r * 0.92
    c.add(c.sw('radome'), sphere(r, seg=20, rings=10), xf=M(o + np.array([0, cy, 0])), node=node)
    if band_h > 0:
        c.add(c.paint, cylinder(r * 0.99, band_h, seg=20, caps=(False, False), y0=-band_h / 2),
              xf=M(o + np.array([0, cy, 0])), node=node)


def capsule(c, pos, r, cyl, ped_h=0.4, ped_r=None):
    """capsule radome (vertical cylinder with a hemispherical top) on a short pedestal."""
    o = np.asarray(pos, float)
    pr = ped_r if ped_r is not None else r * 0.6
    c.add(c.paint, cylinder(pr, ped_h, seg=14, caps=(False, True)), xf=M(o))
    c.add(c.paint, cylinder(r + 0.04, 0.08, seg=20, caps=(True, False), y0=ped_h), xf=M(o))
    c.add(c.sw('radome'), cylinder(r, cyl, seg=20, caps=(False, False), y0=ped_h + 0.08), xf=M(o))
    c.add(c.sw('radome'), sphere(r, seg=20, rings=5, hemi=True), xf=M(o + np.array([0, ped_h + 0.08 + cyl, 0])))


def pal_n(c, pos, yaw=0.0, L=2.1):
    """Pal-N navigation radar: gearbox on a short post, slotted-waveguide bar antenna."""
    o = np.asarray(pos, float)
    R = rot_y(yaw)
    c.add(c.paint, cylinder(0.12, 0.45, seg=8), xf=M(o))
    c.add(c.sw('mid'), rbox(0.45, 0.3, 0.55, r=0.08, seg=1, y0=0.45), xf=M(o, R))
    c.add(c.sw('white'), rbox(L, 0.28, 0.22, r=0.08, seg=1, y0=0.8), xf=M(o, R @ rot_y(math.pi / 2)))


def eo_drum(c, pos, yaw=0.0):
    """optronic director (MTK-201 type): short horizontal drum with glazed end on a yoke and post."""
    o = np.asarray(pos, float)
    R = rot_y(yaw)
    c.add(c.paint, cylinder(0.18, 0.5, seg=10), xf=M(o))
    c.add(c.paint, box(0.9, 0.12, 0.4, center=(0, 0, 0)), xf=M(o + np.array([0, 0.55, 0]), R))
    for s in (-1, 1):
        c.add(c.paint, box(0.08, 0.5, 0.3, center=(0, 0, 0)), xf=M(o + R @ np.array([s * 0.42, 0.8, 0]), R))
    piv = o + np.array([0, 0.92, 0])
    d = R @ np.array([0, 0, 1.0])
    c.add(c.paint, cylinder(0.33, 0.9, seg=14, caps=(True, True), y0=-0.45), xf=M(piv, R @ rot_x(math.pi / 2)))
    c.add(c.sw('glass'), disc(0.22, seg=12), xf=M(piv + d * 0.455, axis_frame(d)))


def esm_box(c, pos, normal, w=0.8, h=1.2, d=0.45):
    """ESM / ECM antenna box on a short bracket, dark radome face along `normal`."""
    n = normalize(np.asarray(normal, float))
    yaw = math.atan2(n[0], n[2])
    o = np.asarray(pos, float)
    R = rot_y(yaw)
    c.add(c.paint, box(0.25, 0.25, 0.4, center=(0, 0, 0)), xf=M(o + n * 0.2, R))
    c.add(c.paint, rbox(w, h, d, r=0.1, seg=2, bevel=0.04, y0=-h / 2), xf=M(o + n * (0.4 + d / 2), R))
    f = o + n * (0.4 + d + 0.005)
    c.add(c.sw('dark'), box(w * 0.8, h * 0.8, 0.02, center=(0, 0, 0)), xf=M(f, R))


def small_dome(c, pos, r=0.32, post=0.25):
    o = np.asarray(pos, float)
    c.add(c.paint, cylinder(r * 0.5, post, seg=8), xf=M(o))
    c.add(c.sw('radome'), sphere(r, seg=12, rings=6), xf=M(o + np.array([0, post + r * 0.85, 0])))


def register(m):
    m.alloc('array', 128, 176, paint_array)
