"""Per-vertex ambient occlusion (stored as glTF COLOR_0) using Embree via trimesh."""
import math
import numpy as np
import trimesh

from meshkit import normalize


def subdivide_soup(P, N, UV, max_len, max_iter=12, min_area=0.0):
    """Triangle soup (n,3,3)/(n,3,3)/(n,3,2): split longest edges until all edges <= max_len."""
    out_P, out_N, out_UV = [], [], []
    for _ in range(max_iter):
        e = np.stack([np.linalg.norm(P[:, 1] - P[:, 0], axis=1), np.linalg.norm(P[:, 2] - P[:, 1], axis=1),
                      np.linalg.norm(P[:, 0] - P[:, 2], axis=1)], axis=1)
        longest = e.max(axis=1)
        area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
        done = (longest <= max_len) | (area <= min_area)
        out_P.append(P[done]); out_N.append(N[done]); out_UV.append(UV[done])
        if np.all(done):
            P = P[:0]; N = N[:0]; UV = UV[:0]
            break
        P, N, UV, e = P[~done], N[~done], UV[~done], e[~done]
        k = np.argmax(e, axis=1)                   # edge k: (k, k+1)
        idx = np.arange(len(P))
        i0 = k; i1 = (k + 1) % 3; i2 = (k + 2) % 3
        a, b, c = P[idx, i0], P[idx, i1], P[idx, i2]
        na, nb, nc = N[idx, i0], N[idx, i1], N[idx, i2]
        ua, ub, uc = UV[idx, i0], UV[idx, i1], UV[idx, i2]
        m = 0.5 * (a + b); nm = normalize(0.5 * (na + nb)); um = 0.5 * (ua + ub)
        # (a, m, c) and (m, b, c) keep the original winding
        P = np.concatenate([np.stack([a, m, c], 1), np.stack([m, b, c], 1)])
        N = np.concatenate([np.stack([na, nm, nc], 1), np.stack([nm, nb, nc], 1)])
        UV = np.concatenate([np.stack([ua, um, uc], 1), np.stack([um, ub, uc], 1)])
    if len(P):
        out_P.append(P); out_N.append(N); out_UV.append(UV)
    return np.concatenate(out_P), np.concatenate(out_N), np.concatenate(out_UV)


def hemisphere_dirs(k, seed=1):
    """stratified cosine-weighted directions around +z"""
    rng = np.random.default_rng(seed)
    n = int(math.ceil(math.sqrt(k)))
    dirs = []
    for i in range(n):
        for j in range(n):
            u1 = (i + rng.random()) / n
            u2 = (j + rng.random()) / n
            r = math.sqrt(u1)
            th = 2 * math.pi * u2
            dirs.append((r * math.cos(th), r * math.sin(th), math.sqrt(max(0.0, 1 - u1))))
    return np.array(dirs[:k])


def compute_ao(occluder_P, occluder_I, pts, nrms, rays=48, max_dist=5.0, eps=0.03, seed=3):
    """returns ambient visibility in [0,1] per point (1 = fully open)."""
    mesh = trimesh.Trimesh(vertices=occluder_P, faces=occluder_I, process=False)
    try:
        inter = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    except Exception:
        inter = mesh.ray
    base = hemisphere_dirs(rays, seed)
    rng = np.random.default_rng(seed)
    n = len(pts)
    vis = np.ones(n)
    chunk = 4000
    for s in range(0, n, chunk):
        p = pts[s:s + chunk]; nn = normalize(nrms[s:s + chunk])
        m = len(p)
        # orthonormal frame per point with a random twist
        helper = np.where(np.abs(nn[:, 1:2]) < 0.9, np.array([[0, 1.0, 0]]), np.array([[1.0, 0, 0]]))
        t1 = normalize(np.cross(helper, nn)); t2 = np.cross(nn, t1)
        tw = rng.random(m) * 2 * math.pi
        c, sn = np.cos(tw)[:, None], np.sin(tw)[:, None]
        t1r = c * t1 + sn * t2; t2r = -sn * t1 + c * t2
        D = (base[None, :, 0:1] * t1r[:, None, :] + base[None, :, 1:2] * t2r[:, None, :] +
             base[None, :, 2:3] * nn[:, None, :])                         # (m, rays, 3)
        O = np.repeat(p + nn * eps, rays, axis=0)
        Dr = D.reshape(-1, 3)
        locs, ray_idx, tri_idx = inter.intersects_location(O, Dr, multiple_hits=False)
        occl = np.zeros(m * rays)
        if len(ray_idx):
            dist = np.linalg.norm(locs - O[ray_idx], axis=1)
            w = np.clip(1.0 - dist / max_dist, 0.0, 1.0) ** 0.7
            np.maximum.at(occl, ray_idx, w)
        vis[s:s + chunk] = 1.0 - occl.reshape(m, rays).mean(axis=1)
    return vis
