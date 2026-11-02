#!/usr/bin/env python3
"""Per-point walkability from the mesh: a 0.25 m footprint, slid over a 5 cm grid.

    systemd-run --user --scope -p MemoryMax=20G python sidewalkbench/gs/annotate/bev_walkable.py <scene>   (sidewalkbench env)

(WORKERS forked processes over strips / face chunks: ucla_sidewalk_backward ~25 s, dark ~60 s, ~3 GB.)

1. Kept faces: a face is kept iff nothing lies below its centroid -- a ray cast straight down
   from just under the centroid (Open3D / Embree) hits no other face. The other faces are
   "filtered out" (canopies, roofs, but also the sides of trunks, poles, walls, cars -- anything
   with something beneath it).
2. The ground is split into PX x PX pixels and every face is clipped to the pixels it overlaps
   (shapely, exact); heights are taken on the face itself inside the pixel (its plane at the
   clipped outline; for vertical faces, the triangle's lower/upper edge along the clipped
   footprint line). Per pixel: the lowest / highest kept height, the area its kept faces cover,
   and each filtered-out face piece's height range.
3. A point (pixel centre) is UNWALKABLE if the 0.25 m footprint (WIN_PX x WIN_PX pixels) centred on it
     * hole:     has a pixel its kept faces do not fully cover (beyond a HOLE_TOL_M2 float
                 tolerance; no kept face at all = all hole; outside the mesh = hole), or
     * gap:      highest minus lowest kept height in the footprint > GAP_M, or
     * obstructed: some filtered-out face piece in the footprint, over its height range, reaches
                 into the band [H + HEADROOM_MIN_M, H + HEADROOM_M], H = highest kept height in
                 the footprint -- poles, trunks, walls, low overhangs; pieces entirely below
                 H + HEADROOM_MIN_M or above H + HEADROOM_M (canopies) do not count,
   and WALKABLE otherwise. The footprint slides in PX steps, so the answer does not depend on
   where grid lines fall: a step narrower than half the footprint lies whole inside some
   footprint wherever it is (a fixed 0.25 m partition could split a curb across two cells).

Output, part of the scene: gs_assets/<scene>/walkable.npz {walkable, hole, gap, obstructed (H, W;
row 0 = min y, column 0 = min x; per pixel = its centre's footprint), origin (xy of pixel
(0, 0)'s corner), cell (= PX)}.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import shapely
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from sidewalkbench.gs.collision_mesh import load_collision_mesh  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PX = 0.05           # grid pixel (m)
WIN_PX = 5          # footprint = WIN_PX x WIN_PX pixels = 0.25 m (odd: centred on a pixel)
GAP_M = 0.10
HEADROOM_M = 2.0
HEADROOM_MIN_M = 0.0
HOLE_TOL_M2 = 4e-6  # float slivers only (0.16 % of a pixel)
RAY_EPS = 1e-4
VERT_NZ = 1e-3      # |unit normal z| below this (within ~0.06 deg of vertical): a vertical face
BATCH = 500_000
FACE_CHUNK = 50_000   # filtered-out faces are clipped this many at a time, one chunk per task
STRIP_PX = 50         # kept faces are processed in strips of this many pixel rows (2.5 m), one strip per task
WORKERS = min(16, os.cpu_count() or 1)  # parallel processes (forked: they share the arrays below)
_W: dict = {}         # the arrays the forked workers read (set before each pool is created)


def kept_faces(mesh: trimesh.Trimesh) -> np.ndarray:
    """Bool per face: no other face lies straight below its centroid."""
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(mesh.vertices.astype(np.float32)), o3d.core.Tensor(mesh.faces.astype(np.uint32)))
    c = mesh.triangles_center.copy()
    c[:, 2] -= RAY_EPS
    rays = np.hstack([c, np.tile([0.0, 0.0, -1.0], (len(c), 1))]).astype(np.float32)
    return np.isinf(scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy())


def _pairs(tri: np.ndarray, lo: np.ndarray):
    """(face, slice i, slice j) for every slice each face's xy bounding box touches."""
    i0 = np.floor((tri[:, :, 0].min(1) - lo[0]) / PX).astype(int)
    i1 = np.floor((tri[:, :, 0].max(1) - lo[0]) / PX).astype(int)
    j0 = np.floor((tri[:, :, 1].min(1) - lo[1]) / PX).astype(int)
    j1 = np.floor((tri[:, :, 1].max(1) - lo[1]) / PX).astype(int)
    ni, nj = i1 - i0 + 1, j1 - j0 + 1
    f = np.repeat(np.arange(len(tri)), ni * nj)
    k = np.arange(len(f)) - np.repeat(np.cumsum(ni * nj) - ni * nj, ni * nj)
    return f, i0[f] + k % ni[f], j0[f] + k // ni[f]


def _clip(geoms: np.ndarray, tri: np.ndarray, lo: np.ndarray, keep_area: bool):
    """Clip each face's xy geometry to the slices it overlaps -> (face, i, j, piece), non-empty pieces."""
    f, pi, pj = _pairs(tri, lo)
    out = [np.zeros(0, int)], [np.zeros(0, int)], [np.zeros(0, int)], [np.zeros(0, object)]  # empty-input safe
    for a in range(0, len(f), BATCH):  # batches: bounded GEOS memory
        b = slice(a, a + BATCH)
        box = shapely.box(lo[0] + pi[b] * PX, lo[1] + pj[b] * PX, lo[0] + (pi[b] + 1) * PX, lo[1] + (pj[b] + 1) * PX)
        pc = shapely.intersection(geoms[f[b]], box)
        ok = ~shapely.is_empty(pc) & ((shapely.area(pc) > 0) if keep_area else True)
        for lst, v in zip(out, (f[b][ok], pi[b][ok], pj[b][ok], pc[ok])):
            lst.append(v)
    return (np.concatenate(x) for x in out)


def _planar_heights(tri: np.ndarray, lo: np.ndarray):
    """Non-vertical faces: (i, j, piece) per clipped piece, plus every clipped outline vertex's (i, j, z) on
    the face and the index of its piece."""
    polys = shapely.polygons(np.concatenate([tri[:, :, :2], tri[:, :1, :2]], 1))
    f, pi, pj, pieces = _clip(polys, tri, lo, keep_area=True)
    xy, o = shapely.get_coordinates(pieces, return_index=True)
    t = tri[f[o]]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    z = t[:, 0, 2] - (n[:, 0] * (xy[:, 0] - t[:, 0, 0]) + n[:, 1] * (xy[:, 1] - t[:, 0, 1])) / n[:, 2]
    z = np.clip(z, t[:, :, 2].min(1), t[:, :, 2].max(1))  # guard near-vertical slivers
    return (pi, pj, pieces), (pi[o], pj[o], z, o)


def _vertical_heights(tri: np.ndarray, lo: np.ndarray):
    """Vertical faces: per (face, slice) piece, the triangle's lowest and highest point over the part
    of its footprint line inside the slice -> (i, j, zmin, zmax)."""
    p = tri[:, :, :2]
    d = np.linalg.norm(p[:, [0, 1, 2]] - p[:, [1, 2, 0]], axis=-1)
    k = d.argmax(1)  # longest footprint edge = the footprint segment
    pa, pb = p[np.arange(len(p)), k], p[np.arange(len(p)), (k + 1) % 3]
    lines = shapely.linestrings(np.stack([pa, pb], 1))
    f, pi, pj, pieces = _clip(lines, tri, lo, keep_area=False)
    u = pb[f] - pa[f]
    L = np.linalg.norm(u, axis=1)
    u = u / np.maximum(L, 1e-12)[:, None]
    s = np.einsum("pkd,pd->pk", p[f] - pa[f][:, None], u)  # (P, 3) vertex positions along the line
    zk = tri[f, :, 2]
    xy, o = shapely.get_coordinates(pieces, return_index=True)
    sp = ((xy - pa[f][o]) * u[o]).sum(1)
    sa, sb = np.full(len(f), np.inf), np.full(len(f), -np.inf)
    np.minimum.at(sa, o, sp)
    np.maximum.at(sb, o, sp)
    cand = np.concatenate([sa[:, None], sb[:, None], np.where((s >= sa[:, None]) & (s <= sb[:, None]), s, np.nan)], 1)
    lo_z, hi_z = np.full(cand.shape, np.inf), np.full(cand.shape, -np.inf)
    for a, b in ((0, 1), (1, 2), (2, 0)):  # each edge, where it spans the candidate position
        s0, s1, z0, z1 = s[:, a, None], s[:, b, None], zk[:, a, None], zk[:, b, None]
        span = (cand >= np.minimum(s0, s1) - 1e-9) & (cand <= np.maximum(s0, s1) + 1e-9)
        same = np.abs(s1 - s0) < 1e-9
        t = np.where(same, 0.0, (cand - s0) / np.where(same, 1.0, s1 - s0))
        ze = z0 + t * (z1 - z0)
        lo_z = np.where(span, np.minimum(lo_z, np.where(same, np.minimum(z0, z1), ze)), lo_z)
        hi_z = np.where(span, np.maximum(hi_z, np.where(same, np.maximum(z0, z1), ze)), hi_z)
    zmin, zmax = np.nanmin(np.where(np.isfinite(lo_z), lo_z, np.nan), 1), np.nanmax(np.where(np.isfinite(hi_z), hi_z, np.nan), 1)
    point = L < 1e-9  # degenerate footprint (a needle): its whole vertical range
    zmin[point], zmax[point] = zk[point].min(1), zk[point].max(1)
    return pi, pj, zmin, zmax


def _kept_strip(r0):
    """Kept faces over pixel rows [r0, r0 + STRIP_PX): (kmin, kmax, covered) of those rows (a face crossing
    a strip edge is clipped in both strips, each keeping only its own rows)."""
    tri, kept, kj0, kj1, lo, w = (_W[k] for k in ("tri", "kept", "kj0", "kj1", "lo", "w"))
    kmin, kmax, covered = np.full((STRIP_PX, w), np.inf), np.full((STRIP_PX, w), -np.inf), np.zeros((STRIP_PX, w))
    sel = kept[(kj1 >= r0) & (kj0 < r0 + STRIP_PX)]
    if len(sel):
        (pi, pj, pieces), (zi, zj, z, _) = _planar_heights(tri[sel], lo)
        zm = (zj >= r0) & (zj < r0 + STRIP_PX)
        np.minimum.at(kmin, (zj[zm] - r0, zi[zm]), z[zm])
        np.maximum.at(kmax, (zj[zm] - r0, zi[zm]), z[zm])
        mine = (pj >= r0) & (pj < r0 + STRIP_PX)
        pi, pj, pieces = pi[mine], pj[mine] - r0, pieces[mine]
        order = np.lexsort((pi, pj))
        sj, si, sp = pj[order], pi[order], pieces[order]
        if len(sp):
            starts = np.flatnonzero(np.r_[True, (sj[1:] != sj[:-1]) | (si[1:] != si[:-1])])
            for a, b in zip(starts, np.r_[starts[1:], len(sp)]):
                covered[sj[a], si[a]] = shapely.area(sp[a]) if b - a == 1 else shapely.area(shapely.union_all(sp[a:b]))
    return r0, kmin, kmax, covered


def _obstructing(i, j, zlo, zhi):
    """Flat indices of the footprints (centred on pixels) whose band [H + HEADROOM_MIN_M, H + HEADROOM_M]
    a filtered-out piece at pixel (i, j) with height range [zlo, zhi] reaches; each piece marks every
    footprint containing its pixel."""
    H, h, w, r = _W["H"], _W["h"], _W["w"], WIN_PX // 2
    out = []
    for dj in range(-r, r + 1):
        for di in range(-r, r + 1):
            qi, qj = i + di, j + dj
            ok = (qi >= 0) & (qi < w) & (qj >= 0) & (qj < h)
            qi, qj, a, b = qi[ok], qj[ok], zlo[ok], zhi[ok]
            Hq = H[qj, qi]
            hit = (b >= Hq + HEADROOM_MIN_M) & (a <= Hq + HEADROOM_M)
            out.append(qj[hit] * w + qi[hit])
    return np.unique(np.concatenate(out))


def _filtered_chunk(c):
    """Filtered-out (non-vertical) faces idx[c:c + FACE_CHUNK] -> (piece count, obstructed footprints)."""
    (fi, fj, fp), (_, _, z, o) = _planar_heights(_W["tri"][_W["idx"][c:c + FACE_CHUNK]], _W["lo"])
    zlo, zhi = np.full(len(fp), np.inf), np.full(len(fp), -np.inf)
    np.minimum.at(zlo, o, z)  # planar: the extremes are at the clipped outline's vertices
    np.maximum.at(zhi, o, z)
    return len(fp), _obstructing(fi, fj, zlo, zhi)


def bev(mesh: trimesh.Trimesh) -> dict:
    import multiprocessing as mp

    from scipy.ndimage import maximum_filter, minimum_filter

    t0 = time.time()
    keep = kept_faces(mesh)
    print(f"[bev] kept {keep.sum():,} / {len(keep):,} faces (nothing below their centroid) in {time.time() - t0:.0f} s")
    lo = np.floor(mesh.vertices[:, :2].min(0) / PX) * PX
    w, h = (np.ceil((mesh.vertices[:, :2].max(0) - lo) / PX).astype(int) + 1)
    vert = np.abs(mesh.face_normals[:, 2]) < VERT_NZ
    tri = mesh.triangles
    ctx = mp.get_context("fork")

    # per pixel: kept heights and coverage, strips in parallel
    kept = np.flatnonzero(keep & ~vert)
    ky = tri[kept, :, 1]
    _W.update(tri=tri, kept=kept, lo=lo, w=w, h=h,
              kj0=np.floor((ky.min(1) - lo[1]) / PX).astype(int), kj1=np.floor((ky.max(1) - lo[1]) / PX).astype(int))
    kmin, kmax, covered = np.full((h, w), np.inf), np.full((h, w), -np.inf), np.zeros((h, w))
    with ctx.Pool(WORKERS) as pool:
        for r0, smin, smax, scov in pool.imap_unordered(_kept_strip, range(0, h, STRIP_PX)):
            n = min(STRIP_PX, h - r0)
            kmin[r0:r0 + n], kmax[r0:r0 + n], covered[r0:r0 + n] = smin[:n], smax[:n], scov[:n]
    vi, vj, vzmin, vzmax = _vertical_heights(tri[keep & vert], lo)
    np.minimum.at(kmin, (vj, vi), vzmin)
    np.maximum.at(kmax, (vj, vi), vzmax)
    print(f"[bev] kept faces: heights + coverage done ({time.time() - t0:.0f} s)")

    # per footprint (centred on each pixel): hole, H, gap
    win = dict(size=WIN_PX, mode="constant")
    hole = maximum_filter(covered < PX * PX - HOLE_TOL_M2, cval=True, **win)
    H = maximum_filter(kmax, cval=-np.inf, **win)
    L = minimum_filter(kmin, cval=np.inf, **win)
    has = np.isfinite(H)
    gap = np.where(has, H - L, np.nan)

    # filtered-out pieces, chunks in parallel: each marks every footprint containing its pixel whose band it reaches
    obstructed = np.zeros(h * w, bool)
    _W.update(H=H, idx=np.flatnonzero(~keep & ~vert))
    n = 0
    with ctx.Pool(WORKERS) as pool:
        for k, flat in pool.imap_unordered(_filtered_chunk, range(0, len(_W["idx"]), FACE_CHUNK)):
            obstructed[flat] = True
            n += k
    obstructed = obstructed.reshape(h, w)
    vi, vj, vzmin, vzmax = _vertical_heights(tri[~keep & vert], lo)
    obstructed.flat[_obstructing(vi, vj, vzmin, vzmax)] = True
    obstructed &= has
    _W.clear()
    print(f"[bev] filtered-out faces: {n:,} + {len(vi):,} vertical pieces ({time.time() - t0:.0f} s)")

    walkable = ~hole & (gap <= GAP_M) & ~obstructed
    return {"walkable": walkable, "hole": hole, "gap": gap, "obstructed": obstructed, "origin": lo, "cell": PX}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    args = ap.parse_args()
    sdir = os.path.join(REPO, "gs_assets", args.scene)
    r = bev(load_collision_mesh(os.path.join(sdir, "mesh.ply")).mesh)
    np.savez_compressed(os.path.join(sdir, "walkable.npz"), **r)
    data = np.isfinite(r["gap"])  # footprints with any kept face
    nw = int(r["walkable"].sum())
    print(f"[bev] {args.scene}: walkable {nw:,} px ({nw * PX * PX:.0f} m^2); unwalkable (overlapping): "
          f"hole {int((r['hole'] & data).sum()):,}, gap > {GAP_M} m {int((data & (r['gap'] > GAP_M)).sum()):,}, "
          f"object {HEADROOM_MIN_M}-{HEADROOM_M} m above ground {int((data & r['obstructed']).sum()):,} -> {sdir}/walkable.npz")

if __name__ == "__main__":
    main()
