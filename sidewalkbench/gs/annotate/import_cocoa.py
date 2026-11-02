#!/usr/bin/env python3
"""Import cocoa reconstructions as eval scenes with generated 'cocoa' routes (sidewalkbench env).

    python sidewalkbench/gs/annotate/import_cocoa.py [SRC] [--routes 10] [--min_length_m 10] [--yaw_jitter_deg 30] [--seed 0]

SRC (default /media/vivek/Extreme SSD/cocoa_reconstructions) holds one folder per capture with
gaussians.ply, mesh.ply, base_trajectory.npz (world_T_base (N, 4, 4), base +x = forward) and
report.json. Each becomes gs_assets/<capture>/ in the eval format:

  mesh.ply, base_trajectory.npz, report.json   copied
  gs.ply                    gaussians.ply (already standard 3DGS: log scales, logit opacity, wxyz rot)
  centerlines.json          one open centerline: the robot trajectory (xy)
  semantic_grid_0p1m.npz    walkable = cells whose centre is within CORRIDOR_M of the trajectory
                            (the off-sidewalk check's region; 0.1 m cells over the mesh bounds)
  walkable.npz              the same corridor on the pedestrians' 0.05 m grid
  annotation.json           ROUTES routes of mode "cocoa" (any other routes and the cones are kept)

A route starts at a uniformly random point of the trajectory, facing the trajectory heading there
(the base's +x) plus a uniform offset within +-(--yaw_jitter_deg), and ends at a uniformly random
later point at least --min_length_m further along it; its centerline is that piece of the
trajectory. A trajectory shorter than --min_length_m gets no routes. Seeded per capture
(--seed + the capture name), so rerunning gives the same routes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import zlib

import numpy as np
import trimesh
from scipy.spatial import cKDTree

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = "/media/vivek/Extreme SSD/cocoa_reconstructions"
CORRIDOR_M = 1.0        # walkable this far on either side of the trajectory
GRID_M, WALK_M = 0.1, 0.05   # semantic grid / pedestrian grid cells
SAMPLE_M = 0.02         # trajectory resampling for the corridor distance


def rnd(a) -> list:
    return np.round(np.asarray(a, dtype=float), 4).tolist()


def corridor(xy: np.ndarray, lo: np.ndarray, hi: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Bool grid (row = y, col = x) over [lo, hi]: cell centres within CORRIDOR_M of the polyline ``xy``."""
    seg = np.diff(xy, axis=0)
    pts = np.vstack([a + np.linspace(0, 1, max(2, int(np.ceil(np.linalg.norm(d) / SAMPLE_M))), endpoint=False)[:, None] * d
                     for a, d in zip(xy[:-1], seg)] + [xy[-1:]])
    w, h = np.ceil((hi - lo) / cell).astype(int)
    xs, ys = lo[0] + (np.arange(w) + 0.5) * cell, lo[1] + (np.arange(h) + 0.5) * cell
    gx, gy = np.meshgrid(xs, ys)
    d, _ = cKDTree(pts).query(np.stack([gx.ravel(), gy.ravel()], 1), distance_upper_bound=CORRIDOR_M + cell)
    return (d <= CORRIDOR_M).reshape(h, w), lo


def make_routes(T: np.ndarray, n: int, min_len: float, jitter_deg: float, rng) -> tuple[np.ndarray, list]:
    xy = T[:, :2, 3]
    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-4]  # drop repeated poses
    xy, fwd = xy[keep], T[keep, :2, 0]
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    at = lambda s: np.array([np.interp(s, cum, xy[:, 0]), np.interp(s, cum, xy[:, 1])])  # noqa: E731
    routes = []
    if cum[-1] < min_len:
        return xy, routes
    for k in range(n):
        s0 = rng.uniform(0.0, cum[-1] - min_len)
        s1 = rng.uniform(s0 + min_len, cum[-1])
        i = int(np.clip(np.searchsorted(cum, s0, side="right") - 1, 0, len(cum) - 2))
        f = fwd[i] + (s0 - cum[i]) / (cum[i + 1] - cum[i]) * (fwd[i + 1] - fwd[i])
        yaw = np.degrees(np.arctan2(f[1], f[0])) + rng.uniform(-jitter_deg, jitter_deg)
        cl = np.vstack([at(s0), xy[(cum > s0) & (cum < s1)], at(s1)])
        routes.append({"id": f"cocoa_{k + 1}", "mode": "cocoa", "centerline_id": "trajectory",
                       "start": {"xy": rnd(cl[0]), "yaw_deg": round(float((yaw + 180) % 360 - 180), 2)},
                       "goal": {"xy": rnd(cl[-1])}, "centerline_xy": rnd(cl),
                       "static_pedestrians": [], "dynamic_pedestrians": []})
    return xy, routes


def convert(src: str, n: int, min_len: float, jitter_deg: float, seed: int) -> None:
    name = os.path.basename(src.rstrip("/"))
    dst = os.path.join(REPO, "gs_assets", name)
    os.makedirs(dst, exist_ok=True)
    for a, b in (("mesh.ply", "mesh.ply"), ("gaussians.ply", "gs.ply"), ("base_trajectory.npz", "base_trajectory.npz"),
                 ("report.json", "report.json")):
        if not os.path.isfile(os.path.join(dst, b)):
            shutil.copy2(os.path.join(src, a), os.path.join(dst, b))
    T = np.load(os.path.join(dst, "base_trajectory.npz"))["world_T_base"]
    xy, routes = make_routes(T, n, min_len, jitter_deg, np.random.default_rng(seed + zlib.crc32(name.encode())))
    json.dump({"scene": name, "centerlines": [{"id": "trajectory", "closed": False, "xy": rnd(xy)}]},
              open(os.path.join(dst, "centerlines.json"), "w"))
    v = np.asarray(trimesh.load(os.path.join(dst, "mesh.ply"), process=False).vertices)
    lo, hi = np.minimum(v[:, :2].min(0), xy.min(0) - 2), np.maximum(v[:, :2].max(0), xy.max(0) + 2)
    walk, org = corridor(xy, lo, hi, GRID_M)
    np.savez_compressed(os.path.join(dst, "semantic_grid_0p1m.npz"), walkable=walk.astype(np.uint8),
                        origin_xy=org.astype(np.float32), resolution_m=np.array([GRID_M], dtype=np.float32))
    walk, org = corridor(xy, lo, hi, WALK_M)
    np.savez_compressed(os.path.join(dst, "walkable.npz"), walkable=walk, origin=org, cell=WALK_M)
    path = os.path.join(dst, "annotation.json")
    ann = json.load(open(path)) if os.path.isfile(path) else {"version": 3, "scene": name, "cones": [], "routes": []}
    ann["routes"] = [r for r in ann["routes"] if r["mode"] != "cocoa"] + routes
    json.dump(ann, open(path, "w"), indent=1)
    length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    print(f"[cocoa] {name}: trajectory {length:.1f} m -> {len(routes)} cocoa route(s)"
          + (f" ({min(len_(r) for r in routes):.1f}-{max(len_(r) for r in routes):.1f} m)" if routes
             else f" (shorter than {min_len:g} m)"), flush=True)


def len_(r) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(r["centerline_xy"]), axis=0), axis=1).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", default=SRC)
    ap.add_argument("--routes", type=int, default=10)
    ap.add_argument("--min_length_m", type=float, default=10.0)
    ap.add_argument("--yaw_jitter_deg", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for d in sorted(os.listdir(args.src)):
        if os.path.isfile(os.path.join(args.src, d, "base_trajectory.npz")):
            convert(os.path.join(args.src, d), args.routes, args.min_length_m, args.yaw_jitter_deg, args.seed)


if __name__ == "__main__":
    main()
