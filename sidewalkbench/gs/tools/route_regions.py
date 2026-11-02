#!/usr/bin/env python3
"""Overlays of each route's walkable region -- the annotated one (route ``region_xy``, drawn in the annotator; what
the eval's off-sidewalk check uses) or, where none is annotated yet, the automatic first guess (annotate/auto_region.py).

    python sidewalkbench/gs/tools/route_regions.py [--out experiments/route_regions] [scene ...]
        -> <out>/<scene>__<route>.png per route + <out>/index.html

A route's region = the part of the walkable surface the route itself runs on; off-sidewalk = more than 1 m outside
it (PG's out_of_region(buffer=1.0) on the route's own surface).

Overlay: the top-down splat (as in the annotator), the route region (green; title says annotated or auto), its 1 m
off-sidewalk boundary (yellow), the fallback rule for comparison (red dashed: within 1 m of the connected walkable
region AND within 2.5 m of the centerline), the centerline (blue), crosswalk centerlines (magenta), start (green
dot) / goal (red dot); right: the automatic guess's raw and smoothed walkable half-widths along the route.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys

import numpy as np
from scipy import ndimage

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from sidewalkbench.gs.annotate.annotate import bev_image  # noqa: E402
from sidewalkbench.gs.route_types import classify, resample  # noqa: E402
from sidewalkbench.gs.sampler import OFF_ROUTE_M  # noqa: E402
from sidewalkbench.gs.annotate.auto_region import (  # noqa: E402
    BUFFER_M, CAP_M, SMOOTH_M, region_polygon, route_region, walkable_grid)
import shapely  # noqa: E402

SCENES = os.path.join(REPO, "gs_assets")


def old_allowed(cl: np.ndarray, walk: np.ndarray, org: np.ndarray, res: float) -> np.ndarray:
    """Today's rule as a grid: within 1 m of the walkable region flood-filled from the centerline, and within
    2.5 m of the centerline (SemanticBounds + OFF_ROUTE_M)."""
    h, w = walk.shape
    ij = np.floor((resample(cl, 0.05) - org) / res).astype(int)
    ij = ij[(ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)]
    lab, _ = ndimage.label(walk)
    seeds = np.unique(lab[ij[:, 1], ij[:, 0]])
    conn = np.isin(lab, seeds[seeds > 0])
    line = np.zeros_like(walk)
    line[ij[:, 1], ij[:, 0]] = True
    return (ndimage.distance_transform_edt(~conn) * res <= BUFFER_M) & \
        (ndimage.distance_transform_edt(~line) * res <= OFF_ROUTE_M)


def render(scene: str, route: dict, bev, cws, walk, org, res, out_png: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cl = np.asarray(route["centerline_xy"], dtype=float)
    rr = route_region(cl, walk, org, res)
    annotated = route.get("region_xy") is not None
    poly = shapely.Polygon(route["region_xy"] if annotated else region_polygon(rr["region"], org, res))
    limit = poly.buffer(BUFFER_M)
    allowed_old = old_allowed(cl, walk, org, res)
    kind = classify({"centerline_xy": cl}, cws)[0]

    img, (x0, x1, y0, y1) = bev
    lo, hi = cl.min(0) - 8.0, cl.max(0) + 8.0
    fig, (ax, axw) = plt.subplots(1, 2, figsize=(17, 8.5), gridspec_kw={"width_ratios": [2.2, 1]})
    ax.imshow(img, origin="lower", extent=(x0, x1, y0, y1), interpolation="nearest")
    ax.fill(*poly.exterior.xy, color=(0.1, 0.9, 0.2), alpha=0.40, lw=0)
    ax.plot(*poly.exterior.xy, color=(0.1, 0.9, 0.2), lw=1.0)
    ax.plot(*limit.exterior.xy, color="yellow", lw=1.6)
    xs = org[0] + (np.arange(walk.shape[1]) + 0.5) * res
    ys = org[1] + (np.arange(walk.shape[0]) + 0.5) * res
    ax.contour(xs, ys, allowed_old.astype(float), levels=[0.5], colors="red", linewidths=1.2, linestyles="--")
    for c in cws:
        ax.plot(c[:, 0], c[:, 1], color="magenta", lw=1.0, alpha=0.8)
    ax.plot(cl[:, 0], cl[:, 1], color="#3b82f6", lw=2.0)
    ax.plot(*route["start"]["xy"], "o", color="lime", ms=9, mec="k")
    ax.plot(*route["goal"]["xy"], "o", color="red", ms=9, mec="k")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_title(f"{scene} / {route['id']}  ({route.get('mode', 'short_horizon')}, {kind}) -- "
                 f"{'ANNOTATED region' if annotated else 'AUTO region (not annotated yet)'}\n"
                 "green = route region · yellow = new limit (region + 1 m) · red dashed = current rule "
                 "(walkable + 1 m ∩ 2.5 m) · magenta = crosswalk centerlines", fontsize=10)
    s = rr["s"]
    axw.plot(s, rr["raw"][:, 0], color="#94a3b8", lw=0.8, label="left raw")
    axw.plot(s, -rr["raw"][:, 1], color="#cbd5e1", lw=0.8, label="right raw")
    axw.plot(s, rr["smooth"][:, 0], color="#16a34a", lw=2, label="left smoothed")
    axw.plot(s, -rr["smooth"][:, 1], color="#15803d", lw=2, label="right smoothed")
    for v in (OFF_ROUTE_M, -OFF_ROUTE_M):
        axw.axhline(v, color="red", ls="--", lw=1)
    for v in (CAP_M, -CAP_M):
        axw.axhline(v, color="k", ls=":", lw=0.8)
    axw.set_ylim(-(CAP_M + 3 * SMOOTH_M + 0.5), CAP_M + 3 * SMOOTH_M + 0.5)
    axw.set_xlabel("distance along route (m)")
    axw.set_ylabel("half-width (m): left +, right -")
    axw.set_title("walkable half-width along the route\n(red dashed = ±2.5 m, dotted = ±5 m cap)", fontsize=10)
    axw.legend(fontsize=8, loc="upper right")
    axw.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
    sm = rr["smooth"].sum(1)
    return {"annotated": annotated, "scene": scene, "route": route["id"], "mode": route.get("mode", "short_horizon"), "type": kind,
            "length_m": float(rr["s"][-1]), "width_median_m": float(np.median(sm)), "width_min_m": float(sm.min()),
            "width_max_m": float(sm.max()), "png": os.path.basename(out_png)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--out", default=os.path.join(REPO, "experiments", "route_regions"))
    args = ap.parse_args()
    skip = set(open(os.path.join(REPO, "sidewalkbench", "gs", "scenes_skip.txt")).read().split())
    scenes = args.scenes or sorted(
        s for s in os.listdir(SCENES) if s not in skip and all(
            os.path.isfile(os.path.join(SCENES, s, f)) for f in ("annotation.json", "mesh.ply", "gs.ply")))
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for scene in scenes:
        sdir = os.path.join(SCENES, scene)
        routes = [r for r in json.load(open(os.path.join(sdir, "annotation.json")))["routes"]
                  if r.get("mode", "short_horizon") in ("short_horizon", "long_horizon")]
        if not routes:
            continue
        cws = [np.asarray(c["xy"], dtype=float) for c in
               json.load(open(os.path.join(sdir, "centerlines.json")))["centerlines"] if c.get("crosswalk")]
        walk, org, res = walkable_grid(sdir)
        bev = bev_image(sdir)
        for r in routes:
            row = render(scene, r, bev, cws, walk, org, res, os.path.join(args.out, f"{scene}__{r['id']}.png"))
            rows.append(row)
            print(f"[regions] {scene:28s} {r['id']:9s} {row['mode']:13s} {row['type']:9s} width median "
                  f"{row['width_median_m']:.1f} m (min {row['width_min_m']:.1f}, max {row['width_max_m']:.1f})", flush=True)
    cards = "".join(
        f'<section><h2>{html.escape(r["scene"])} / {html.escape(r["route"])} <span>{"annotated" if r["annotated"] else "AUTO"} · {r["mode"].replace("_", " ")} · '
        f'{r["type"]} · {r["length_m"]:.1f} m · auto width median {r["width_median_m"]:.1f} m '
        f'(auto min {r["width_min_m"]:.1f}, max {r["width_max_m"]:.1f})</span></h2>'
        f'<img src="{html.escape(r["png"])}" loading="lazy"></section>' for r in rows)
    open(os.path.join(args.out, "index.html"), "w").write(
        '<!doctype html><html><head><meta charset="utf-8"><title>Route Regions</title><style>'
        'body{margin:0;background:#0f1115;color:#e6e8ec;font:14px system-ui,sans-serif;padding:16px 24px}'
        'h1{font-size:18px}h2{font-size:15px;margin:22px 0 6px}h2 span{color:#8a92a3;font-weight:400;font-size:13px}'
        'img{max-width:100%;border-radius:6px;background:#fff}</style></head><body>'
        f'<h1>Route regions — {sum(r["annotated"] for r in rows)}/{len(rows)} annotated</h1>{cards}</body></html>')
    print(f"[regions] {len(rows)} routes -> {os.path.join(args.out, 'index.html')}")


if __name__ == "__main__":
    main()
