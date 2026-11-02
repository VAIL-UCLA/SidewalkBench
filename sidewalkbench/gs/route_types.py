#!/usr/bin/env python3
"""Count the short-horizon routes of each type: crosswalk, curved, straight (mutually exclusive).

    python sidewalkbench/gs/route_types.py [scene ...]   (default: every scene in gs_assets/; -v: per route)

* crosswalk -- the route runs >= CROSSWALK_MIN_M along a crosswalk centerline (``"crosswalk": true`` in
  centerlines.json: a coco-gs coverage path drawn with only two control points, curb to curb);
* curved    -- otherwise, its centerline turns >= CURVED_MIN_DEG in total (headings over STEP_M chords);
* straight  -- the rest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCENES = os.path.join(REPO, "gs_assets")
CROSSWALK_MIN_M = 2.0
CURVED_MIN_DEG = 30.0
STEP_M = 1.0


def resample(xy: np.ndarray, step: float) -> np.ndarray:
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    s = np.arange(0.0, cum[-1], step)
    return np.column_stack([np.interp(s, cum, xy[:, 0]), np.interp(s, cum, xy[:, 1])])


def dist_to(polys: list[np.ndarray], p: np.ndarray) -> np.ndarray:
    """Distance from each point p to the nearest of the polylines."""
    best = np.full(len(p), np.inf)
    for c in polys:
        a, ab = c[:-1], np.diff(c, axis=0)
        t = np.clip(((p[:, None] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
        best = np.minimum(best, np.linalg.norm(a + t[..., None] * ab - p[:, None], axis=-1).min(1))
    return best


def classify(route: dict, crosswalks: list[np.ndarray]) -> tuple[str, float, float]:
    """-> (type, metres along crosswalk centerlines, total turning in degrees)."""
    cl = np.asarray(route["centerline_xy"], dtype=float)
    p = resample(cl, 0.1)
    on_cross = float((dist_to(crosswalks, p) < 0.05).sum() * 0.1) if crosswalks else 0.0
    q = resample(cl, STEP_M)
    h = np.arctan2(*np.diff(q, axis=0)[:, ::-1].T)
    turn = float(np.degrees(np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi).sum())) if len(h) > 1 else 0.0
    kind = "crosswalk" if on_cross >= CROSSWALK_MIN_M else "curved" if turn >= CURVED_MIN_DEG else "straight"
    return kind, on_cross, turn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every route")
    args = ap.parse_args()
    scenes = args.scenes or sorted(d for d in os.listdir(SCENES) if os.path.isfile(os.path.join(SCENES, d, "annotation.json")))
    total, per_scene = {"straight": 0, "curved": 0, "crosswalk": 0}, {}
    for sc in scenes:
        routes = [r for r in json.load(open(os.path.join(SCENES, sc, "annotation.json")))["routes"] if r["mode"] == "short_horizon"]
        if not routes:
            continue
        lines = json.load(open(os.path.join(SCENES, sc, "centerlines.json")))["centerlines"]
        crosswalks = [np.asarray(c["xy"], dtype=float) for c in lines if c.get("crosswalk")]
        counts = dict.fromkeys(total, 0)
        for r in routes:
            kind, on_cross, turn = classify(r, crosswalks)
            counts[kind] += 1
            if args.verbose:
                print(f"  {sc:26s} {r['id']:9s} {kind:9s}  on crosswalk {on_cross:5.1f} m, turning {turn:6.1f} deg")
        per_scene[sc] = counts
        for k in total:
            total[k] += counts[k]
    print(f"\n{'scene':28s}{'straight':>9s}{'curved':>8s}{'crosswalk':>11s}{'total':>7s}")
    for sc, c in per_scene.items():
        print(f"{sc:28s}{c['straight']:9d}{c['curved']:8d}{c['crosswalk']:11d}{sum(c.values()):7d}")
    print(f"{'all short-horizon routes':28s}{total['straight']:9d}{total['curved']:8d}{total['crosswalk']:11d}{sum(total.values()):7d}")


if __name__ == "__main__":
    sys.exit(main())
