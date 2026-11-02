#!/usr/bin/env python3
"""Create gs_assets/<scene>/ from a coco-gs scene (sidewalkbench env).

    python sidewalkbench/gs/annotate/make_scene_v2.py [scene ...]   (default: all 11 benchmark scenes)

Copies mesh.ply, scene.ply -> gs.ply and the semantic grid (semantic_grid_0p1m.npz), writes
centerlines.json from the coco-gs coverage-path centerlines and their connections (which centerlines join, e.g.
a crosswalk and the sidewalks at its ends: routes may cross over there), and an empty annotation.json if the
scene has none yet (an existing one is kept). walkable.npz is built separately
(sidewalkbench/gs/annotate/bev_walkable.py <scene>).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
COCO = "/home/vivek/vivek/research/wanderland/coco-gs/scenes"
SCENES = ("ucla_crosswalk", "beverly_hills_crosswalk", "beverly_hills_intersection", "loop", "cars", "triangle",
          "ucla_sidewalk_backward", "midvale", "hill", "create", "dark")
CLOSED_GAP_M = 2.0  # a coverage path whose ends are this close is a closed loop


def make(scene: str) -> None:
    src, dst = os.path.join(COCO, scene), os.path.join(REPO, "gs_assets", scene)
    os.makedirs(dst, exist_ok=True)
    for a, b in (("mesh.ply", "mesh.ply"), ("scene.ply", "gs.ply"),
                 ("annotations/bev/v1_assets/semantic_grid_0p1m.npz", "semantic_grid_0p1m.npz")):
        if not os.path.isfile(os.path.join(dst, b)):
            shutil.copy2(os.path.join(src, a), os.path.join(dst, b))
    bev = yaml.safe_load(open(os.path.join(src, "annotations/bev/v1.yaml")))
    lines, ids = [], {}
    for p in bev["coverage_paths"]:
        xy = np.asarray(p["centerline_points_xy"], dtype=float)
        closed = bool(np.linalg.norm(xy[0] - xy[-1]) < CLOSED_GAP_M)
        ids[p["id"]] = f"centerline_{len(lines)}"
        lines.append({"id": ids[p["id"]], "closed": closed, "crosswalk": len(p["control_points_xy"]) == 2,  # a crosswalk:
                      "xy": np.round(xy, 4).tolist()})                                         # drawn curb to curb
    conns = [[ids[a] for a in c["coverage_path_ids"]] for c in bev.get("centerline_connections") or []]
    json.dump({"scene": scene, "centerlines": lines, "connections": conns}, open(os.path.join(dst, "centerlines.json"), "w"))
    ann = os.path.join(dst, "annotation.json")
    if not os.path.isfile(ann):
        json.dump({"version": 3, "scene": scene, "cones": [], "routes": []}, open(ann, "w"), indent=1)
    print(f"[v2] {scene}: {len(lines)} centerline(s) ({sum(c['closed'] for c in lines)} closed, "
          f"{sum(c['crosswalk'] for c in lines)} crosswalk), "
          f"{len(conns)} connection(s) -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="*", default=SCENES)
    for s in ap.parse_args().scenes:
        make(s)


if __name__ == "__main__":
    main()
