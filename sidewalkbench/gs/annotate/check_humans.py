#!/usr/bin/env python3
"""Check that a scene's pedestrians are ready for the eval (headless, no Isaac, ~1 min).

    python sidewalkbench/gs/annotate/check_humans.py <scene>   (sidewalkbench env; prepare_scene.sh runs it)

Builds the eval's HumanCrowd (sidewalkbench/gs/humans.py: RVO2 on gs_assets/<scene>/walkable.npz within the 2 m
route corridor, the cones and the route's static pedestrians) and, on each route with dynamic
pedestrians, lets them walk with no robot nearby until each has turned around twice (a round trip).
A route passes when every walker does so within MAX_SIM_S and none stands still for STUCK_S (jammed
against an obstacle, the corridor edge or another pedestrian). The identities' baked avatars are
checked when the annotation loads. Exit code 1 if anything fails.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

from sidewalkbench.gs.annotation import load_scene_annotation  # noqa: E402
from sidewalkbench.gs.humans import HumanCrowd  # noqa: E402

DT = 0.05        # the eval's env step
MAX_SIM_S = 600.0
STUCK_S = 2.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    args = ap.parse_args()
    sdir = os.path.join(REPO, "gs_assets", args.scene)
    ann = load_scene_annotation(os.path.join(sdir, "annotation.json"), scene_name=args.scene)
    print(f"[humans] {args.scene}: {sum(len(r.static_pedestrians) for r in ann.routes)} static pedestrian(s)")
    routes = [r for r in ann.routes if r.dynamic_pedestrians]
    if not routes:
        print(f"[humans] {args.scene}: no dynamic pedestrians, nothing more to check")
        return 0
    walkable = os.path.join(sdir, "walkable.npz")
    if not os.path.isfile(walkable):
        print(f"[humans] {args.scene}: missing {walkable} (build it: sidewalkbench/gs/annotate/bev_walkable.py {args.scene})")
        return 1
    crowd = HumanCrowd(walkable, 1, DT, lambda x, y: 0.0, ann.cones, routes, device="cpu")
    failed = []
    for r in routes:
        crowd.reset(0, r, ann.route_cones(r))
        ws = crowd.walkers[0]
        turns, heading, still, worst = [0] * len(ws), [w["heading"] for w in ws], [0] * len(ws), [0] * len(ws)
        steps = 0
        while min(turns) < 2 and steps * DT < MAX_SIM_S:
            p0 = np.array(crowd.xy(0)[:len(ws)])
            crowd.step(np.array([[1e4, 1e4]]), np.zeros((1, 2)))  # robot parked far away
            steps += 1
            moved = np.linalg.norm(np.array(crowd.xy(0)[:len(ws)]) - p0, axis=1)
            for k, w in enumerate(ws):
                still[k] = still[k] + 1 if moved[k] < 0.01 else 0
                worst[k] = max(worst[k], still[k])
                if w["heading"] != heading[k]:
                    turns[k], heading[k] = turns[k] + 1, w["heading"]
        ok = min(turns) >= 2 and max(worst) * DT < STUCK_S
        failed += [] if ok else [r.route_id]
        print(f"[humans] {args.scene} {r.route_id}: {len(ws)} walker(s), round trips {'done' if min(turns) >= 2 else 'NOT done'} "
              f"in {steps * DT:.0f} s, longest standstill {max(worst) * DT:.1f} s -> {'ok' if ok else 'FAIL'}", flush=True)
    print(f"[humans] {args.scene}: {len(routes) - len(failed)}/{len(routes)} routes with dynamic pedestrians ok"
          + (f"; failed {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
