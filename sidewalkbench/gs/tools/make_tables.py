#!/usr/bin/env python3
"""Paper-table rows for the GS (real-world scanned) scenes, from finished eval runs.

    python sidewalkbench/gs/tools/make_tables.py --sh experiments/<sh_exp> [...] --lh experiments/<lh_exp> [...]

Short horizon (routes.jsonl of every run): per model and route type (straight / curve / crosswalk,
sbgs route_types), SR% = mean success, SPL% = mean SPL, RC% = mean route completion, each x100.
Long horizon (results.pkl of every run = PG's RecoveryTracker summary), pooled over all scenes of a model:
  C  = (collisions + pedestrian collisions) per 100 m,  OL = off-sidewalk per 100 m,  F = freezes per 100 m,
  FC = C + OL + F,  V = total successful route length / total time (m/s)  -- "per 100 m" of the successful
  route length, as PG's RecoveryTracker.
Prints LaTeX rows (models in the paper's order, "--" where a model has no runs) and a plain summary.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from collections import defaultdict

import joblib

# paper row order: (display name, eval model_type)
MODELS = [("ViNT~\\cite{shah2023vint}", "ViNT"), ("NoMaD~\\cite{sridhar2024nomad}", "NoMaD"),
          ("MBRA~\\cite{hirose2025learning}", "MBRA"), ("CityWalker~\\cite{liu2025citywalker}", "CityWalker"),
          ("InternVLA-DualVLN~\\cite{wei2026ground}", "InternVLA_N1"),
          ("Cosmos3-Policy~\\cite{agarwal2026cosmos}", "Cosmos3"), None,
          ("MIMIC~\\cite{he2026learning}", "MIMIC"), ("S2E~\\cite{he2025seeing}", "S2E"),
          ("FlowPilot~\\cite{he2026imitation}", "FlowPilot"), ("OpenPilot~\\cite{openpilot}", "Coco_GL_SW1k")]
TYPES = [("straight", "Straight"), ("curved", "Curve"), ("crosswalk", "Crosswalk")]


def run_dirs(exps: list[str], tag: str) -> list[tuple[str, str]]:
    """(model, run dir) of every run under the experiment folders whose name contains ``tag``."""
    out = []
    for exp in exps:
        for d in sorted(glob.glob(os.path.join(exp, "*", ""))):
            name = os.path.basename(os.path.normpath(d))
            if tag in name:
                config_file = os.path.join(d, "run_config.json")
                config = json.loads(Path(config_file).read_text()) if os.path.isfile(config_file) else {}
                model = config.get("model_type") or (name.split("_")[0] if not name.startswith(("Coco_", "InternVLA_")) else
                            "_".join(name.split("_")[:3 if name.startswith("Coco_") else 2]))
                # Never pool different embodiments, goal modes, or controllers.
                out.append((model + config.get("variant", ""), d))
    return out


def display_models(keys):
    """Keep the paper order and display additional variants as separate rows."""
    rows = []
    for entry in MODELS:
        rows.append(entry)
        if entry is not None:
            label, model = entry
            for key in sorted(keys):
                if key.startswith(model + "_"):
                    suffix = key[len(model) + 1:].replace("_", r"\_")
                    rows.append((label + r" (\texttt{" + suffix + "})", key))
    return rows


def short_horizon(exps: list[str]) -> dict:
    """{model: {type: [route rows]}} from the routes.jsonl of every short-horizon run (one row per route; a
    route evaluated twice counts twice)."""
    res = defaultdict(lambda: defaultdict(list))
    for model, d in run_dirs(exps, "_gs_0_"):
        p = os.path.join(d, "routes.jsonl")
        for line in open(p) if os.path.isfile(p) else ():
            r = json.loads(line)
            if r["mode"] == "short_horizon":
                res[model][r["type"]].append(r)
    return res


def long_horizon(exps: list[str]) -> dict:
    """{model: pooled RecoveryTracker sums} over every long-horizon run's results.pkl."""
    res = defaultdict(lambda: defaultdict(float))
    for model, d in run_dirs(exps, "_gs_long_horizon_"):
        p = os.path.join(d, "results.pkl")
        if not os.path.isfile(p):
            continue
        r = joblib.load(p)
        for k in ("successes", "static_collisions", "dynamic_collisions", "oob_count", "freeze_count",
                  "total_steps", "total_route_length"):
            res[model][k] += float(r[k])
        res[model]["runs"] += 1
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sh", nargs="*", default=[], help="short-horizon experiment folders")
    ap.add_argument("--lh", nargs="*", default=[], help="long-horizon experiment folders")
    ap.add_argument("--dt", type=float, default=0.05, help="env step (s) the long-horizon total_steps count")
    args = ap.parse_args()
    sh, lh = short_horizon(args.sh), long_horizon(args.lh)
    models = display_models(set(sh) | set(lh))

    print("% Long horizon -- Real-World Scanned Scenes columns: FC & C & OL & F & V")
    for m in models:
        if m is None:
            print("\\midrule")
            continue
        name, key = m
        s = lh.get(key)
        if not s or s["total_route_length"] <= 0:
            print(f"{name} & -- & -- & -- & -- & --  % {'no runs' if not s else 'no successful route: per-100 m undefined'}")
            continue
        per = 100.0 / s["total_route_length"]
        c = (s["static_collisions"] + s["dynamic_collisions"]) * per
        ol, f = s["oob_count"] * per, s["freeze_count"] * per
        v = s["total_route_length"] / (s["total_steps"] * args.dt) if s["total_steps"] else 0.0
        print(f"{name} & {c + ol + f:.2f} & {c:.2f} & {ol:.2f} & {f:.2f} & {v:.2f}"
              f"  % {int(s['runs'])} run(s), {int(s['successes'])} successful routes, {s['total_route_length']:.0f} m")

    print("\n% Short horizon -- Straight (SR SPL RC) & Curve (SR SPL RC) & Crosswalk (SR SPL RC), %")
    for m in models:
        if m is None:
            print("\\midrule")
            continue
        name, key = m
        cells, counts = [], []
        for t, _ in TYPES:
            rs = sh.get(key, {}).get(t, [])
            counts.append(f"{t} {len(rs)}")
            if not rs:
                cells += ["--"] * 3
                continue
            n = len(rs)
            cells += [f"{100 * sum(r['outcome'] == 'success' for r in rs) / n:.1f}",
                      f"{100 * sum(r.get('spl', 0.0) for r in rs) / n:.1f}",
                      f"{100 * sum(r['completion'] for r in rs) / n:.1f}"]
        print(f"{name} & " + " & ".join(cells) + " \\\\  % routes: " + ", ".join(counts))


if __name__ == "__main__":
    main()
