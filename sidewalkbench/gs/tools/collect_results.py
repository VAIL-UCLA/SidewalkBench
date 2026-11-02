import csv
import json
import os
import pickle
import sys
from pathlib import Path
from glob import glob

"""PG-format results of an experiment folder -> <exp>/results.csv (one row per run's results.pkl):
    python sidewalkbench/gs/tools/collect_results.py experiments/<exp_name>
"""
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from sidewalkbench.gs.model_spec import ZOO_CLASSES  # noqa: E402  the PG release zoo (sidewalkbench/models)
MODEL_TYPES = sorted(ZOO_CLASSES)

SCENARIOS = [
    "frontal_approach",
    "pedestrian_obstruction",
    "robot_overtaking",
    "intersection",
    "empty",
    "obstacle_obstruction",
]

import sys
results_dir = sys.argv[1]  
pkl_files = sorted(glob(os.path.join(results_dir, "*/results.pkl")))

rows = []
for pkl_path in pkl_files:
    name = os.path.basename(os.path.dirname(pkl_path))

    # Extract model_type by matching known prefixes
    model_type = None
    for mt in sorted(MODEL_TYPES, key=len, reverse=True):
        if name.startswith(mt + "_"):
            model_type = mt
            break
    if model_type is None:
        print(f"Skipping unknown model: {name}")
        continue

    remainder = name[len(model_type) + 1:]  # e.g. "sidewalk_frontal_approach_left"

    # Extract scenario by matching known scenarios
    terrain = None
    scenario = None
    direction = ""
    for sc in sorted(SCENARIOS, key=len, reverse=True):
        idx = remainder.find(sc)
        if idx != -1:
            terrain = remainder[:idx - 1]  # everything before "_scenario"
            scenario = sc
            after = remainder[idx + len(sc):]
            if after.startswith("_"):
                direction = after[1:]
            break

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    config_path = os.path.join(os.path.dirname(pkl_path), "run_config.json")
    config = json.loads(Path(config_path).read_text()) if os.path.isfile(config_path) else {}

    rows.append({
        "model_type": model_type,
        "terrain": terrain,
        "scenario": scenario,
        "direction": direction,
        "run": name,
        "robot_type": config.get("robot_type", "coco"),
        "use_goal": config.get("use_goal", False),
        "controller": config.get("resolved_controller", {"type": "pd"}),
        **data,
    })

out_path = os.path.join(results_dir, "results.csv")
metric_keys = sorted(k for k in rows[0] if k not in ("model_type", "terrain", "scenario", "direction"))
fields = ["model_type", "terrain", "scenario", "direction"] + metric_keys
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f"Saved {len(rows)} rows to {out_path}")
