#!/usr/bin/env bash
# Run the GS (gaussian-splat scene) evaluation for one model on one scene, writing
# PG-format outputs under experiments/<exp_name>/<model>_<scene>_gs_0_<seed>/:
#   results.pkl              (EvalTracker schema, read by sidewalkbench/gs/tools/collect_results.py)
#   evaluation_metrics.csv   (coco-gs per-route CSV, extra)
#   visualize_reset<k>.mp4   (with --save_video; one per route, PG composite)
#   log_images/trial_<i>_<outcome>/   (with --save_images)
#
# Usage:
#   sidewalkbench/gs/run_gs_eval.sh <model_type> <scene> [isaac_env.py args...]
# Examples:
#   sidewalkbench/gs/run_gs_eval.sh FlowPilot beverly_hills_crosswalk
#   sidewalkbench/gs/run_gs_eval.sh FlowPilot beverly_hills_crosswalk --save_video
#   sidewalkbench/gs/run_gs_eval.sh FlowPilot beverly_hills_crosswalk --num_envs 4 --exp_name gs_v1
# Works from any shell (interpreter resolved by _python.sh). All models on one scene:
# run_all_models.sh; all models on all scenes: run_all.sh.
set -euo pipefail
MODEL="${1:?usage: run_gs_eval.sh <model_type> <scene> [args...]}"
SCENE="${2:?usage: run_gs_eval.sh <model_type> <scene> [args...]}"
shift 2
cd "$(dirname "$0")/../.."
. sidewalkbench/gs/_python.sh; sbgs_activate || exit 1; PY="$(sbgs_python)" || exit 1
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" sidewalkbench/gs/isaac_env.py --model_type "$MODEL" --scene "$SCENE" --viz none "$@"
