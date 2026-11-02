#!/usr/bin/env bash
# Evaluate one model on every short- and long-horizon route (all scenes under gs_assets/), with videos, and build
# an HTML report of all of them: experiments/<exp_name>/index.html (tools/make_report.py).
#
# Usage:
#   sidewalkbench/gs/run_model_report.sh <model> [--exp_name NAME] [isaac_env.py args, e.g. --num_envs 4]
#   (default NAME: report_<model>_<YYYY-mm-dd>; runs that already have results are skipped, --overwrite to redo)
# Scenes must be prepared first (cones settled: prepare_scene.sh / settle_cones.sh); a scene that fails is reported
# by run_all.sh and left out of the report.
set -uo pipefail
MODEL="${1:?usage: run_model_report.sh <model> [--exp_name NAME] [args...]}"
shift
EXP="report_${MODEL}_$(date +%F)"
ARGS=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--exp_name" ]; then EXP="${2:?--exp_name needs a value}"; shift 2; else ARGS+=("$1"); shift; fi
done
cd "$(dirname "$0")/../.."
. sidewalkbench/gs/_python.sh; sbgs_activate || exit 1; PY="$(sbgs_python)" || exit 1
for MODE in short_horizon long_horizon; do
  echo "======================== $MODEL: $MODE ========================"
  sidewalkbench/gs/run_all.sh --models "$MODEL" --mode "$MODE" --save_video --viz none --exp_name "$EXP" "${ARGS[@]}"
done
"$PY" sidewalkbench/gs/tools/make_report.py "experiments/$EXP"
echo "report: $PWD/experiments/$EXP/index.html"
