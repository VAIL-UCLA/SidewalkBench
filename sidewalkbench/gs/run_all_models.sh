#!/usr/bin/env bash
# Run the GS eval for the default models (sidewalkbench/gs/models.txt) on one scene (one Kit process
# per model, sequential), then collect PG-format results into a CSV.
#
# Usage:
#   sidewalkbench/gs/run_all_models.sh <scene> [--models A,B,C] [isaac_env.py args...]
# Examples:
#   sidewalkbench/gs/run_all_models.sh beverly_hills_crosswalk
#   sidewalkbench/gs/run_all_models.sh beverly_hills_crosswalk --models FlowPilot,NoMaD --exp_name gs_v1
# Default models: sidewalkbench/gs/models.txt; every runnable one: `python sidewalkbench/gs/model_spec.py --runnable`.
# For every scene at once use sidewalkbench/gs/run_all.sh.
# A model whose run fails is reported and the batch continues; runs whose
# results.pkl already exists are skipped by isaac_env.py (--overwrite to redo).
# With --mode long_horizon / cocoa, a scene without routes of that mode is skipped.
set -uo pipefail
SCENE="${1:?usage: run_all_models.sh <scene> [--models A,B] [args...]}"
shift
MODELS=""
if [ "${1:-}" = "--models" ]; then MODELS="${2:?--models needs a comma-separated list}"; shift 2; fi
cd "$(dirname "$0")/../.."
. sidewalkbench/gs/_python.sh; sbgs_activate || exit 1; PY="$(sbgs_python)" || exit 1
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
EXP="$("$PY" -m sidewalkbench.gs.config --print_field exp_name "$@")" || exit 1
MODE="$("$PY" -m sidewalkbench.gs.config --print_field mode "$@")" || exit 1
if ! "$PY" - "$SCENE" "$MODE" <<'PYEOF'
import json, sys
sys.exit(not any(r.get("mode") == sys.argv[2] for r in json.load(open(f"gs_assets/{sys.argv[1]}/annotation.json"))["routes"]))
PYEOF
then
  echo "[run_all_models] $SCENE has no $MODE routes: skipped"; exit 0
fi
if [ -z "$MODELS" ]; then MODELS="$(grep -v '^\s*#' sidewalkbench/gs/models.txt | xargs)"; fi
ok=(); failed=()
for m in ${MODELS//,/ }; do
  echo; echo "################ $m / $SCENE (exp_name=$EXP) ################"
  # Kit's fast shutdown can exit 0 even after a crash, so also require the output file.
  run_dir="$("$PY" -m sidewalkbench.gs.config --model_type "$m" --scene "$SCENE" "$@")" || exit 1
  out="$run_dir/results.pkl"
  if sidewalkbench/gs/run_gs_eval.sh "$m" "$SCENE" "$@" && [ -f "$out" ]; then ok+=("$m")
  else failed+=("$m"); echo "[run_all_models] FAILED: $m / $SCENE (no $out)" >&2; fi
done
echo; echo "################ batch summary ################"
echo "ok     (${#ok[@]}): ${ok[*]:-}"
echo "failed (${#failed[@]}): ${failed[*]:-}"
# (run_all.sh collects once at the end instead: SBGS_NO_COLLECT=1)
if [ -z "${SBGS_NO_COLLECT:-}" ] && [ "$MODE" != "long_horizon" ] && ls "experiments/$EXP"/*/results.pkl >/dev/null 2>&1; then
  "$PY" sidewalkbench/gs/tools/collect_results.py "experiments/$EXP" || true
fi
[ "${#failed[@]}" -eq 0 ]
