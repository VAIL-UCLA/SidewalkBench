#!/usr/bin/env bash
# Full evaluation: the default models (sidewalkbench/gs/models.txt) on every scene under gs_assets/
# (a scene = a folder with mesh.ply, gs.ply, annotation.json; its routes are
# the scenarios). One Kit process per (model, scene), sequential; runs whose
# results.pkl already exists are skipped (--overwrite to redo).
#
# Usage:
#   sidewalkbench/gs/run_all.sh [--scenes A,B] [--models X,Y] [isaac_env.py args...]
# Examples:
#   sidewalkbench/gs/run_all.sh --exp_name full_v1
#   sidewalkbench/gs/run_all.sh --scenes residential_loop,residential_triangle --models FlowPilot
#   sidewalkbench/gs/run_all.sh --models FlowPilot --mode long_horizon --save_video --exp_name lh_v1
#     (long horizon: only scenes with long_horizon routes run)
#   sidewalkbench/gs/run_all.sh --models FlowPilot --mode cocoa --save_video --exp_name cocoa_v1
#     (the generated cocoa routes of the imported cocoa captures, sidewalkbench/gs/annotate/import_cocoa.py)
# Output: experiments/<exp_name>/<model>_<scene>_gs_0_<seed>/ (PG format) and
#         experiments/<exp_name>/results.csv (sidewalkbench/gs/tools/collect_results.py);
#         long horizon: <model>_<scene>_gs_long_horizon_0_<seed>/ with PG's recovery summary.
set -uo pipefail
cd "$(dirname "$0")/../.."
. sidewalkbench/gs/_python.sh; sbgs_activate || exit 1; PY="$(sbgs_python)" || exit 1
SCENES=""; MODELS_ARG=()
while [ $# -gt 0 ]; do
  case "$1" in
    --scenes) SCENES="${2:?--scenes needs a comma-separated list}"; shift 2 ;;
    --models) MODELS_ARG=(--models "${2:?--models needs a comma-separated list}"); shift 2 ;;
    *) break ;;
  esac
done
if [ -z "$SCENES" ]; then
  # every scene with annotation.json + mesh.ply + gs.ply, except those in sidewalkbench/gs/scenes_skip.txt
  SKIP="$(grep -v '^\s*#' sidewalkbench/gs/scenes_skip.txt 2>/dev/null | xargs)"
  SCENES=$(for d in gs_assets/*/; do s="$(basename "$d")"; [ -f "$d/annotation.json" ] && [ -f "$d/mesh.ply" ] && [ -f "$d/gs.ply" ] \
           && [[ " $SKIP " != *" $s "* ]] && echo "$s"; done | paste -sd, -)
fi
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
EXP="$("$PY" -m sidewalkbench.gs.config --print_field exp_name "$@")" || exit 1
MODE="$("$PY" -m sidewalkbench.gs.config --print_field mode "$@")" || exit 1
echo "scenes:   ${SCENES}"
echo "exp_name: $EXP"
failed=()
for s in ${SCENES//,/ }; do
  echo; echo "======================== SCENE $s ========================"
  SBGS_NO_COLLECT=1 sidewalkbench/gs/run_all_models.sh "$s" "${MODELS_ARG[@]}" "$@" || failed+=("$s")
done
echo; echo "======================== full-eval summary ========================"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
# collect_results.py reads the short-horizon EvalTracker results; long horizon = per-run recovery summaries
if [ "$MODE" != "long_horizon" ] && ls "experiments/$EXP"/*/results.pkl >/dev/null 2>&1; then
  "$PY" sidewalkbench/gs/tools/collect_results.py "experiments/$EXP" || true
fi
echo "scenes with failures (${#failed[@]}): ${failed[*]:-none}"
[ "${#failed[@]}" -eq 0 ]
