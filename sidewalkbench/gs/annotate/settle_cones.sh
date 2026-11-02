#!/usr/bin/env bash
# Settle the cones of every scene under gs_assets/ that has cones (or of the listed scenes):
# each marked cone is dropped onto mesh.ply once and its resting pose is stored in
# annotation.json (sidewalkbench/gs/annotate/settle_cones.py). Run this after adding or editing cones; the eval
# refuses scenes with unsettled cones. Exits 1 if any scene fails (nothing is written for it).
#
# Usage:
#   sidewalkbench/gs/annotate/settle_cones.sh [scene ...] [-- settle_cones.py args, e.g. --max_shift_m 0.2]
set -uo pipefail
cd "$(dirname "$0")/../../.."
. sidewalkbench/gs/_python.sh; sbgs_activate || exit 1; PY="$(sbgs_python)" || exit 1
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
SCENES=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do SCENES+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
if [ ${#SCENES[@]} -eq 0 ]; then
  for d in gs_assets/*/; do
    "$PY" -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get('cones') else 1)" \
      "$d/annotation.json" 2>/dev/null && SCENES+=("$(basename "$d")")
  done
fi
failed=()
for s in "${SCENES[@]}"; do
  echo "======================== settle $s ========================"
  "$PY" sidewalkbench/gs/annotate/settle_cones.py "$s" "$@" 2>&1 | grep -E "^\[settle\]|Error|Traceback" || true
  [ "${PIPESTATUS[0]}" -eq 0 ] || failed+=("$s")
done
echo "scenes with failures (${#failed[@]}): ${failed[*]:-none}"
[ "${#failed[@]}" -eq 0 ]
