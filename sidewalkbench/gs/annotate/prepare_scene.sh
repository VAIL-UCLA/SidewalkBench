#!/usr/bin/env bash
# Make a scene ready for the eval, from the coco-gs scene to the checked pedestrians.
# Each step runs only when needed:
#   1. folder      -- gs_assets/<scene>/{mesh.ply, gs.ply, semantic_grid_0p1m.npz, centerlines.json,
#                     annotation.json (empty)} from coco-gs (sidewalkbench/gs/annotate/make_scene_v2.py)
#   2. walkable    -- the pedestrians' walkability grid gs_assets/<scene>/walkable.npz
#                     (sidewalkbench/gs/annotate/bev_walkable.py, ~1 min, ~3 GB)
#   (annotate the routes, cones and pedestrians: sidewalkbench/gs/annotate/annotate.py)
#   3. cones       -- drop any cone without a settled pose onto mesh.ply (settle_cones.sh, Isaac)
#   4. humans      -- sidewalkbench/gs/annotate/check_humans.py (the dynamic pedestrians' round trip on every route)
# The pedestrians need their identities' baked avatars gs_assets/pedestrians/<identity>/{walk,stand}.npz.
#
# Usage:
#   sidewalkbench/gs/annotate/prepare_scene.sh <scene> [--rebuild_walkable]   # rebuild walkable.npz (mesh changed)
set -euo pipefail
SCENE="${1:?usage: prepare_scene.sh <scene> [--rebuild_walkable]}"
shift
cd "$(dirname "$0")/../../.."
. sidewalkbench/gs/_python.sh; sbgs_activate || exit 1; PY="$(sbgs_python)" || exit 1
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
DIR="gs_assets/$SCENE"

echo "[prepare] 1. folder: $DIR"
"$PY" sidewalkbench/gs/annotate/make_scene_v2.py "$SCENE"

if [ "${1:-}" = "--rebuild_walkable" ] || [ ! -f "$DIR/walkable.npz" ]; then
  echo "[prepare] 2. walkable: building $DIR/walkable.npz"
  CAP=(); command -v systemd-run >/dev/null && CAP=(systemd-run --user --scope -q -p MemoryMax=20G -p MemorySwapMax=0)
  "${CAP[@]}" "$PY" sidewalkbench/gs/annotate/bev_walkable.py "$SCENE"
else
  echo "[prepare] 2. walkable: $DIR/walkable.npz exists (--rebuild_walkable to redo it)"
fi

UNSETTLED=$("$PY" -c "import json, sys; print(sum('settled' not in c for c in json.load(open(sys.argv[1])).get('cones') or []))" \
  "$DIR/annotation.json")
if [ "$UNSETTLED" -gt 0 ]; then
  echo "[prepare] 3. cones: settling $UNSETTLED cone(s)"
  sidewalkbench/gs/annotate/settle_cones.sh "$SCENE"
else
  echo "[prepare] 3. cones: nothing to settle"
fi

if "$PY" -c "import json, sys; sys.exit(not json.load(open(sys.argv[1]))['routes'])" "$DIR/annotation.json"; then
  echo "[prepare] 4. humans"
  "$PY" sidewalkbench/gs/annotate/check_humans.py "$SCENE"
  echo "[prepare] $SCENE is ready"
else
  echo "[prepare] $SCENE has no routes yet: annotate it (sidewalkbench/gs/annotate/annotate.py), then rerun"
fi
