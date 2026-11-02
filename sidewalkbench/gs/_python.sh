# Sourced by the sidewalkbench/gs/*.sh drivers (from the repo root) so they work from any shell.
# sbgs_activate: use the repo's uv environment (.venv, `uv sync`) if there is one, else make the
# `sidewalkbench` conda env active in this shell (real
# `conda activate`, so nvcc / CXX / CUDA flags are set -- gsplat's CUDA backend
# needs nvcc on PATH; the bare interpreter alone gives "No CUDA toolkit found").
#   SBGS_CONDA_ENV   env name to activate (default: sidewalkbench)
#   SBGS_PYTHON      skip activation and use this interpreter as-is
# sbgs_python: print the interpreter to use (after sbgs_activate).
SBGS_CONDA_ENV="${SBGS_CONDA_ENV:-sidewalkbench}"
sbgs_activate() {
  if [ -n "${SBGS_PYTHON:-}" ]; then return 0; fi
  if [ -x "$PWD/.venv/bin/python" ]; then  # uv sync's environment
    export VIRTUAL_ENV="$PWD/.venv" PATH="$PWD/.venv/bin:$PATH" SBGS_PYTHON="$PWD/.venv/bin/python"
    # gsplat's small scene-packing kernel compiles on first use (~10 s): it needs the CUDA toolkit (CUDA_HOME / nvcc)
    if [ -z "${CUDA_HOME:-}" ]; then
      if command -v nvcc >/dev/null; then export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
      elif [ -x /usr/local/cuda/bin/nvcc ]; then export CUDA_HOME=/usr/local/cuda
      else echo "sidewalkbench.gs: set CUDA_HOME to a CUDA 12.8 toolkit (gsplat compiles a small kernel on first use)" >&2; fi
    fi
    [ -n "${CUDA_HOME:-}" ] && export PATH="$CUDA_HOME/bin:$PATH"
    export MAX_JOBS="${MAX_JOBS:-2}"  # bound any compile (an unbounded parallel CUDA compile exhausts RAM)
    return 0
  fi
  if [ "${CONDA_DEFAULT_ENV:-}" = "$SBGS_CONDA_ENV" ] && [ -x "${CONDA_PREFIX:-}/bin/python" ]; then return 0; fi
  local base
  for base in "${CONDA_EXE:+${CONDA_EXE%/bin/conda}}" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda; do
    if [ -n "$base" ] && [ -f "$base/etc/profile.d/conda.sh" ] && [ -d "$base/envs/$SBGS_CONDA_ENV" ]; then
      # conda's activation hooks are not `set -u` clean
      set +u; . "$base/etc/profile.d/conda.sh"; conda activate "$SBGS_CONDA_ENV"; local rc=$?; set -u
      return $rc
    fi
  done
  echo "sidewalkbench.gs: conda env '$SBGS_CONDA_ENV' not found -- activate it yourself or set SBGS_PYTHON=/path/to/python" >&2
  return 1
}
sbgs_python() {
  if [ -n "${SBGS_PYTHON:-}" ] && [ -x "$SBGS_PYTHON" ]; then echo "$SBGS_PYTHON"; return 0; fi
  if [ -x "${CONDA_PREFIX:-}/bin/python" ]; then echo "$CONDA_PREFIX/bin/python"; return 0; fi
  echo "sidewalkbench.gs: no Python interpreter (run sbgs_activate first)" >&2
  return 1
}

# The value given to an isaac_env.py flag in an argument list (the last one wins), else a default:
#   EXP="$(sbgs_arg --exp_name gs "$@")"
sbgs_arg() {
  local flag=$1 val=$2 prev="" a; shift 2
  for a in "$@"; do [ "$prev" = "$flag" ] && val="$a"; prev="$a"; done
  echo "$val"
}
