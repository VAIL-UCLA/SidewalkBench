#!/usr/bin/env bash
# Download every checkpoint the DualVLN navigator needs, into a folder laid out
# exactly as inference.py expects.
#
# Final layout (default target = <script_dir>/ckpts):
#   <target>/
#   ├── InternVLA-N1-DualVLN/                          (HF repo, ~8 GB)
#   │   ├── config.json
#   │   ├── model-00001-of-00004.safetensors
#   │   └── ...
#   └── depth_anything_v2_metric_hypersim_vits.pth     (~95 MB)
#
# Usage:
#   bash download_checkpoints.sh                       # default target
#   bash download_checkpoints.sh /some/other/dir       # custom target
#
# Env vars:
#   HF_TOKEN          access token (in case the repo is ever gated)
#   HF_ENDPOINT       mirror URL, e.g. https://hf-mirror.com (useful from CN)
#   FORCE=1           re-download even if files already exist
#   NO_HF_TRANSFER=1  disable the hf_transfer accelerator
#   PYTHON=/path/python  override the interpreter used

set -euo pipefail

VLN_REPO="InternRobotics/InternVLA-N1-DualVLN"
DEPTH_REPO="depth-anything/Depth-Anything-V2-Metric-Hypersim-Small"
DEPTH_FILE="depth_anything_v2_metric_hypersim_vits.pth"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-${SCRIPT_DIR}/ckpts}"
TARGET="$(realpath -m "${TARGET}")"
mkdir -p "${TARGET}"
echo ">> target: ${TARGET}"

# ---- pick python ----------------------------------------------------------
PY="${PYTHON:-$(command -v python3 || command -v python)}"
echo ">> python: ${PY}"

# ---- ensure deps ----------------------------------------------------------
"${PY}" -c "import huggingface_hub" 2>/dev/null || {
  echo ">> installing huggingface_hub"
  "${PY}" -m pip install -q "huggingface_hub>=0.24"
}
if [[ "${NO_HF_TRANSFER:-0}" != "1" ]]; then
  "${PY}" -c "import hf_transfer" 2>/dev/null || {
    echo ">> installing hf_transfer"
    "${PY}" -m pip install -q hf_transfer
  }
  export HF_HUB_ENABLE_HF_TRANSFER=1
fi

# ---- auth / mirror --------------------------------------------------------
[[ -n "${HF_TOKEN:-}" ]] && export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
[[ -n "${HF_ENDPOINT:-}" ]] && { echo ">> endpoint: ${HF_ENDPOINT}"; export HF_ENDPOINT; }

FORCE="${FORCE:-0}"

# ---- run --------------------------------------------------------------------
"${PY}" - <<PYEOF
import os, pathlib
from huggingface_hub import snapshot_download, hf_hub_download

target = pathlib.Path(r"""${TARGET}""")
force  = ${FORCE} != 0
token  = os.environ.get("HUGGING_FACE_HUB_TOKEN")

# 1) InternVLA-N1-DualVLN (HF model dir)
vln_dir = target / "InternVLA-N1-DualVLN"
if (vln_dir / "config.json").exists() and not force:
    print(f">> SKIP  {vln_dir}  (config.json present; FORCE=1 to redownload)")
else:
    print(f">> FETCH ${VLN_REPO} -> {vln_dir}")
    snapshot_download(
        repo_id="${VLN_REPO}",
        local_dir=str(vln_dir),
        local_dir_use_symlinks=False,    # real files, not symlinks into the HF cache
        resume_download=True,
        max_workers=8,
        token=token,
    )

# 2) DepthAnything backbone (single .pth)
depth_path = target / "${DEPTH_FILE}"
if depth_path.exists() and depth_path.stat().st_size > 0 and not force:
    mb = depth_path.stat().st_size / 1024 / 1024
    print(f">> SKIP  {depth_path}  ({mb:.1f} MB present; FORCE=1 to redownload)")
else:
    print(f">> FETCH ${DEPTH_REPO}/${DEPTH_FILE} -> {depth_path}")
    hf_hub_download(
        repo_id="${DEPTH_REPO}",
        filename="${DEPTH_FILE}",
        local_dir=str(target),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=token,
    )

# ---- summary ----
print()
print(">> done. final layout:")
for p in sorted(target.iterdir()):
    if p.is_symlink():
        kind = "↪"
    elif p.is_dir():
        kind = "/"
    else:
        kind = " "
    if p.is_file() or p.is_symlink():
        sz = p.stat().st_size / 1024 / 1024
        print(f"   {kind} {p.name}    ({sz:.1f} MB)")
    else:
        files = [f for f in p.rglob('*') if f.is_file()]
        n = len(files)
        gb = sum(f.stat().st_size for f in files) / 1024**3
        print(f"   {kind} {p.name}    ({n} files, {gb:.2f} GB)")
PYEOF
