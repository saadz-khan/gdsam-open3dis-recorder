#!/bin/bash
# One-time setup for a Grounded-SAM worker node. Idempotent: safe to re-run.
#
#   bash setup.sh                      # creates ./venv and installs everything into it
#   bash setup.sh /path/to/python      # use an existing interpreter (e.g. a conda env)
#
# Debian/Ubuntu mark the system Python as externally managed (PEP 668) and refuse pip installs
# into it. Rather than passing --break-system-packages, which can damage the OS python, this
# builds a venv and installs there. The resolved interpreter is written to .python_path so
# run_node.sh picks it up with no further arguments.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REQ="${1:-}"
mkdir -p "$HERE/ckpt"

pip_works() {  # can this interpreter install into itself?
  "$1" -m pip install --dry-run --quiet --no-deps packaging >/dev/null 2>&1
}

echo "== 0/5 interpreter =="
if [ -n "$REQ" ] && pip_works "$REQ"; then
  PY="$REQ"; echo "  using $PY"
else
  BOOT="${REQ:-python3}"
  [ -n "$REQ" ] && echo "  $REQ is externally managed (PEP 668) or cannot pip install"
  if [ ! -x "$HERE/venv/bin/python" ]; then
    echo "  creating venv at $HERE/venv"
    "$BOOT" -m venv "$HERE/venv" 2>/dev/null || {
      echo "  ERROR: python venv module missing. Install it first:"
      echo "         sudo apt install -y python3-venv python3-full"; exit 1; }
  fi
  PY="$HERE/venv/bin/python"
  "$PY" -m pip install -q --upgrade pip setuptools wheel
  echo "  using $PY"
fi
echo "$PY" > "$HERE/.python_path"

echo "== 1/5 torch =="
if "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "  torch already present with CUDA"
else
  # cu128 wheels cover Ampere (A6000 sm_86), Ada (4090 sm_89) and Blackwell (5090 sm_120).
  "$PY" -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi

echo "== 2/5 python deps =="
# transformers MUST stay <5: v5 removed BertModel.get_extended_attention_mask, which
# GroundingDINO's bertwarper calls directly, and the failure is an obscure AttributeError.
"$PY" -m pip install -q "transformers==4.44.2" "numpy<2.3" addict yapf supervision timm \
                        imageio scipy opencv-python-headless pycocotools

echo "== 3/5 GroundingDINO (builds a CUDA op for THIS python + GPU arch) =="
[ -d "$HERE/GroundingDINO" ] || git clone -q https://github.com/IDEA-Research/GroundingDINO.git "$HERE/GroundingDINO"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
# TORCH_CUDA_ARCH_LIST is left unset so the op autodetects the local card. A prebuilt _C.so from
# another python version or GPU arch will NOT load.
( cd "$HERE/GroundingDINO" && "$PY" -m pip install -q -e . --no-build-isolation )

echo "== 4/5 segment-anything =="
"$PY" -m pip install -q git+https://github.com/facebookresearch/segment-anything.git

echo "== 5/5 checkpoints (3.1 GB, skipped if present) =="
[ -f "$HERE/ckpt/groundingdino_swint_ogc.pth" ] || wget -q --show-progress -O "$HERE/ckpt/groundingdino_swint_ogc.pth" \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
[ -f "$HERE/ckpt/sam_vit_h_4b8939.pth" ] || wget -q --show-progress -O "$HERE/ckpt/sam_vit_h_4b8939.pth" \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

"$PY" - <<'EOF'
import torch, groundingdino, segment_anything
from groundingdino.models.GroundingDINO import ms_deform_attn as M
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}  gpus={torch.cuda.device_count()}")
assert getattr(M, "_C", None) is not None, "GroundingDINO CUDA op did not build"
print("  GroundingDINO CUDA op OK")
EOF
echo
echo "setup complete. interpreter recorded in .python_path"
echo "next:  bash fetch_scannet.sh ./scans <MY_SHARDS> <TOTAL_SHARDS>"
echo "       bash run_node.sh ./scans ./out <MY_SHARDS> <TOTAL_SHARDS>"
