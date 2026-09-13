#!/bin/bash
# One-time setup for a Grounded-SAM worker node. Idempotent: safe to re-run.
#   bash setup.sh [ENV_PY]      ENV_PY defaults to `python3`
set -euo pipefail
PY="${1:-python3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/ckpt"

echo "== 1/4 python deps =="
# transformers MUST stay <5: v5 removed BertModel.get_extended_attention_mask, which
# GroundingDINO's bertwarper calls directly, and the failure is an obscure AttributeError.
$PY -m pip install -q "transformers==4.44.2" addict yapf supervision timm imageio scipy \
                      opencv-python pycocotools

echo "== 2/4 GroundingDINO (builds a CUDA op for THIS python + GPU arch) =="
if [ ! -d "$HERE/GroundingDINO" ]; then
  git clone -q https://github.com/IDEA-Research/GroundingDINO.git "$HERE/GroundingDINO"
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
# Leave TORCH_CUDA_ARCH_LIST unset so it autodetects the local card (8.6 A6000, 8.9 4090, 12.0 5090).
( cd "$HERE/GroundingDINO" && $PY -m pip install -q -e . --no-build-isolation )

echo "== 3/4 segment-anything =="
$PY -m pip install -q git+https://github.com/facebookresearch/segment-anything.git

echo "== 4/4 checkpoints (3.1 GB, skipped if present) =="
[ -f "$HERE/ckpt/groundingdino_swint_ogc.pth" ] || wget -q --show-progress -O "$HERE/ckpt/groundingdino_swint_ogc.pth" \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
[ -f "$HERE/ckpt/sam_vit_h_4b8939.pth" ] || wget -q --show-progress -O "$HERE/ckpt/sam_vit_h_4b8939.pth" \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

$PY - <<'EOF'
import torch, groundingdino, segment_anything
from groundingdino.models.GroundingDINO import ms_deform_attn as M
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"gpus={torch.cuda.device_count()}")
assert getattr(M, "_C", None) is not None, "GroundingDINO CUDA op did not build"
print("  GroundingDINO CUDA op OK — setup complete")
EOF
