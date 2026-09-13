#!/bin/bash
# Run this machine's shards of the 312-scene job.
#
#   bash run_node.sh <SCANS_DIR> <OUT_DIR> <MY_SHARDS> <TOTAL_SHARDS> [PY] [CHUNK] [GD_BATCH]
#
# TOTAL_SHARDS is a shared constant across the whole fleet; MY_SHARDS is the comma-separated list
# this machine owns. Shards are deliberately decoupled from GPU count so an uneven fleet works:
# use MORE shards than GPUs and hand the faster cards more of them, so every machine finishes at
# roughly the same time instead of the fleet waiting on the slowest box.
#
# At most one worker runs per GPU at a time -- a single worker already saturates a modern card at
# this batch size, so extra concurrent processes only trade memory for nothing. A GPU given several
# shards works through them IN SEQUENCE.
#
#   10 shards over 5090 (fast) + A6000 (mid) + 4090 laptop (slow):
#     5090        : bash run_node.sh ./scans ./out 0,1,2,3,4 10
#     A6000       : bash run_node.sh ./scans ./out 5,6,7     10
#     4090 laptop : bash run_node.sh ./scans ./out 8,9       10   "" 1 6
set -uo pipefail
SCANS="$1"; OUT="$2"; MY="$3"; TOTAL="$4"; CHUNK="${6:-1}"; GDB="${7:-10}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${5:-}"; [ -z "$PY" ] && PY="$( [ -f "$HERE/.python_path" ] && cat "$HERE/.python_path" || echo python3 )"
IFS=',' read -ra SHARDS <<< "$MY"
NGPU=$($PY -c "import torch;print(torch.cuda.device_count())")
[ "$NGPU" -lt 1 ] && { echo "no CUDA GPU visible"; exit 1; }
mkdir -p "$OUT" "$HERE/logs"

T=$(( $(nproc) / (NGPU < ${#SHARDS[@]} ? NGPU : ${#SHARDS[@]}) / 2 )); [ "$T" -lt 1 ] && T=1
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "this machine: ${#SHARDS[@]} shard(s) of $TOTAL across $NGPU GPU(s), chunk=$CHUNK, batch=$GDB"

# Deal the shards round-robin to GPUs, then each GPU runs its queue sequentially.
run_queue() {
  local gpu="$1"; shift
  for S in "$@"; do
    echo "  GPU $gpu -> shard $S/$TOTAL   (logs/shard_${S}.log)"
    CUDA_VISIBLE_DEVICES=$gpu "$PY" "$HERE/record_gdsam.py" \
      --scans "$SCANS" --out "$OUT" --scenes "$HERE/scannetv2_val_312.txt" \
      --vocab "$HERE/scannet200_classes.txt" \
      --gd-config "$HERE/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
      --gd-ckpt "$HERE/ckpt/groundingdino_swint_ogc.pth" \
      --sam-ckpt "$HERE/ckpt/sam_vit_h_4b8939.pth" \
      --chunk "$CHUNK" --max-frames 200 --stride 10 --gd-batch "$GDB" \
      --shard "$S" --nshards "$TOTAL" \
      >> "$HERE/logs/shard_${S}.log" 2>&1
  done
}

for g in $(seq 0 $((NGPU-1))); do
  q=(); i=0
  for S in "${SHARDS[@]}"; do [ $(( i % NGPU )) -eq "$g" ] && q+=("$S"); i=$((i+1)); done
  [ ${#q[@]} -gt 0 ] && run_queue "$g" "${q[@]}" &
done
echo
echo "progress:   ls $OUT/*.pkl | wc -l        # target 312 across the fleet"
wait
echo "this machine's shards are done."
