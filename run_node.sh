#!/bin/bash
# Run this machine's shards of the 312-scene job, one process per GPU.
#
#   bash run_node.sh <SCANS_DIR> <OUT_DIR> <MY_SHARDS> <TOTAL_SHARDS> [PY] [CHUNK]
#
# TOTAL_SHARDS is the number of GPUs across ALL machines; MY_SHARDS is the comma-separated list
# this machine owns. Using an explicit list (rather than deriving it from the local GPU count)
# is what makes a heterogeneous fleet work: machines have different numbers of GPUs, and a faster
# card can simply be given more shards.
#
#   6 GPUs total = 1x5090 + 1xA6000 + 4x4090
#     5090  : bash run_node.sh /data/scannet/scans ./out 0     6
#     A6000 : bash run_node.sh /data/scannet/scans ./out 1     6
#     4090s : bash run_node.sh /data/scannet/scans ./out 2,3,4,5 6
#
# Shards are disjoint and each scene file is written atomically, so no machine coordinates with
# any other. Merge at the end with:  rsync -a nodeN:out/ ./out/
set -uo pipefail
SCANS="$1"; OUT="$2"; MY="$3"; TOTAL="$4"; CHUNK="${6:-1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
# default to whatever setup.sh resolved (its venv, or the interpreter you gave it)
PY="${5:-$( [ -f "$HERE/.python_path" ] && cat "$HERE/.python_path" || echo python3 )}"
IFS=',' read -ra SHARDS <<< "$MY"
NGPU=$($PY -c "import torch;print(torch.cuda.device_count())")
[ "$NGPU" -lt 1 ] && { echo "no CUDA GPU visible"; exit 1; }
mkdir -p "$OUT" "$HERE/logs"

# One worker per GPU: a single worker already saturates a modern card at this batch size, so extra
# processes per GPU buy nothing and risk OOM. Threads are split so the .sens decode of one worker
# overlaps the GPU work of another instead of oversubscribing the box.
T=$(( $(nproc) / ${#SHARDS[@]} / 2 )); [ "$T" -lt 1 ] && T=1
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "this machine: ${#SHARDS[@]} shard(s) of $TOTAL, $NGPU GPU(s), chunk=$CHUNK, $T threads/proc"
i=0
for S in "${SHARDS[@]}"; do
  G=$(( i % NGPU ))
  CUDA_VISIBLE_DEVICES=$G $PY "$HERE/record_gdsam.py" \
    --scans "$SCANS" --out "$OUT" --scenes "$HERE/scannetv2_val_312.txt" \
    --vocab "$HERE/scannet200_classes.txt" \
    --gd-config "$HERE/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
    --gd-ckpt "$HERE/ckpt/groundingdino_swint_ogc.pth" \
    --sam-ckpt "$HERE/ckpt/sam_vit_h_4b8939.pth" \
    --chunk "$CHUNK" --max-frames 200 --stride 10 --gd-batch 10 \
    --shard "$S" --nshards "$TOTAL" \
    > "$HERE/logs/shard_${S}.log" 2>&1 &
  echo "  GPU $G -> shard $S/$TOTAL   (logs/shard_${S}.log)"
  i=$((i+1))
done
echo
echo "progress:   ls $OUT/*.pkl | wc -l        # target 312"
echo "per shard:  tail -f $HERE/logs/shard_${SHARDS[0]}.log"
wait
echo "this machine's shards are done."
