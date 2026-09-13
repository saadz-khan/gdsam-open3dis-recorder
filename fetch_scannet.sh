#!/bin/bash
# Fetch the ScanNet validation scans this recorder needs — only this machine's shard of them.
#
#   bash fetch_scannet.sh <OUT_DIR> [MY_SHARDS] [TOTAL_SHARDS] [PARALLEL]
#
#     bash fetch_scannet.sh ./scans                 # all 312 scenes  (~95 GB)
#     bash fetch_scannet.sh ./scans 2,3,4,5 6       # just those shards (~63 GB)
#     bash fetch_scannet.sh ./scans 1 6             # one shard (~16 GB)
#
# LICENSE: ScanNet is released under its own Terms of Use. By downloading you are agreeing to
# them; see http://www.scan-net.org/ and the ScanNet ToU. This script only retrieves the same
# files the official toolkit does — it grants you no rights you do not already have.
#
# Two things the bundled download-scannet.py gets wrong for this use, both found by measurement:
#
#   The .sens stream is NOT under v2/scans. That path 404s for every scan tried, while the mesh
#   and the .txt return 200 from it. The RGB-D stream is served from v1/scans (~290 MB a scan,
#   not the gigabytes one might assume).
#
#   A single stream runs at roughly 150 KB/s, so one scan is ~30 minutes and 312 would be weeks
#   serially. The transfer is latency-bound rather than bandwidth-bound, so scans are fetched
#   concurrently; lower PARALLEL if the host starts refusing connections.
#
# Resumable twice over: curl -C - continues a partial file, and a scan whose parts already match
# their advertised length is skipped entirely. Safe to re-run until it reports 0 incomplete.
set -uo pipefail
OUT="${1:?usage: fetch_scannet.sh <OUT_DIR> [MY_SHARDS] [TOTAL_SHARDS] [PARALLEL]}"
MY="${2:-}"; TOTAL="${3:-1}"; PAR="${4:-8}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE=http://kaldir.vc.in.tum.de/scannet
LIST="$HERE/scannetv2_val_312.txt"
mkdir -p "$OUT"

if [ -n "$MY" ]; then
  mapfile -t SCENES < <(python3 -c "
import sys
sh=[int(x) for x in '$MY'.split(',')]; T=int('$TOTAL')
sc=[l.strip() for l in open('$LIST') if l.strip()]
[print(x) for s in sh for x in sc[s::T]]")
else
  mapfile -t SCENES < <(grep -v '^[[:space:]]*$' "$LIST")
fi

fetch_one() {
  local s="$1" base="$2" out="$3" ok=1
  mkdir -p "$out/$s"
  # Only the five file types the recorder and the official evaluation actually read. The 2D
  # label/instance zips are the bulk of the release and nothing here opens them.
  for spec in "v1:.sens" "v2:.txt" "v2:_vh_clean_2.ply" \
              "v2:_vh_clean_2.0.010000.segs.json" "v2:.aggregation.json"; do
    local rel="${spec%%:*}" ext="${spec#*:}"
    local dest="$out/$s/$s$ext" url="$base/$rel/scans/$s/$s$ext" want
    want=$(curl -sIL --max-time 30 "$url" 2>/dev/null | grep -i '^content-length' | tail -1 | tr -dc '0-9')
    if [ -n "$want" ] && [ -f "$dest" ] && [ "$(stat -c%s "$dest" 2>/dev/null)" = "$want" ]; then
      continue
    fi
    curl -sL -C - --retry 5 --retry-delay 5 --retry-connrefused \
         --connect-timeout 30 -o "$dest" "$url" 2>/dev/null
    if [ -n "$want" ] && [ "$(stat -c%s "$dest" 2>/dev/null)" != "$want" ]; then ok=0; fi
  done
  if [ "$ok" = 1 ]; then echo "  OK   $s ($(du -sh "$out/$s" | cut -f1))"
  else                   echo "  PART $s -- rerun to resume"; fi
}
export -f fetch_one

echo "fetching ${#SCENES[@]} scenes, $PAR at a time, into $OUT"
printf '%s\n' "${SCENES[@]}" | xargs -P "$PAR" -I{} bash -c 'fetch_one "$@"' _ {} "$BASE" "$OUT"

echo
echo "scan dirs      : $(ls -d "$OUT"/*/ 2>/dev/null | wc -l) / ${#SCENES[@]}"
echo "complete .sens : $(find "$OUT" -name '*.sens' -size +1M | wc -l)"
echo "total size     : $(du -sh "$OUT" 2>/dev/null | cut -f1)"
