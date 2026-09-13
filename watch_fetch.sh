#!/bin/bash
# Live progress for a running fetch_scannet.sh.
#
#   bash watch_fetch.sh <SCANS_DIR> [MY_SHARDS] [TOTAL_SHARDS] [INTERVAL_SEC]
#
#     bash watch_fetch.sh ./scans                # watching a full 312-scene fetch
#     bash watch_fetch.sh ./scans 5,6,7 10       # watching one machine's shards
#
# fetch_scannet.sh only prints a line when an ENTIRE scene finishes, and a single .sens can be
# 3 GB, so at a slow link it looks stalled for a long time while it is in fact working. This
# samples the directory instead, so you see movement within seconds.
set -uo pipefail
OUT="${1:?usage: watch_fetch.sh <SCANS_DIR> [MY_SHARDS] [TOTAL_SHARDS] [INTERVAL]}"
MY="${2:-}"; TOTAL="${3:-1}"; IV="${4:-15}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ -n "$MY" ]; then
  EXPECT=$(python3 -c "
sh=[int(x) for x in '$MY'.split(',')]; T=int('$TOTAL')
sc=[l.strip() for l in open('$HERE/scannetv2_val_312.txt') if l.strip()]
print(sum(len(sc[s::T]) for s in sh))")
else
  EXPECT=$(grep -vc '^[[:space:]]*$' "$HERE/scannetv2_val_312.txt")
fi
# Measured over all 312 validation scenes: 714 MB per scene on average, 99% of it the .sens.
AVG_MB=714

prev=0; prev_t=0
while true; do
  now=$(date +%s)
  bytes=$(du -sb "$OUT" 2>/dev/null | cut -f1); bytes=${bytes:-0}
  dirs=$(ls -d "$OUT"/*/ 2>/dev/null | wc -l)
  # a scene counts as done only when its .sens is present and not obviously truncated
  done_n=$(find "$OUT" -name '*.sens' -size +1M 2>/dev/null | wc -l)
  curls=$(pgrep -c curl 2>/dev/null); curls=${curls:-0}; curls=${curls%%$'\n'*}

  if [ "$prev_t" -gt 0 ]; then
    dt=$(( now - prev_t )); [ "$dt" -lt 1 ] && dt=1
    rate=$(( (bytes - prev) / dt ))
    total_mb=$(( EXPECT * AVG_MB ))
    left_mb=$(( total_mb - bytes/1048576 ))
    if [ "$rate" -gt 0 ] && [ "$left_mb" -gt 0 ]; then
      eta=$(( left_mb * 1048576 / rate ))
      etas=$(printf '%dh%02dm' $((eta/3600)) $(((eta%3600)/60)))
    else etas="--"; fi
    printf '%s  %4d/%-4d scenes  %7.1f GB  %8.2f MB/s  %d curl  ETA %s\n' \
      "$(date +%H:%M:%S)" "$done_n" "$EXPECT" \
      "$(echo "$bytes" | awk '{printf "%.1f", $1/1e9}')" \
      "$(echo "$rate" | awk '{printf "%.2f", $1/1048576}')" "$curls" "$etas"
  else
    printf '%s  %4d/%-4d scenes  %7.1f GB  (sampling...)  %3s curl\n' \
      "$(date +%H:%M:%S)" "$done_n" "$EXPECT" \
      "$(awk -v b="$bytes" 'BEGIN{printf "%.1f", b/1e9}')" "$curls"
  fi
  [ "$done_n" -ge "$EXPECT" ] && { echo "all $EXPECT scenes have a .sens -- rerun fetch_scannet.sh once to verify sizes"; break; }
  prev=$bytes; prev_t=$now
  sleep "$IV"
done
