#!/bin/bash
# Print the scenes a shard owns, so a machine only needs to hold ITS slice of ScanNet
# (~16 GB for one shard of six, instead of the full ~94 GB).
#
#   bash shard_scenes.sh 2,3,4,5 6                 # list them
#   bash shard_scenes.sh 2,3,4,5 6 | \
#     rsync -a --files-from=- host:/data/scannet/scans/ ./scans/
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
python3 - "$1" "$2" <<'EOF'
import sys
shards = [int(x) for x in sys.argv[1].split(",")]
total = int(sys.argv[2])
sc = [l.strip() for l in open(__import__("os").path.join(
    __import__("os").path.dirname(__import__("os").path.abspath(sys.argv[0] if False else ".")),
    "scannetv2_val_312.txt")) if l.strip()] if False else None
EOF
python3 -c "
import sys
sh=[int(x) for x in '$1'.split(',')]; T=int('$2')
sc=[l.strip() for l in open('$HERE/scannetv2_val_312.txt') if l.strip()]
for s in sh:
    for x in sc[s::T]: print(x)
"
