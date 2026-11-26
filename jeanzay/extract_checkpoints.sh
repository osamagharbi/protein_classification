#!/bin/bash
# Pull ONLY the checkpoints needed by the point-2 analysis out of the zstd run
# archive (streamed: nothing else is extracted). ~150 small run dirs, ~450 inodes.
#
#   ./jeanzay/extract_checkpoints.sh          # -> $SCRATCH/fw_ckpts/runs/...
#   ARCHIVE=... DEST=... ./jeanzay/extract_checkpoints.sh
set -uo pipefail

ARCHIVE="${ARCHIVE:-$WORK/protein_classification/runs_20260907.tar.zst}"
DEST="${DEST:-$SCRATCH/fw_ckpts}"
[ -f "$ARCHIVE" ] || { echo "archive not found: $ARCHIVE"; exit 1; }
command -v zstd >/dev/null || module load zstd
mkdir -p "$DEST" && cd "$DEST"

echo "streaming $ARCHIVE (~5 min for 100 GB) ..."
zstd -dc "$ARCHIVE" | tar -xf - --wildcards \
    'runs/shrec19_recheck/*_gvp_h256_s*/best_model.pt' \
    'runs/shrec19_recheck/*_gvp_h256_s*/final_results.json' \
    'runs/shrec_crosseval/*shrec25*_gvp_h256_s*/best_model.pt' \
    'runs/shrec_crosseval/*shrec25*_gvp_h256_s*/final_results.json' 2>&1 | grep -v "Not found in archive" || true

n19=$(find "$DEST/runs/shrec19_recheck" -name best_model.pt 2>/dev/null | wc -l)
n25=$(find "$DEST/runs/shrec_crosseval" -name best_model.pt 2>/dev/null | wc -l)
echo "checkpoints extracted: shrec19_recheck=$n19  shrec_crosseval(SHREC25)=$n25  -> $DEST/runs"
if [ "$n19" -eq 0 ]; then
  echo "!! no checkpoint matched. List what the archive holds for one run with:"
  echo "   zstd -dc $ARCHIVE | tar -tf - | grep 'shrec19_recheck.*gvp_h256_s1/' | head"
fi
