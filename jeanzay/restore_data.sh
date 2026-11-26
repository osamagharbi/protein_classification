#!/bin/bash
# Put every graph folder the future-work jobs read under one data root.
#
#   ./jeanzay/restore_data.sh            # -> $SCRATCH/fw_data (the default data root of all fw scripts once it exists)
#   DATA_ROOT=$WORK ./jeanzay/restore_data.sh    # restore into $WORK instead (costs ~60k inodes there)
#
# For each folder, in order:
#   1. already in the data root with enough graphs  -> keep;
#   2. found as a folder in $WORK, $SCRATCH or $STORE -> symlink it (no copy, no inodes);
#   3. found as an archive (<name>.tar, .tar.gz, .tgz, .tar.zst) -> extract it. The archive's
#      internal layout is detected (top folder, nested path, or bare train_set/test_set).
# Nothing outside the data root is modified. Safe to rerun.
set -uo pipefail

DEST="${DATA_ROOT:-$SCRATCH/fw_data}"
SEARCH="${SEARCH:-$WORK $SCRATCH ${STORE:-}}"
mkdir -p "$DEST"
echo "data root: $DEST"

npt() { find -L "$1" -name '*.pt' 2>/dev/null | wc -l; }

find_dir() {   # first folder called $1 with at least $2 graphs, outside DEST
  local d=$1 min=$2 c
  for c in $(find $SEARCH -maxdepth 4 -type d -name "$1" -not -path "$DEST/*" 2>/dev/null); do
    [ "$(npt "$c")" -ge "$min" ] && { echo "$c"; return 0; }
  done
  return 1
}

find_tar() {
  find $SEARCH -maxdepth 4 -type f \( -name "$1.tar" -o -name "$1.tar.gz" -o -name "$1.tgz" \
       -o -name "$1.tar.zst" \) -not -path "$DEST/*" 2>/dev/null | head -1
}

extract() {    # extract <archive> <name>
  local tb=$1 d=$2 tmp="$DEST/.restore_$2_$$" src=""
  mkdir -p "$tmp"
  echo "    extracting $tb"
  case "$tb" in
    *.zst) command -v zstd >/dev/null || module load zstd; zstd -dc "$tb" | tar -xf - -C "$tmp" ;;
    *)     tar -xf "$tb" -C "$tmp" ;;
  esac || { echo "    !! extraction failed (partial files left in $tmp)"; return 1; }
  if [ -d "$tmp/train_set" ] || [ -d "$tmp/test_set" ]; then src="$tmp"
  else src=$(find "$tmp" -type d -name "$d" -prune 2>/dev/null | head -1)
       [ -n "$src" ] || { src=$(find "$tmp" -type d -name train_set 2>/dev/null | head -1); src=${src%/train_set}; }
  fi
  [ -n "$src" ] || { echo "    !! no train_set/test_set inside; left in $tmp"; return 1; }
  if [ "$src" = "$tmp" ]; then mv "$tmp" "$DEST/$d"
  else mv "$src" "$DEST/$d" && rm -rf "$tmp"; fi
}

missing=()
restore() {    # restore <name> <min graphs> [optional]
  local d=$1 min=$2 opt=${3:-} n src tb
  if [ -e "$DEST/$d" ]; then
    n=$(npt "$DEST/$d")
    if [ "$n" -ge "$min" ]; then printf "  OK       %-50s %6d graphs\n" "$d" "$n"; return; fi
    echo "  !! $DEST/$d has only $n graphs (expected >= $min); move it away and rerun"; missing+=("$d"); return
  fi
  if src=$(find_dir "$d" "$min"); then
    ln -s "$src" "$DEST/$d"; printf "  LINKED   %-50s -> %s\n" "$d" "$src"; return
  fi
  tb=$(find_tar "$d")
  if [ -n "$tb" ]; then
    printf "  RESTORE  %-50s\n" "$d"
    if extract "$tb" "$d"; then printf "           %-50s %6d graphs\n" "" "$(npt "$DEST/$d")"; return; fi
  fi
  if [ -n "$opt" ]; then printf "  absent   %-50s %s\n" "$d" "$opt"
  else printf "  MISSING  %-50s no folder or archive found under: %s\n" "$d" "$SEARCH"; missing+=("$d"); fi
}

echo "SHREC2019"
MIN19="${MIN19:-5000}"; MIN25="${MIN25:-11000}"   # minimum graph files per folder
restore ssl_graphs_shared_proteinsplit "$MIN19"
for k in 8 16 32 64 128 256; do restore ssl_graphs_shared_proteinsplit_patchassign_k$k "$MIN19"; done
for d in graphs_pyg_shared_proteins graphs_pyg_shared_species; do   # label CSVs
  if [ -e "$DEST/$d" ]; then echo "  OK       $d"
  elif [ -e "$WORK/$d/train_labels.csv" ]; then ln -s "$WORK/$d" "$DEST/$d"; echo "  LINKED   $d -> $WORK/$d"
  else echo "  MISSING  $d (label CSVs)"; missing+=("$d"); fi
done

echo "SHREC2025"
restore ssl_graphs_1k_geom_enriched_ri "$MIN25"
restore ssl_graphs_1k_patchassign_k64 "$MIN25"
for k in 8 32 128; do restore ssl_graphs_1k_patchassign_k$k "$MIN25" "(only if SHREC2025 checkpoints use k=$k)"; done
for k in 16 256; do restore ssl_graphs_1k_patchassign_k$k "$MIN25" "(built by: sbatch jeanzay/build_patchassign.slurm)"; done

echo
if [ ${#missing[@]} -eq 0 ]; then
  echo "all required inputs are in $DEST"
else
  echo "still missing: ${missing[*]}"
fi
[ "$DEST" = "$SCRATCH/fw_data" ] && echo "the fw scripts use $DEST automatically (no DATA_ROOT needed)." \
  || echo "run every fw command with: export DATA_ROOT=$DEST"
echo "note: \$SCRATCH deletes files unused for 30 days; rerun this script to restore them."
