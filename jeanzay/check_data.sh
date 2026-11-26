#!/bin/bash
# Check that every input of the future-work jobs is present under DATA_ROOT.
#   ./jeanzay/check_data.sh            (DATA_ROOT defaults to $WORK)
DATA_ROOT="${DATA_ROOT:-$([ -d "$SCRATCH/fw_data" ] && echo "$SCRATCH/fw_data" || echo "$WORK")}"
PROJECT="${PROJECT:-$WORK/protein_classification}"
ok=1
need() {  # need <path> <used by> [hint]
  if [ -e "$DATA_ROOT/$1" ] || [ -e "$1" ]; then printf "  OK       %-62s %s\n" "$1" "$2"
  else printf "  MISSING  %-62s %s\n" "$1" "$2"; [ -n "${3:-}" ] && echo "           -> $3"; ok=0; fi
}
echo "DATA_ROOT=$DATA_ROOT"
echo "SHREC2019"
for d in ssl_graphs_shared_proteinsplit ssl_graphs_shared_proteinsplit_patchassign_k16 \
         ssl_graphs_shared_proteinsplit_patchassign_k64 ssl_graphs_shared_proteinsplit_patchassign_k256; do
  need "$d" "rotation, multiscale19, analysis"
done
for d in proteins species; do
  need "graphs_pyg_shared_$d/train_labels.csv" "SHREC2019 labels"
  need "graphs_pyg_shared_$d/test_labels.csv" "SHREC2019 labels"
done
for k in 8 32 128; do need "ssl_graphs_shared_proteinsplit_patchassign_k$k" "analysis (patch-count sweep)"; done
echo "SHREC2025"
need ssl_graphs_1k_geom_enriched_ri "rotation, multiscale25, analysis"
need ssl_graphs_1k_geom_enriched_ri/train_split.csv "SHREC2025 labels (paper release)"
need ssl_graphs_1k_patchassign_k64 "rotation, multiscale25, analysis"
for k in 16 256; do need "ssl_graphs_1k_patchassign_k$k" "multiscale25 only" "sbatch jeanzay/build_patchassign.slurm"; done
echo "repository label sets"
for f in shrec25_final/train_split.csv shrec25_final/val_split.csv shrec25_final/test_labels.csv \
         shrec25_update_281025/test_labels.csv; do need "$PROJECT/data_labels/$f" "FINAL release / re-scoring"; done
echo "checkpoints for the analysis"
need "$WORK/protein_classification/runs_20260907.tar.zst" "extract_checkpoints.sh"
[ $ok = 1 ] && echo "all inputs present" \
  || echo "some inputs are missing: run ./jeanzay/restore_data.sh (links or extracts them into \$SCRATCH/fw_data)"
