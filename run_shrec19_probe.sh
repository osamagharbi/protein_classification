#!/bin/bash
# ===========================================================================
#  run_shrec19_probe.sh — correct the paper's SHREC19 test-set numbers WITHOUT
#  retraining: re-score the EXISTING checkpoints from the study behind
#  master_summary.csv / master_summary_species.csv against the corrected
#  (post .cla-parser-fix) test_labels.csv, using qos_gpu-dev since this is
#  pure inference (each task loads one best_model.pt and runs one forward
#  pass over the test set -- no training loop).
#
#  This does NOT tell you whether retraining on the corrected (slightly
#  larger) train set would help further -- that is run_shrec19_recheck.sh
#  (already launched, full retrain, qos_gpu-t3). This script answers a
#  narrower, cheaper question: "how much does completing the test set alone
#  change the numbers already printed in the paper?"
#
#  PREREQUISITES:
#   1. Phase A done: DATA_ROOT has the corrected graphs_pyg_shared_{proteins,
#      species}/test_labels.csv (recovered surfaces present) and their .pt
#      graph files.
#   2. Two SEPARATE original study roots (confirmed on Jean-Zay -- protein and
#      species were never in the same study):
#        protein : runs/shrec19_hsnet     (10 seeds/config, master_summary.csv)
#        species : runs/shrec_crosseval   (5 seeds/config only,
#                                           master_summary_species.csv)
#      Each structured as <root>/<dataset>_<feature>_k<K>/runs/<tag>/
#      (best_model.pt + final_results.json). Species seeds 6-10 do NOT exist
#      as checkpoints (never trained beyond 5 seeds) so species cells are
#      probed at 5 seeds, protein cells at 10 -- probing can't manufacture
#      seeds that were never trained; getting species to 10 seeds still
#      requires run_shrec19_recheck.sh's full retrain.
#
#  Usage:
#     ./run_shrec19_probe.sh                       # uses the defaults below
#     ORIG_ROOT_PROTEIN=... ORIG_ROOT_SPECIES=... ./run_shrec19_probe.sh
#     DRY_RUN=1 ./run_shrec19_probe.sh              # preview only
# ===========================================================================
set -euo pipefail

ORIG_ROOT_PROTEIN="${ORIG_ROOT_PROTEIN:-$WORK/protein_classification/runs/shrec19_hsnet}"
ORIG_ROOT_SPECIES="${ORIG_ROOT_SPECIES:-$WORK/protein_classification/runs/shrec_crosseval}"

PROBE_NAME="${PROBE_NAME:-shrec19_probe_corrected_test}"
PROTEIN_SEEDS="${PROTEIN_SEEDS:-1 2 3 4 5 6 7 8 9 10}"
SPECIES_SEEDS="${SPECIES_SEEDS:-1 2 3 4 5}"

# ------------------------------- CLUSTER -----------------------------------
ACCOUNT="${ACCOUNT:-acu@v100}"
CONSTRAINT="${CONSTRAINT:-v100-32g}"
QOS="${QOS:-qos_gpu-dev}"          # inference only -> cheap, dev QOS is fine
TIME="${TIME:-00:30:00}"
CPUS="${CPUS:-6}"
CONCURRENCY="${CONCURRENCY:-20}"   # dev partition has far fewer nodes than t3
DATA_ROOT="${DATA_ROOT:-$WORK}"
PROJECT="${PROJECT:-$WORK/protein_classification}"
DRY_RUN="${DRY_RUN:-0}"
# ---------------------------------------------------------------------------

cd "$PROJECT"
mkdir -p "$PROJECT/logs"

# same 11 cells as run_shrec19_recheck.sh -- the minimal set actually cited
# in the paper's tables (dataset|feature_group|patch_k|backbones|hidden)
CELLS=(
  "shrec19_protein|geom|64|gvp gps gt gin msg_gin sage pna|256"
  "shrec19_protein|geom|64|sgformer gatv2 gat gcn|512"
  "shrec19_protein|geom|8|gvp|256"
  "shrec19_protein|geom|16|gvp|256"
  "shrec19_protein|geom|32|gvp|256"
  "shrec19_protein|geom|128|gvp|256"
  "shrec19_protein|geom|256|gvp|256"
  "shrec19_protein|enriched|64|gvp|256"
  "shrec19_species|geom|64|gvp gps gin gt gcn|256"
  "shrec19_species|geom|64|gat|512"
  "shrec19_species|enriched|64|gvp|256"
)

PROBE_ROOT="$PROJECT/runs/$PROBE_NAME"
OUT_ROOT="$PROBE_ROOT/results"
mkdir -p "$OUT_ROOT"
MANIFEST="$PROBE_ROOT/MANIFEST.tsv"
echo -e "tag\torig_run_dir\tgraph_dir\tassign_dir\ttest_labels_csv" > "$MANIFEST"

echo "[manifest] resolving corrected paths for ${#CELLS[@]} cells via grid_hiergin.resolve_paths ..."
for CELL in "${CELLS[@]}"; do
  IFS='|' read -r DS FG K BACKBONES HIDDEN_DIMS <<< "$CELL"
  read -r GRAPH_DIR ASSIGN_DIR TEST_CSV <<< "$(python3 -c "
from grid_hiergin import resolve_paths
g, a, tr, te, fm = resolve_paths('$DATA_ROOT', '$DS', '$FG', $K)
print(g, a, te)
")"
  if [ "$DS" = "shrec19_protein" ]; then
    ORIG_ROOT="$ORIG_ROOT_PROTEIN"; SEEDS="$PROTEIN_SEEDS"
  else
    ORIG_ROOT="$ORIG_ROOT_SPECIES"; SEEDS="$SPECIES_SEEDS"
  fi
  ORIG_CELL_DIR="$ORIG_ROOT/${DS}_${FG}_k${K}/runs"
  for BB in $BACKBONES; do
    for S in $SEEDS; do
      TAG="${DS}_${FG}_k${K}_${BB}_h${HIDDEN_DIMS}_s${S}"
      echo -e "${TAG}\t${ORIG_CELL_DIR}/${TAG}\t${GRAPH_DIR}\t${ASSIGN_DIR}\t${TEST_CSV}" >> "$MANIFEST"
    done
  done
done

N=$(( $(wc -l < "$MANIFEST") - 1 ))
LAST=$(( N - 1 ))

cat <<BANNER
===========================================================================
 PROBE: $PROBE_NAME   (dry_run=$DRY_RUN)
 orig checkpoints (protein) : $ORIG_ROOT_PROTEIN   [$PROTEIN_SEEDS]
 orig checkpoints (species) : $ORIG_ROOT_SPECIES   [$SPECIES_SEEDS]
 corrected data   : $DATA_ROOT
 total tasks      : $N   (inference only, one checkpoint each)
 concurrency      : $CONCURRENCY   qos: $QOS   time/task cap: $TIME
 manifest         : $MANIFEST
 results          : $OUT_ROOT/<tag>.json  ->  aggregated by agg_probe.py
===========================================================================
BANNER

# ------------------------- preflight: every checkpoint present -------------
echo "[preflight] checking all $N original checkpoints exist ..."
MISSING=0
tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r TAG RUN_DIR GRAPH_DIR ASSIGN_DIR TEST_CSV; do
  if [ ! -f "$RUN_DIR/best_model.pt" ] || [ ! -f "$RUN_DIR/final_results.json" ]; then
    echo "  MISSING checkpoint: $RUN_DIR"
    echo "1" >> "$PROBE_ROOT/.preflight_fail"
  fi
done
if [ -f "$PROBE_ROOT/.preflight_fail" ]; then
  rm -f "$PROBE_ROOT/.preflight_fail"
  echo "[preflight] fix ORIG_ROOT_PROTEIN/ORIG_ROOT_SPECIES or re-run the missing configs first, then re-run."
  exit 1
fi
echo "[preflight] all $N checkpoints found."

if [ "$DRY_RUN" = "1" ]; then
  echo; echo "DRY_RUN=1 -> nothing submitted. Unset DRY_RUN to launch."; exit 0
fi

# ------------------------------- submit array -------------------------------
export MANIFEST OUT_ROOT
JID=$(sbatch --parsable \
    --job-name="PROBE_${PROBE_NAME}" \
    --array=0-${LAST}%${CONCURRENCY} \
    --time="$TIME" --account="$ACCOUNT" --constraint="$CONSTRAINT" --qos="$QOS" \
    --cpus-per-task="$CPUS" --gres=gpu:1 --hint=nomultithread \
    --output="$PROJECT/logs/log_%x_%A_%a.out" \
    --error="$PROJECT/logs/log_%x_%A_%a.err" \
    slurm_probe_hiergin.slurm)

AGG_JID=$(sbatch --parsable \
    --job-name="PROBE_AGG" \
    --dependency=afterany:${JID} \
    --account="$ACCOUNT" --qos="$QOS" \
    --cpus-per-task=4 --hint=nomultithread --time=00:10:00 \
    --output="$PROJECT/logs/log_%x_%j.out" --error="$PROJECT/logs/log_%x_%j.err" \
    --wrap="module purge; module load pytorch-gpu/py3/2.8.0; source $WORK/myvenv/bin/activate; cd $PROJECT; python -u agg_probe.py --out-root $OUT_ROOT --csv $PROBE_ROOT/master_summary_corrected_test.csv")

echo
echo "submitted array $JID ($N tasks) + agg job $AGG_JID (afterany)."
echo "watch  : squeue -u \$USER"
echo "results: $PROBE_ROOT/master_summary_corrected_test.csv  (orig vs corrected vs delta per config)"
echo "re-agg anytime: python agg_probe.py --out-root $OUT_ROOT --csv $PROBE_ROOT/master_summary_corrected_test.csv"
