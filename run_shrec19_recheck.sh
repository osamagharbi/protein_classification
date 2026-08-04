#!/bin/bash
# ===========================================================================
#  run_shrec19_recheck.sh — re-validate SHREC2019 protein + species against
#  the corrected .cla labels (convert_cla_file.py parser-bug fix, see
#  papers/paper_hiergin_shrec19/TODO.md) using ONLY the minimal set of
#  configs actually cited in the paper's tables, at 10 seeds each.
#
#  This is NOT the full run_study.sh cartesian sweep: it is 11 explicit,
#  hand-picked cells (dataset x feature-group x patch-k x backbones x
#  hidden), because each cited backbone needs a specific hidden size, not
#  every combination. Reuses grid_hiergin.py / slurm_grid_hiergin.slurm
#  unchanged as the run engine.
#
#  PREREQUISITE (blocking): DATA_ROOT must contain graphs rebuilt from the
#  corrected proteins_labels.csv / species_labels.csv (17 + 54 recovered
#  surface IDs) -- i.e. graphs_pyg_shared_proteins/{train,test}_labels.csv,
#  graphs_pyg_shared_species/{train,test}_labels.csv, and the
#  ssl_graphs_shared_*split*/*.pt graph files must all be regenerated from
#  the corrected manifest BEFORE this script is run. It will not do that
#  rebuild itself.
#
#  Usage (on Jean-Zay, from $WORK/protein_classification):
#     ./run_shrec19_recheck.sh              # submit everything
#     DRY_RUN=1 ./run_shrec19_recheck.sh     # print the plan, submit nothing
#
#  240 runs total (170 protein + 70 species), all qos_gpu-t3, each array's
#  concurrency = its own size (fully parallel), peak ~240 GPUs (cap 640).
# ===========================================================================
set -euo pipefail

STUDY_NAME="${STUDY_NAME:-shrec19_recheck}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"                     # 10, both tasks

# ------------------------------- CLUSTER -----------------------------------
ACCOUNT="${ACCOUNT:-acu@v100}"
CONSTRAINT="${CONSTRAINT:-v100-32g}"
QOS="${QOS:-qos_gpu-t3}"
TIME="${TIME:-12:00:00}"
CPUS="${CPUS:-10}"
DATA_ROOT="${DATA_ROOT:-$WORK}"
PROJECT="${PROJECT:-$WORK/protein_classification}"
DRY_RUN="${DRY_RUN:-0}"
EST_MIN_PER_RUN="${EST_MIN_PER_RUN:-35}"
# ---------------------------------------------------------------------------

cd "$PROJECT"
mkdir -p "$PROJECT/logs"

# ---- 11 explicit cells: dataset|feature_group|patch_k|backbones|hidden ----
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

ns=$(echo $SEEDS | wc -w)
TOTAL=0
for CELL in "${CELLS[@]}"; do
  IFS='|' read -r DS FG K BACKBONES HIDDEN_DIMS <<< "$CELL"
  nb=$(echo $BACKBONES | wc -w)
  TOTAL=$(( TOTAL + nb * ns ))
done
EST_GPUH=$(( TOTAL * EST_MIN_PER_RUN / 60 ))

STUDY_ROOT="$PROJECT/runs/$STUDY_NAME"
mkdir -p "$STUDY_ROOT"

cat <<BANNER
===========================================================================
 STUDY: $STUDY_NAME   (dry_run=$DRY_RUN)
 cells (dataset/feature/k -> backbones @hidden):
BANNER
for CELL in "${CELLS[@]}"; do
  IFS='|' read -r DS FG K BACKBONES HIDDEN_DIMS <<< "$CELL"
  nb=$(echo $BACKBONES | wc -w)
  printf "   %-16s %-9s k%-4s -> %-34s @h%-4s  (%d x %d seeds = %d runs)\n" \
    "$DS" "$FG" "$K" "$BACKBONES" "$HIDDEN_DIMS" "$nb" "$ns" "$(( nb * ns ))"
done
cat <<BANNER
 ---------------------------------------------------------------------------
 seeds         : $SEEDS
 TOTAL runs    : $TOTAL   (peak GPUs ~ $TOTAL, cap 640, one array per cell,
                 each array fully parallel: --array=0-N%N)
 est. compute  : ~$EST_GPUH GPU-hours (@ ${EST_MIN_PER_RUN} min/run avg)
 qos/time      : $QOS / $TIME
 study root    : $STUDY_ROOT
===========================================================================
BANNER

# ------------------------- preflight: every cell's data present ------------
echo "[preflight] validating data paths for all ${#CELLS[@]} cells ..."
FAIL=0
for CELL in "${CELLS[@]}"; do
  IFS='|' read -r DS FG K BACKBONES HIDDEN_DIMS <<< "$CELL"
  if ! python grid_hiergin.py --mode check --dataset "$DS" --feature-group "$FG" \
        --patch-k "$K" --data-root "$DATA_ROOT" >/dev/null 2>&1; then
    echo "  MISSING data for cell: $DS / $FG / k$K"
    python grid_hiergin.py --mode check --dataset "$DS" --feature-group "$FG" \
        --patch-k "$K" --data-root "$DATA_ROOT" 2>&1 | sed 's/^/      /'
    FAIL=1
  fi
done
[ "$FAIL" -eq 0 ] && echo "[preflight] all cells OK." || { echo "[preflight] fix the above (rebuild corrected graphs first), then re-run."; exit 1; }

if [ "$DRY_RUN" = "1" ]; then
  echo; echo "DRY_RUN=1 -> nothing submitted. Unset DRY_RUN to launch."; exit 0
fi

# ------------------------------- submit cells ------------------------------
MANIFEST="$STUDY_ROOT/MANIFEST.tsv"
echo -e "cell\tgrid_root\tarray_jid\tn_tasks" > "$MANIFEST"
ARRAY_JIDS=()
for CELL in "${CELLS[@]}"; do
  IFS='|' read -r DS FG K BACKBONES HIDDEN_DIMS <<< "$CELL"
  nb=$(echo $BACKBONES | wc -w)
  N=$(( nb * ns )); LAST=$(( N - 1 ))
  GRID_ROOT="$STUDY_ROOT/${DS}_${FG}_k${K}"
  mkdir -p "$GRID_ROOT/runs"
  export DATA_ROOT GRID_ROOT SEEDS
  export BACKBONES HIDDEN_DIMS="$HIDDEN_DIMS"
  export DATASET="$DS" FEATURE_GROUP="$FG" PATCH_K="$K"

  JID=$(sbatch --parsable \
      --job-name="RECHK_${DS}_${FG}_k${K}" \
      --array=0-${LAST}%${N} \
      --time="$TIME" --account="$ACCOUNT" --constraint="$CONSTRAINT" --qos="$QOS" \
      --cpus-per-task="$CPUS" --gres=gpu:1 --hint=nomultithread \
      --output="$PROJECT/logs/log_%x_%A_%a.out" \
      --error="$PROJECT/logs/log_%x_%A_%a.err" \
      slurm_grid_hiergin.slurm)
  ARRAY_JIDS+=("$JID")
  echo -e "${DS}_${FG}_k${K}\t$GRID_ROOT\t$JID\t$N" >> "$MANIFEST"
  printf "  submitted  %-34s  array=%s  (%d tasks, full concurrency)\n" "${DS}/${FG}/k${K}" "$JID" "$N"
done

# ---------------------- master aggregation (afterany all) ------------------
DEP=$(IFS=:; echo "${ARRAY_JIDS[*]}")
AGG_JID=$(sbatch --parsable \
    --job-name="RECHK_MASTER_AGG" \
    --dependency=afterany:${DEP} \
    --account="$ACCOUNT" --qos=qos_gpu-t3 --gres=gpu:1 \
    --constraint="$CONSTRAINT" --cpus-per-task="$CPUS" --hint=nomultithread --time=00:20:00 \
    --output="$PROJECT/logs/log_%x_%j.out" --error="$PROJECT/logs/log_%x_%j.err" \
    --wrap="module purge; module load pytorch-gpu/py3/2.8.0; source $WORK/myvenv/bin/activate; cd $PROJECT; python -u grid_hiergin.py --mode agg-study --study-root $STUDY_ROOT")

echo
echo "submitted ${#CELLS[@]} cell-arrays + 1 master-agg ($AGG_JID, afterany all)."
echo "manifest : $MANIFEST"
echo "watch    : squeue -u \$USER   |   progress: find $STUDY_ROOT -name final_results.json | wc -l   (/$TOTAL)"
echo "results  : $STUDY_ROOT/master_summary.{csv,json}   (built when everything finishes)"
echo "re-agg anytime: python grid_hiergin.py --mode agg-study --study-root $STUDY_ROOT"
