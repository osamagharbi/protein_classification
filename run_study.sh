#!/bin/bash
# ===========================================================================
#  run_study.sh — launch the FULL HSNet benchmark study in one shot.
#
#  A "study" is the Cartesian product of every axis below. Each cell
#  (dataset x feature-group x patch-k) is submitted as ONE Slurm array that
#  grids over backbone x hidden x seed; every array task is one independent
#  single-GPU training. All cells go out together, then a single master
#  aggregation job collates everything into one table.
#
#  Usage:
#     ./run_study.sh              # submit the study
#     DRY_RUN=1 ./run_study.sh    # print the plan + accounting, submit nothing
#
#  Every knob is an env var with a sane default — override only what you need,
#  e.g.:  CONCURRENCY=60 FEATURE_GROUPS="geom" ./run_study.sh
# ===========================================================================
set -euo pipefail

# --------------------------- STUDY AXES (edit here) ------------------------
STUDY_NAME="${STUDY_NAME:-shrec19_hsnet}"
DATASETS=( ${DATASETS:-shrec19_protein} )                 # + shrec19_species shrec25
FEATURE_GROUPS=( ${FEATURE_GROUPS:-geom enriched} )       # geom | enriched
PATCH_KS=( ${PATCH_KS:-8 16 32 64 128 256} )
BACKBONES="${BACKBONES:-gin sage gcn gat gatv2 pna msg_gin gvp sgformer gt gps}"  # 11
HIDDEN_DIMS="${HIDDEN_DIMS:-128 256 512}"                 # 3
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"                    # 10

# ------------------------------- CLUSTER -----------------------------------
ACCOUNT="${ACCOUNT:-acu@v100}"
CONSTRAINT="${CONSTRAINT:-v100-32g}"
QOS="${QOS:-qos_gpu-t3}"                # t3 = <=20h, up to 640 GPU/user
TIME="${TIME:-20:00:00}"
CPUS="${CPUS:-10}"
CONCURRENCY="${CONCURRENCY:-40}"        # running GPUs PER cell-array (peak = cells x this)
DATA_ROOT="${DATA_ROOT:-$WORK}"         # dir containing the graph dirs
PROJECT="${PROJECT:-$WORK/protein_classification}"
DRY_RUN="${DRY_RUN:-0}"
EST_MIN_PER_RUN="${EST_MIN_PER_RUN:-35}"   # rough, for the GPU-hour estimate only
# ---------------------------------------------------------------------------

cd "$PROJECT"
mkdir -p "$PROJECT/logs"

nb=$(echo $BACKBONES | wc -w); nh=$(echo $HIDDEN_DIMS | wc -w); ns=$(echo $SEEDS | wc -w)
PER_CELL=$(( nb * nh * ns ))
N_CELLS=$(( ${#DATASETS[@]} * ${#FEATURE_GROUPS[@]} * ${#PATCH_KS[@]} ))
TOTAL=$(( N_CELLS * PER_CELL ))
PEAK_GPU=$(( N_CELLS * CONCURRENCY ))
EST_GPUH=$(( TOTAL * EST_MIN_PER_RUN / 60 ))

STUDY_ROOT="$PROJECT/runs/$STUDY_NAME"
mkdir -p "$STUDY_ROOT"

cat <<BANNER
===========================================================================
 STUDY: $STUDY_NAME     (dry_run=$DRY_RUN)
 datasets      : ${DATASETS[*]}
 feature groups: ${FEATURE_GROUPS[*]}
 patch-k       : ${PATCH_KS[*]}
 backbones ($nb): $BACKBONES
 hidden ($nh)    : $HIDDEN_DIMS
 seeds ($ns)     : $SEEDS
 ---------------------------------------------------------------------------
 cells         : $N_CELLS   (dataset x feature x patch-k)
 runs / cell   : $PER_CELL   (backbone x hidden x seed)
 TOTAL runs    : $TOTAL
 concurrency   : $CONCURRENCY GPUs/cell  ->  peak ~$PEAK_GPU GPUs (cap 640)
 est. compute  : ~$EST_GPUH GPU-hours  (@ ${EST_MIN_PER_RUN} min/run avg)
 qos/time      : $QOS / $TIME
 study root    : $STUDY_ROOT
===========================================================================
BANNER

if [ "$PEAK_GPU" -gt 640 ]; then
  echo "!! peak ~$PEAK_GPU GPUs exceeds the 640 cap. Lower CONCURRENCY (<= $(( 640 / N_CELLS )))."
  exit 1
fi

# ------------------------- preflight: every cell's data present ------------
echo "[preflight] validating data paths for all $N_CELLS cells ..."
FAIL=0
for DS in "${DATASETS[@]}"; do for FG in "${FEATURE_GROUPS[@]}"; do for K in "${PATCH_KS[@]}"; do
  if ! python grid_hiergin.py --mode check --dataset "$DS" --feature-group "$FG" \
        --patch-k "$K" --data-root "$DATA_ROOT" >/dev/null 2>&1; then
    echo "  MISSING data for cell: $DS / $FG / k$K"
    python grid_hiergin.py --mode check --dataset "$DS" --feature-group "$FG" \
        --patch-k "$K" --data-root "$DATA_ROOT" 2>&1 | sed 's/^/      /'
    FAIL=1
  fi
done; done; done
[ "$FAIL" -eq 0 ] && echo "[preflight] all cells OK." || { echo "[preflight] fix the above, then re-run."; exit 1; }

if [ "$DRY_RUN" = "1" ]; then
  echo; echo "DRY_RUN=1 -> nothing submitted. Unset DRY_RUN to launch."; exit 0
fi

# ------------------------------- submit cells ------------------------------
MANIFEST="$STUDY_ROOT/MANIFEST.tsv"
echo -e "cell\tgrid_root\tarray_jid\tn_tasks" > "$MANIFEST"
ARRAY_JIDS=()
for DS in "${DATASETS[@]}"; do for FG in "${FEATURE_GROUPS[@]}"; do for K in "${PATCH_KS[@]}"; do
  N=$PER_CELL; LAST=$(( N - 1 ))
  GRID_ROOT="$STUDY_ROOT/${DS}_${FG}_k${K}"
  mkdir -p "$GRID_ROOT/runs"
  export DATA_ROOT GRID_ROOT BACKBONES HIDDEN_DIMS SEEDS
  export DATASET="$DS" FEATURE_GROUP="$FG" PATCH_K="$K"

  JID=$(sbatch --parsable \
      --job-name="HS_${DS}_${FG}_k${K}" \
      --array=0-${LAST}%${CONCURRENCY} \
      --time="$TIME" --account="$ACCOUNT" --constraint="$CONSTRAINT" --qos="$QOS" \
      --cpus-per-task="$CPUS" --gres=gpu:1 --hint=nomultithread \
      --output="$PROJECT/logs/log_%x_%A_%a.out" \
      --error="$PROJECT/logs/log_%x_%A_%a.err" \
      slurm_grid_hiergin.slurm)
  ARRAY_JIDS+=("$JID")
  echo -e "${DS}_${FG}_k${K}\t$GRID_ROOT\t$JID\t$N" >> "$MANIFEST"
  printf "  submitted  %-34s  array=%s  (%d tasks)\n" "${DS}/${FG}/k${K}" "$JID" "$N"
done; done; done

# ---------------------- master aggregation (afterany all) ------------------
DEP=$(IFS=:; echo "${ARRAY_JIDS[*]}")
AGG_JID=$(sbatch --parsable \
    --job-name="HS_MASTER_AGG" \
    --dependency=afterany:${DEP} \
    --account="$ACCOUNT" --qos=qos_gpu-dev --gres=gpu:1 \
    --constraint="$CONSTRAINT" --cpus-per-task="$CPUS" --hint=nomultithread --time=00:20:00 \
    --output="$PROJECT/logs/log_%x_%j.out" --error="$PROJECT/logs/log_%x_%j.err" \
    --wrap="module purge; module load pytorch-gpu/py3/2.8.0; source $WORK/myvenv/bin/activate; cd $PROJECT; python -u grid_hiergin.py --mode agg-study --study-root $STUDY_ROOT")

echo
echo "submitted $N_CELLS cell-arrays + 1 master-agg ($AGG_JID, afterany all)."
echo "manifest : $MANIFEST"
echo "watch    : squeue -u \$USER   |   progress: find $STUDY_ROOT -name final_results.json | wc -l   (/$TOTAL)"
echo "results  : $STUDY_ROOT/master_summary.{csv,json}   (built when everything finishes)"
echo "re-agg anytime: python grid_hiergin.py --mode agg-study --study-root $STUDY_ROOT"
