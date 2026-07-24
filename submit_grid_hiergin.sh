#!/bin/bash
# ===========================================================================
# submit_grid_hiergin.sh — launch the HSNet backbone x hidden x seed GRID on
# Jean-Zay as max-GPU-parallel array jobs, then chain per-cell aggregation.
#
# EDIT THE CELL below, then run:  ./submit_grid_hiergin.sh
#
# A "cell" = (dataset, feature-group, patch-k). PATCH_K may be a LIST: the script
# submits ONE independent array (+ agg) per patch-k value, so each array stays a
# clean 330-task job (11 backbones x 3 hidden x 10 seeds) instead of one giant
# array. Inside each array every task is one independent single-GPU training; the
# scheduler runs up to CONCURRENCY of them at once (the "%N" in --array). Set
# CONCURRENCY to the max GPUs your allocation allows — riconv-style "use all the
# GPUs", NOT one job spread over 4 GPUs.
# ===========================================================================
set -euo pipefail

# ----------------------------- THE CELL (edit me) --------------------------
DATASET="${DATASET:-shrec19_protein}"      # shrec19_protein | shrec19_species | shrec25
FEATURE_GROUP="${FEATURE_GROUP:-geom}"     # geom | enriched
PATCH_K_LIST="${PATCH_K_LIST:-8 16 32 64 128 256}"   # sweep; SHREC25 valid: 8 32 64 128

# STUDY: optional stable study name. When set, each cell's outputs live in a fixed
# dir runs/<STUDY>/<dataset>_<feature>_k<K>/ (no timestamp), so SEVERAL submissions
# for the SAME (dataset,feature,patch-k) — e.g. safe backbones on dev + risky on
# t3 — accumulate into ONE runs/ dir and aggregate together. Unset -> timestamped.
STUDY="${STUDY:-}"

# ----------------------------- THE GRID AXES -------------------------------
# BACKBONE_SET picks a named group (or override BACKBONES directly):
#   safe  = < ~2h/run, dev-friendly (local ops + fast gvp)   [8]
#   risky = global-attention transformers, ~2h+  -> use t3   [3]
#   all   = everything                                        [11]
BACKBONE_SET="${BACKBONE_SET:-all}"
case "$BACKBONE_SET" in
  safe)  _BB="gin sage gcn gat gatv2 pna msg_gin gvp" ;;
  risky) _BB="sgformer gt gps" ;;
  all)   _BB="gin sage gcn gat gatv2 pna msg_gin gvp sgformer gt gps" ;;
  *) echo "bad BACKBONE_SET=$BACKBONE_SET (use safe|risky|all)"; exit 1 ;;
esac
BACKBONES="${BACKBONES:-$_BB}"
HIDDEN_DIMS="${HIDDEN_DIMS:-128 256 512}"                                           # 3
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"                                             # 10

# ----------------------------- CLUSTER / SCALE -----------------------------
DATA_ROOT="${DATA_ROOT:-$WORK}"            # dir containing the graph dirs (same level as SHREC25 ri)
PROJECT="${PROJECT:-$WORK/protein_classification}"
ACCOUNT="${ACCOUNT:-acu@v100}"
CONSTRAINT="${CONSTRAINT:-v100-32g}"
QOS="${QOS:-qos_gpu-t3}"                   # t3=<=20h ; use qos_gpu-t4 for very long
TIME="${TIME:-12:00:00}"
CPUS="${CPUS:-10}"
CONCURRENCY="${CONCURRENCY:-16}"           # max GPUs to run at once -> set to your cap
# SERIALIZE=1 : chain the patch-k cells with SLURM dependencies so only ONE array
# runs at a time (submit once, they fire back-to-back). Needed on dev, where 6
# simultaneous arrays would blow the 32-GPU / 10-job caps. On t3 leave it unset
# (0) so all patch-k run in parallel.
SERIALIZE="${SERIALIZE:-0}"
# NO_AGG=1 : do NOT submit the per-cell aggregation job. On dev every job counts
# against the ~10-job submit cap, so skip agg here and run it once at the end
# (grid_hiergin.py --mode agg). Leave 0 on t3.
NO_AGG="${NO_AGG:-0}"
# ---------------------------------------------------------------------------

cd "$PROJECT"
mkdir -p "$PROJECT/logs"

echo "cell axes : dataset=$DATASET feature=$FEATURE_GROUP  study=${STUDY:-<timestamped>}"
echo "patch-k   : $PATCH_K_LIST"
echo "backbones : set=$BACKBONE_SET -> [$BACKBONES]"
echo "grid      : hidden=[$HIDDEN_DIMS] seeds=[$SEEDS]  qos=$QOS time=$TIME"
echo "scale     : CONCURRENCY=$CONCURRENCY  (max GPUs at once)  serialize=$SERIALIZE"
echo

PREV_ARRAY=""      # for SERIALIZE: chain each cell after the previous one's array
for PATCH_K in $PATCH_K_LIST; do
    export DATA_ROOT DATASET FEATURE_GROUP PATCH_K BACKBONES HIDDEN_DIMS SEEDS

    # Resolve grid size + validate paths via the planner (single source of truth).
    PLAN=$(python grid_hiergin.py --mode plan \
            --dataset "$DATASET" --feature-group "$FEATURE_GROUP" --patch-k "$PATCH_K" \
            --data-root "$DATA_ROOT" \
            --backbones $BACKBONES --hidden-dims $HIDDEN_DIMS --seeds $SEEDS)
    N=$(echo "$PLAN" | sed -n 's/^GRID_SIZE=//p')
    [ -n "$N" ] || { echo "!! could not determine grid size for patch-k=$PATCH_K"; echo "$PLAN"; exit 1; }
    LAST=$(( N - 1 ))

    if [ -n "$STUDY" ]; then
        # Stable per-cell root: multiple submissions (safe/dev + risky/t3) for the
        # same (dataset,feature,patch-k) accumulate into ONE runs/ dir -> one agg.
        GRID_ROOT="$PROJECT/runs/${STUDY}/${DATASET}_${FEATURE_GROUP}_k${PATCH_K}"
    else
        STAMP=$(date +%Y%m%d_%H%M%S)
        GRID_ROOT="$PROJECT/runs/grid_${DATASET}_${FEATURE_GROUP}_k${PATCH_K}_${STAMP}"
    fi
    mkdir -p "$GRID_ROOT/runs"
    export GRID_ROOT
    cat > "$GRID_ROOT/CELL_${BACKBONE_SET}.env" <<EOF
DATASET=$DATASET
FEATURE_GROUP=$FEATURE_GROUP
PATCH_K=$PATCH_K
BACKBONES=$BACKBONES
HIDDEN_DIMS=$HIDDEN_DIMS
SEEDS=$SEEDS
DATA_ROOT=$DATA_ROOT
GRID_SIZE=$N
EOF

    echo "=== patch-k=$PATCH_K : $N tasks (array 0-$LAST %$CONCURRENCY) ==="
    echo "$PLAN" | sed -n '2,4p'
    echo "grid root: $GRID_ROOT"

    # ---------- grid array (one GPU per task) ----------
    # SERIALIZE: make this cell's array wait for the previous cell's array, so only
    # one patch-k runs at a time (keeps dev within its 32-GPU / 10-job caps).
    DEP_ARG=""
    if [ "$SERIALIZE" = "1" ] && [ -n "$PREV_ARRAY" ]; then
        DEP_ARG="--dependency=afterany:${PREV_ARRAY}"
    fi
    ARRAY_JID=$(sbatch --parsable $DEP_ARG \
        --job-name="HGRID_${DATASET}_${FEATURE_GROUP}_${BACKBONE_SET}_k${PATCH_K}" \
        --array=0-${LAST}%${CONCURRENCY} \
        --time="$TIME" --account="$ACCOUNT" --constraint="$CONSTRAINT" --qos="$QOS" \
        --cpus-per-task="$CPUS" --gres=gpu:1 --hint=nomultithread \
        --output="$PROJECT/logs/log_%x_%A_%a.out" \
        --error="$PROJECT/logs/log_%x_%A_%a.err" \
        slurm_grid_hiergin.slurm)
    echo "  array : $ARRAY_JID${DEP_ARG:+  ($DEP_ARG)}"
    PREV_ARRAY="$ARRAY_JID"

    # ---------- aggregate after the whole array (afterany) ----------
    if [ "$NO_AGG" = "1" ]; then
        echo "  agg   : skipped (NO_AGG=1) — run grid_hiergin.py --mode agg at the end"
    else
        AGG_JID=$(sbatch --parsable \
            --job-name="HGRID_AGG_${BACKBONE_SET}_k${PATCH_K}" \
            --dependency=afterany:${ARRAY_JID} \
            --account="$ACCOUNT" --qos=qos_gpu-dev --gres=gpu:1 \
            --constraint="$CONSTRAINT" --cpus-per-task="$CPUS" --hint=nomultithread \
            --time=00:20:00 \
            --output="$PROJECT/logs/log_%x_%j.out" --error="$PROJECT/logs/log_%x_%j.err" \
            --wrap="module purge; module load pytorch-gpu/py3/2.8.0; source $WORK/myvenv/bin/activate; cd $PROJECT; python -u grid_hiergin.py --mode agg --dataset $DATASET --feature-group $FEATURE_GROUP --patch-k $PATCH_K --grid-root $GRID_ROOT")
        echo "  agg   : $AGG_JID (afterany: $ARRAY_JID)"
    fi
    echo
done

echo "watch  : squeue -u \$USER"
if [ -n "$STUDY" ]; then
  echo "study root : $PROJECT/runs/$STUDY/"
  echo "per-cell   : runs/$STUDY/${DATASET}_${FEATURE_GROUP}_k<K>/summary.{csv,json}"
  echo
  echo "NOTE: with STUDY set, safe+risky submissions share each cell dir. After BOTH"
  echo "      finish, re-run agg per cell to get the COMBINED table, e.g.:"
  echo "        for K in $PATCH_K_LIST; do"
  echo "          python grid_hiergin.py --mode agg --dataset $DATASET \\"
  echo "            --feature-group $FEATURE_GROUP --patch-k \$K \\"
  echo "            --grid-root runs/$STUDY/${DATASET}_${FEATURE_GROUP}_k\$K ; done"
else
  echo "summary per cell: runs/grid_${DATASET}_${FEATURE_GROUP}_k<K>_*/summary.{csv,json}"
fi
echo "DOWNLOAD few files: scp -r <jz>:$PROJECT/runs/ ... /summary.json ./"
