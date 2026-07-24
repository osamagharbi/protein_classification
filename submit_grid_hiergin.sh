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

# ----------------------------- THE GRID AXES -------------------------------
BACKBONES="${BACKBONES:-gin sage gcn gat gatv2 pna gt gps sgformer msg_gin gvp}"   # 11
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
# ---------------------------------------------------------------------------

cd "$PROJECT"
mkdir -p "$PROJECT/logs"

echo "cell axes : dataset=$DATASET feature=$FEATURE_GROUP"
echo "patch-k   : $PATCH_K_LIST"
echo "grid      : backbones=[$BACKBONES] hidden=[$HIDDEN_DIMS] seeds=[$SEEDS]"
echo "scale     : CONCURRENCY=$CONCURRENCY  (max GPUs at once)"
echo

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

    STAMP=$(date +%Y%m%d_%H%M%S)
    GRID_ROOT="$PROJECT/runs/grid_${DATASET}_${FEATURE_GROUP}_k${PATCH_K}_${STAMP}"
    mkdir -p "$GRID_ROOT/runs"
    export GRID_ROOT
    cat > "$GRID_ROOT/CELL.env" <<EOF
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
    ARRAY_JID=$(sbatch --parsable \
        --job-name="HGRID_${DATASET}_${FEATURE_GROUP}_k${PATCH_K}" \
        --array=0-${LAST}%${CONCURRENCY} \
        --time="$TIME" --account="$ACCOUNT" --constraint="$CONSTRAINT" --qos="$QOS" \
        --cpus-per-task="$CPUS" --gres=gpu:1 --hint=nomultithread \
        --output="$PROJECT/logs/log_%x_%A_%a.out" \
        --error="$PROJECT/logs/log_%x_%A_%a.err" \
        slurm_grid_hiergin.slurm)
    echo "  array : $ARRAY_JID"

    # ---------- aggregate after the whole array (afterany) ----------
    AGG_JID=$(sbatch --parsable \
        --job-name="HGRID_AGG_k${PATCH_K}" \
        --dependency=afterany:${ARRAY_JID} \
        --account="$ACCOUNT" --qos=qos_gpu-dev --gres=gpu:1 \
        --constraint="$CONSTRAINT" --cpus-per-task="$CPUS" --hint=nomultithread \
        --time=00:20:00 \
        --output="$PROJECT/logs/log_%x_%j.out" --error="$PROJECT/logs/log_%x_%j.err" \
        --wrap="module purge; module load pytorch-gpu/py3/2.8.0; source $WORK/myvenv/bin/activate; cd $PROJECT; python -u grid_hiergin.py --mode agg --dataset $DATASET --feature-group $FEATURE_GROUP --patch-k $PATCH_K --grid-root $GRID_ROOT")
    echo "  agg   : $AGG_JID (afterany: $ARRAY_JID)"
    echo
done

echo "watch  : squeue -u \$USER"
echo "summary per cell: runs/grid_${DATASET}_${FEATURE_GROUP}_k<K>_*/summary.{csv,json}"
echo "DOWNLOAD few files: scp -r <jz>:$PROJECT/runs/grid_*/summary.json ./"
