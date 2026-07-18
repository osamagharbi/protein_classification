#!/bin/bash
# Orchestrate the per-backbone flat-GNN Optuna search on Jean-Zay (or any SLURM
# cluster). Mirrors submit_sweep_jeanzay.sh, but submits ONE search array PER
# backbone (each with its own shared journal study), then chains a retrain+agg job
# that depends on all of them.
#
#   per backbone : (array size) x (--n-trials) trials, %N concurrent GPUs
#   backbones    : gin gcn gat sage sgformer   (flat-GNN-supported, published)
#
# Usage:
#   ./submit_flatgnn_search.sh [WORKERS_PER_BACKBONE] [CONCURRENCY]
#
# Examples:
#   ./submit_flatgnn_search.sh              # 4 workers/backbone, 4 concurrent
#   ./submit_flatgnn_search.sh 8 4          # 8 workers/backbone (80 trials), 4 at a time
set -euo pipefail

WORKERS="${1:-4}"
CONCURRENCY="${2:-4}"
LAST=$(( WORKERS - 1 ))

PROJECT="$WORK/protein_classification"
ACCOUNT="acu@v100"
BACKBONES=(gin gcn gat sage sgformer)

mkdir -p "$PROJECT/logs"
cd "$PROJECT"

SEARCH_ROOT="$PROJECT/runs/flatgnn_search_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SEARCH_ROOT/trials"
SEARCH_TS=$(basename "$SEARCH_ROOT" | grep -oE '[0-9]{8}_[0-9]{6}' | head -1)

echo "search root : $SEARCH_ROOT"
echo "per backbone: array 0-$LAST (%$CONCURRENCY)  x  --n-trials(10) = $(( WORKERS * 10 )) trials"
echo "backbones   : ${BACKBONES[*]}"

# ---------- Step 1: one search array per backbone ----------
SEARCH_JIDS=()
for bb in "${BACKBONES[@]}"; do
    JID=$(sbatch --parsable \
        --job-name="FLATGNN_OPT_${bb}" \
        --array=0-${LAST}%${CONCURRENCY} \
        --output="$PROJECT/logs/log_%x_%A_%a.out" \
        --error="$PROJECT/logs/log_%x_%A_%a.err" \
        slurm_flatgnn_optuna.slurm "$bb" "$SEARCH_ROOT")
    SEARCH_JIDS+=("$JID")
    echo "  submitted search array for $bb : $JID"
done

# join dependency list: retrain waits for ALL search arrays (afterok)
DEP=$(IFS=:; echo "${SEARCH_JIDS[*]}")

# ---------- Step 2: retrain-best array (5 backbones x 3 seeds = 15 tasks) ----------
# afterok on every search array so best_config_<bb>.json exist first.
RETRAIN_JID=$(sbatch --parsable \
    --dependency=afterok:${DEP} \
    slurm_flatgnn_retrain_best.slurm "$SEARCH_ROOT")
echo "submitted retrain array : $RETRAIN_JID  (afterok: $DEP)"

# ---------- Step 3: aggregate (after the whole retrain array) ----------
AGG_JID=$(sbatch --parsable \
    --job-name=FLATGNN_AGG \
    --dependency=afterany:${RETRAIN_JID} \
    --account="$ACCOUNT" --qos=qos_gpu-dev --gres=gpu:1 \
    --constraint=v100-32g --cpus-per-task=10 --hint=nomultithread --time=00:20:00 \
    --output="$PROJECT/logs/log_%x_%j.out" --error="$PROJECT/logs/log_%x_%j.err" \
    --wrap="module purge; module load pytorch-gpu/py3/2.8.0; source $WORK/myvenv/bin/activate; cd $PROJECT; python -u optuna_search_flatgnn.py --mode agg --backbones gin gcn gat sage sgformer --output-dir $SEARCH_ROOT --trial-output-dir $SEARCH_ROOT/retrain --seeds 1 2 3")
echo "submitted agg job       : $AGG_JID  (afterany: $RETRAIN_JID)"

echo
echo "watch:    squeue -u \$USER"
echo "per-bb best:  $SEARCH_ROOT/best_config_<bb>.json + best_command_<bb>.sh"
echo "final:        $SEARCH_ROOT/summary_flatgnn.json  (mean +/- std per backbone)"
