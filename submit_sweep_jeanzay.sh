#!/bin/bash
# Orchestrate the parallel seed-averaged RI-HierNet sweep on Jean-Zay:
#   1. gen   : sample N_CONFIGS configs ONCE  -> $SWEEP_ROOT/configs.json
#              (skipped when EXISTING_SWEEP_ROOT is supplied — reuses its configs.json)
#   2. array : N_CONFIGS x N_SEEDS GPU tasks  (one training per GPU)
#              seed_id = (task % N_SEEDS) + SEED_OFFSET
#   3. agg   : after the array, group by config -> mean +/- std + best command
#
# Usage:
#   ./submit_sweep_jeanzay.sh [N_CONFIGS] [N_SEEDS] [SEED_OFFSET] [EXISTING_SWEEP_ROOT]
#
# Examples:
#   ./submit_sweep_jeanzay.sh                          # new sweep,   seeds 0-4
#   ./submit_sweep_jeanzay.sh 100 5 5 $PREV_SWEEP      # extend sweep, seeds 5-9 (same configs.json)
#   ./submit_sweep_jeanzay.sh 100 5 10 $PREV_SWEEP     # extend sweep, seeds 10-14
set -euo pipefail

N_CONFIGS="${1:-100}"
N_SEEDS="${2:-5}"
SEED_OFFSET="${3:-0}"
EXISTING_SWEEP_ROOT="${4:-}"

TOTAL=$(( N_CONFIGS * N_SEEDS ))
LAST=$(( TOTAL - 1 ))

PROJECT="$WORK/protein_classification"
DATA="$WORK/ssl_graphs_1k_geom_enriched_ri"
ACCOUNT="acu@v100"
mkdir -p "$PROJECT/logs"
cd "$PROJECT"

# ---- sweep root: reuse existing or create new ----
if [ -n "$EXISTING_SWEEP_ROOT" ]; then
    SWEEP_ROOT="$EXISTING_SWEEP_ROOT"
    [ -f "$SWEEP_ROOT/configs.json" ] || { echo "ERROR: $SWEEP_ROOT/configs.json not found"; exit 1; }
    echo "extending sweep : $SWEEP_ROOT  (reusing configs.json)"
else
    SWEEP_ROOT="$PROJECT/runs/sweep_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$SWEEP_ROOT/results"
fi

echo "sweep root  : $SWEEP_ROOT"
echo "tasks       : $N_CONFIGS configs x $N_SEEDS seeds = $TOTAL  (array 0-$LAST)"
echo "seed range  : $SEED_OFFSET .. $(( SEED_OFFSET + N_SEEDS - 1 ))"

# ---------- Step 1: generate configs (skipped when extending) ----------
if [ -z "$EXISTING_SWEEP_ROOT" ]; then
    module purge
    module load pytorch-gpu/py3/2.8.0
    source "$WORK/myvenv/bin/activate"
    python -u optuna_search_ri_hiernet.py \
      --mode gen --n-configs "$N_CONFIGS" --n-seeds "$N_SEEDS" \
      --configs-file "$SWEEP_ROOT/configs.json"
    [ -f "$SWEEP_ROOT/configs.json" ] || { echo "config generation failed"; exit 1; }
fi

# ---------- Step 2: submit the training array ----------
# Pass SWEEP_ROOT and SEED_OFFSET as positional args to the slurm script.
ARRAY_JID=$(sbatch --parsable \
  --job-name=RIHIER_SWEEP \
  --array=0-${LAST} \
  --output="$PROJECT/logs/log_%x_%A_%a.out" \
  --error="$PROJECT/logs/log_%x_%A_%a.err" \
  slurm_sweep_jeanzay.slurm "$SWEEP_ROOT" "$SEED_OFFSET")
echo "submitted array job : $ARRAY_JID"

# ---------- Step 3: aggregate after the array finishes ----------
AGG_JID=$(sbatch --parsable \
  --job-name=RIHIER_AGG \
  --dependency=afterany:${ARRAY_JID} \
  --account="$ACCOUNT" --qos=qos_gpu-dev --gres=gpu:1 \
  --constraint=v100-32g --cpus-per-task=10 --hint=nomultithread --time=00:20:00 \
  --output="$PROJECT/logs/log_%x_%j.out" --error="$PROJECT/logs/log_%x_%j.err" \
  --wrap="module purge; module load pytorch-gpu/py3/2.8.0; source $WORK/myvenv/bin/activate; cd $PROJECT; python -u optuna_search_ri_hiernet.py --mode agg --result-dir $SWEEP_ROOT/results --graph-dir $DATA --train-labels-csv $DATA/train_split.csv --test-labels-csv $DATA/test_labels.csv")
echo "submitted agg job   : $AGG_JID  (runs after array)"

echo
echo "watch:    squeue -u \$USER"
echo "results:  $SWEEP_ROOT/results/   (one cfgXXX_seedY.json per task)"
echo "summary:  $SWEEP_ROOT/results/summary.json  +  best_command.sh  (after agg)"
