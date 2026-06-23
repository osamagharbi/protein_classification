#!/bin/bash
# Orchestrate the parallel seed-averaged RI-HierNet sweep on Jean-Zay:
#   1. gen   : sample N_CONFIGS configs ONCE  -> $SWEEP_ROOT/configs.json
#   2. array : N_CONFIGS x N_SEEDS GPU tasks  (one training per GPU)
#   3. agg   : after the array, group by config -> mean +/- std + best command
#
# Usage:   ./submit_sweep_jeanzay.sh [N_CONFIGS] [N_SEEDS]
# Default: 100 configs x 5 seeds = 500 tasks (<= 512 GPU/user on qos_gpu-t3).
set -euo pipefail

N_CONFIGS="${1:-100}"
N_SEEDS="${2:-5}"
TOTAL=$(( N_CONFIGS * N_SEEDS ))
LAST=$(( TOTAL - 1 ))

PROJECT="$WORK/protein_classification"
DATA="$WORK/ssl_graphs_1k_geom_enriched_ri"
ACCOUNT="acu@v100"
SWEEP_ROOT="$PROJECT/runs/sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SWEEP_ROOT/results" "$PROJECT/logs"
cd "$PROJECT"

echo "sweep root : $SWEEP_ROOT"
echo "tasks      : $N_CONFIGS configs x $N_SEEDS seeds = $TOTAL  (array 0-$LAST)"

# ---------- Step 1: generate configs (CPU only, fast, deterministic) ----------
module purge
module load pytorch-gpu/py3/2.8.0
source "$WORK/myvenv/bin/activate"
python -u optuna_search_ri_hiernet.py \
  --mode gen --n-configs "$N_CONFIGS" --n-seeds "$N_SEEDS" \
  --configs-file "$SWEEP_ROOT/configs.json"
[ -f "$SWEEP_ROOT/configs.json" ] || { echo "config generation failed"; exit 1; }

# ---------- Step 2: submit the training array ----------
ARRAY_JID=$(sbatch --parsable \
  --job-name=RIHIER_SWEEP \
  --array=0-${LAST} \
  --output="$PROJECT/logs/log_%x_%A_%a.out" \
  --error="$PROJECT/logs/log_%x_%A_%a.err" \
  slurm_sweep_jeanzay.slurm "$SWEEP_ROOT")
echo "submitted array job : $ARRAY_JID"

# ---------- Step 3: aggregate after the array finishes ----------
# afterok fires only if all tasks succeed; use afterany so a few OOM/failed
# tasks still let the summary run over whatever completed.
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
