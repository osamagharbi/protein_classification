#!/bin/bash
# Focused SECOND sweep for RI-HierNet on Jean-Zay.
#
# Narrowed search space (optuna_search_ri_hiernet.py: suggest_params_focused),
# concentrated on the winning region of the first 100-config sweep:
#   patch_layers=1 · {sage,gat,gin,graphormer} · max-based readout ·
#   nsample{16,32} · patch_knn{4,8} · lr[3e-4,1e-3] · dropout[0.20,0.40].
#
# Fully separate from the first sweep — nothing is overwritten:
#   * its own timestamped folder  runs/sweep_focused_<ts>/
#   * its own job names           RIHIER_FSWEEP / RIHIER_FAGG
#   * its own W&B group           sweep_focused_<ts>  (same project)
# It reuses slurm_sweep_jeanzay.slurm for the array (train mode is space-agnostic).
#
# Usage:
#   ./submit_sweep_focused_jeanzay.sh [N_CONFIGS] [N_SEEDS] [SEED_OFFSET] [EXISTING_FOCUSED_ROOT]
# Examples:
#   ./submit_sweep_focused_jeanzay.sh                 # new focused sweep, 80x5, seeds 0-4
#   ./submit_sweep_focused_jeanzay.sh 80 5 5 $PREV    # extend it with seeds 5-9
set -euo pipefail

N_CONFIGS="${1:-80}"
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

# ---- sweep root: reuse existing focused root or create a NEW one ----
if [ -n "$EXISTING_SWEEP_ROOT" ]; then
    SWEEP_ROOT="$EXISTING_SWEEP_ROOT"
    [ -f "$SWEEP_ROOT/configs.json" ] || { echo "ERROR: $SWEEP_ROOT/configs.json not found"; exit 1; }
    echo "extending focused sweep : $SWEEP_ROOT  (reusing configs.json)"
else
    SWEEP_ROOT="$PROJECT/runs/sweep_focused_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$SWEEP_ROOT/results"
fi

# sweep timestamp (YYYYMMDD_HHMMSS) pulled from the root name; used in the log dir
SWEEP_TS=$(basename "$SWEEP_ROOT" | grep -oE '[0-9]{8}_[0-9]{6}' | head -1)
[ -n "$SWEEP_TS" ] || SWEEP_TS=$(date +%Y%m%d_%H%M%S)

echo "FOCUSED sweep root : $SWEEP_ROOT"
echo "tasks              : $N_CONFIGS configs x $N_SEEDS seeds = $TOTAL  (array 0-$LAST)"
echo "seed range         : $SEED_OFFSET .. $(( SEED_OFFSET + N_SEEDS - 1 ))"

# ---------- Step 1: generate FOCUSED configs (skipped when extending) ----------
if [ -z "$EXISTING_SWEEP_ROOT" ]; then
    module purge
    module load pytorch-gpu/py3/2.8.0
    source "$WORK/myvenv/bin/activate"
    python -u optuna_search_ri_hiernet.py \
      --mode gen --space focused --n-configs "$N_CONFIGS" --n-seeds "$N_SEEDS" \
      --configs-file "$SWEEP_ROOT/configs.json"
    [ -f "$SWEEP_ROOT/configs.json" ] || { echo "config generation failed"; exit 1; }
fi

# ---------- Step 2: submit the training array ----------
# Logs go to logs/run_<arrayjobid>_<sweepts>/ . SLURM substitutes %A (array job id)
# in the path but will NOT create the dir, so we mkdir it right after sbatch
# returns the id (well before any task is scheduled and opens its .out file).
ARRAY_JID=$(sbatch --parsable \
  --job-name=RIHIER_FSWEEP \
  --array=0-${LAST} \
  --output="$PROJECT/logs/run_%A_${SWEEP_TS}/log_%x_%A_%a.out" \
  --error="$PROJECT/logs/run_%A_${SWEEP_TS}/log_%x_%A_%a.err" \
  slurm_sweep_jeanzay.slurm "$SWEEP_ROOT" "$SEED_OFFSET")
LOG_DIR="$PROJECT/logs/run_${ARRAY_JID}_${SWEEP_TS}"
mkdir -p "$LOG_DIR"
echo "submitted focused array job : $ARRAY_JID"
echo "logs ->                       $LOG_DIR/"

# ---------- Step 3: aggregate after the array finishes ----------
AGG_JID=$(sbatch --parsable \
  --job-name=RIHIER_FAGG \
  --dependency=afterany:${ARRAY_JID} \
  --account="$ACCOUNT" --qos=qos_gpu-dev --gres=gpu:1 \
  --constraint=v100-32g --cpus-per-task=10 --hint=nomultithread --time=00:20:00 \
  --output="$LOG_DIR/log_%x_%j.out" --error="$LOG_DIR/log_%x_%j.err" \
  --wrap="module purge; module load pytorch-gpu/py3/2.8.0; source $WORK/myvenv/bin/activate; cd $PROJECT; python -u optuna_search_ri_hiernet.py --mode agg --result-dir $SWEEP_ROOT/results --graph-dir $DATA --train-labels-csv $DATA/train_split.csv --test-labels-csv $DATA/test_labels.csv")
echo "submitted focused agg job   : $AGG_JID  (runs after array)"

echo
echo "watch:    squeue -u \$USER"
echo "results:  $SWEEP_ROOT/results/   (one cfgXXX_seedY.json per task)"
echo "summary:  $SWEEP_ROOT/results/summary.json  +  best_command.sh  (after agg)"
echo
echo "first sweep is untouched (separate folder, job names, and W&B group)."
