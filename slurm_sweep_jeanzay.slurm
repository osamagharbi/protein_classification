#!/bin/bash
#SBATCH --job-name=RIHIER_SWEEP
#SBATCH --time=20:00:00
#SBATCH --array=0-499
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --constraint=v100-32g
#SBATCH --qos=qos_gpu-t3
#SBATCH --account=acu@v100
#SBATCH --output=logs/log_%x_%A_%a.out
#SBATCH --error=logs/log_%x_%A_%a.err
#
# Parallel seed-averaged sweep — ONE training per GPU.
#   array task i  ->  config_id = i // N_SEEDS,  seed_id = i % N_SEEDS
# 100 configs x 5 seeds = 500 tasks (qos_gpu-t3 allows up to 512 GPUs/user).
#
# Do NOT submit this directly — use ./submit_sweep_jeanzay.sh, which generates
# configs.json first, sizes --array to match, and chains the aggregation job.
# (If you do submit by hand: pass SWEEP_ROOT as the first argument and make sure
#  $SWEEP_ROOT/configs.json already exists.)

set -euo pipefail
set -x

N_SEEDS=5
SWEEP_ROOT="${1:?pass SWEEP_ROOT as arg1 (dir holding configs.json)}"
SEED_OFFSET="${2:-0}"

# ---- optional loggers ----
# TensorBoard: always on (writes to $SWEEP_ROOT/tb/<run>/)
# W&B: set WANDB_PROJECT to your project name to enable, leave empty to disable
WANDB_PROJECT="ri-hiernet-sweep"
WANDB_ARGS=""
if [ -n "$WANDB_PROJECT" ]; then
  # Jean-Zay compute nodes have no internet — run offline, sync later from login node:
  #   wandb sync $SWEEP_ROOT/wandb/run-*
  export WANDB_MODE=offline
  export WANDB_DIR="$SWEEP_ROOT/wandb"
  mkdir -p "$WANDB_DIR"
  WANDB_ARGS="--wandb-project $WANDB_PROJECT --wandb-group $(basename $SWEEP_ROOT)"
fi

# ---------- Environment ----------
module purge
module load pytorch-gpu/py3/2.8.0
source "$WORK/myvenv/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

PROJECT="$WORK/protein_classification"
DATA="$WORK/ssl_graphs_1k_geom_enriched_ri"
cd "$PROJECT"

CONFIG_ID=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_ID=$(( SLURM_ARRAY_TASK_ID % N_SEEDS + SEED_OFFSET ))
echo "task ${SLURM_ARRAY_TASK_ID} -> config ${CONFIG_ID} seed ${SEED_ID} (offset=${SEED_OFFSET}) on $(hostname)"
nvidia-smi -L || { echo "no GPU"; exit 1; }

# Stagger the start so 500 tasks don't open ~9k Lustre files at the same instant.
sleep $(( (SLURM_ARRAY_TASK_ID % 50) * 2 ))

srun python -u optuna_search_ri_hiernet.py \
  --mode train \
  --configs-file "$SWEEP_ROOT/configs.json" \
  --config-id    "$CONFIG_ID" \
  --seed-id      "$SEED_ID" \
  --result-dir   "$SWEEP_ROOT/results" \
  --graph-dir        "$DATA" \
  --train-labels-csv "$DATA/train_split.csv" \
  --val-labels-csv   "$DATA/val_split.csv" \
  --test-labels-csv  "$DATA/test_labels.csv" \
  --search-epochs 300 \
  --patience      30 \
  --warmup-epochs 5 \
  --batch-size    64 \
  --num-workers   0 \
  --select-metric accuracy \
  --eval-test \
  --tb-dir "$SWEEP_ROOT/tb" \
  $WANDB_ARGS

echo "task ${SLURM_ARRAY_TASK_ID} done -> $SWEEP_ROOT/results/cfg$(printf %03d $CONFIG_ID)_seed${SEED_ID}.json"
