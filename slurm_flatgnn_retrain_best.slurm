#!/bin/bash
#SBATCH --job-name=FLATGNN_BEST
#SBATCH --time=20:00:00
#SBATCH --array=0-14%4
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
# PRE-REQUISITE (run once on login node before first sbatch):
#   mkdir -p $WORK/protein_classification/logs
#
# Full-epoch retrain of the best config per backbone. ONE (backbone, seed) per
# array task so trainings run in parallel across GPUs (5 backbones x 3 seeds =
# 15 tasks, array 0-14). Aggregation is a SEPARATE dependency job (--mode agg),
# submitted by submit_flatgnn_search.sh after this array completes.
#
# Usage:
#   sbatch slurm_flatgnn_retrain_best.slurm <SEARCH_ROOT>
#
# Example:
#   sbatch slurm_flatgnn_retrain_best.slurm \
#       $WORK/protein_classification/runs/flatgnn_search_20260718_210000
#
set -euo pipefail
set -x

SEARCH_ROOT="${1:?Usage: sbatch slurm_flatgnn_retrain_best.slurm <SEARCH_ROOT>}"

# ---------- Task -> (backbone, seed) mapping ----------
BACKBONES=(gin gcn gat sage sgformer)
SEEDS=(1 2 3)
N_SEEDS=${#SEEDS[@]}
BB=${BACKBONES[$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}

# ---------- Environment ----------
module purge
module load pytorch-gpu/py3/2.8.0
source "$WORK/myvenv/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

# ---------- Run ----------
PROJECT="$WORK/protein_classification"
DATA="$WORK/ssl_graphs_1k"                    # train_set/ test_set/ + train_split.csv/val_split.csv
mkdir -p "$PROJECT/logs"
cd "$PROJECT"

echo "Node : $(hostname)  ARRAY_JOB=${SLURM_ARRAY_JOB_ID}  TASK=${SLURM_ARRAY_TASK_ID}"
echo "Retrain: backbone=$BB  seed=$SEED  search_root=$SEARCH_ROOT"
[ -f "$SEARCH_ROOT/best_config_${BB}.json" ] || { echo "ERROR: best_config_${BB}.json not found — run search first"; exit 1; }
nvidia-smi -L || { echo "no GPU"; exit 1; }

srun python -u optuna_search_flatgnn.py \
  --mode             retrain-best \
  --backbones        "$BB" \
  --seeds            "$SEED" \
  --data-dir         "$DATA" \
  --output-dir       "$SEARCH_ROOT" \
  --trial-output-dir "$SEARCH_ROOT/retrain" \
  --retrain-epochs   120 \
  --batch-size       64 \
  --num-workers      "${SLURM_CPUS_PER_TASK}"

echo "===== RETRAIN COMPLETE: $BB seed $SEED ====="
echo "Checkpoints and logs saved under $SEARCH_ROOT/retrain/"
