#!/bin/bash
#SBATCH --job-name=FLATGNN_BEST
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --constraint=v100-32g
#SBATCH --qos=qos_gpu-t3
#SBATCH --account=acu@v100
#SBATCH --output=logs/log_%x_%j.out
#SBATCH --error=logs/log_%x_%j.err
#
# Full-epoch, multi-seed retrain of the best config found per backbone, followed
# by aggregation to mean +/- std. Mirrors the branch's slurm_retrain_best_jeanzay.
#
# Usage:
#   sbatch slurm_flatgnn_retrain_best.slurm <SEARCH_ROOT>
#
set -euo pipefail
set -x

SEARCH_ROOT="${1:?Usage: sbatch slurm_flatgnn_retrain_best.slurm <SEARCH_ROOT>}"

module purge
module load pytorch-gpu/py3/2.8.0
source "$WORK/myvenv/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

PROJECT="$WORK/protein_classification"
DATA="$WORK/ssl_graphs_1k"
cd "$PROJECT"
mkdir -p "$PROJECT/logs"

echo "Node: $(hostname)  JOB=${SLURM_JOB_ID}  search_root=$SEARCH_ROOT"
nvidia-smi -L || { echo "no GPU"; exit 1; }

# retrain every backbone's best config across seeds {1,2,3}, then aggregate
srun python -u optuna_search_flatgnn.py \
  --mode           retrain-best \
  --backbones      gin gcn gat sage sgformer \
  --data-dir       "$DATA" \
  --output-dir     "$SEARCH_ROOT" \
  --trial-output-dir "$SEARCH_ROOT/retrain" \
  --retrain-epochs 120 \
  --seeds          1 2 3 \
  --batch-size     64 \
  --num-workers    "${SLURM_CPUS_PER_TASK}"

srun python -u optuna_search_flatgnn.py \
  --mode           agg \
  --backbones      gin gcn gat sage sgformer \
  --output-dir     "$SEARCH_ROOT" \
  --trial-output-dir "$SEARCH_ROOT/retrain" \
  --seeds          1 2 3

echo "===== retrain + agg complete ====="
echo "summary: $SEARCH_ROOT/summary_flatgnn.json"
