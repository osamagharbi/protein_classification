#!/bin/bash
#SBATCH --job-name=RIHIER_BEST
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
# PRE-REQUISITE (run once on login node before first sbatch):
#   mkdir -p $WORK/protein_classification/logs
#
# Full retrain of the best config found by the sweep.
# Reads best_command.sh from the sweep results dir and executes it.
#
# Usage:
#   sbatch slurm_retrain_best_jeanzay.slurm <SWEEP_ROOT>
#
# Example:
#   sbatch slurm_retrain_best_jeanzay.slurm \
#       $WORK/protein_classification/runs/sweep_20260623_180544

set -euo pipefail
set -x

SWEEP_ROOT="${1:?Usage: sbatch slurm_retrain_best_jeanzay.slurm <SWEEP_ROOT>}"
BEST_CMD="$SWEEP_ROOT/results/best_command.sh"

[ -f "$BEST_CMD" ] || { echo "ERROR: $BEST_CMD not found — run agg first"; exit 1; }

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
mkdir -p "$PROJECT/logs"
cd "$PROJECT"

echo "Node     : $(hostname)   JOB_ID=${SLURM_JOB_ID}"
echo "Sweep    : $SWEEP_ROOT"
echo "Command  :"; cat "$BEST_CMD"; echo

nvidia-smi -L || { echo "no GPU"; exit 1; }

srun bash "$BEST_CMD"

echo "===== RETRAIN COMPLETE ====="
echo "Checkpoints and logs saved under $PROJECT/runs/"
