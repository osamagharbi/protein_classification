#!/bin/bash
#SBATCH --job-name=FW
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --constraint=v100-32g
#SBATCH --qos=qos_gpu-t3
#SBATCH --account=acu@v100
#SBATCH --output=logs/fw/%x_%A_%a.out
#SBATCH --error=logs/fw/%x_%A_%a.err
#
# Array body for jeanzay/submit_fw.sh: one array task = one (arm, seed) training.
# Env (exported by submit_fw.sh): STUDY SEEDS ARMS DATA_ROOT PROJECT SMOKE_FLAG
set -euo pipefail

module purge
module load pytorch-gpu/py3/2.8.0
source "$WORK/myvenv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" PYTHONUNBUFFERED=1

cd "$PROJECT"
echo "node=$(hostname) job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} study=$STUDY"
nvidia-smi -L

srun python -u jeanzay/fw_grid.py --mode run --study "$STUDY" \
    --task-id "${SLURM_ARRAY_TASK_ID}" --seeds $SEEDS ${ARMS:+--arms $ARMS} \
    --data-root "$DATA_ROOT" --num-workers 8 ${SMOKE_FLAG:-}
