#!/bin/bash
#SBATCH --job-name=HIERGRID
#SBATCH --time=12:00:00
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
# One array task = ONE grid point (backbone x hidden x seed) = ONE GPU.
# Max-GPU parallelism comes from the array's %CONCURRENCY (set by submit script),
# NOT from splitting a single job over 4 GPUs. Identical philosophy to the riconv /
# flat-GNN launches: many independent single-GPU trainings running at once.
#
# The cell (dataset / feature-group / patch-k) and the grid axes are passed via
# environment variables exported by submit_grid_hiergin.sh, so this file never
# needs editing.
#
# Required env (exported by submit script):
#   DATA_ROOT      dir containing the graph dirs           ($WORK on Jean-Zay)
#   GRID_ROOT      output root for this study
#   DATASET        shrec19_protein | shrec19_species | shrec25
#   FEATURE_GROUP  geom | enriched
#   PATCH_K        16 | 32 | 64 | 128 | 256
#   BACKBONES HIDDEN_DIMS SEEDS   space-separated lists (must match submit-time plan)
#
set -euo pipefail
set -x

module purge
module load pytorch-gpu/py3/2.8.0
source "$WORK/myvenv/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

PROJECT="$WORK/protein_classification"
cd "$PROJECT"
mkdir -p "$PROJECT/logs"

echo "Node=$(hostname)  ARRAY_JOB=${SLURM_ARRAY_JOB_ID}  TASK=${SLURM_ARRAY_TASK_ID}"
echo "cell: dataset=$DATASET feature=$FEATURE_GROUP patch_k=$PATCH_K root=$GRID_ROOT"
nvidia-smi -L || { echo "no GPU"; exit 1; }

srun python -u grid_hiergin.py \
  --mode          run \
  --task-id       "${SLURM_ARRAY_TASK_ID}" \
  --dataset       "$DATASET" \
  --feature-group "$FEATURE_GROUP" \
  --patch-k       "$PATCH_K" \
  --data-root     "$DATA_ROOT" \
  --grid-root     "$GRID_ROOT" \
  --backbones     $BACKBONES \
  --hidden-dims   $HIDDEN_DIMS \
  --seeds         $SEEDS \
  --device        0 \
  --num-workers   "${SLURM_CPUS_PER_TASK}"

echo "===== grid task ${SLURM_ARRAY_TASK_ID} done ====="
