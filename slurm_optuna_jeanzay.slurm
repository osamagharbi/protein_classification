#!/bin/bash
#SBATCH --job-name=RIHIER_OPTUNA
#SBATCH --time=08:00:00
#SBATCH --array=0-7%4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --constraint=v100-32g
#SBATCH --qos=qos_gpu-t3
#SBATCH --account=acu@v100
#SBATCH --output=/lustre/fswork/projects/rech/acu/uzh45fh/protein_classification/logs/log_%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/acu/uzh45fh/protein_classification/logs/log_%x_%A_%a.err
#
# Distributed Optuna HPO for RI-HierNet on SHREC25.
# One Slurm array task = one Optuna worker = one V100. All tasks in the array
# share ONE study through a journal file (Lustre/NFS-safe symlink lock), so the
# TPE sampler + Hyperband pruner coordinate across workers.
#
#   total trials  = (array size) x (--n-trials per worker)   here 8 x 8 = 64
#   concurrency   = the %N in --array (here 4 GPUs at a time)
# Scale by editing --array and --n-trials below. Each worker exits when it has
# done its --n-trials OR hits --timeout (kept < walltime so summaries flush).

set -euo pipefail
set -x

# ---------- Environment ----------
module purge
module load pytorch-gpu/py3/2.8.0
source "$WORK/myvenv/bin/activate"          # must contain: optuna, pandas, scikit-learn, tqdm

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

# ---------- Paths ----------
PROJECT="$WORK/protein_classification"
DATA="$WORK/ssl_graphs_1k_geom_enriched_ri"            # has train_set/ test_set/ + *.csv
cd "$PROJECT" || { echo "cd $PROJECT failed"; exit 1; }
mkdir -p "$PROJECT/logs"

echo "Node: $(hostname)"
echo "PWD : $(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}  TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"
nvidia-smi || { echo "nvidia-smi failed"; exit 1; }

# ---------- Shared Optuna study (one per array submission) ----------
SEARCH_ROOT="$PROJECT/runs/optuna_ri_${SLURM_ARRAY_JOB_ID}"
JOURNAL="${SEARCH_ROOT}/optuna_journal.log"
STUDY_NAME="ri_hiernet_v1"
mkdir -p "${SEARCH_ROOT}/logs"

# ---------- Run one worker ----------
# batch 64 is safe for the heaviest sampled config (width_scale=2, nsample=64)
# on a 32 GB V100; OOM trials are caught and pruned rather than crashing.
srun python -u optuna_search_ri_hiernet.py \
  --graph-dir        "$DATA" \
  --train-labels-csv "$DATA/train_split.csv" \
  --val-labels-csv   "$DATA/val_split.csv" \
  --test-labels-csv  "$DATA/test_labels.csv" \
  --study-name       "$STUDY_NAME" \
  --journal-file     "$JOURNAL" \
  --output-dir       "$SEARCH_ROOT" \
  --worker-id        "${SLURM_ARRAY_TASK_ID}" \
  --n-trials         8 \
  --search-epochs    40 \
  --warmup-epochs    5 \
  --batch-size       64 \
  --num-workers      0 \
  --select-metric    accuracy \
  --timeout          27000

# ---------- Link Slurm logs into the run folder ----------
SLURM_OUT="$PROJECT/logs/log_${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
SLURM_ERR="$PROJECT/logs/log_${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
[ -f "${SLURM_OUT}" ] && ln -sf "$(realpath "${SLURM_OUT}")" "${SEARCH_ROOT}/logs/task_${SLURM_ARRAY_TASK_ID}.out"
[ -f "${SLURM_ERR}" ] && ln -sf "$(realpath "${SLURM_ERR}")" "${SEARCH_ROOT}/logs/task_${SLURM_ARRAY_TASK_ID}.err"

echo "Search artifacts:"
echo "  ${JOURNAL}                       (shared study)"
echo "  ${SEARCH_ROOT}/${STUDY_NAME}/best_config.json"
echo "  ${SEARCH_ROOT}/${STUDY_NAME}/best_command.sh"
echo "  ${SEARCH_ROOT}/logs/"
