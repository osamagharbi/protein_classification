#!/bin/bash
#SBATCH --job-name=FLATGNN_OPT
#SBATCH --time=08:00:00
#SBATCH --array=0-5%4
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
# One Optuna worker for a SINGLE backbone on one GPU. Every array task shares ONE
# per-backbone study through a journal file (NFS/Lustre-safe), so the TPE sampler
# coordinates across workers -- identical mechanism to the branch's
# slurm_optuna_jeanzay.slurm, but per-backbone and targeting the flat GNN.
#
# The search space now includes a `feature_set` dimension (4 cumulative base-8D
# combinations: coords / +normals / +normal_potential / +potential), so the budget
# is raised to ~60 trials/backbone to cover it.
#   total trials / backbone = (array size) x (--n-trials per worker)  = 6 x 10 = 60
#   concurrency             = the %N in --array (here 4 GPUs at a time)
#
# Usage (normally submitted by submit_flatgnn_search.sh, which sets --array):
#   sbatch slurm_flatgnn_optuna.slurm <BACKBONE> <SEARCH_ROOT>
#
set -euo pipefail
set -x

BACKBONE="${1:?Usage: sbatch slurm_flatgnn_optuna.slurm <BACKBONE> <SEARCH_ROOT>}"
SEARCH_ROOT="${2:?missing SEARCH_ROOT}"

# ---------- Environment ----------
module purge
module load pytorch-gpu/py3/2.8.0
source "$WORK/myvenv/bin/activate"          # must contain: optuna, pandas, scikit-learn, dgl, tqdm

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

PROJECT="$WORK/protein_classification"
DATA="$WORK/ssl_graphs_1k_geom_enriched_ri"  # self-contained (train/val/test CSVs inside); 21D features
cd "$PROJECT" || { echo "cd $PROJECT failed"; exit 1; }
mkdir -p "$PROJECT/logs"

echo "Node: $(hostname)  backbone=$BACKBONE  ARRAY_JOB=${SLURM_ARRAY_JOB_ID}  TASK=${SLURM_ARRAY_TASK_ID}"
nvidia-smi -L || { echo "no GPU"; exit 1; }

JOURNAL="${SEARCH_ROOT}/journal_${BACKBONE}.log"
OUT="${SEARCH_ROOT}"
TRIALS="${SEARCH_ROOT}/trials"
mkdir -p "$OUT" "$TRIALS"

# ---------- One worker joins the shared per-backbone study ----------
srun python -u optuna_search_flatgnn.py \
  --mode           search \
  --backbone       "$BACKBONE" \
  --data-dir       "$DATA" \
  --journal-file   "$JOURNAL" \
  --output-dir     "$OUT" \
  --trial-output-dir "$TRIALS" \
  --worker-id      "${SLURM_ARRAY_TASK_ID}" \
  --n-trials       10 \
  --search-epochs  60 \
  --warmup-epochs  5 \
  --batch-size     64 \
  --num-workers    "${SLURM_CPUS_PER_TASK}" \
  --timeout        27000

echo "===== worker done: ${BACKBONE} task ${SLURM_ARRAY_TASK_ID} ====="
echo "artifacts: ${OUT}/best_config_${BACKBONE}.json + best_command_${BACKBONE}.sh"
