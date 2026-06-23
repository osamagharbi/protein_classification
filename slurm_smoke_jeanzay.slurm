#!/bin/bash
#SBATCH --job-name=RIHIER_SMOKE
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --constraint=v100-32g
#SBATCH --qos=qos_gpu-t3
#SBATCH --account=acu@v100
#SBATCH --output=/lustre/fswork/projects/rech/acu/uzh45fh/protein_classification/logs/log_%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/acu/uzh45fh/protein_classification/logs/log_%x_%j.err
#
# Pre-flight smoke test — run this ONCE before submitting slurm_optuna_jeanzay.slurm.
# Two stages on one V100:
#   (A) smoke_test.py --heavy : data + paths + heaviest model fwd/bwd, prints
#                               peak GPU mem so you know batch-size fits 32 GB.
#   (B) one real Optuna trial through the journal storage : validates the exact
#       distributed entrypoint (sampling, journal write, summary) end-to-end.
# If both print success, the full array job is safe to launch.
# (For faster scheduling during debugging you can switch --qos to qos_gpu-dev.)

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
DATA="$WORK/ssl_graphs_1k_geom_enriched_ri"
cd "$PROJECT" || { echo "cd $PROJECT failed"; exit 1; }
mkdir -p "$PROJECT/logs"

echo "Node: $(hostname)   JOB_ID=${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"
nvidia-smi || { echo "nvidia-smi failed"; exit 1; }

# ---------- (A) data + heaviest-model smoke ----------
echo "===== STAGE A: data + model smoke (heavy config) ====="
srun python -u smoke_test.py \
  --graph-dir        "$DATA" \
  --train-labels-csv "$DATA/train_split.csv" \
  --val-labels-csv   "$DATA/val_split.csv" \
  --test-labels-csv  "$DATA/test_labels.csv" \
  --batch-size 16 \
  --heavy

# ---------- (B) one real Optuna trial via the journal ----------
echo "===== STAGE B: one Optuna trial through journal storage ====="
SMOKE_ROOT="$PROJECT/runs/smoke_${SLURM_JOB_ID}"
mkdir -p "$SMOKE_ROOT"
srun python -u optuna_search_ri_hiernet.py \
  --graph-dir        "$DATA" \
  --train-labels-csv "$DATA/train_split.csv" \
  --val-labels-csv   "$DATA/val_split.csv" \
  --test-labels-csv  "$DATA/test_labels.csv" \
  --study-name       smoke \
  --journal-file     "$SMOKE_ROOT/optuna_journal.log" \
  --output-dir       "$SMOKE_ROOT" \
  --worker-id        0 \
  --n-trials         1 \
  --search-epochs    2 \
  --warmup-epochs    1 \
  --batch-size       64 \
  --num-workers      0 \
  --select-metric    accuracy

echo "===== SMOKE COMPLETE ====="
echo "If STAGE A printed [SMOKE PASSED] and STAGE B wrote:"
echo "  $SMOKE_ROOT/smoke/best_config.json"
echo "then submit:  sbatch slurm_optuna_jeanzay.slurm"
