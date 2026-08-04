#!/bin/bash
#SBATCH --job-name=PROBE
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --constraint=v100-32g
#SBATCH --qos=qos_gpu-dev
#SBATCH --account=acu@v100
#SBATCH --output=logs/log_%x_%A_%a.out
#SBATCH --error=logs/log_%x_%A_%a.err
#
# Array body for run_shrec19_probe.sh. One array task = re-score ONE existing
# checkpoint on the corrected test set (inference only, no training) -- cheap
# enough for qos_gpu-dev. Reads its (tag, orig_run_dir, graph_dir, assign_dir,
# test_labels_csv) row from MANIFEST.tsv.
#
# Required env (exported by run_shrec19_probe.sh):
#   MANIFEST   path to the manifest.tsv (columns: tag orig_run_dir graph_dir assign_dir test_labels_csv)
#   OUT_ROOT   where to write each task's final_results_corrected_test.json
#
set -euo pipefail
set -x

module purge
module load pytorch-gpu/py3/2.8.0
source "$WORK/myvenv/bin/activate"

export PYTHONUNBUFFERED=1

PROJECT="$WORK/protein_classification"
cd "$PROJECT"
mkdir -p "$PROJECT/logs"

# task-id is 0-based; manifest row 1 is the header
LINE=$(( SLURM_ARRAY_TASK_ID + 2 ))
ROW=$(sed -n "${LINE}p" "$MANIFEST")
[ -z "$ROW" ] && { echo "[probe] no manifest row at line $LINE"; exit 1; }

TAG=$(echo "$ROW"        | cut -f1)
RUN_DIR=$(echo "$ROW"    | cut -f2)
GRAPH_DIR=$(echo "$ROW"  | cut -f3)
ASSIGN_DIR=$(echo "$ROW" | cut -f4)
TEST_CSV=$(echo "$ROW"   | cut -f5)

echo "Node=$(hostname)  ARRAY_JOB=${SLURM_ARRAY_JOB_ID}  TASK=${SLURM_ARRAY_TASK_ID}  tag=$TAG"
nvidia-smi -L || { echo "no GPU"; exit 1; }

mkdir -p "$OUT_ROOT"
srun python -u probe_checkpoint.py \
  --run-dir          "$RUN_DIR" \
  --graph-dir        "$GRAPH_DIR" \
  --assign-dir       "$ASSIGN_DIR" \
  --test-labels-csv  "$TEST_CSV" \
  --device           0 \
  --num-workers      "${SLURM_CPUS_PER_TASK}" \
  --out              "$OUT_ROOT/${TAG}.json"

echo "===== probe task ${SLURM_ARRAY_TASK_ID} ($TAG) done ====="
