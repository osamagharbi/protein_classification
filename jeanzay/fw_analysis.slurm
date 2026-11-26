#!/bin/bash
#SBATCH --job-name=FW_ANALYSIS
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --constraint=v100-32g
#SBATCH --qos=qos_gpu-t3
#SBATCH --account=acu@v100
#SBATCH --output=logs/fw/%x_%j.out
#SBATCH --error=logs/fw/%x_%j.err
#
# Point 2: ensembles across patch resolutions + patch attribution + rotation test
# (+ SHREC2025 re-scoring on the 28 Oct 2025 labels), from existing checkpoints.
# Run jeanzay/extract_checkpoints.sh first.   sbatch jeanzay/fw_analysis.slurm
set -euo pipefail

module purge
module load pytorch-gpu/py3/2.8.0
source "$WORK/myvenv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" PYTHONUNBUFFERED=1

PROJECT="${PROJECT:-$WORK/protein_classification}"
CKPT="${CKPT:-$SCRATCH/fw_ckpts/runs}"
DATA_ROOT="${DATA_ROOT:-$([ -d "$SCRATCH/fw_data" ] && echo "$SCRATCH/fw_data" || echo "$WORK")}"
OUT="${OUT:-$PROJECT/runs/fw_analysis}"
cd "$PROJECT"

A="python -u tools/ensemble_attribution.py --data-root $DATA_ROOT --rotations 5 --num-workers 8"

# SHREC2019 protein: patch-count sweep k = 8..256 (10 seeds each) -> ensembles + attribution
$A --runs-root "$CKPT/shrec19_recheck" --include '^shrec19_protein_geom_k[0-9]+_gvp_h256_s[0-9]+$' \
   --out "$OUT/shrec19_protein"

# SHREC2019 species: k = 64 only in the recheck -> attribution + rotation test
$A --runs-root "$CKPT/shrec19_recheck" --include '^shrec19_species_geom_k[0-9]+_gvp_h256_s[0-9]+$' \
   --out "$OUT/shrec19_species"

# SHREC2025: rotation test + re-scoring on the organisers' updated test labels
if [ -d "$DATA_ROOT/ssl_graphs_1k_geom_enriched_ri" ] && [ -d "$DATA_ROOT/ssl_graphs_1k_patchassign_k64" ]; then
  $A --runs-root "$CKPT/shrec_crosseval" --include '^shrec25_[a-z]+_k[0-9]+_gvp_h256_s[0-9]+$' \
     --relabel-test-csv data_labels/shrec25_update_281025/test_labels.csv \
     --out "$OUT/shrec25"
else
  echo "SKIPPED SHREC2025: graphs not in $DATA_ROOT (run jeanzay/restore_data.sh, then resubmit;"
  echo "        finished SHREC2019 stages are cached and not recomputed)"
fi

echo "reports:"; ls -1 "$OUT"/*/report.md 2>/dev/null || true
