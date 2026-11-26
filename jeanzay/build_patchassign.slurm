#!/bin/bash
#SBATCH --job-name=FW_PATCHES
#SBATCH --time=03:00:00
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
# Build missing patch-assignment sidecars (CPU work; a GPU is requested only because
# acu@v100 rejects CPU-only jobs). Default: the SHREC2025 resolutions needed by the
# multiscale25 study. Each k adds ~11.5k files (inodes) for SHREC2025.
#
#   sbatch jeanzay/build_patchassign.slurm
#   KS="16 256" GRAPHS=ssl_graphs_1k_geom_enriched_ri PREFIX=ssl_graphs_1k_patchassign sbatch ...
set -euo pipefail

module purge
module load pytorch-gpu/py3/2.8.0
source "$WORK/myvenv/bin/activate"

PROJECT="${PROJECT:-$WORK/protein_classification}"
DATA_ROOT="${DATA_ROOT:-$WORK}"
GRAPHS="${GRAPHS:-ssl_graphs_1k_geom_enriched_ri}"
PREFIX="${PREFIX:-ssl_graphs_1k_patchassign}"
KS="${KS:-16 256}"
cd "$PROJECT"

# sanity check: the builder must reproduce an existing k=64 sidecar exactly
python -u tools/build_patchassign.py --input "$DATA_ROOT/$GRAPHS" \
    --verify-against "$DATA_ROOT/${PREFIX}_k64" --num-patches 64 --max-graphs 20

for k in $KS; do
  python -u tools/build_patchassign.py --input "$DATA_ROOT/$GRAPHS" \
      --out "$DATA_ROOT/${PREFIX}_k${k}" --num-patches "$k" --workers "${SLURM_CPUS_PER_TASK}"
done
