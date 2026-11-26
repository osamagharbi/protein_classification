#!/bin/bash
# Submit a future-work training study as one Slurm array (one GPU per run).
#
#   STUDY=rotation      ./jeanzay/submit_fw.sh    # point 3 (6 arms x 10 seeds = 60 runs)
#   STUDY=multiscale19  ./jeanzay/submit_fw.sh    # point 4, SHREC2019 (4 arms x 10 = 40 runs)
#   STUDY=multiscale25  ./jeanzay/submit_fw.sh    # point 4, SHREC2025 (needs k16/k256 sidecars)
#   STUDY=edges         ./jeanzay/submit_fw.sh    # bidirectional message passing (4 arms x 10 = 40 runs)
#
# Options (env): SMOKE=1 (1 epoch, seed 1, dev QoS: check everything runs first)
#                DRY_RUN=1  SEEDS="1 2 3"  ARMS="s25final_inv s25final_base"
#                CONCURRENCY=30  DATA_ROOT=$WORK  QOS  TIME
set -euo pipefail

export STUDY="${STUDY:?set STUDY=rotation|multiscale19|multiscale25|edges}"
export SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
export ARMS="${ARMS:-}"
export DATA_ROOT="${DATA_ROOT:-$([ -d "$SCRATCH/fw_data" ] && echo "$SCRATCH/fw_data" || echo "$WORK")}"
export PROJECT="${PROJECT:-$WORK/protein_classification}"
CONCURRENCY="${CONCURRENCY:-30}"
QOS="${QOS:-qos_gpu-t3}"
TIME="${TIME:-20:00:00}"
export SMOKE_FLAG=""
if [ "${SMOKE:-0}" = "1" ]; then
  export SMOKE_FLAG="--smoke" SEEDS="1"
  QOS="qos_gpu-dev"; TIME="01:30:00"
fi

cd "$PROJECT"
mkdir -p logs/fw
module purge >/dev/null 2>&1 || true
module load pytorch-gpu/py3/2.8.0 >/dev/null 2>&1 || true

PLAN=$(python jeanzay/fw_grid.py --mode plan --study "$STUDY" --seeds $SEEDS ${ARMS:+--arms $ARMS} ${SMOKE_FLAG})
echo "$PLAN"
N=$(echo "$PLAN" | sed -n 's/^TASKS=//p')
[ -n "$N" ] && [ "$N" -gt 0 ] || { echo "empty plan"; exit 1; }
echo "study=$STUDY tasks=$N qos=$QOS time=$TIME concurrency=$CONCURRENCY data_root=$DATA_ROOT"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1: nothing submitted"; exit 0
fi
sbatch --job-name="FW_${STUDY}" --array="0-$((N - 1))%${CONCURRENCY}" \
       --qos="$QOS" --time="$TIME" --export=ALL jeanzay/fw_train.slurm
echo "when done:  python jeanzay/fw_grid.py --mode agg --study $STUDY ${SMOKE_FLAG}"
