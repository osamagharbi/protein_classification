#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RUN_NAME="${1:-upstream_dgcnn_shrec25_$(date +%Y%m%d_%H%M%S)}"
LOG="runs_pc/${RUN_NAME}.log"

mkdir -p runs_pc

python train_dgcnn_shrec25.py \
  --run-name "${RUN_NAME}" \
  --data-dir /home/ogharbi/datasets/SHREC25/ssl_graphs_1k_geom_enriched_ri \
  --input-signal pos \
  --num-points 1000 \
  --epochs 250 \
  --patience 50 \
  --batch-size 8 \
  --test-batch-size 16 \
  --grad-accum 2 \
  --num-workers 6 \
  --optimizer adam \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --scheduler step \
  --step-size 20 \
  --gamma 0.7 \
  --dropout 0.5 \
  --k 20 \
  --emb-dims 1024 \
  --monitor f1_weighted \
  2>&1 | tee "${LOG}"
