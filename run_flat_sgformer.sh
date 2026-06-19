#!/bin/bash
# Flat (non-hierarchical) SGFormer on SHREC25
# Waits for the hierarchical SGFormer run (PID 3825061) to finish, then launches.
# Uses bs=64 once GPU is clear (~6GB freed); fall back to bs=32 if needed.
set -e
PY=/home/ogharbi/miniconda3/envs/graphmae/bin/python3
SCRIPT=/home/ogharbi/projects/protein_classification/classify_gnn.py
GRAPH_DIR=/home/ogharbi/datasets/SHREC25/ssl_graphs_1k
RUNS=/home/ogharbi/projects/protein_classification/runs

echo "=== Waiting for hierarchical SGFormer (PID 3825061) to finish ==="
while kill -0 3825061 2>/dev/null; do
    echo "  [$(date '+%H:%M:%S')] still running, sleeping 60s..."
    sleep 60
done
echo "=== Hierarchical run done. Launching flat SGFormer ==="

$PY $SCRIPT \
    --graph-dir "$GRAPH_DIR" \
    --model sgformer \
    --hidden-dim 256 \
    --num-layers 3 \
    --feature-mode coords_potential \
    --class-weight-method inverse_sqrt \
    --loss-type ce \
    --batch-size 64 \
    --num-workers 4 \
    --epochs 80 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --run-name flat_sgformer_cosine \
    --output-dir "$RUNS" \
    --device 0 \
    --seed 42

echo "=== Done ==="
