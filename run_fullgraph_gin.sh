#!/bin/bash
# Hierarchical GIN on full-resolution SSL graphs (avg 16k nodes, max 40k).
# Waits for patch assignment generation (PID 3912625) to finish first.
#
# Memory budget (profiled):
#   bs=4, avg 16k nodes/graph  → 2.2 GB
#   bs=4, worst 40k nodes/graph → 5.0 GB
#   GraphMAE co-process         → 6.7 GB
#   Total worst case            → 11.7 GB  (within 12 GB RTX A2000)
#
# Epochs: 20 full passes over 7395 graphs (2311 batches/epoch at bs=4)
#   = 46k gradient steps vs 9k for 1k-graph run → well past convergence.

set -e
PY=/home/ogharbi/miniconda3/envs/graphmae/bin/python3
SCRIPT=/home/ogharbi/projects/protein_classification/classify_hierarchical_gnn.py
GRAPH_DIR=/home/ogharbi/datasets/SHREC25/ssl_graphs
ASSIGN_DIR=/home/ogharbi/datasets/SHREC25/ssl_graphs_patchassign_k64
RUNS=/home/ogharbi/projects/protein_classification/runs

echo "=== Waiting for patch generation (PID 3912625) ==="
while kill -0 3912625 2>/dev/null; do
    DONE=$(ls "$ASSIGN_DIR"/train_set/ 2>/dev/null | wc -l)
    echo "  [$(date '+%H:%M:%S')] $DONE/9244 train patches done..."
    sleep 60
done
echo "=== Patch generation complete. Checking output ==="
DONE_TRAIN=$(ls "$ASSIGN_DIR"/train_set/ 2>/dev/null | wc -l)
DONE_TEST=$(ls  "$ASSIGN_DIR"/test_set/  2>/dev/null | wc -l)
echo "  Train patches: $DONE_TRAIN | Test patches: $DONE_TEST"

echo "=== Launching full-graph hierarchical GIN ==="
$PY $SCRIPT \
    --graph-dir  "$GRAPH_DIR" \
    --assign-dir "$ASSIGN_DIR" \
    --hidden-dim 512 --node-layers 3 --patch-layers 2 \
    --pool attn --readout max \
    --feature-mode coords_potential \
    --class-weight-method inverse_sqrt \
    --loss-type ce \
    --batch-size 4 --num-workers 4 \
    --epochs 20 --lr 5e-4 --weight-decay 1e-4 \
    --lr-schedule cosine --warmup-epochs 3 \
    --run-name hier_gin_fullgraph \
    --output-dir "$RUNS" \
    --device 0 --seed 42

echo "=== Done ==="
