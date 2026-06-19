#!/bin/bash
# Improvement sweep on SHREC25 hierarchical GNN.
# Best baseline: GIN / coords+pot (4-dim) / attn pool → 0.652 macro-F1
# Six experiments:
#   [1] focal loss (gamma=2)
#   [2] oversampling (WeightedRandomSampler, power=0.5)
#   [3] focal + oversampling
#   [4] cosine LR schedule (no early stop)
#   [5] Graph Transformer node backbone
#   [6] two-level attention (attn pool + attn readout) — best combo
set -e
PY=/home/ogharbi/miniconda3/envs/graphmae/bin/python3
SCRIPT=/home/ogharbi/projects/protein_classification/classify_hierarchical_gnn.py
GRAPH_DIR=/home/ogharbi/datasets/SHREC25/ssl_graphs_1k
ASSIGN_DIR=/home/ogharbi/datasets/SHREC25/ssl_graphs_1k_patchassign_k64
RUNS=/home/ogharbi/projects/protein_classification/runs

COMMON="--graph-dir $GRAPH_DIR --assign-dir $ASSIGN_DIR
        --hidden-dim 256 --node-layers 3 --patch-layers 2
        --pool attn --readout max
        --feature-mode coords_potential
        --class-weight-method inverse_sqrt
        --batch-size 64 --num-workers 4
        --epochs 80 --patience 30
        --lr 5e-4 --weight-decay 1e-4
        --output-dir $RUNS --device 0 --seed 42"

echo "=== [1/6] focal loss (gamma=2) ==="
$PY $SCRIPT $COMMON \
    --loss-type focal --focal-gamma 2.0 \
    --run-name hier_focal

echo "=== [2/6] oversampling (power=0.5) ==="
$PY $SCRIPT $COMMON \
    --balanced-sampler --sampler-power 0.5 \
    --run-name hier_oversample

echo "=== [3/6] focal + oversampling ==="
$PY $SCRIPT $COMMON \
    --loss-type focal --focal-gamma 2.0 \
    --balanced-sampler --sampler-power 0.5 \
    --run-name hier_focal_oversample

echo "=== [4/6] cosine LR (80 epochs, warmup=5) ==="
$PY $SCRIPT $COMMON \
    --lr-schedule cosine --warmup-epochs 5 \
    --run-name hier_cosine

echo "=== [5/6] Graph Transformer node backbone ==="
$PY $SCRIPT $COMMON \
    --node-backbone gt \
    --run-name hier_gt_node

echo "=== [6/6] two-level attn (attn pool + attn readout) ==="
$PY $SCRIPT $COMMON \
    --readout attn \
    --run-name hier_attn2

echo "=== All done ==="
