#!/bin/bash
# Round-2 sweep — new baseline: cosine LR → 0.691 macro-F1
# Experiments:
#   [1] cosine + 150 epochs (more time to converge)
#   [2] cosine + oversampling (combine best two improvements)
#   [3] cosine + hidden 512 (more capacity)
#   [4] cosine + RWPE-16 (global topology encoding)
#   [5] cosine + RWPE-16 + oversampling (best combo candidate)
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
        --loss-type ce
        --batch-size 64 --num-workers 4
        --lr 5e-4 --weight-decay 1e-4
        --lr-schedule cosine --warmup-epochs 5
        --output-dir $RUNS --device 0 --seed 42"

echo "=== [1/5] cosine + 150 epochs ==="
$PY $SCRIPT $COMMON \
    --epochs 150 \
    --run-name hier_cosine_150ep

echo "=== [2/5] cosine + oversampling ==="
$PY $SCRIPT $COMMON \
    --epochs 80 \
    --balanced-sampler --sampler-power 0.5 \
    --run-name hier_cosine_oversample

echo "=== [3/5] cosine + hidden 512 ==="
$PY $SCRIPT $COMMON \
    --epochs 80 \
    --hidden-dim 512 \
    --run-name hier_cosine_h512

echo "=== [4/5] cosine + RWPE-16 ==="
$PY $SCRIPT $COMMON \
    --epochs 80 \
    --rwpe-dim 16 \
    --run-name hier_cosine_rwpe16

echo "=== [5/5] cosine + RWPE-16 + oversampling ==="
$PY $SCRIPT $COMMON \
    --epochs 80 \
    --rwpe-dim 16 \
    --balanced-sampler --sampler-power 0.5 \
    --run-name hier_cosine_rwpe16_oversample

echo "=== All done ==="
