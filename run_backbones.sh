#!/bin/bash
# Backbone sweep — best config: cosine LR / h512 / attn pool / coords_potential / k64
# Batch sizes chosen from memory profiling on RTX A2000 12GB:
#   gin/gcn/gat/gatv2/sage → bs=64 (≤6.1GB peak)
#   gps                    → bs=32 (~5.2GB peak)
#   pna                    → bs=16 (~4.1GB peak)
set -e
PY=/home/ogharbi/miniconda3/envs/graphmae/bin/python3
SCRIPT=/home/ogharbi/projects/protein_classification/classify_hierarchical_gnn.py
GRAPH_DIR=/home/ogharbi/datasets/SHREC25/ssl_graphs_1k
ASSIGN_DIR=/home/ogharbi/datasets/SHREC25/ssl_graphs_1k_patchassign_k64
RUNS=/home/ogharbi/projects/protein_classification/runs

COMMON="--graph-dir $GRAPH_DIR --assign-dir $ASSIGN_DIR
        --hidden-dim 512 --node-layers 3 --patch-layers 2
        --pool attn --readout max
        --feature-mode coords_potential
        --class-weight-method inverse_sqrt
        --loss-type ce --num-workers 4
        --epochs 80 --lr 5e-4 --weight-decay 1e-4
        --lr-schedule cosine --warmup-epochs 5
        --output-dir $RUNS --device 0 --seed 42"

echo "=== [1/7] GIN (baseline, bs=64) ==="
$PY $SCRIPT $COMMON --batch-size 64 --node-backbone gin --run-name hier_bb_gin_cosine

echo "=== [2/7] GCN (bs=64, peak=1.6GB) ==="
$PY $SCRIPT $COMMON --batch-size 64 --node-backbone gcn --run-name hier_bb_gcn_cosine

echo "=== [3/7] GAT (bs=64, peak=2.0GB) ==="
$PY $SCRIPT $COMMON --batch-size 64 --node-backbone gat --run-name hier_bb_gat_cosine

echo "=== [4/7] GATv2 (bs=64, peak=6.0GB) ==="
$PY $SCRIPT $COMMON --batch-size 64 --node-backbone gatv2 --run-name hier_bb_gatv2_cosine

echo "=== [5/7] GraphSAGE (bs=64, peak=1.8GB) ==="
$PY $SCRIPT $COMMON --batch-size 64 --node-backbone sage --run-name hier_bb_sage_cosine

echo "=== [6/7] GPS (GIN+global attn, bs=32, peak~5.2GB) ==="
$PY $SCRIPT $COMMON --batch-size 32 --node-backbone gps --run-name hier_bb_gps_cosine

echo "=== [7/7] PNA (4 aggr x 3 scalers, bs=16, peak~4.1GB) ==="
$PY $SCRIPT $COMMON --batch-size 16 --node-backbone pna --run-name hier_bb_pna_cosine

echo "=== All done ==="
