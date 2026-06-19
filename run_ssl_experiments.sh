#!/bin/bash
# Run SSL-embedding experiments sequentially (safe for 12GB GPU)
# node-layers=0 uses NO node GNN — direct pool of 256-dim SSL embs to patches (cheapest)
# node-layers=3 uses GIN on full mesh — batch=16 + num-workers=0 to avoid edge-activation OOM
#   (32×5000 edges × 256-dim × 3 layers × 2 directions ≈ 1GB just for node GNN activations)
set -e
PY=/home/ogharbi/miniconda3/envs/graphmae/bin/python3
SCRIPT=/home/ogharbi/projects/protein_classification/classify_hierarchical_gnn.py
GRAPH_DIR=/home/ogharbi/datasets/SHREC25/ssl_graphs_1k
ASSIGN_DIR=/home/ogharbi/datasets/SHREC25/ssl_graphs_1k_patchassign_k64
SSL_DIR=/home/ogharbi/datasets/SHREC25/ssl_embeddings_graphsage_coordspotential
RUNS=/home/ogharbi/projects/protein_classification/runs
COMMON_BASE="--graph-dir $GRAPH_DIR --assign-dir $ASSIGN_DIR
             --hidden-dim 256 --patch-layers 2
             --pool meanmax --readout max
             --lr 5e-4 --epochs 60 --patience 30
             --class-weight-method inverse_sqrt
             --output-dir $RUNS --device 0 --seed 42"

echo "=== [1/3] SSL + no node GNN (direct pool of SSL embs → patches) ==="
# Safe: no DGL message-passing on 1000-node graphs at all
$PY $SCRIPT $COMMON_BASE \
    --ssl-emb-dir $SSL_DIR \
    --node-layers 0 \
    --batch-size 32 --num-workers 2 \
    --run-name hier_k64_ssl_nonode

echo "=== [2/3] SSL + GIN node backbone (3L) — reduced batch to avoid edge-activation OOM ==="
$PY $SCRIPT $COMMON_BASE \
    --ssl-emb-dir $SSL_DIR \
    --node-layers 3 --node-backbone gin \
    --batch-size 16 --num-workers 0 \
    --run-name hier_k64_ssl_gin

echo "=== [3/3] SSL + GraphSAGE node backbone (3L) ==="
$PY $SCRIPT $COMMON_BASE \
    --ssl-emb-dir $SSL_DIR \
    --node-layers 3 --node-backbone sage \
    --batch-size 16 --num-workers 0 \
    --run-name hier_k64_ssl_sage

echo "All SSL experiments done."
