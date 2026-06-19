#!/bin/bash
set -e
PY=/home/ogharbi/miniconda3/envs/graphmae/bin/python3
SCRIPT=/home/ogharbi/projects/protein_classification/classify_hierarchical_gnn.py
ASSIGN=/home/ogharbi/datasets/SHREC25/ssl_graphs_1k_patchassign_k64
RUNS=/home/ogharbi/projects/protein_classification/runs
BASE="--assign-dir $ASSIGN --hidden-dim 256 --node-layers 3 --patch-layers 2
      --readout max --lr 5e-4 --batch-size 96 --num-workers 4
      --epochs 60 --patience 30 --class-weight-method inverse_sqrt
      --output-dir $RUNS --device 0 --seed 42"

echo "=== [1/4] coords + potential (4-dim), GIN, meanmax pool ==="
$PY $SCRIPT $BASE \
    --feature-mode coords_potential \
    --node-backbone gin --pool meanmax \
    --run-name hier_k64_coords_pot

echo "=== [2/4] all features (8-dim), GAT backbone, meanmax pool ==="
$PY $SCRIPT $BASE \
    --feature-mode all \
    --node-backbone gat --pool meanmax \
    --run-name hier_k64_gat

echo "=== [3/4] all features (8-dim), GCN backbone, meanmax pool ==="
$PY $SCRIPT $BASE \
    --feature-mode all \
    --node-backbone gcn --pool meanmax \
    --run-name hier_k64_gcn

echo "=== [4/4] all features (8-dim), GIN backbone, attention patch pool ==="
$PY $SCRIPT $BASE \
    --feature-mode all \
    --node-backbone gin --pool attn \
    --run-name hier_k64_attn_pool

echo "=== All done ==="
