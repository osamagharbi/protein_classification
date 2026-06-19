#!/bin/bash
# Step 1: Extract remaining SSL embeddings on CPU (no GPU conflict with training)
# Step 2: Verify all files are readable
# Step 3: Run SSL training experiments sequentially
set -e

PY=/home/ogharbi/miniconda3/envs/graphmae/bin/python3
SCRIPT=/home/ogharbi/projects/protein_classification/classify_hierarchical_gnn.py
GRAPH_DIR=/home/ogharbi/datasets/SHREC25/ssl_graphs_1k
ASSIGN_DIR=/home/ogharbi/datasets/SHREC25/ssl_graphs_1k_patchassign_k64
SSL_DIR=/home/ogharbi/datasets/SHREC25/ssl_embeddings_graphsage_coordspotential
RUNS=/home/ogharbi/projects/protein_classification/runs

# ── Step 1: extract remaining embeddings on CPU ──────────────────────────────
echo "=== Step 1: Extracting remaining SSL embeddings (CPU) ==="
CUDA_VISIBLE_DEVICES="" $PY /home/ogharbi/projects/protein_classification/extract_ssl_embeddings.py
echo "Extraction done."

# ── Step 2: verify integrity ─────────────────────────────────────────────────
echo "=== Step 2: Verifying embedding integrity ==="
BAD=$($PY -c "
import torch
from pathlib import Path
bad = 0
for split in ('train', 'test'):
    d = Path('$SSL_DIR') / split
    if not d.exists(): continue
    for p in d.glob('*.pt'):
        try: torch.load(p, map_location='cpu', weights_only=False)
        except: bad += 1
print(bad)
" 2>/dev/null)
echo "Corrupt files: $BAD"
if [ "$BAD" -gt 0 ]; then
    echo "ERROR: $BAD corrupt files remain. Aborting." >&2
    exit 1
fi
echo "All files OK."

# ── Step 3: SSL training experiments ────────────────────────────────────────
COMMON="--graph-dir $GRAPH_DIR --assign-dir $ASSIGN_DIR
        --ssl-emb-dir $SSL_DIR
        --hidden-dim 256 --patch-layers 2
        --pool meanmax --readout max
        --lr 5e-4 --epochs 60 --patience 30
        --class-weight-method inverse_sqrt
        --output-dir $RUNS --device 0 --seed 42"

echo "=== [1/3] SSL + no node GNN (direct SSL pool → patches) ==="
$PY $SCRIPT $COMMON \
    --node-layers 0 \
    --batch-size 32 --num-workers 2 \
    --run-name hier_k64_ssl_nonode

echo "=== [2/3] SSL + GIN (3L node backbone) ==="
$PY $SCRIPT $COMMON \
    --node-layers 3 --node-backbone gin \
    --batch-size 16 --num-workers 0 \
    --run-name hier_k64_ssl_gin

echo "=== [3/3] SSL + GraphSAGE (3L node backbone) ==="
$PY $SCRIPT $COMMON \
    --node-layers 3 --node-backbone sage \
    --batch-size 16 --num-workers 0 \
    --run-name hier_k64_ssl_sage

echo "=== All done ==="
