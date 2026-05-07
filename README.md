# Protein Classification

Independent classification scripts for SHREC25 protein dataset.

## Structure

```
protein_classification/
├── classify_gnn.py       # GNN classifier (GAT/GCN/GIN)
├── classify_mlp.py       # MLP baseline classifier
└── runs/                 # Training runs (auto-generated)
```

## Usage

### GNN Classifier

Train with default GAT model:
```bash
cd /home/osm/projects/protein_classification
python classify_gnn.py --device 0
```

Train with GIN model:
```bash
python classify_gnn.py --device 0 --model gin --hidden-dim 512 --num-layers 5
```

Key arguments:
- `--model`: gat, gcn, or gin (default: gat)
- `--hidden-dim`: Hidden layer size (default: 256)
- `--num-layers`: Number of GNN layers (default: 3)
- `--batch-size`: Batch size (default: 16)
- `--epochs`: Training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-3)
- `--val-split`: Validation split ratio (default: 0.2)

### MLP Baseline

Train simple MLP baseline:
```bash
python classify_mlp.py --device 0
```

## Output

GNN classifier creates timestamped run directories:
```
runs/
└── gin_h512_l5_20251217_092345/
    ├── best_model.pth
    ├── training_history.json
    ├── final_results.json
    └── checkpoint_epoch*.pth
```

MLP classifier saves directly to current directory:
- `best_mlp_model.pth`
- `mlp_training_history.json`

## Dataset

Expected data directory: `/home/osm/datasets/SHREC25/normalized_8feat/`

Features:
- 8-dimensional node features (pos_x, pos_y, pos_z, normal_x, normal_y, normal_z, potential_norm, normal_potential_norm)
- 99 protein classes
- Train/val/test splits
