#!/usr/bin/env python3
"""
GNN-based classification for SHREC25 using graph structure.

Uses GAT/GCN/GIN/GraphSAGE/EGNN layers to learn from graph topology.

Usage:
    # Basic training
    python classify_gnn.py --device 0
    python classify_gnn.py --device 0 --model gat --hidden-dim 256 --num-layers 3
    python classify_gnn.py --device 0 --model gin --hidden-dim 512 --num-layers 5
    python classify_gnn.py --device 0 --model sage --hidden-dim 384 --num-layers 3
    python classify_gnn.py --device 0 --model egnn --hidden-dim 256 --num-layers 4
    
    # With rotation augmentation (SSL-inspired)
    python classify_gnn.py --device 0 --model gat --rotation-aug
    python classify_gnn.py --device 0 --model gcn --rotation-ssl --rotation-weight 0.1
    python classify_gnn.py --device 0 --model gat --rotation-ssl --test-time-aug
    
    # Maximum rotation robustness
    python classify_gnn.py --device 0 --model gat --rotation-ssl --rotation-weight 0.1 --test-time-aug
    
See ROTATION_AUGMENTATION_README.md for detailed documentation.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

import dgl
from dgl.nn.pytorch import GATConv, GraphConv, GINConv, SAGEConv
from gvp_dgl import DGLGVPEncoder
from shrec25_loader import SHREC25SplitDataset
from fix_training import compute_class_weights, create_robust_train_val_split


def resolve_split_dir(root_dir, split):
    """Resolve train/test split directory across naming variants."""
    root = Path(root_dir)
    candidates = [root / split]
    if split in ('train', 'test'):
        candidates.append(root / f'{split}_set')
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def resolve_graph_file(split_dir, graph_name):
    """Resolve graph file for a graph id across supported formats."""
    # Try exact id, reduced-graph id with _graph suffix, and reverse mapping.
    candidates = [graph_name]
    if not graph_name.endswith('_graph'):
        candidates.append(f"{graph_name}_graph")
    else:
        candidates.append(graph_name[:-6])

    for stem in candidates:
        for suffix in ('.bin', '.pt'):
            p = split_dir / f"{stem}{suffix}"
            if p.exists():
                return p
    return split_dir / f"{graph_name}.bin"


def canonical_graph_id(graph_id: str) -> str:
    """Normalize graph ids across variants (e.g., *_graph reduced format)."""
    gid = str(graph_id)
    return gid[:-6] if gid.endswith('_graph') else gid


def load_graph_file(graph_file):
    """Load a graph from .bin or .pt into DGL format."""
    if graph_file.suffix == '.bin':
        graphs, _ = dgl.load_graphs(str(graph_file))
        return graphs[0]

    if graph_file.suffix == '.pt':
        data = torch.load(graph_file, map_location='cpu', weights_only=False)
        if not isinstance(data, dict):
            raise TypeError(f"Unsupported .pt content in {graph_file}")

        edge_index = data.get('edge_index_mesh', data.get('edge_index'))
        feat = data.get('x', data.get('pos'))
        if edge_index is None or feat is None:
            raise KeyError(
                f"{graph_file} missing required keys. Expected edge_index_mesh/edge_index and x/pos"
            )

        if not torch.is_tensor(edge_index):
            edge_index = torch.as_tensor(edge_index)
        if not torch.is_tensor(feat):
            feat = torch.as_tensor(feat)

        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=int(feat.shape[0]))
        feat = feat.float()
        # Clip z-scored scalars (indices 0=Potential_norm, 1=NormalPotential_norm) at ±5σ
        # to prevent outliers (observed up to ±12.5σ) from dominating max pooling
        feat[:, :2] = feat[:, :2].clamp(-5.0, 5.0)
        g.ndata['attr'] = feat
        if 'pos' in data and torch.is_tensor(data['pos']):
            g.ndata['pos'] = data['pos'].float()
        return g

    raise ValueError(f"Unsupported graph format: {graph_file}")


# ============================================================================
# ROTATION AUGMENTATION UTILITIES
# ============================================================================

def rotate_graph_z(g, angle_deg):
    """
    Rotate graph coordinates around Z-axis.
    
    Args:
        g: DGL graph with 'attr' node features (first 3 dims are XYZ)
        angle_deg: Rotation angle in degrees (0, 90, 180, or 270)
    
    Returns:
        Rotated graph (new copy)
    """
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation_matrix = torch.tensor([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0,      0,     1]
    ], dtype=torch.float32, device=g.ndata['attr'].device)
    
    # Clone graph
    g_rot = g.clone()
    
    # Extract and rotate XYZ coordinates (first 3 features)
    feat = g_rot.ndata['attr'].clone()
    xyz = feat[:, :3]  # [N, 3]
    xyz_rot = torch.matmul(xyz, rotation_matrix.T)
    
    # Update features with rotated coordinates
    feat[:, :3] = xyz_rot
    g_rot.ndata['attr'] = feat
    
    return g_rot


def apply_random_rotation(g, angles=[0, 90, 180, 270]):
    """Apply random rotation from predefined angles."""
    angle = np.random.choice(angles)
    return rotate_graph_z(g, angle), angle


def create_rotated_batch(g, angles=[0, 90, 180, 270]):
    """
    Create batch with all rotation augmentations.
    
    Args:
        g: Single DGL graph
        angles: List of rotation angles
    
    Returns:
        batch_g: Batched graph with len(angles) copies
        rotation_labels: Tensor of rotation indices [0, 1, 2, 3]
    """
    rotated_graphs = []
    for angle in angles:
        g_rot = rotate_graph_z(g, angle)
        rotated_graphs.append(g_rot)
    
    batch_g = dgl.batch(rotated_graphs)
    rotation_labels = torch.arange(len(angles), dtype=torch.long)
    
    return batch_g, rotation_labels


class EGNNLayer(nn.Module):
    """
    E(n) Equivariant Graph Neural Network Layer.
    
    Based on: "E(n) Equivariant Graph Neural Networks" (Satorras et al., 2021)
    https://arxiv.org/abs/2102.09844
    
    This layer is equivariant to rotations, translations, and reflections.
    It updates both node features AND coordinates.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, edge_dim=1, act_fn=nn.SiLU(), 
                 residual=True, normalize=False, coords_agg='mean'):
        super().__init__()
        self.residual = residual
        self.normalize = normalize
        self.coords_agg = coords_agg
        
        # Edge MLP: takes node features of i, j and edge attributes (distances)
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )
        
        # Node MLP: updates node features
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, out_dim)
        )
        
        # Coordinate MLP: predicts coordinate update weights
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        # Initialize coord_mlp last layer to small values for stability
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)
    
    def forward(self, h, pos, edge_index, edge_attr=None):
        """
        Args:
            h: Node features [N, in_dim]
            pos: Node coordinates [N, 3]
            edge_index: Edge indices [2, E]
            edge_attr: Edge attributes [E, edge_dim] (optional, uses distances if None)
        
        Returns:
            h_out: Updated node features [N, out_dim]
            pos_out: Updated coordinates [N, 3]
        """
        src, dst = edge_index[0], edge_index[1]
        
        # Compute relative positions and distances
        rel_pos = pos[src] - pos[dst]  # [E, 3]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # [E, 1]
        
        if edge_attr is None:
            edge_attr = dist
        
        # Edge features: concatenate source, destination features, and edge attributes
        edge_feat = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        edge_out = self.edge_mlp(edge_feat)  # [E, hidden_dim]
        
        # Coordinate update
        coord_weights = self.coord_mlp(edge_out)  # [E, 1]
        coord_diff = rel_pos * coord_weights  # [E, 3]
        
        # Aggregate coordinate updates
        num_nodes = h.shape[0]
        coord_agg = torch.zeros_like(pos)
        coord_agg.index_add_(0, dst, coord_diff)
        
        if self.coords_agg == 'mean':
            # Normalize by degree
            degree = torch.zeros(num_nodes, device=h.device)
            degree.index_add_(0, dst, torch.ones(src.shape[0], device=h.device))
            degree = degree.clamp(min=1).unsqueeze(-1)
            coord_agg = coord_agg / degree
        
        pos_out = pos + coord_agg
        
        # Node feature update: aggregate edge messages
        msg_agg = torch.zeros(num_nodes, edge_out.shape[-1], device=h.device)
        msg_agg.index_add_(0, dst, edge_out)
        
        # Update node features
        node_input = torch.cat([h, msg_agg], dim=-1)
        h_out = self.node_mlp(node_input)
        
        # Residual connection (if dimensions match)
        if self.residual and h.shape[-1] == h_out.shape[-1]:
            h_out = h + h_out
        
        return h_out, pos_out


class GNNClassifier(nn.Module):
    """
    GNN classifier using graph structure.

    Architecture:
        Input → GNN layers → Pool → MLP → Output
    """
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=3,
                 model_type='gat', num_heads=4, pooling='mean', dropout=0.3, 
                 feature_indices=None, use_rotation_head=False):
        super().__init__()

        # Enforce minimum layers for consistent architecture
        assert num_layers >= 2, f"num_layers must be >= 2 (got {num_layers})"

        self.num_layers = num_layers
        self.pooling = pooling
        self.model_type = model_type
        self.dropout = dropout
        self.feature_indices = feature_indices
        self.use_rotation_head = use_rotation_head
        raw_in_dim = in_dim
        
        # Adjust input dimension based on feature selection
        if feature_indices is not None:
            in_dim = len(feature_indices)
            print(f"  Using {in_dim} features: indices {feature_indices}")

        # GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        if model_type == 'gat':
            # GAT requires hidden_dim divisible by num_heads
            assert hidden_dim % num_heads == 0, \
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

            # GAT already has feat_drop and attn_drop, so we use lighter dropout after activation
            # to avoid over-regularization
            post_dropout = dropout * 0.5  # Reduce post-activation dropout

            # First layer - no activation in conv, we apply it after norm
            self.convs.append(GATConv(in_dim, hidden_dim // num_heads, num_heads,
                                     feat_drop=dropout, attn_drop=dropout, activation=None))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(post_dropout))

            # Hidden layers (not including last)
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, num_heads,
                                         feat_drop=dropout, attn_drop=dropout, activation=None))
                self.norms.append(nn.LayerNorm(hidden_dim))
                self.dropouts.append(nn.Dropout(post_dropout))

            # Last layer - no dropout needed after final layer
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, num_heads,
                                     feat_drop=dropout, attn_drop=dropout, activation=None))
            self.norms.append(nn.LayerNorm(hidden_dim))

        elif model_type == 'gcn':
            # First layer - no activation in conv
            self.convs.append(GraphConv(in_dim, hidden_dim, activation=None))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))

            # Hidden layers (not including last)
            for _ in range(num_layers - 2):
                self.convs.append(GraphConv(hidden_dim, hidden_dim, activation=None))
                self.norms.append(nn.LayerNorm(hidden_dim))
                self.dropouts.append(nn.Dropout(dropout))

            # Last layer - no dropout after final layer
            if num_layers > 1:
                self.convs.append(GraphConv(hidden_dim, hidden_dim, activation=None))
                self.norms.append(nn.LayerNorm(hidden_dim))
        
        elif model_type == 'resgcn':
            # ResGCN - GCN with residual connections for deeper, more powerful networks
            # Input projection to hidden_dim for residual connections
            self.input_proj = nn.Linear(in_dim, hidden_dim)
            self.input_norm = nn.LayerNorm(hidden_dim)
            
            # All layers have same dim for residual connections
            for _ in range(num_layers):
                self.convs.append(GraphConv(hidden_dim, hidden_dim, activation=None))
                self.norms.append(nn.LayerNorm(hidden_dim))
                self.dropouts.append(nn.Dropout(dropout))

        elif model_type == 'gin':
            # GIN uses MLP for each layer
            # First layer
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, 'sum', learn_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))

            # Hidden layers (not including last)
            for _ in range(num_layers - 2):
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.convs.append(GINConv(mlp, 'sum', learn_eps=True))
                self.norms.append(nn.LayerNorm(hidden_dim))
                self.dropouts.append(nn.Dropout(dropout))

            # Last layer - no dropout after final layer
            if num_layers > 1:
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.convs.append(GINConv(mlp, 'sum', learn_eps=True))
                self.norms.append(nn.LayerNorm(hidden_dim))

        elif model_type == 'sage':
            # GraphSAGE uses neighborhood sampling and aggregation
            # First layer - no activation in conv
            self.convs.append(SAGEConv(in_dim, hidden_dim, aggregator_type='pool', activation=None))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))

            # Hidden layers (not including last)
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type='pool', activation=None))
                self.norms.append(nn.LayerNorm(hidden_dim))
                self.dropouts.append(nn.Dropout(dropout))

            # Last layer - no dropout after final layer
            if num_layers > 1:
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type='pool', activation=None))
                self.norms.append(nn.LayerNorm(hidden_dim))

        elif model_type == 'egnn':
            # E(n) Equivariant GNN - rotation/translation equivariant
            # EGNN operates on coordinates and features separately
            # First layer - project input features to hidden_dim
            self.input_proj = nn.Linear(in_dim, hidden_dim)
            
            # EGNN layers (using custom EGNNLayer)
            for i in range(num_layers):
                self.convs.append(EGNNLayer(hidden_dim, hidden_dim, hidden_dim, edge_dim=1, residual=True))
                self.norms.append(nn.LayerNorm(hidden_dim))
                if i < num_layers - 1:
                    self.dropouts.append(nn.Dropout(dropout))

        elif model_type == 'gvp':
            self.gvp_encoder = DGLGVPEncoder(
                in_dim=raw_in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                feature_indices=feature_indices,
            )

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Optional rotation prediction head (for SSL multi-task learning)
        if use_rotation_head:
            self.rotation_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 4)  # 4 rotation classes (0°, 90°, 180°, 270°)
            )

    def forward(self, g, feat, pos=None):
        """Forward pass through GNN.
        
        Args:
            g: DGL graph
            feat: Node features [N, in_dim]
            pos: Node positions [N, 3] (required for EGNN, extracted from feat if not provided)
        """
        # GVP uses the full raw feature tensor and handles scalar/vector splitting internally.
        if self.model_type == 'gvp':
            h = self.gvp_encoder(g, feat)

        else:
            # Select specific features if requested
            if self.feature_indices is not None:
                h = feat[:, self.feature_indices]
            else:
                h = feat

        # EGNN requires special handling with coordinates
        if self.model_type == 'egnn':
            # Use stored 3D positions; fall back to last 3 feature dims (x_ssl, y_ssl, z_ssl)
            if pos is None:
                if 'pos' in g.ndata:
                    pos = g.ndata['pos']
                else:
                    pos = feat[:, -3:].clone()

            # Project (already feature-selected) h to hidden dim
            h = self.input_proj(h)
            
            # Convert DGL graph to PyG edge_index format
            src, dst = g.edges()
            edge_index = torch.stack([src, dst], dim=0)
            
            # Compute edge attributes (distances - rotation invariant)
            edge_attr = torch.norm(pos[src] - pos[dst], dim=1, keepdim=True)
            
            # EGNN layers
            for i, conv in enumerate(self.convs):
                h, pos = conv(h, pos, edge_index, edge_attr)
                h = self.norms[i](h)
                
                if i < len(self.convs) - 1:
                    h = F.elu(h)
                    if i < len(self.dropouts):
                        h = self.dropouts[i](h)
        else:
            # Standard GNN forward pass (GCN, GIN, SAGE, ResGCN)
            # ResGCN: project input first
            if self.model_type == 'resgcn':
                h = self.input_proj(h)
                h = self.input_norm(h)
                h = F.elu(h)
            
            for i, conv in enumerate(self.convs):
                # Save residual for ResGCN
                if self.model_type == 'resgcn':
                    residual = h
                
                # Apply convolution
                if self.model_type == 'gin':
                    h = conv(g, h)
                elif self.model_type == 'sage':
                    h = conv(g, h)
                elif self.model_type == 'gat':
                    h = conv(g, h)
                    # DGL GATConv returns [N, num_heads, out_dim]; flatten heads to [N, hidden_dim].
                    h = h.flatten(1)
                else:  # gcn or resgcn
                    h = conv(g, h)

                # Add residual connection for ResGCN
                if self.model_type == 'resgcn':
                    h = h + residual  # Residual connection
                
                h = self.norms[i](h)

                # Apply activation and dropout (except on last layer)
                if i < len(self.convs) - 1:
                    h = F.elu(h)
                    # Safer: check if dropout exists for this layer
                    if i < len(self.dropouts):
                        h = self.dropouts[i](h)

        if self.model_type == 'gvp':
            h = F.elu(h)

        # Pool to graph-level
        with g.local_scope():
            g.ndata['h'] = h
            if self.pooling == 'mean':
                hg = dgl.mean_nodes(g, 'h')
            elif self.pooling == 'max':
                hg = dgl.max_nodes(g, 'h')
            elif self.pooling == 'sum':
                hg = dgl.sum_nodes(g, 'h')
            else:
                raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Classify
        class_logits = self.mlp(hg)
        
        # Optional rotation prediction
        if self.use_rotation_head:
            rotation_logits = self.rotation_head(hg)
            return class_logits, rotation_logits
        
        return class_logits


def load_labels(dataset_root, split='train', labels_csv=None):
    """Load labels from the best-matching CSV for the target split.

    Selection is based on overlap between CSV protein_id values and graph stems
    in the resolved split directory.
    """
    split_dir = resolve_split_dir(dataset_root, split)
    graph_stems = {canonical_graph_id(p.stem) for p in split_dir.glob('*.bin')}
    if not graph_stems:
        graph_stems = {canonical_graph_id(p.stem) for p in split_dir.glob('*.pt')}

    root = Path(dataset_root)
    parent = root.parent
    candidates = []
    if labels_csv is not None:
        candidates.append(Path(labels_csv))
    else:
        candidates.extend([
            root / f"{split}_labels.csv",
            parent / 'graphs' / f"{split}_labels.csv",
            parent / 'graphs_old' / f"{split}_labels.csv",
            parent / 'graphs_pyg' / f"{split}_labels.csv",
            parent / 'graphs_dgl' / f"{split}_labels.csv",
            parent / 'pc' / f"{split}_labels.csv",
            parent / 'pc_8f' / f"{split}_labels.csv",
        ])

    best = None
    best_overlap = -1
    best_df = None
    for cand in candidates:
        if not cand.exists():
            continue
        df = pd.read_csv(cand)
        if 'protein_id' not in df.columns or 'class_id' not in df.columns:
            continue
        csv_ids = {canonical_graph_id(x) for x in df['protein_id'].astype(str)}
        overlap = len(csv_ids & graph_stems)
        if overlap > best_overlap:
            best_overlap = overlap
            best = cand
            best_df = df

    if best is None:
        attempted = "\n".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"No valid label file found for split '{split}'. Tried:\n{attempted}"
        )

    if graph_stems and best_overlap == 0:
        raise ValueError(
            f"Selected label file {best} has zero overlap with graph ids in {split_dir}"
        )

    df = best_df
    print(f"Using labels from: {best} (overlap={best_overlap}/{len(graph_stems)})")

    # Create mapping from graph name to label
    label_dict = {}
    for _, row in df.iterrows():
        graph_name = str(row['protein_id'])
        label = int(row['class_id'])  # Already 0-indexed
        label_dict[graph_name] = label

    num_classes = len(df['class_id'].unique())

    print(f"Loaded {len(label_dict)} labels for {split} set")
    print(f"Number of classes: {num_classes}")

    return label_dict, num_classes


def select_topk_populated_classes(train_labels, test_labels, topk):
    """Keep only top-k most populated classes from train labels and remap ids.

    Returns filtered/remapped train labels, filtered/remapped test labels,
    and a mapping old_class_id -> new_class_id.
    """
    if topk is None:
        return train_labels, test_labels, None

    if topk <= 0:
        raise ValueError(f"--topk-classes must be > 0, got {topk}")

    # Count class frequency on training set only.
    class_counts = {}
    for cls in train_labels.values():
        class_counts[cls] = class_counts.get(cls, 0) + 1

    if topk > len(class_counts):
        raise ValueError(
            f"--topk-classes={topk} exceeds available train classes ({len(class_counts)})"
        )

    top_classes = [cls for cls, _ in sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:topk]]
    class_map = {old: new for new, old in enumerate(top_classes)}

    train_filtered = {
        k: class_map[v]
        for k, v in train_labels.items()
        if v in class_map
    }
    test_filtered = {
        k: class_map[v]
        for k, v in test_labels.items()
        if v in class_map
    }

    print("\nTop-k populated class mode enabled")
    print(f"  Requested top-k: {topk}")
    print(f"  Train samples kept: {len(train_filtered)}/{len(train_labels)}")
    print(f"  Test samples kept:  {len(test_filtered)}/{len(test_labels)}")
    print(f"  Remapped classes: 0..{topk-1}")

    return train_filtered, test_filtered, class_map


def collate_fn(batch):
    """Collate function for dataloader."""
    graphs = []
    labels = []

    for g, label in batch:
        # Self-loops and _ID removal now done in dataset __getitem__
        graphs.append(g)
        labels.append(label)

    batch_g = dgl.batch(graphs)
    batch_labels = torch.stack(labels, dim=0)  # stack scalars into 1D tensor

    return batch_g, batch_labels


class LabeledSHREC25Dataset(SHREC25SplitDataset):
    """SHREC25 dataset with actual labels."""

    def __init__(self, root_dir, split='train', label_dict=None):
        # Don't call super().__init__() yet - we'll build graph_files from CSV
        self.root_dir = Path(root_dir)
        self.split = split
        self.label_dict = label_dict
        
        # Build graph_files list from label_dict keys (only files in CSV)
        if self.label_dict is not None:
            split_dir = resolve_split_dir(self.root_dir, split)
            self.graph_files = []
            missing = 0
            missing_examples = []
            
            for graph_name in self.label_dict.keys():
                graph_file = resolve_graph_file(split_dir, graph_name)
                if graph_file.exists():
                    self.graph_files.append(graph_file)
                else:
                    missing += 1
                    if len(missing_examples) < 20:
                        missing_examples.append(graph_name)
            
            if missing > 0:
                for name in missing_examples:
                    print(f"  Warning: {name} graph in CSV but not found in {split_dir}")
                if missing > len(missing_examples):
                    print(f"  ... and {missing - len(missing_examples)} more missing graphs")
                print(f"  Missing {missing} graphs from CSV")
            print(f"  Loaded {len(self.graph_files)} graphs from CSV")
        else:
            # Fallback: load all files (original behavior)
            super().__init__(root_dir, split)
            print(f"  Graphs: {len(self.graph_files)}")

    def __getitem__(self, idx):
        """Load graph and its true label."""
        graph_file = self.graph_files[idx]

        # Load graph from .bin or .pt
        dgl_graph = load_graph_file(graph_file)

        # Add self-loops once at load time (not every batch)
        # Remove existing self-loops first to avoid duplicates
        dgl_graph = dgl_graph.remove_self_loop().add_self_loop()

        # Remove _ID fields if present
        if '_ID' in dgl_graph.ndata:
            dgl_graph.ndata.pop('_ID')
        if '_ID' in dgl_graph.edata:
            dgl_graph.edata.pop('_ID')

        # Get true label
        if self.label_dict is not None:
            graph_name = canonical_graph_id(graph_file.stem)
            label = self.label_dict[graph_name]
        else:
            raise ValueError("label_dict must be provided for LabeledSHREC25Dataset")

        # Return label as 1D tensor (not 2D) - safer for all batch sizes
        return dgl_graph, torch.tensor(label, dtype=torch.long)


def compute_metrics(y_true, y_pred, num_classes=None):
    """Compute all classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Total number of classes (fixes sklearn warning for imbalanced datasets)
    """
    # Define all possible class labels to avoid sklearn warning
    labels = None if num_classes is None else list(range(num_classes))

    # Use macro recall over a fixed class list when num_classes is known.
    # This keeps balanced accuracy comparable across splits that may miss rare classes
    # and avoids sklearn warnings when y_pred contains labels absent from y_true.
    if labels is None:
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    else:
        balanced_accuracy = recall_score(
            y_true,
            y_pred,
            average='macro',
            zero_division=0,
            labels=labels,
        )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy,
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0, labels=labels),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0, labels=labels),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0, labels=labels),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0, labels=labels),
    }
    return metrics


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weighting."""

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce

        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss
        return loss.mean()


def train_epoch(model, loader, optimizer, criterion, device, num_classes=None, 
                use_rotation_aug=False, rotation_weight=0.1, test_time_aug=False):
    """Train for one epoch with optional rotation augmentation.
    
    Args:
        model: GNN model
        loader: DataLoader
        optimizer: Optimizer
        criterion: Loss function for classification
        device: Device
        num_classes: Number of classes
        use_rotation_aug: If True, apply random rotations during training
        rotation_weight: Weight for rotation prediction loss (multi-task learning)
        test_time_aug: If True, ensemble predictions from multiple rotations
    """
    model.train()
    total_loss = 0
    total_class_loss = 0
    total_rot_loss = 0
    all_preds = []
    all_labels = []

    # Efficient running accuracy tracking
    correct = 0
    total = 0

    # Rotation criterion (if needed)
    if model.use_rotation_head:
        rotation_criterion = nn.CrossEntropyLoss()

    pbar = tqdm(loader, desc="Training")
    for batch_g, labels in pbar:
        batch_g = batch_g.to(device)
        labels = labels.view(-1).to(device)  # Safe for batch_size=1
        feat = batch_g.ndata['attr']

        # Apply random rotation augmentation (data augmentation only)
        if use_rotation_aug and not model.use_rotation_head:
            # Simple data augmentation - apply random rotation
            graphs_list = dgl.unbatch(batch_g)
            rotated_graphs = []
            for g in graphs_list:
                g_rot, _ = apply_random_rotation(g)
                rotated_graphs.append(g_rot)
            batch_g = dgl.batch(rotated_graphs).to(device)
            feat = batch_g.ndata['attr']

        # Multi-task learning with rotation prediction
        elif model.use_rotation_head:
            # Create rotated versions for multi-task learning
            graphs_list = dgl.unbatch(batch_g)
            all_rotated = []
            rotation_labels = []
            expanded_class_labels = []
            
            for i, g in enumerate(graphs_list):
                # Create 4 rotations of each graph
                for rot_idx, angle in enumerate([0, 90, 180, 270]):
                    g_rot = rotate_graph_z(g, angle)
                    all_rotated.append(g_rot)
                    rotation_labels.append(rot_idx)
                    expanded_class_labels.append(labels[i])
            
            batch_g = dgl.batch(all_rotated).to(device)
            feat = batch_g.ndata['attr']
            rotation_labels = torch.tensor(rotation_labels, dtype=torch.long, device=device)
            expanded_class_labels = torch.tensor(expanded_class_labels, dtype=torch.long, device=device)

        # Forward
        optimizer.zero_grad()
        
        if model.use_rotation_head:
            class_logits, rotation_logits = model(batch_g, feat)
            class_loss = criterion(class_logits, expanded_class_labels)
            rot_loss = rotation_criterion(rotation_logits, rotation_labels)
            loss = class_loss + rotation_weight * rot_loss
            
            # Track losses separately
            total_class_loss += class_loss.item()
            total_rot_loss += rot_loss.item()
            
            # Use original labels for predictions (average over rotations)
            with torch.no_grad():
                # Reshape to [batch_size, 4, num_classes]
                class_logits_reshaped = class_logits.reshape(len(labels), 4, -1)
                # Average predictions over rotations
                logits = class_logits_reshaped.mean(dim=1)
        else:
            logits = model(batch_g, feat)
            loss = criterion(logits, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Collect predictions
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Efficient running accuracy
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        acc = correct / total

        if model.use_rotation_head:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{class_loss.item():.4f}',
                'rot': f'{rot_loss.item():.4f}',
                'acc': f'{acc:.4f}'
            })
        else:
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})

    # Compute all metrics
    metrics = compute_metrics(all_labels, all_preds, num_classes=num_classes)
    metrics['loss'] = total_loss / len(loader)

    return metrics


def evaluate(model, loader, criterion, device, desc="Evaluating", num_classes=None, 
             test_time_aug=False):
    """Evaluate on validation or test set with optional test-time augmentation.
    
    Args:
        model: GNN model
        loader: DataLoader
        criterion: Loss function
        device: Device
        desc: Progress bar description
        num_classes: Number of classes
        test_time_aug: If True, ensemble predictions from multiple rotations (0°, 90°, 180°, 270°)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_g, labels in tqdm(loader, desc=desc):
            batch_g = batch_g.to(device)
            labels = labels.view(-1).to(device)  # Safe for batch_size=1
            feat = batch_g.ndata['attr']

            if test_time_aug:
                # Test-time augmentation: average predictions over rotations
                graphs_list = dgl.unbatch(batch_g)
                batch_logits = []
                
                for g in graphs_list:
                    # Get predictions for all rotations
                    rot_logits = []
                    for angle in [0, 90, 180, 270]:
                        g_rot = rotate_graph_z(g, angle).to(device)
                        feat_rot = g_rot.ndata['attr']
                        
                        # Handle models with rotation head
                        output = model(g_rot, feat_rot)
                        if isinstance(output, tuple):
                            class_logits, _ = output  # Ignore rotation predictions
                        else:
                            class_logits = output
                        
                        rot_logits.append(class_logits.squeeze(0))  # Remove batch dim [1, C] -> [C]
                    
                    # Average logits over rotations: [4, num_classes] -> [num_classes]
                    avg_logits = torch.stack(rot_logits).mean(dim=0)
                    batch_logits.append(avg_logits)
                
                # Stack back to batch: list of [num_classes] -> [batch_size, num_classes]
                logits = torch.stack(batch_logits)
            else:
                # Standard evaluation
                output = model(batch_g, feat)
                if isinstance(output, tuple):
                    logits, _ = output  # Ignore rotation predictions
                else:
                    logits = output
            
            loss = criterion(logits, labels)

            # Collect predictions
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute all metrics
    metrics = compute_metrics(all_labels, all_preds, num_classes=num_classes)
    metrics['loss'] = total_loss / len(loader)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="GNN classification for SHREC25")

    # Data
    parser.add_argument('--data-dir', type=str, default='/home/ogharbi/datasets/SHREC25/graphs/normalized_8feat_1k',
                       help='Dataset directory')
    parser.add_argument('--train-labels-csv', type=str, default=None,
                       help='Path to train labels CSV (defaults to <data-dir>/train_labels.csv)')
    parser.add_argument('--test-labels-csv', type=str, default=None,
                       help='Path to test labels CSV (defaults to <data-dir>/test_labels.csv)')
    parser.add_argument('--val-labels-csv', type=str, default=None,
                       help='Path to val labels CSV for intra-class split (e.g. Mamba3D val_split.csv). '
                            'When provided, skips internal train/val split and loads val as a separate dataset.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Dataloader workers')

    # Model
    parser.add_argument('--model', type=str, default='gat', choices=['gat', 'gcn', 'resgcn', 'gin', 'sage', 'egnn', 'gvp'],
                       help='GNN model type (resgcn=ResidualGCN, egnn=E(n) Equivariant GNN, gvp=Geometric Vector Perceptron GNN)')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads (GAT)')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'max', 'sum'],
                       help='Pooling method')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--drop-features', type=int, nargs='*', default=None,
                       help='Feature indices to drop (e.g., --drop-features 0 1 2 to drop XYZ). '
                            'Features: [0-2]=XYZ coords, [3-5]=normals, [6-7]=surface properties')
    parser.add_argument('--keep-features', type=int, nargs='*', default=None,
                       help='Feature indices to keep (e.g., --keep-features 3 4 5 6 7). '
                            'Cannot use with --drop-features')
    parser.add_argument('--skip-xyz', action='store_true',
                       help='Shortcut for --drop-features 0 1 2 (skip XYZ coordinates)')

    # Rotation Augmentation (SSL-inspired)
    parser.add_argument('--rotation-aug', action='store_true',
                       help='Apply random rotation augmentation during training (rotation invariance)')
    parser.add_argument('--rotation-ssl', action='store_true',
                       help='Multi-task learning: add rotation prediction auxiliary task')
    parser.add_argument('--rotation-weight', type=float, default=0.1,
                       help='Weight for rotation prediction loss (default: 0.1)')
    parser.add_argument('--test-time-aug', action='store_true',
                       help='Test-time augmentation: ensemble predictions from 4 rotations (0°/90°/180°/270°)')

    # Training
    parser.add_argument('--epochs', type=int, default=None,
                       help='Max epochs (default: 500 with early stopping, or specify exact number)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=30,
                       help='Early stopping patience (epochs without val improvement)')
    parser.add_argument('--no-early-stopping', action='store_true',
                       help='Disable early stopping and run for full --epochs')
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio (0.2 = 20%%)')
    parser.add_argument('--loss-type', type=str, default='ce', choices=['ce', 'focal'],
                       help='Classification loss type (default: ce)')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                       help='Focal loss gamma (used when --loss-type focal)')
    parser.add_argument('--balanced-sampler', action='store_true',
                       help='Use class-balanced WeightedRandomSampler for training batches')
    parser.add_argument('--sampler-power', type=float, default=1.0,
                       help='Sampler weight exponent: w=1/(class_count^power)')
    parser.add_argument('--topk-classes', type=int, default=None,
                       help='Train/evaluate only on top-k most populated train classes (optional)')

    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-interval', type=int, default=10, help='Save progress every N epochs')
    parser.add_argument('--run-name', type=str, default=None, help='Run name (auto-generated if not provided)')
    parser.add_argument('--output-dir', type=str, default='runs', help='Output directory for runs')

    args = parser.parse_args()

    # Set default max epochs if not specified
    if args.epochs is None:
        args.epochs = 500 if not args.no_early_stopping else 200

    # Validate arguments
    assert 0.0 <= args.val_split < 1.0, \
        f"--val-split must be in range [0.0, 1.0), got {args.val_split}"
    assert args.sampler_power >= 0.0, \
        f"--sampler-power must be >= 0, got {args.sampler_power}"

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create run directory
    if args.run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{args.model}_h{args.hidden_dim}_l{args.num_layers}_{timestamp}"
    else:
        run_name = args.run_name

    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {run_dir}")

    # Device
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"\n{'='*70}")
    print(f"SHREC25 GNN CLASSIFICATION")
    print(f"{'='*70}")
    print(f"Run name:      {run_name}")
    print(f"Data dir:      {args.data_dir}")
    print(f"Model:         {args.model.upper()}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Hidden dim:    {args.hidden_dim}")
    print(f"Num layers:    {args.num_layers}")
    if args.model == 'gat':
        print(f"Num heads:     {args.num_heads}")
    print(f"Pooling:       {args.pooling}")
    print(f"Dropout:       {args.dropout}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs:        {args.epochs}")
    print(f"Val split:     {args.val_split:.1%}")
    print(f"Loss type:     {args.loss_type}")
    if args.loss_type == 'focal':
        print(f"Focal gamma:   {args.focal_gamma}")
    print(f"Sampler:       {'balanced' if args.balanced_sampler else 'standard shuffle'}")
    print(f"{'='*70}\n")

    # Determine which features to use (before loading datasets)
    feature_indices = None
    if args.drop_features is not None and args.keep_features is not None:
        raise ValueError("Cannot use both --drop-features and --keep-features")
    
    # Note: We'll validate and print actual feature info after loading a sample graph
    if args.skip_xyz:
        print("Feature mode:  Skipping XYZ coordinates (features 0-2)")
    elif args.drop_features is not None:
        print(f"Feature mode:  Dropping features {args.drop_features}")
    elif args.keep_features is not None:
        print(f"Feature mode:  Keeping only features {args.keep_features}")
    else:
        print("Feature mode:  Using all features")
    print()

    # Load labels
    print("Loading labels...")
    train_labels, num_classes = load_labels(args.data_dir, 'train', args.train_labels_csv)
    test_labels, _ = load_labels(args.data_dir, 'test', args.test_labels_csv)

    # Optional: filter to top-k most populated classes and remap labels to 0..k-1
    train_labels, test_labels, _ = select_topk_populated_classes(
        train_labels, test_labels, args.topk_classes
    )
    if args.topk_classes is not None:
        num_classes = args.topk_classes

    # Create datasets
    print("\nCreating datasets...")
    full_train_dataset = LabeledSHREC25Dataset(args.data_dir, 'train', train_labels)
    test_dataset = LabeledSHREC25Dataset(args.data_dir, 'test', test_labels)

    # Split train into train/val:
    # Option A (--val-labels-csv): load val as a separate dataset from the given CSV.
    #   Val files live in the train_set directory (intra-class split like Mamba3D).
    # Option B (default): internal random stratified split via create_robust_train_val_split.
    if args.val_labels_csv is not None:
        print(f"\nUsing external val split from: {args.val_labels_csv}")
        val_labels, _ = load_labels(args.data_dir, 'train', args.val_labels_csv)
        train_dataset = full_train_dataset
        val_dataset = LabeledSHREC25Dataset(args.data_dir, 'train', val_labels)
        # Update num_classes to cover both splits (union of class ids)
        all_class_ids = set(train_labels.values()) | set(val_labels.values())
        num_classes = len(all_class_ids)
    elif args.val_split > 0:
        train_indices, val_indices = create_robust_train_val_split(
            full_train_dataset,
            train_labels,
            val_split=args.val_split,
            min_samples_for_stratify=2,  # Only singletons kept in training only
            seed=args.seed
        )
        train_dataset = Subset(full_train_dataset, train_indices)
        val_dataset = Subset(full_train_dataset, val_indices)
    else:
        # No validation split - use all training data
        print("\nNo validation split - using all training data")
        train_dataset = full_train_dataset
        val_dataset = None

    print(f"Train samples: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Val samples:   {len(val_dataset)}")
    else:
        print(f"Val samples:   0 (no validation)")
    print(f"Test samples:  {len(test_dataset)}")
    print(f"Num classes:   {num_classes}")

    # Get input dimension and finalize feature selection
    sample_graph, _ = full_train_dataset[0]
    in_dim = sample_graph.ndata['attr'].shape[1]
    
    # Finalize which features to use
    if args.skip_xyz:
        # Shortcut: drop features 0, 1, 2 (XYZ)
        feature_indices = list(range(3, in_dim))
        print(f"Input dim:     {in_dim} → {len(feature_indices)} (skipping XYZ: features 0-2)")
    elif args.drop_features is not None:
        # Drop specific features
        feature_indices = [i for i in range(in_dim) if i not in args.drop_features]
        print(f"Input dim:     {in_dim} → {len(feature_indices)} (dropped features: {args.drop_features})")
    elif args.keep_features is not None:
        # Keep only specific features
        feature_indices = args.keep_features
        print(f"Input dim:     {in_dim} → {len(feature_indices)} (keeping features: {args.keep_features})")
    else:
        # Use all features
        print(f"Input dim:     {in_dim} (using all features)")
    
    # Validate feature indices
    if feature_indices is not None:
        if not all(0 <= i < in_dim for i in feature_indices):
            raise ValueError(f"Feature indices must be in range [0, {in_dim-1})")
        if len(feature_indices) == 0:
            raise ValueError("Must keep at least one feature")

    # Build per-sample train labels for optional balanced sampling.
    if args.val_labels_csv is not None:
        # External val split: train_dataset == full_train_dataset (all train files)
        train_sample_indices = list(range(len(full_train_dataset.graph_files)))
    elif args.val_split > 0:
        train_sample_indices = train_indices
    else:
        train_sample_indices = list(range(len(full_train_dataset.graph_files)))

    train_sample_labels = []
    for i in train_sample_indices:
        sid = canonical_graph_id(full_train_dataset.graph_files[i].stem)
        train_sample_labels.append(train_labels[sid])

    sampler = None
    shuffle_train = True
    if args.balanced_sampler:
        # Sample graphs with probability inverse to class frequency.
        class_counts_arr = np.bincount(train_sample_labels, minlength=num_classes).astype(np.float64)
        sample_weights = []
        for cls in train_sample_labels:
            count = max(class_counts_arr[cls], 1.0)
            sample_weights.append(1.0 / (count ** args.sampler_power))
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle_train = False
        print(
            f"Using balanced sampler (power={args.sampler_power}, "
            f"num_samples={len(sample_weights)})"
        )

    # Create dataloaders with optimizations for GPU training
    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_train,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_cuda,  # Faster CPU-to-GPU transfer
        persistent_workers=args.num_workers > 0  # Keep workers alive between epochs
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=use_cuda,
            persistent_workers=args.num_workers > 0
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0
    )

    # Create model
    print("\nInitializing model...")
    model = GNNClassifier(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.num_layers,
        model_type=args.model,
        num_heads=args.num_heads,
        pooling=args.pooling,
        dropout=args.dropout,
        feature_indices=feature_indices,
        use_rotation_head=args.rotation_ssl
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Print rotation augmentation config
    if args.rotation_aug or args.rotation_ssl or args.test_time_aug:
        print(f"\n{'='*70}")
        print("ROTATION AUGMENTATION CONFIG")
        print(f"{'='*70}")
        if args.rotation_aug:
            print("✓ Random rotation augmentation enabled (training)")
        if args.rotation_ssl:
            print(f"✓ Multi-task learning: rotation prediction (weight={args.rotation_weight})")
        if args.test_time_aug:
            print("✓ Test-time augmentation: 4-rotation ensemble")
        print(f"{'='*70}\n")

    # Compute class weights to handle extreme imbalance
    print("\nComputing class weights for imbalanced dataset...")
    # Build train label dict using actual dataset filenames (not CSV order)
    train_label_dict = {}
    if args.val_labels_csv is not None or args.val_split == 0:
        # External val split or no val: use full train_labels for class weights
        train_label_dict = train_labels
    else:
        # Internal split: use only training subset indices for class weights
        for i in train_indices:
            graph_name = canonical_graph_id(full_train_dataset.graph_files[i].stem)
            train_label_dict[graph_name] = train_labels[graph_name]

    class_weights = compute_class_weights(
        train_label_dict,
        num_classes,
        device,
        method='inverse_sqrt'  # Smoother than 'inverse', better for extreme imbalance
    )

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    if args.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted CrossEntropyLoss with class weights")
    else:
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
        print(f"Using FocalLoss (gamma={args.focal_gamma}) with class weights")

    # Training loop
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}\n")

    best_val_acc = 0
    best_val_f1 = 0
    best_train_f1 = -1  # Track best train F1 when no validation
    early_stopping_patience = args.patience
    epochs_without_improvement = 0
    
    # Print early stopping configuration
    if args.no_early_stopping:
        print(f"Early stopping: DISABLED (will run all {args.epochs} epochs)")
    else:
        print(f"Early stopping: ENABLED (patience={early_stopping_patience} epochs)")

    # Initialize history structure based on whether we have validation
    if val_loader is not None:
        history = {
            'train': [],
            'val': []
        }
    else:
        history = {
            'train': [],
            'val': None  # Explicitly None instead of empty list
        }

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            num_classes=num_classes,
            use_rotation_aug=args.rotation_aug,
            rotation_weight=args.rotation_weight
        )

        scheduler.step()

        # Validate (if validation set exists)
        if val_loader is not None:
            val_metrics = evaluate(
                model, val_loader, criterion, device, 
                desc="Validating", 
                num_classes=num_classes,
                test_time_aug=args.test_time_aug
            )
            history['val'].append(val_metrics)
        else:
            val_metrics = None

        # Save history (always append to train, only append to val if it's a list)
        history['train'].append(train_metrics)

        # Print summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | "
              f"Bal Acc: {train_metrics['balanced_accuracy']:.4f} | F1: {train_metrics['f1_macro']:.4f}")
        if val_metrics is not None:
            print(f"  Val   - Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
                  f"Bal Acc: {val_metrics['balanced_accuracy']:.4f} | F1: {val_metrics['f1_macro']:.4f}")
            if not args.no_early_stopping and epochs_without_improvement > 0:
                print(f"  ⏱ No improvement for {epochs_without_improvement}/{early_stopping_patience} epochs")

        # Save best model (based on validation F1 if available, otherwise training F1)
        if val_metrics is not None:
            # Use validation metrics for early stopping and best model
            if val_metrics['f1_macro'] > best_val_f1:
                best_val_f1 = val_metrics['f1_macro']
                best_val_acc = val_metrics['accuracy']
                epochs_without_improvement = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                }, run_dir / 'best_model.pth')
                print(f"  ✓ New best val F1: {best_val_f1:.4f} (Acc: {best_val_acc:.4f})")
            else:
                epochs_without_improvement += 1
                # Only apply early stopping if not disabled
                if not args.no_early_stopping and epochs_without_improvement >= early_stopping_patience:
                    print(f"\n  ⚠ Early stopping triggered at epoch {epoch+1}")
                    print(f"  No improvement in validation F1 for {early_stopping_patience} epochs")
                    print(f"  Best validation F1: {best_val_f1:.4f}")
                    break
        else:
            # No validation - track best by training F1
            # Save both best (by train F1) and last epoch
            current_train_f1 = train_metrics['f1_macro']

            # Save last epoch model (always overwrite)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
            }, run_dir / 'last_model.pth')

            # Save best model (only when train F1 improves)
            if current_train_f1 > best_train_f1:
                best_train_f1 = current_train_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                }, run_dir / 'best_model.pth')
                print(f"  ✓ New best train F1: {best_train_f1:.4f} (saved to best_model.pth)")
            else:
                print(f"  ✓ Epoch {epoch+1} saved to last_model.pth (train F1: {current_train_f1:.4f}, best: {best_train_f1:.4f})")

        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = run_dir / f'checkpoint_epoch{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, checkpoint_path)

            # Save history
            with open(run_dir / 'training_history.json', 'w') as f:
                json.dump(history, f, indent=2)

            print(f"  ✓ Checkpoint saved: {checkpoint_path}")

    # Final results
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")

    # Load best model for final test evaluation
    best_checkpoint = torch.load(run_dir / 'best_model.pth')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    final_test_metrics = evaluate(
        model, test_loader, criterion, device, 
        desc="Final Test", 
        num_classes=num_classes,
        test_time_aug=args.test_time_aug
    )

    if 'val_metrics' in best_checkpoint:
        # Had validation set - show best validation metrics
        print(f"\nBest Validation Metrics:")
        print(f"  Accuracy:          {best_checkpoint['val_metrics']['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {best_checkpoint['val_metrics']['balanced_accuracy']:.4f}")
        print(f"  F1 (macro):        {best_checkpoint['val_metrics']['f1_macro']:.4f}")
        print(f"  F1 (weighted):     {best_checkpoint['val_metrics']['f1_weighted']:.4f}")
        print(f"  Precision:         {best_checkpoint['val_metrics']['precision_macro']:.4f}")
        print(f"  Recall:            {best_checkpoint['val_metrics']['recall_macro']:.4f}")
        print(f"\nFinal Test Metrics (from best val model):")
    else:
        # No validation set - show best training metrics
        print(f"\nBest Training Metrics (epoch {best_checkpoint['epoch']+1}):")
        print(f"  Accuracy:          {best_checkpoint['train_metrics']['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {best_checkpoint['train_metrics']['balanced_accuracy']:.4f}")
        print(f"  F1 (macro):        {best_checkpoint['train_metrics']['f1_macro']:.4f}")
        print(f"  F1 (weighted):     {best_checkpoint['train_metrics']['f1_weighted']:.4f}")
        print(f"  Precision:         {best_checkpoint['train_metrics']['precision_macro']:.4f}")
        print(f"  Recall:            {best_checkpoint['train_metrics']['recall_macro']:.4f}")
        print(f"\nFinal Test Metrics (from best train F1 model):")

    print(f"  Accuracy:          {final_test_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {final_test_metrics['balanced_accuracy']:.4f}")
    print(f"  F1 (macro):        {final_test_metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):     {final_test_metrics['f1_weighted']:.4f}")
    print(f"  Precision:         {final_test_metrics['precision_macro']:.4f}")
    print(f"  Recall:            {final_test_metrics['recall_macro']:.4f}")

    # Save final history
    with open(run_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Save final test results
    with open(run_dir / 'final_results.json', 'w') as f:
        results = {
            'final_test_metrics': final_test_metrics,
            'config': vars(args)
        }
        if 'val_metrics' in best_checkpoint:
            results['best_val_metrics'] = best_checkpoint['val_metrics']
        else:
            results['best_train_metrics'] = best_checkpoint['train_metrics']
        json.dump(results, f, indent=2)

    print(f"\n✓ All outputs saved to: {run_dir}")
    print(f"  - training_history.json")
    print(f"  - final_results.json")
    print(f"  - best_model.pth", end="")
    if args.val_split == 0:
        print(" (best by train F1)")
        print(f"  - last_model.pth (last epoch)")
    else:
        print(" (best by val F1)")
    print(f"  - checkpoint_epoch*.pth")


if __name__ == "__main__":
    main()
