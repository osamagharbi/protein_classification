#!/usr/bin/env python3
"""
Point cloud classifier for SHREC25 protein graphs.

Reads ssl_graphs_1k .pt files, extracts point clouds (pos + features),
and trains PointNet / PointNet++ / DGCNN.

Diagnostic goal: test whether the raw data is informative for classification,
independent of graph connectivity.

Usage:
    python classify_pointcloud.py --model pointnet2
    python classify_pointcloud.py --model dgcnn
    python classify_pointcloud.py --model pointnet
    python classify_pointcloud.py --model pointnet2 --features electrostatic
    python classify_pointcloud.py --model pointnet2 --data-dir datasets/SHREC25/ssl_graphs_1k
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_split_guard import resolve_dataset_local_split_csvs
from sklearn.metrics import balanced_accuracy_score, f1_score
from shrec25_loader import infer_feature_order
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Feature sets
# ─────────────────────────────────────────────────────────────────────────────

SCALAR_FEATURE_NAMES = ['Potential_norm', 'NormalPotential_norm']
NORMAL_FEATURE_NAMES = ['nx', 'ny', 'nz']
COORD_FEATURE_NAMES = ['x_ssl', 'y_ssl', 'z_ssl']
CURVATURE_FEATURE_NAMES = [
    'principal_curvature_min',
    'principal_curvature_max',
    'mean_curvature',
    'gaussian_curvature',
]
ESSENTIAL_FEATURE_NAMES = [
    *SCALAR_FEATURE_NAMES,
    *NORMAL_FEATURE_NAMES,
    *COORD_FEATURE_NAMES,
]
POINTCLOUD_FEATURE_CHOICES = [
    'all',
    'legacy8',
    'electrostatic',
    'scalars',
    'geometric',
    'normals',
    'xyz',
    'coords',
    'no_xyz',
    'curvature',
    'legacy8_curvature',
    'coords_normals',
    'coords_normals_curvature',
    'scalars_curvature',
]

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def canonical_id(stem: str) -> str:
    return stem.replace('_graph', '')


def resolve_split_dir(data_dir: Path, split: str) -> Path:
    split_dir = data_dir / f'{split}_set'
    if split_dir.exists():
        return split_dir
    split_dir = data_dir / split
    if split_dir.exists():
        return split_dir
    raise FileNotFoundError(f'Split dir not found for {split} in {data_dir}')


def infer_dataset_feature_names(split_dir: Path):
    files = sorted(split_dir.glob('*.pt'))
    if not files:
        raise FileNotFoundError(f'No .pt files found in {split_dir}')
    data = torch.load(files[0], map_location='cpu', weights_only=False)
    feat = torch.as_tensor(data['x'])
    meta = data.get('meta', {}) if isinstance(data.get('meta', {}), dict) else {}
    return infer_feature_order(int(feat.shape[1]), meta)


def _resolve_named_feature_indices(feature_names, target_names, preset_name):
    lower_to_index = {name.lower(): idx for idx, name in enumerate(feature_names)}
    indices = []
    missing = []
    for name in target_names:
        idx = lower_to_index.get(name.lower())
        if idx is None:
            missing.append(name)
        else:
            indices.append(idx)
    if missing:
        available = ', '.join(feature_names)
        raise ValueError(
            f"Point-cloud feature preset '{preset_name}' is missing required features {missing}. "
            f"Available features: {available}"
        )
    return indices


def resolve_pointcloud_feature_selection(feature_names, preset):
    preset = preset.lower()
    preset_map = {
        'legacy8': ESSENTIAL_FEATURE_NAMES,
        'electrostatic': SCALAR_FEATURE_NAMES,
        'scalars': SCALAR_FEATURE_NAMES,
        'geometric': NORMAL_FEATURE_NAMES,
        'normals': NORMAL_FEATURE_NAMES,
        'xyz': COORD_FEATURE_NAMES,
        'coords': COORD_FEATURE_NAMES,
        'curvature': CURVATURE_FEATURE_NAMES,
        'legacy8_curvature': [*ESSENTIAL_FEATURE_NAMES, *CURVATURE_FEATURE_NAMES],
        'coords_normals': [*NORMAL_FEATURE_NAMES, *COORD_FEATURE_NAMES],
        'coords_normals_curvature': [*NORMAL_FEATURE_NAMES, *COORD_FEATURE_NAMES, *CURVATURE_FEATURE_NAMES],
        'scalars_curvature': [*SCALAR_FEATURE_NAMES, *CURVATURE_FEATURE_NAMES],
    }

    if preset == 'all':
        indices = list(range(len(feature_names)))
    elif preset == 'no_xyz':
        xyz_indices = set(_resolve_named_feature_indices(feature_names, COORD_FEATURE_NAMES, 'coords'))
        indices = [idx for idx in range(len(feature_names)) if idx not in xyz_indices]
    else:
        if preset not in preset_map:
            raise ValueError(f"Unknown point-cloud feature preset '{preset}'")
        indices = _resolve_named_feature_indices(feature_names, preset_map[preset], preset)

    selected_feature_names = [feature_names[idx] for idx in indices]
    return indices, selected_feature_names


def load_labels(data_dir: Path, split: str, labels_csv=None):
    """Load labels for the given split, searching standard candidate paths."""
    split_dir = resolve_split_dir(data_dir, split)

    graph_stems = {canonical_id(p.stem) for p in split_dir.glob('*.pt')}

    if labels_csv is not None:
        candidates = [Path(labels_csv)]
    else:
        candidates = [
            data_dir / f'{split}_labels.csv',
            data_dir.parent / 'graphs' / f'{split}_labels.csv',
            data_dir.parent / 'graphs_old' / f'{split}_labels.csv',
            data_dir.parent / 'pc' / f'{split}_labels.csv',
            data_dir.parent / 'pc_8f' / f'{split}_labels.csv',
        ]

    best, best_overlap, best_df = None, -1, None
    for cand in candidates:
        if not cand.exists():
            continue
        df = pd.read_csv(cand)
        if 'protein_id' not in df.columns or 'class_id' not in df.columns:
            continue
        csv_ids = {canonical_id(str(x)) for x in df['protein_id']}
        overlap = len(csv_ids & graph_stems)
        if overlap > best_overlap:
            best_overlap, best, best_df = overlap, cand, df

    if best is None or best_overlap == 0:
        raise FileNotFoundError(f'No valid label CSV found for {split} split in {data_dir}')

    print(f'Labels [{split}]: {best.name}  overlap={best_overlap}/{len(graph_stems)}')
    label_dict = {canonical_id(str(r['protein_id'])): int(r['class_id'])
                  for _, r in best_df.iterrows()}
    num_classes = int(best_df['class_id'].nunique())
    return label_dict, num_classes


def train_val_split(label_dict, val_ratio=0.2, min_for_stratify=2, seed=42):
    """Stratified split; rare classes (< min_for_stratify) go to train only."""
    rng = np.random.default_rng(seed)
    counts = Counter(label_dict.values())
    ids = list(label_dict.keys())
    labels = [label_dict[i] for i in ids]

    train_idx, val_idx = [], []
    for cls, cls_ids in Counter(label_dict).items():
        pass  # rebuild per class below

    by_class = {}
    for i, (sid, lbl) in enumerate(zip(ids, labels)):
        by_class.setdefault(lbl, []).append(i)

    for cls, idxs in by_class.items():
        idxs = np.array(idxs)
        if len(idxs) < min_for_stratify:
            train_idx.extend(idxs.tolist())
        else:
            rng.shuffle(idxs)
            n_val = max(1, int(round(len(idxs) * val_ratio)))
            val_idx.extend(idxs[:n_val].tolist())
            train_idx.extend(idxs[n_val:].tolist())

    return train_idx, val_idx


class ProteinPointCloudDataset(Dataset):
    def __init__(self, split_dir: Path, label_dict: dict, feature_indices: list,
                 n_points: int = 1000):
        self.split_dir = split_dir
        self.feature_indices = feature_indices
        self.n_points = n_points

        self.samples = []  # (path, label)
        for stem, label in label_dict.items():
            # Try direct name and _graph suffix variant
            for name in [stem, f'{stem}_graph']:
                p = split_dir / f'{name}.pt'
                if p.exists():
                    self.samples.append((p, label))
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = torch.load(path, map_location='cpu', weights_only=False)

        pos  = data['pos'].float()                      # [N, 3] unit-ball xyz
        feat = data['x'].float()                        # [N, F]
        feat[:, :2] = feat[:, :2].clamp(-5.0, 5.0)    # clip electrostatic outliers

        if self.feature_indices:
            feat = feat[:, self.feature_indices]        # [N, F]

        N = pos.shape[0]
        if N >= self.n_points:
            idx_sel = torch.randperm(N)[:self.n_points]
            pos  = pos[idx_sel]
            feat = feat[idx_sel]
        else:
            # Repeat-pad if somehow fewer points
            rep = (self.n_points + N - 1) // N
            pos  = pos.repeat(rep, 1)[:self.n_points]
            feat = feat.repeat(rep, 1)[:self.n_points]

        return pos, feat, label


def collate_fn(batch):
    pos  = torch.stack([b[0] for b in batch])   # [B, N, 3]
    feat = torch.stack([b[1] for b in batch])   # [B, N, F]
    lbl  = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return pos, feat, lbl


# ─────────────────────────────────────────────────────────────────────────────
# PointNet
# ─────────────────────────────────────────────────────────────────────────────

class SharedMLP(nn.Sequential):
    def __init__(self, dims, bn=True):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Conv1d(dims[i], dims[i+1], 1, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class PointNet(nn.Module):
    def __init__(self, in_dim, num_classes, dropout=0.3):
        super().__init__()
        self.mlp = SharedMLP([in_dim + 3, 64, 128, 256, 1024])  # +3 for xyz
        self.head = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, pos, feat):
        # feat: [B, N, F]  pos: [B, N, 3]
        x = torch.cat([feat, pos], dim=-1).permute(0, 2, 1)  # [B, F+3, N]
        x = self.mlp(x)                                        # [B, 1024, N]
        x = x.max(dim=2).values                               # [B, 1024]
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# PointNet++ helpers (pure torch, no torch_cluster)
# ─────────────────────────────────────────────────────────────────────────────

def fps_batch(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Farthest Point Sampling on [B, N, 3] → [B, n_samples] indices."""
    B, N, _ = xyz.shape
    selected = torch.zeros(B, n_samples, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), float('inf'), device=xyz.device)

    # Start from centroid-nearest point
    centroid = xyz.mean(dim=1, keepdim=True)               # [B, 1, 3]
    curr = ((xyz - centroid) ** 2).sum(-1).argmin(dim=1)  # [B]

    for i in range(n_samples):
        selected[:, i] = curr
        curr_xyz = xyz[torch.arange(B), curr].unsqueeze(1)  # [B, 1, 3]
        d = ((xyz - curr_xyz) ** 2).sum(-1)                  # [B, N]
        dist = torch.minimum(dist, d)
        curr = dist.argmax(dim=1)                            # [B]
    return selected


def index_points(pts: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by indices. pts [B,N,C], idx [B,M] or [B,M,K] → [B,M,C] or [B,M,K,C]"""
    B = pts.shape[0]
    if idx.dim() == 2:
        return pts[torch.arange(B, device=pts.device).unsqueeze(1).expand_as(idx), idx]
    else:  # [B, M, K]
        B, M, K = idx.shape
        flat = idx.reshape(B, -1)
        out = pts[torch.arange(B, device=pts.device).unsqueeze(1).expand_as(flat), flat]
        return out.reshape(B, M, K, pts.shape[-1])


def knn_query(xyz: torch.Tensor, query: torch.Tensor, k: int) -> torch.Tensor:
    """kNN: xyz [B,N,3], query [B,M,3] → [B,M,k] neighbour indices."""
    dists = torch.cdist(query, xyz)          # [B, M, N]
    return dists.topk(k, dim=-1, largest=False).indices  # [B, M, k]


class SetAbstractionMSG(nn.Module):
    """Multi-scale grouping set abstraction."""
    def __init__(self, n_points, radii_k_pairs, in_dim, mlps):
        """
        n_points: number of output centroids (None = global)
        radii_k_pairs: list of k values (using kNN instead of ball query for simplicity)
        in_dim: input feature dim
        mlps: list of mlp channel lists, one per scale
        """
        super().__init__()
        self.n_points = n_points
        self.ks = radii_k_pairs  # list of k values
        self.convs = nn.ModuleList()
        for i, channels in enumerate(mlps):
            layers = SharedMLP([in_dim + 3] + channels)
            self.convs.append(layers)

    def forward(self, xyz, feat):
        # xyz [B,N,3], feat [B,N,C] or None
        B, N, _ = xyz.shape

        if self.n_points is None:
            # Global: aggregate all points
            x = feat if feat is not None else xyz
            if feat is not None:
                x = torch.cat([xyz, feat], dim=-1)  # [B, N, 3+C]
            else:
                x = xyz
            x = x.permute(0, 2, 1)                  # [B, 3+C, N]
            outs = []
            for conv in self.convs:
                o = conv(x).max(dim=-1).values       # [B, out_C]
                outs.append(o)
            return None, torch.cat(outs, dim=-1)     # None, [B, sum_out]

        # FPS centroids
        cent_idx = fps_batch(xyz, self.n_points)     # [B, n_points]
        cent_xyz = index_points(xyz, cent_idx)       # [B, n_points, 3]

        outs = []
        for k, conv in zip(self.ks, self.convs):
            nn_idx = knn_query(xyz, cent_xyz, k)     # [B, n_points, k]
            grouped_xyz = index_points(xyz, nn_idx)  # [B, n_points, k, 3]
            grouped_xyz = grouped_xyz - cent_xyz.unsqueeze(2)  # relative coords

            if feat is not None:
                grouped_feat = index_points(feat, nn_idx)           # [B, n_points, k, C]
                grouped = torch.cat([grouped_xyz, grouped_feat], -1) # [B, n_points, k, 3+C]
            else:
                grouped = grouped_xyz                                # [B, n_points, k, 3]

            # Apply shared MLP: [B, n_points, k, channels] → [B, channels, k, n_points]
            grouped = grouped.permute(0, 3, 2, 1)    # [B, 3+C, k, n_points]
            # Reshape for Conv1d
            grouped = grouped.reshape(B * self.n_points, grouped.shape[1], k)
            o = conv(grouped)                         # [B*n_points, out_C, k]
            o = o.max(dim=-1).values                  # [B*n_points, out_C]
            o = o.reshape(B, self.n_points, -1)       # [B, n_points, out_C]
            outs.append(o)

        new_feat = torch.cat(outs, dim=-1)            # [B, n_points, sum_out]
        return cent_xyz, new_feat


class PointNet2(nn.Module):
    def __init__(self, in_dim, num_classes, dropout=0.4):
        super().__init__()
        # SA1: 256 centroids, 2 scales (k=16, k=32)
        self.sa1 = SetAbstractionMSG(
            n_points=256,
            radii_k_pairs=[16, 32],
            in_dim=in_dim,
            mlps=[[32, 32, 64], [64, 64, 128]],
        )
        # SA2: 64 centroids, 2 scales (k=16, k=32)
        self.sa2 = SetAbstractionMSG(
            n_points=64,
            radii_k_pairs=[16, 32],
            in_dim=64 + 128,
            mlps=[[64, 64, 128], [128, 128, 256]],
        )
        # SA3: global
        self.sa3 = SetAbstractionMSG(
            n_points=None,
            radii_k_pairs=[None],
            in_dim=128 + 256,
            mlps=[[256, 512, 1024]],
        )
        self.head = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, pos, feat):
        xyz1, f1 = self.sa1(pos, feat)
        xyz2, f2 = self.sa2(xyz1, f1)
        _,    f3 = self.sa3(xyz2, f2)
        return self.head(f3)


# ─────────────────────────────────────────────────────────────────────────────
# DGCNN
# ─────────────────────────────────────────────────────────────────────────────

def get_graph_feature(x: torch.Tensor, k: int) -> torch.Tensor:
    """Build kNN in feature space, return edge features [B, 2C, k, N]."""
    B, C, N = x.shape
    # pairwise distances in feature space
    inner = -2 * torch.bmm(x.transpose(1, 2), x)   # [B, N, N]
    xx    = (x * x).sum(dim=1, keepdim=True)         # [B, 1, N]
    dist  = xx + inner + xx.transpose(1, 2)          # [B, N, N]
    idx   = dist.topk(k=k, dim=-1, largest=False).indices  # [B, N, k]

    idx_base = torch.arange(B, device=x.device).view(-1, 1, 1) * N
    idx_flat  = (idx + idx_base).reshape(-1)

    x_t = x.transpose(1, 2).contiguous().reshape(B * N, C)
    neighbors = x_t[idx_flat].reshape(B, N, k, C).permute(0, 3, 2, 1)  # [B,C,k,N]
    x_expand  = x.unsqueeze(2).expand_as(neighbors)                      # [B,C,k,N]
    return torch.cat([x_expand, neighbors - x_expand], dim=1)            # [B,2C,k,N]


class EdgeConv(nn.Module):
    def __init__(self, in_dim, out_dim, k):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim * 2, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        # x: [B, C, N]
        edge_feat = get_graph_feature(x, self.k)   # [B, 2C, k, N]
        x = self.conv(edge_feat)                    # [B, out, k, N]
        return x.max(dim=2).values                  # [B, out, N]


class DGCNN(nn.Module):
    def __init__(self, in_dim, num_classes, k=20, emb_dims=1024, dropout=0.5):
        super().__init__()
        self.conv1 = EdgeConv(in_dim + 3, 64, k)  # +3: pos always prepended
        self.conv2 = EdgeConv(64,      64, k)
        self.conv3 = EdgeConv(64,     128, k)
        self.conv4 = EdgeConv(128,    256, k)
        self.conv5 = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, emb_dims, 1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(emb_dims * 2, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, pos, feat):
        # Use both position and features as input signal
        x = torch.cat([feat, pos], dim=-1).permute(0, 2, 1)  # [B, F+3, N]
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x  = self.conv5(torch.cat([x1, x2, x3, x4], dim=1))  # [B, emb, N]
        x_max = x.max(dim=-1).values                           # [B, emb]
        x_avg = x.mean(dim=-1)                                 # [B, emb]
        return self.head(torch.cat([x_max, x_avg], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, criterion, device, desc='Eval'):
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []
    with torch.no_grad():
        for pos, feat, lbl in tqdm(loader, desc=desc, leave=False):
            pos, feat, lbl = pos.to(device), feat.to(device), lbl.to(device)
            logits = model(pos, feat)
            total_loss += criterion(logits, lbl).item() * lbl.size(0)
            all_pred.append(logits.argmax(1).cpu())
            all_true.append(lbl.cpu())
    pred = torch.cat(all_pred).numpy()
    true = torch.cat(all_true).numpy()
    n = len(true)
    acc     = (pred == true).mean()
    bal_acc = balanced_accuracy_score(true, pred)
    f1      = f1_score(true, pred, average='macro', zero_division=0)
    return total_loss / n, acc, bal_acc, f1


def main():
    parser = argparse.ArgumentParser(description='Point cloud classifier for SHREC25')
    parser.add_argument('--data-dir', default=os.getenv('SHREC25_POINTCLOUD_DATA_DIR', 'datasets/SHREC25/ssl_graphs_1k'))
    parser.add_argument('--train-labels-csv', default=None,
                        help='Path to train labels CSV (overrides auto-discovery)')
    parser.add_argument('--val-labels-csv', default=None,
                        help='Path to val labels CSV for intra-class split (skips internal split)')
    parser.add_argument('--allow-split-mismatch', action='store_true',
                        help='Allow overriding dataset-local train_split.csv/val_split.csv with custom label CSVs')
    parser.add_argument('--model', choices=['pointnet', 'pointnet2', 'dgcnn'], default='pointnet2')
    parser.add_argument('--features', choices=POINTCLOUD_FEATURE_CHOICES, default='all',
                        help='Feature subset to use. Supports curvature-aware presets when present in the dataset metadata')
    parser.add_argument('--n-points', type=int, default=1000, help='Points per cloud (random crop)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dgcnn-k', type=int, default=20, help='DGCNN k-NN')
    parser.add_argument('--output-dir', default=os.getenv('SHREC25_POINTCLOUD_OUTPUT_DIR', 'runs_pc'))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    data_dir = Path(args.data_dir)
    args.train_labels_csv, args.val_labels_csv, split_guard_messages = resolve_dataset_local_split_csvs(
        data_dir,
        args.train_labels_csv,
        args.val_labels_csv,
        allow_mismatch=args.allow_split_mismatch,
        train_flag='--train-labels-csv',
        val_flag='--val-labels-csv',
    )
    for message in split_guard_messages:
        print(message)
    if args.val_labels_csv is not None and args.val_split > 0:
        print('Validation strategy: external split CSV in use; --val-split is ignored.')

    train_dir = resolve_split_dir(data_dir, 'train')
    test_dir = resolve_split_dir(data_dir, 'test')
    feature_names = infer_dataset_feature_names(train_dir)
    feat_idx, selected_feature_names = resolve_pointcloud_feature_selection(feature_names, args.features)
    print(f'Available features: {feature_names}')
    print(f'Selected features: {selected_feature_names}')

    # ── Labels ───────────────────────────────────────────────────────────────
    train_labels, nc_train = load_labels(data_dir, 'train', args.train_labels_csv)
    test_labels,  nc_test  = load_labels(data_dir, 'test')
    num_classes = max(nc_train, nc_test)
    print(f'Classes: {num_classes}  |  Train proteins: {len(train_labels)}  |  Test proteins: {len(test_labels)}')

    # ── Split ─────────────────────────────────────────────────────────────────
    if args.val_labels_csv is not None:
        val_labels, _ = load_labels(data_dir, 'train', args.val_labels_csv)
        num_classes = max(num_classes, len(set(train_labels.values()) | set(val_labels.values())))
        train_ds = ProteinPointCloudDataset(train_dir, train_labels, feat_idx, args.n_points)
        val_ds   = ProteinPointCloudDataset(train_dir, val_labels,   feat_idx, args.n_points)
    else:
        train_idx, val_idx = train_val_split(train_labels, val_ratio=args.val_split, seed=args.seed)
        full_train_ds = ProteinPointCloudDataset(train_dir, train_labels, feat_idx, args.n_points)
        train_ds = Subset(full_train_ds, train_idx)
        val_ds   = Subset(full_train_ds, val_idx)

    test_ds = ProteinPointCloudDataset(test_dir, test_labels, feat_idx, args.n_points)

    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    in_dim = len(feat_idx) + 3  # features + xyz position (always appended)

    if args.model == 'pointnet':
        model = PointNet(len(feat_idx), num_classes).to(device)
    elif args.model == 'pointnet2':
        model = PointNet2(len(feat_idx), num_classes).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(len(feat_idx), num_classes, k=args.dgcnn_k).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {args.model}  |  Params: {n_params:,}  |  Features: {args.features} {selected_feature_names}')
    print(f'Device: {device}')

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    # ── Output dir ────────────────────────────────────────────────────────────
    from datetime import datetime
    run_name = f'{args.model}_{args.features}_h{args.epochs}ep_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1, best_epoch = 0.0, 0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for pos, feat, lbl in tqdm(train_loader, desc=f'Ep {epoch:3d}/{args.epochs}', leave=False):
            pos, feat, lbl = pos.to(device), feat.to(device), lbl.to(device)
            optimizer.zero_grad()
            logits = model(pos, feat)
            loss = criterion(logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * lbl.size(0)
            correct += (logits.argmax(1) == lbl).sum().item()
            total += lbl.size(0)

        scheduler.step()

        tr_loss = total_loss / total
        tr_acc  = correct / total

        val_loss, val_acc, val_bal, val_f1 = evaluate(model, val_loader, criterion, device, 'Val')

        elapsed = time.time() - t0
        print(f'Ep {epoch:3d} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | '
              f'val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_bal={val_bal:.4f} val_f1={val_f1:.4f} | '
              f'{elapsed:.1f}s')

        history.append({'epoch': epoch, 'tr_loss': tr_loss, 'tr_acc': tr_acc,
                        'val_acc': val_acc, 'val_bal': val_bal, 'val_f1': val_f1})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), run_dir / 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f'\nEarly stopping at epoch {epoch} (best val F1={best_val_f1:.4f} at ep {best_epoch})')
                break

    # ── Final test ────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(run_dir / 'best_model.pth', map_location=device))
    test_loss, test_acc, test_bal, test_f1 = evaluate(model, test_loader, criterion, device, 'Test')

    print('\n' + '='*70)
    print('FINAL RESULTS')
    print('='*70)
    print(f'Best val F1: {best_val_f1:.4f}  (epoch {best_epoch})')
    print(f'Test  acc:   {test_acc:.4f}')
    print(f'Test  bal:   {test_bal:.4f}')
    print(f'Test  F1:    {test_f1:.4f}')
    print(f'Saved to:    {run_dir}')

    results = {
        'model': args.model, 'features': args.features,
        'best_val_f1': best_val_f1, 'best_epoch': best_epoch,
        'test': {'acc': test_acc, 'balanced_acc': test_bal, 'f1_macro': test_f1},
    }
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    with open(run_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()
