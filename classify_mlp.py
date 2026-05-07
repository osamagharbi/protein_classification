#!/usr/bin/env python3
"""
Simple MLP-based classification for SHREC25 protein graphs.

Uses graph pooling + MLP instead of GraphMAE pretraining.

Usage:
    python classify_mlp.py --device 0
    python classify_mlp.py --device 0 --pooling mean --hidden 256 --epochs 100
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

import dgl
from shrec25_loader import SHREC25SplitDataset


class GraphPoolingMLP(nn.Module):
    """
    Simple MLP classifier with graph pooling.

    Architecture:
        Graph Pooling → MLP → Output
    """
    def __init__(self, in_dim, hidden_dim, num_classes, pooling='mean', dropout=0.5):
        super().__init__()
        self.pooling = pooling

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def pool_graph(self, g, feat):
        """Pool node features to graph-level representation."""
        with g.local_scope():
            g.ndata['h'] = feat

            if self.pooling == 'mean':
                return dgl.mean_nodes(g, 'h')
            elif self.pooling == 'max':
                return dgl.max_nodes(g, 'h')
            elif self.pooling == 'sum':
                return dgl.sum_nodes(g, 'h')
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(self, g, feat):
        """Forward pass: pool then classify."""
        # Pool to graph-level
        graph_feat = self.pool_graph(g, feat)

        # MLP classification
        logits = self.mlp(graph_feat)
        return logits


def load_labels(dataset_root, split='train'):
    """Load actual labels from CSV files."""
    label_file = Path(dataset_root).parent / f"{split}_labels.csv"

    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    df = pd.read_csv(label_file)

    # Create mapping from graph name to label
    label_dict = {}
    for _, row in df.iterrows():
        graph_name = str(row['protein_id'])  # Use protein_id column
        label = int(row['class_id'])  # Already 0-indexed (0 to 98)
        label_dict[graph_name] = label

    num_classes = len(df['class_id'].unique())

    print(f"Loaded {len(label_dict)} labels for {split} set")
    print(f"Number of classes: {num_classes}")

    return label_dict, num_classes


def collate_fn(batch):
    """Collate function for dataloader."""
    graphs = []
    labels = []

    for g, label in batch:
        # Add self-loops
        g = g.add_self_loop()

        # Remove _ID fields to avoid schema mismatch
        if '_ID' in g.ndata:
            g.ndata.pop('_ID')
        if '_ID' in g.edata:
            g.edata.pop('_ID')

        graphs.append(g)
        labels.append(label)

    # Batch graphs
    batch_g = dgl.batch(graphs)
    batch_labels = torch.cat(labels, dim=0)

    return batch_g, batch_labels


class LabeledSHREC25Dataset(SHREC25SplitDataset):
    """SHREC25 dataset with actual labels."""

    def __init__(self, root_dir, split='train', label_dict=None):
        super().__init__(root_dir, split)
        self.label_dict = label_dict

    def __getitem__(self, idx):
        """Load graph and its true label."""
        graph_file = self.graph_files[idx]

        # Load graph
        graphs, _ = dgl.load_graphs(str(graph_file))
        dgl_graph = graphs[0]

        # Get true label
        if self.label_dict is not None:
            graph_name = graph_file.stem  # Filename without extension
            label = self.label_dict.get(graph_name, 0)  # Default to 0 if not found
        else:
            label = 0

        return dgl_graph, torch.tensor([label], dtype=torch.long)


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for batch_g, labels in pbar:
        batch_g = batch_g.to(device)
        labels = labels.to(device).squeeze()
        feat = batch_g.ndata['attr']

        # Forward
        optimizer.zero_grad()
        logits = model(batch_g, feat)
        loss = criterion(logits, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_g, labels in tqdm(loader, desc="Evaluating"):
            batch_g = batch_g.to(device)
            labels = labels.to(device).squeeze()
            feat = batch_g.ndata['attr']

            # Forward
            logits = model(batch_g, feat)
            loss = criterion(logits, labels)

            # Metrics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def main():
    parser = argparse.ArgumentParser(description="Simple MLP classification for SHREC25")

    # Data
    parser.add_argument('--data-dir', type=str, default='/home/osm/datasets/SHREC25/normalized_8feat',
                       help='Dataset directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Dataloader workers')

    # Model
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'max', 'sum'],
                       help='Pooling method')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')

    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--device', type=int, default=0, help='GPU device')

    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"\n{'='*70}")
    print(f"SHREC25 MLP CLASSIFICATION")
    print(f"{'='*70}")
    print(f"Data dir:      {args.data_dir}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Hidden dim:    {args.hidden_dim}")
    print(f"Pooling:       {args.pooling}")
    print(f"Dropout:       {args.dropout}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs:        {args.epochs}")
    print(f"{'='*70}\n")

    # Load labels
    print("Loading labels...")
    train_labels, num_classes = load_labels(args.data_dir, 'train')
    test_labels, _ = load_labels(args.data_dir, 'test')

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = LabeledSHREC25Dataset(args.data_dir, 'train', train_labels)
    test_dataset = LabeledSHREC25Dataset(args.data_dir, 'test', test_labels)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    print(f"Num classes:   {num_classes}")

    # Get input dimension from first graph
    sample_graph, _ = train_dataset[0]
    in_dim = sample_graph.ndata['attr'].shape[1]
    print(f"Input dim:     {in_dim}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    # Create model
    print("\nInitializing model...")
    model = GraphPoolingMLP(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        pooling=args.pooling,
        dropout=args.dropout
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}\n")

    best_test_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # Print summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.4f}")

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_mlp_model.pth')
            print(f"  ✓ New best test accuracy: {best_test_acc:.4f}")

    # Final results
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Best Test Accuracy: {best_test_acc:.4f}")
    print(f"Final Train Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final Test Accuracy: {history['test_acc'][-1]:.4f}")

    # Save history
    with open('mlp_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Training history saved to mlp_training_history.json")
    print(f"✓ Best model saved to best_mlp_model.pth")


if __name__ == "__main__":
    main()
