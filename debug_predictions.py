#!/usr/bin/env python3
"""Debug what's happening with predictions."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pandas as pd
from pathlib import Path
from collections import Counter
import dgl

from classify_gnn import GNNClassifier, LabeledSHREC25Dataset, collate_fn, load_labels
from fix_training import create_robust_train_val_split

# Setup
device = torch.device('cuda:0')
data_dir = '/home/ogharbi/datasets/SHREC25/graphs/normalized_8feat_1k/'

# Load labels
train_labels, num_classes = load_labels(data_dir, 'train')

# Create dataset
full_train_dataset = LabeledSHREC25Dataset(data_dir, 'train', train_labels)

# Create split
train_indices, val_indices = create_robust_train_val_split(
    full_train_dataset,
    train_labels,
    val_split=0.15,
    min_samples_for_stratify=10,
    seed=42
)

val_dataset = Subset(full_train_dataset, val_indices)

# Create loader
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

print(f"Validation set: {len(val_dataset)} samples")

# Check what labels are in validation
print("\nChecking first batch...")
batch_g, batch_labels = next(iter(val_loader))

print(f"Batch size: {len(batch_labels)}")
print(f"Batch labels shape: {batch_labels.shape}")
print(f"Batch labels: {batch_labels[:10]}")
print(f"Unique labels in batch: {torch.unique(batch_labels)}")

# Count all labels in validation
all_labels = []
for batch_g, batch_labels in val_loader:
    all_labels.extend(batch_labels.cpu().numpy())

label_counts = Counter(all_labels)
print(f"\nTotal validation labels collected: {len(all_labels)}")
print(f"Unique classes: {len(label_counts)}")
print(f"\nTop 10 classes:")
for class_id, count in label_counts.most_common(10):
    print(f"  Class {class_id}: {count} samples")

if len(label_counts) == 1:
    print(f"\n❌ PROBLEM: All validation labels are the same class!")
else:
    print(f"\n✅ Validation has {len(label_counts)} different classes")
