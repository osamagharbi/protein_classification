#!/usr/bin/env python3
"""Check what's happening with the train/val split."""

import pandas as pd
from pathlib import Path
from fix_training import create_robust_train_val_split
from collections import Counter

# Load labels
data_dir = '/home/ogharbi/datasets/SHREC25/graphs/normalized_8feat_1k/'
labels_file = Path(data_dir) / 'train_labels.csv'
df = pd.read_csv(labels_file)

train_labels = {}
for _, row in df.iterrows():
    train_labels[str(row['protein_id'])] = int(row['class_id'])

print(f"Total samples: {len(train_labels)}")
print(f"Total classes: {len(set(train_labels.values()))}")

# Create split
train_indices, val_indices = create_robust_train_val_split(
    None,  # Don't need dataset object
    train_labels,
    val_split=0.15,
    min_samples_for_stratify=10,
    seed=42
)

# Get labels for train and val
all_ids = list(train_labels.keys())
all_labels = list(train_labels.values())

train_split_labels = [all_labels[i] for i in train_indices]
val_split_labels = [all_labels[i] for i in val_indices]

print(f"\n{'='*80}")
print("CHECKING VALIDATION SET")
print(f"{'='*80}")

val_class_counts = Counter(val_split_labels)
print(f"\nValidation set:")
print(f"  Total samples: {len(val_split_labels)}")
print(f"  Unique classes: {len(val_class_counts)}")
print(f"\nClass distribution in validation:")
for class_id, count in sorted(val_class_counts.items())[:20]:
    print(f"  Class {class_id}: {count} samples")

if len(val_class_counts) == 1:
    print(f"\n❌ PROBLEM: Validation has only ONE class: {list(val_class_counts.keys())[0]}")
    print(f"   This explains the 100% accuracy!")
