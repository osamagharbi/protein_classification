#!/usr/bin/env python3
"""Debug what's happening with predictions."""

import argparse
import os
import torch
from torch.utils.data import DataLoader, Subset
from collections import Counter

from classify_gnn import LabeledSHREC25Dataset, collate_fn, load_labels
from fix_training import create_robust_train_val_split

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect validation labels for robust split")
    parser.add_argument(
        "--data-dir",
        default=os.getenv("SHREC25_GRAPH_DATA_DIR", "datasets/SHREC25/graphs/normalized_8feat_1k"),
        help="Dataset root containing train split and labels",
    )
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--min-samples-for-stratify", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()

    train_labels, _ = load_labels(args.data_dir, "train")
    full_train_dataset = LabeledSHREC25Dataset(args.data_dir, "train", train_labels)

    _, val_indices = create_robust_train_val_split(
        full_train_dataset,
        train_labels,
        val_split=args.val_split,
        min_samples_for_stratify=args.min_samples_for_stratify,
        seed=args.seed,
    )

    val_dataset = Subset(full_train_dataset, val_indices)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"Validation set: {len(val_dataset)} samples")

    print("\nChecking first batch...")
    _, batch_labels = next(iter(val_loader))

    print(f"Batch size: {len(batch_labels)}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print(f"Batch labels: {batch_labels[:10]}")
    print(f"Unique labels in batch: {torch.unique(batch_labels)}")

    all_labels = []
    for _, labels in val_loader:
        all_labels.extend(labels.cpu().numpy())

    label_counts = Counter(all_labels)
    print(f"\nTotal validation labels collected: {len(all_labels)}")
    print(f"Unique classes: {len(label_counts)}")
    print(f"\nTop 10 classes:")
    for class_id, count in label_counts.most_common(10):
        print(f"  Class {class_id}: {count} samples")

    if len(label_counts) == 1:
        print("\n❌ PROBLEM: All validation labels are the same class!")
    else:
        print(f"\n✅ Validation has {len(label_counts)} different classes")


if __name__ == "__main__":
    main()
