#!/usr/bin/env python3
"""Check what's happening with the train/val split."""

import argparse
import os
import pandas as pd
from pathlib import Path
from fix_training import create_robust_train_val_split
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect train/val split class coverage")
    parser.add_argument(
        "--data-dir",
        default=os.getenv("SHREC25_GRAPH_DATA_DIR", "datasets/SHREC25/graphs/normalized_8feat_1k"),
        help="Path containing train_labels.csv",
    )
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--min-samples-for-stratify", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    labels_file = Path(args.data_dir) / "train_labels.csv"
    df = pd.read_csv(labels_file)

    train_labels = {}
    for _, row in df.iterrows():
        train_labels[str(row["protein_id"])] = int(row["class_id"])

    print(f"Total samples: {len(train_labels)}")
    print(f"Total classes: {len(set(train_labels.values()))}")

    train_indices, val_indices = create_robust_train_val_split(
        None,
        train_labels,
        val_split=args.val_split,
        min_samples_for_stratify=args.min_samples_for_stratify,
        seed=args.seed,
    )

    all_labels = list(train_labels.values())
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
        print("   This explains the 100% accuracy!")


if __name__ == "__main__":
    main()
