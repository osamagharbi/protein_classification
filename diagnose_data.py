#!/usr/bin/env python3
"""
Diagnostic script for SHREC25 dataset and labels.
Checks for data issues, class imbalance, feature distributions, etc.
"""

import pandas as pd
import numpy as np
import dgl
import argparse
import os
from pathlib import Path

def load_labels(labels_dir):
    """Load train and test labels from CSV files."""
    labels_dir = Path(labels_dir)

    train_csv = labels_dir / 'train_labels.csv'
    test_csv = labels_dir / 'test_labels.csv'

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    return train_df, test_df

def analyze_label_distribution(df, split_name):
    """Analyze class distribution in a dataset split."""
    print(f"\n{'='*80}")
    print(f"{split_name.upper()} LABEL ANALYSIS")
    print(f"{'='*80}")

    print(f"\nTotal samples: {len(df)}")
    print(f"Number of unique classes: {df['class_id'].nunique()}")
    print(f"Class ID range: [{df['class_id'].min()}, {df['class_id'].max()}]")

    # Class distribution
    class_counts = df['class_id'].value_counts().sort_index()
    print(f"\nClass distribution statistics:")
    print(f"  Mean samples per class: {class_counts.mean():.2f}")
    print(f"  Median samples per class: {class_counts.median():.2f}")
    print(f"  Std dev: {class_counts.std():.2f}")
    print(f"  Min samples in a class: {class_counts.min()} (class {class_counts.idxmin()})")
    print(f"  Max samples in a class: {class_counts.max()} (class {class_counts.idxmax()})")

    # Imbalance ratio
    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"  Imbalance ratio (max/min): {imbalance_ratio:.2f}x")

    # Classes with very few samples
    few_samples_threshold = 5
    few_sample_classes = class_counts[class_counts < few_samples_threshold]
    if len(few_sample_classes) > 0:
        print(f"\n⚠️  Classes with < {few_samples_threshold} samples: {len(few_sample_classes)}")
        for class_id, count in few_sample_classes.items():
            print(f"    Class {class_id}: {count} samples")

    return class_counts

def check_label_coverage(train_df, test_df):
    """Check if all classes appear in both train and test."""
    print(f"\n{'='*80}")
    print("TRAIN/TEST CLASS COVERAGE")
    print(f"{'='*80}")

    train_classes = set(train_df['class_id'].unique())
    test_classes = set(test_df['class_id'].unique())

    print(f"\nClasses in train: {len(train_classes)}")
    print(f"Classes in test: {len(test_classes)}")

    # Classes only in train
    train_only = train_classes - test_classes
    if train_only:
        print(f"\n⚠️  Classes ONLY in train: {len(train_only)}")
        print(f"    {sorted(train_only)[:20]}{'...' if len(train_only) > 20 else ''}")

    # Classes only in test
    test_only = test_classes - train_classes
    if test_only:
        print(f"\n⚠️  Classes ONLY in test: {len(test_only)}")
        print(f"    {sorted(test_only)[:20]}{'...' if len(test_only) > 20 else ''}")

    # Shared classes
    shared = train_classes & test_classes
    print(f"\n✓ Classes in BOTH train and test: {len(shared)}")

    return train_classes, test_classes, shared

def analyze_graph_files(data_dir, split_name, label_df, max_samples=50):
    """Analyze graph files for a given split."""
    print(f"\n{'='*80}")
    print(f"{split_name.upper()} GRAPH FILES ANALYSIS")
    print(f"{'='*80}")

    split_dir = Path(data_dir) / split_name

    if not split_dir.exists():
        print(f"❌ Directory not found: {split_dir}")
        return None

    # Get all .bin files
    graph_files = list(split_dir.glob('*.bin'))
    print(f"\nTotal .bin files: {len(graph_files)}")
    print(f"Expected from labels: {len(label_df)}")

    if len(graph_files) != len(label_df):
        print(f"⚠️  MISMATCH: {abs(len(graph_files) - len(label_df))} files difference!")

    # Check if protein IDs match
    file_ids = {f.stem for f in graph_files}
    label_ids = set(label_df['protein_id'].astype(str))

    missing_files = label_ids - file_ids
    if missing_files:
        print(f"\n⚠️  Labels without graph files: {len(missing_files)}")
        print(f"    Examples: {list(missing_files)[:10]}")

    extra_files = file_ids - label_ids
    if extra_files:
        print(f"\n⚠️  Graph files without labels: {len(extra_files)}")
        print(f"    Examples: {list(extra_files)[:10]}")

    # Sample some graphs to check structure
    print(f"\nSampling {min(max_samples, len(graph_files))} graphs to analyze structure...")

    node_counts = []
    edge_counts = []
    feature_dims = []
    feature_stats = []

    sample_files = np.random.choice(graph_files, min(max_samples, len(graph_files)), replace=False)

    for graph_file in sample_files:
        try:
            graphs, _ = dgl.load_graphs(str(graph_file))
            g = graphs[0]

            node_counts.append(g.num_nodes())
            edge_counts.append(g.num_edges())

            if 'attr' in g.ndata:
                feat = g.ndata['attr']
                feature_dims.append(feat.shape[1])
                feature_stats.append({
                    'mean': feat.mean(dim=0).cpu().numpy(),
                    'std': feat.std(dim=0).cpu().numpy(),
                    'min': feat.min(dim=0)[0].cpu().numpy(),
                    'max': feat.max(dim=0)[0].cpu().numpy()
                })
        except Exception as e:
            print(f"❌ Error loading {graph_file.name}: {e}")

    if node_counts:
        print(f"\nGraph structure statistics (from {len(node_counts)} samples):")
        print(f"  Nodes per graph:")
        print(f"    Mean: {np.mean(node_counts):.2f}")
        print(f"    Median: {np.median(node_counts):.2f}")
        print(f"    Range: [{np.min(node_counts)}, {np.max(node_counts)}]")
        print(f"  Edges per graph:")
        print(f"    Mean: {np.mean(edge_counts):.2f}")
        print(f"    Median: {np.median(edge_counts):.2f}")
        print(f"    Range: [{np.min(edge_counts)}, {np.max(edge_counts)}]")

        if feature_dims:
            unique_dims = set(feature_dims)
            if len(unique_dims) == 1:
                print(f"  Feature dimension: {feature_dims[0]} (consistent ✓)")
            else:
                print(f"  ⚠️  Feature dimensions vary: {unique_dims}")

            # Aggregate feature statistics
            if feature_stats:
                all_means = np.array([s['mean'] for s in feature_stats])
                all_stds = np.array([s['std'] for s in feature_stats])

                print(f"\n  Feature statistics (per dimension):")
                feat_dim = all_means.shape[1]
                for i in range(feat_dim):
                    mean_of_means = all_means[:, i].mean()
                    std_of_means = all_means[:, i].std()
                    mean_of_stds = all_stds[:, i].mean()
                    print(f"    Dim {i}: mean={mean_of_means:8.4f} (±{std_of_means:.4f}), "
                          f"std={mean_of_stds:.4f}")

    return {
        'num_files': len(graph_files),
        'node_counts': node_counts,
        'edge_counts': edge_counts,
        'feature_dims': feature_dims
    }

def check_stratification_feasibility(class_counts, val_split=0.2):
    """Check if stratified split is feasible with given validation split."""
    print(f"\n{'='*80}")
    print(f"STRATIFIED SPLIT FEASIBILITY (val_split={val_split})")
    print(f"{'='*80}")

    min_samples_needed = int(1 / val_split) + 1  # Need at least this many for stratified split

    insufficient_classes = class_counts[class_counts < min_samples_needed]

    print(f"\nFor stratified split with {val_split*100}% validation:")
    print(f"  Minimum samples needed per class: {min_samples_needed}")
    print(f"  Classes with insufficient samples: {len(insufficient_classes)}")

    if len(insufficient_classes) > 0:
        print(f"\n⚠️  WARNING: {len(insufficient_classes)} classes cannot be stratified!")
        print(f"  These classes have < {min_samples_needed} samples:")
        for class_id, count in insufficient_classes.items():
            print(f"    Class {class_id}: {count} samples")
        print(f"\n  This may cause sklearn's train_test_split to fail or")
        print(f"  result in some classes appearing only in train or val!")
    else:
        print(f"\n✓ All classes have sufficient samples for stratified split")

    return len(insufficient_classes) == 0

def main():
    parser = argparse.ArgumentParser(description="Diagnose SHREC25 labels and graph files")
    parser.add_argument(
        "--data-dir",
        default=os.getenv("SHREC25_GRAPH_DATA_DIR", "datasets/SHREC25/graphs/normalized_8feat_1k"),
        help="Root directory containing train/test splits and labels CSV files",
    )
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    data_dir = args.data_dir
    labels_dir = data_dir

    print("="*80)
    print("SHREC25 DATASET DIAGNOSTIC")
    print("="*80)
    print(f"\nData directory: {data_dir}")
    print(f"Labels directory: {labels_dir}")

    # Load labels
    print("\nLoading labels...")
    train_df, test_df = load_labels(labels_dir)

    # Analyze label distributions
    train_counts = analyze_label_distribution(train_df, 'train')
    test_counts = analyze_label_distribution(test_df, 'test')

    # Check coverage
    train_classes, test_classes, shared = check_label_coverage(train_df, test_df)

    # Check stratification feasibility
    feasible = check_stratification_feasibility(train_counts, val_split=args.val_split)

    # Analyze graph files
    analyze_graph_files(data_dir, 'train', train_df, max_samples=args.max_samples)
    analyze_graph_files(data_dir, 'test', test_df, max_samples=args.max_samples)

    # Final summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    issues = []

    if len(train_classes) < 99:
        issues.append(f"❌ Train set missing {99 - len(train_classes)} classes")
    if len(test_classes) < 99:
        issues.append(f"❌ Test set missing {99 - len(test_classes)} classes")
    if len(train_classes - test_classes) > 0:
        issues.append(f"⚠️  {len(train_classes - test_classes)} classes only in train")
    if len(test_classes - train_classes) > 0:
        issues.append(f"❌ {len(test_classes - train_classes)} classes only in test (will fail!)")
    if not feasible:
        issues.append("❌ Stratified split not feasible for some classes")

    if train_counts.max() / train_counts.min() > 10:
        issues.append(f"⚠️  High class imbalance: {train_counts.max() / train_counts.min():.1f}x")

    if issues:
        print("\n🔴 ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ No major issues detected!")

    print(f"\n{'='*80}")

if __name__ == '__main__':
    main()
