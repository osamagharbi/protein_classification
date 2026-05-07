#!/usr/bin/env python
"""
Detect high signal nodes per graph per feature in the normalized_8feat_1k dataset.
Uses statistical thresholding to identify nodes with significantly high feature values.
"""

import torch
import dgl
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json


def load_graphs(root_dir, num_samples=20, seed=42):
    """
    Load a sample of graphs.

    Args:
        root_dir: Path to the dataset
        num_samples: Number of graphs to sample

    Returns:
        list_of_graphs: List of DGL graphs
        list_of_features: List of feature arrays, each of shape (num_nodes, 8)
        graph_names: List of graph names
    """
    root_path = Path(root_dir)
    train_dir = root_path / "train"

    # Get all graph files
    graph_files = sorted(train_dir.glob("*.bin"))

    # Sample random graphs
    np.random.seed(seed)
    sampled_files = np.random.choice(graph_files, min(num_samples, len(graph_files)), replace=False)

    print(f"Loading {len(sampled_files)} graphs...")
    list_of_graphs = []
    list_of_features = []
    graph_names = []

    for graph_file in tqdm(sampled_files):
        graphs, _ = dgl.load_graphs(str(graph_file))
        dgl_graph = graphs[0]

        # Extract node features
        features = dgl_graph.ndata['attr'].numpy()  # Shape: (num_nodes, 8)

        list_of_graphs.append(dgl_graph)
        list_of_features.append(features)
        graph_names.append(graph_file.stem)

    return list_of_graphs, list_of_features, graph_names


def detect_high_signal_nodes(features, method='zscore', threshold=2.0):
    """
    Detect high signal nodes for each feature using statistical methods.

    Args:
        features: Array of shape (num_nodes, num_features)
        method: Detection method ('zscore', 'percentile', 'iqr', 'mad')
        threshold: Threshold value (depends on method)
                  - 'zscore': number of standard deviations (default: 2.0)
                  - 'percentile': percentile cutoff (default: 95)
                  - 'iqr': IQR multiplier (default: 1.5)
                  - 'mad': MAD multiplier (default: 3.0)

    Returns:
        high_signal_mask: Boolean array of shape (num_nodes, num_features)
        statistics: Dictionary with statistics for each feature
    """
    num_nodes, num_features = features.shape
    high_signal_mask = np.zeros((num_nodes, num_features), dtype=bool)
    statistics = {}

    for feat_idx in range(num_features):
        feature_data = features[:, feat_idx]

        if method == 'zscore':
            # Z-score method: flag values > mean + threshold*std
            mean = np.mean(feature_data)
            std = np.std(feature_data)
            cutoff = mean + threshold * std
            mask = feature_data > cutoff

            statistics[feat_idx] = {
                'method': 'zscore',
                'mean': float(mean),
                'std': float(std),
                'threshold': float(threshold),
                'cutoff': float(cutoff),
                'num_high_signal': int(np.sum(mask)),
                'percentage': float(100 * np.sum(mask) / num_nodes)
            }

        elif method == 'percentile':
            # Percentile method: flag values above the Nth percentile
            cutoff = np.percentile(feature_data, threshold)
            mask = feature_data > cutoff

            statistics[feat_idx] = {
                'method': 'percentile',
                'percentile': float(threshold),
                'cutoff': float(cutoff),
                'num_high_signal': int(np.sum(mask)),
                'percentage': float(100 * np.sum(mask) / num_nodes)
            }

        elif method == 'iqr':
            # IQR (Interquartile Range) method: flag outliers
            q1 = np.percentile(feature_data, 25)
            q3 = np.percentile(feature_data, 75)
            iqr = q3 - q1
            cutoff = q3 + threshold * iqr
            mask = feature_data > cutoff

            statistics[feat_idx] = {
                'method': 'iqr',
                'q1': float(q1),
                'q3': float(q3),
                'iqr': float(iqr),
                'threshold': float(threshold),
                'cutoff': float(cutoff),
                'num_high_signal': int(np.sum(mask)),
                'percentage': float(100 * np.sum(mask) / num_nodes)
            }

        elif method == 'mad':
            # MAD (Median Absolute Deviation) method: robust to outliers
            median = np.median(feature_data)
            mad = np.median(np.abs(feature_data - median))
            # MAD-based z-score
            cutoff = median + threshold * mad * 1.4826  # 1.4826 is scaling factor
            mask = feature_data > cutoff

            statistics[feat_idx] = {
                'method': 'mad',
                'median': float(median),
                'mad': float(mad),
                'threshold': float(threshold),
                'cutoff': float(cutoff),
                'num_high_signal': int(np.sum(mask)),
                'percentage': float(100 * np.sum(mask) / num_nodes)
            }

        else:
            raise ValueError(f"Unknown method: {method}")

        high_signal_mask[:, feat_idx] = mask

    return high_signal_mask, statistics


def visualize_high_signal_detection(features, high_signal_mask, feature_names, graph_name, statistics):
    """
    Visualize high signal detection for a single graph.

    Args:
        features: Array of shape (num_nodes, 8)
        high_signal_mask: Boolean array of shape (num_nodes, 8)
        feature_names: List of feature names
        graph_name: Name of the graph
        statistics: Statistics dictionary
    """
    num_features = features.shape[1]

    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(f'High Signal Detection - {graph_name}\nMethod: {statistics[0]["method"]}',
                 fontsize=16, fontweight='bold')

    axes = axes.flatten()

    for feat_idx, (ax, name) in enumerate(zip(axes, feature_names)):
        feature_data = features[:, feat_idx]
        mask = high_signal_mask[:, feat_idx]

        # Plot histogram with high signal nodes highlighted
        ax.hist(feature_data[~mask], bins=40, alpha=0.6, color='steelblue',
               label='Normal', edgecolor='black', linewidth=0.5)
        ax.hist(feature_data[mask], bins=40, alpha=0.8, color='red',
               label='High Signal', edgecolor='black', linewidth=0.5)

        # Draw cutoff line
        cutoff = statistics[feat_idx]['cutoff']
        ax.axvline(cutoff, color='darkred', linestyle='--', linewidth=2, label=f'Cutoff: {cutoff:.3f}')

        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Node Count', fontsize=10)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

        # Add statistics
        num_high = statistics[feat_idx]['num_high_signal']
        percentage = statistics[feat_idx]['percentage']

        stats_text = f'High Signal:\n{num_high} nodes\n({percentage:.1f}%)'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.7),
                fontsize=9, fontweight='bold')

    plt.tight_layout()
    filename = f'high_signal_{graph_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ High signal visualization saved to '{filename}'")
    plt.close()


def analyze_high_signal_across_graphs(list_of_features, graph_names, feature_names, method='zscore', threshold=2.0):
    """
    Analyze high signal detection across multiple graphs.

    Args:
        list_of_features: List of feature arrays
        graph_names: List of graph names
        feature_names: List of feature names
        method: Detection method
        threshold: Threshold value

    Returns:
        all_results: Dictionary with results for each graph
    """
    all_results = {}

    print(f"\nDetecting high signal nodes using method='{method}', threshold={threshold}")
    print("=" * 80)

    for graph_idx, (features, graph_name) in enumerate(zip(list_of_features, graph_names)):
        # Detect high signal nodes
        high_signal_mask, statistics = detect_high_signal_nodes(features, method=method, threshold=threshold)

        # Store results
        all_results[graph_name] = {
            'features': features,
            'high_signal_mask': high_signal_mask,
            'statistics': statistics
        }

        # Print summary
        print(f"\nGraph {graph_idx+1}: {graph_name}")
        print("-" * 80)
        for feat_idx, name in enumerate(feature_names):
            stats = statistics[feat_idx]
            print(f"  {name:25s} | Cutoff: {stats['cutoff']:7.3f} | "
                  f"High Signal: {stats['num_high_signal']:4d} nodes ({stats['percentage']:5.2f}%)")

    return all_results


def create_summary_heatmap(all_results, feature_names, graph_names):
    """
    Create a heatmap showing percentage of high signal nodes per graph per feature.

    Args:
        all_results: Dictionary with results for each graph
        feature_names: List of feature names
        graph_names: List of graph names
    """
    num_graphs = len(graph_names)
    num_features = len(feature_names)

    # Create matrix of percentages
    heatmap_data = np.zeros((num_graphs, num_features))

    for graph_idx, graph_name in enumerate(graph_names):
        stats = all_results[graph_name]['statistics']
        for feat_idx in range(num_features):
            heatmap_data[graph_idx, feat_idx] = stats[feat_idx]['percentage']

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, max(8, num_graphs * 0.5)))
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=20)

    # Set ticks and labels
    ax.set_xticks(np.arange(num_features))
    ax.set_yticks(np.arange(num_graphs))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels([f"G{i+1}: {name[:20]}" for i, name in enumerate(graph_names)])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('% High Signal Nodes', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(num_graphs):
        for j in range(num_features):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title('High Signal Detection Heatmap (% of nodes per feature)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Graphs', fontsize=12)

    plt.tight_layout()
    plt.savefig('high_signal_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ High signal heatmap saved to 'high_signal_heatmap.png'")
    plt.close()


def save_high_signal_results(all_results, feature_names, output_file='high_signal_results.json'):
    """
    Save high signal detection results to JSON file.

    Args:
        all_results: Dictionary with results for each graph
        feature_names: List of feature names
        output_file: Output JSON file path
    """
    # Convert results to JSON-serializable format
    json_results = {}

    for graph_name, result in all_results.items():
        graph_stats = {}
        for feat_idx, name in enumerate(feature_names):
            graph_stats[name] = result['statistics'][feat_idx]
        json_results[graph_name] = graph_stats

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"✓ High signal results saved to '{output_file}'")


def main():
    parser = argparse.ArgumentParser(description="Detect high-signal nodes in SHREC25 graphs")
    parser.add_argument(
        "--data-dir",
        default=os.getenv("SHREC25_GRAPH_DATA_DIR", "datasets/SHREC25/graphs/normalized_8feat_1k"),
        help="Root directory containing graph splits",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--method", choices=["zscore", "percentile", "iqr", "mad"], default="zscore")
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="high_signal_results.json")
    args = parser.parse_args()

    root_dir = args.data_dir
    num_samples = args.num_samples

    # Feature names
    feature_names = [
        "pos_x",
        "pos_y",
        "pos_z",
        "normal_x",
        "normal_y",
        "normal_z",
        "potential_norm",
        "normal_potential_norm"
    ]

    # Detection parameters
    method = args.method
    threshold = args.threshold

    print("=" * 80)
    print("High Signal Detection Analysis")
    print("=" * 80)
    print(f"Dataset: {root_dir}")
    print(f"Method: {method}")
    print(f"Threshold: {threshold}")
    print(f"Sampling {num_samples} graphs...")
    print()

    # Load graphs
    list_of_graphs, list_of_features, graph_names = load_graphs(root_dir, num_samples, seed=args.seed)

    print(f"\nLoaded {len(list_of_features)} graphs")
    print(f"Nodes per graph: {[f.shape[0] for f in list_of_features]}")
    print(f"Features per node: {list_of_features[0].shape[1]}")

    # Analyze high signal across all graphs
    all_results = analyze_high_signal_across_graphs(
        list_of_features, graph_names, feature_names,
        method=method, threshold=threshold
    )

    # Create summary heatmap
    print("\n" + "=" * 80)
    print("Creating summary visualizations...")
    print("=" * 80)
    create_summary_heatmap(all_results, feature_names, graph_names)

    # Visualize first 3 graphs in detail
    print("\nCreating detailed visualizations for first 3 graphs...")
    for graph_idx in range(min(3, len(graph_names))):
        graph_name = graph_names[graph_idx]
        result = all_results[graph_name]
        visualize_high_signal_detection(
            result['features'],
            result['high_signal_mask'],
            feature_names,
            graph_name,
            result['statistics']
        )

    # Save results to JSON
    save_high_signal_results(all_results, feature_names, output_file=args.output_json)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - high_signal_heatmap.png: Summary heatmap of all graphs")
    print("  - high_signal_<graph_name>.png: Detailed visualizations (first 3 graphs)")
    print("  - high_signal_results.json: Complete results in JSON format")


if __name__ == "__main__":
    main()
