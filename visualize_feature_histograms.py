#!/usr/bin/env python
"""
Visualize histograms of all 8 features in the normalized_8feat_1k dataset.
Shows per-graph feature distributions with overlaid histograms from multiple graphs.
"""

import torch
import dgl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def load_sample_graphs(root_dir, num_samples=10):
    """
    Load a sample of graphs, keeping them separate (not concatenated).

    Args:
        root_dir: Path to the dataset
        num_samples: Number of graphs to sample

    Returns:
        list_of_features: List of arrays, each of shape (num_nodes, 8)
        graph_names: List of graph names
    """
    root_path = Path(root_dir)
    train_dir = root_path / "train"

    # Get all graph files
    graph_files = sorted(train_dir.glob("*.bin"))

    # Sample random graphs
    np.random.seed(42)
    sampled_files = np.random.choice(graph_files, min(num_samples, len(graph_files)), replace=False)

    print(f"Loading {len(sampled_files)} graphs...")
    list_of_features = []
    graph_names = []

    for graph_file in tqdm(sampled_files):
        graphs, _ = dgl.load_graphs(str(graph_file))
        dgl_graph = graphs[0]

        # Extract node features
        features = dgl_graph.ndata['attr'].numpy()  # Shape: (num_nodes, 8)
        list_of_features.append(features)
        graph_names.append(graph_file.stem)

    return list_of_features, graph_names


def plot_feature_histograms_per_graph(list_of_features, feature_names, graph_names):
    """
    Plot histograms for all features, showing per-graph distributions.
    Each subplot shows overlaid histograms from different graphs.

    Args:
        list_of_features: List of feature arrays, each of shape (num_nodes, 8)
        feature_names: List of feature names
        graph_names: List of graph identifiers
    """
    num_features = list_of_features[0].shape[1]
    num_graphs = len(list_of_features)

    # Create color map for different graphs
    colors = plt.cm.tab10(np.linspace(0, 1, num_graphs))

    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle('Per-Graph Feature Distributions (Node-level Histograms)', fontsize=16, fontweight='bold')

    axes = axes.flatten()

    for feat_idx, (ax, name) in enumerate(zip(axes, feature_names)):
        # Plot histogram for each graph with transparency
        for graph_idx, (features, graph_name) in enumerate(zip(list_of_features, graph_names)):
            feature_data = features[:, feat_idx]

            # Plot histogram with transparency
            ax.hist(feature_data, bins=40, alpha=0.4, color=colors[graph_idx],
                   label=f'Graph {graph_idx+1}', edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Value', fontsize=11)
        ax.set_ylabel('Node Count', fontsize=11)
        ax.set_title(f'{name}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add legend for first plot only
        if feat_idx == 0:
            ax.legend(loc='upper right', fontsize=8, ncol=2)

        # Compute statistics across all graphs
        all_data = np.concatenate([f[:, feat_idx] for f in list_of_features])
        mean_val = np.mean(all_data)
        std_val = np.std(all_data)
        min_val = np.min(all_data)
        max_val = np.max(all_data)

        stats_text = f'Overall Stats:\nMean: {mean_val:.3f}\nStd: {std_val:.3f}\nRange: [{min_val:.3f}, {max_val:.3f}]'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                fontsize=8)

    plt.tight_layout()
    plt.savefig('feature_histograms_per_graph.png', dpi=300, bbox_inches='tight')
    print("✓ Per-graph histogram saved to 'feature_histograms_per_graph.png'")
    plt.show()


def plot_single_graph_histograms(features, feature_names, graph_name):
    """
    Plot histograms for a single graph showing node feature distributions.

    Args:
        features: Array of shape (num_nodes, 8)
        feature_names: List of feature names
        graph_name: Name/ID of the graph
    """
    num_features = features.shape[1]
    num_nodes = features.shape[0]

    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Node Feature Distributions - {graph_name}\n({num_nodes} nodes)',
                 fontsize=16, fontweight='bold')

    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        feature_data = features[:, i]

        # Plot histogram
        ax.hist(feature_data, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Node Count', fontsize=10)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_val = np.mean(feature_data)
        std_val = np.std(feature_data)
        min_val = np.min(feature_data)
        max_val = np.max(feature_data)

        stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMin: {min_val:.3f}\nMax: {max_val:.3f}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

    plt.tight_layout()
    filename = f'feature_histograms_single_{graph_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Single graph histogram saved to '{filename}'")
    plt.show()


def main():
    # Dataset configuration
    root_dir = "/home/ogharbi/datasets/SHREC25/graphs/normalized_8feat_1k"
    num_samples = 10  # Number of graphs to sample for visualization

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

    print("="*70)
    print("Per-Graph Node Feature Histogram Visualization")
    print("="*70)
    print(f"Dataset: {root_dir}")
    print(f"Sampling {num_samples} graphs...")
    print()

    # Load features from sample graphs (keeping them separate)
    list_of_features, graph_names = load_sample_graphs(root_dir, num_samples)

    print(f"\nLoaded {len(list_of_features)} graphs")
    print(f"Nodes per graph: {[f.shape[0] for f in list_of_features]}")
    print(f"Features per node: {list_of_features[0].shape[1]}")
    print()

    # Plot 1: Overlaid histograms showing multiple graphs
    print("Creating overlaid histograms (all graphs)...")
    plot_feature_histograms_per_graph(list_of_features, feature_names, graph_names)

    # Plot 2: Single graph detailed view (first graph)
    print("\nCreating detailed histogram for first graph...")
    plot_single_graph_histograms(list_of_features[0], feature_names, graph_names[0])

    # Print per-graph statistics
    print("\n" + "="*70)
    print("Per-Graph Statistics Summary:")
    print("="*70)
    for graph_idx, (features, graph_name) in enumerate(zip(list_of_features, graph_names)):
        print(f"\nGraph {graph_idx+1}: {graph_name} ({features.shape[0]} nodes)")
        print("-" * 70)
        for feat_idx, name in enumerate(feature_names):
            feature_data = features[:, feat_idx]
            print(f"  {name:25s} | Mean: {np.mean(feature_data):7.3f} | Std: {np.std(feature_data):7.3f} | "
                  f"Range: [{np.min(feature_data):7.3f}, {np.max(feature_data):7.3f}]")


if __name__ == "__main__":
    main()
