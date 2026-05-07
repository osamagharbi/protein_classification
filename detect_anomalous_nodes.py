#!/usr/bin/env python
"""
Detect anomalous nodes that don't follow uniformity in the normalized_8feat_1k dataset.
Identifies nodes with unusual feature values (both high and low) that deviate from the typical distribution.
"""

import torch
import dgl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json


def load_graphs(root_dir, num_samples=20):
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
    np.random.seed(42)
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


def detect_anomalous_nodes(features, method='zscore', threshold=2.5):
    """
    Detect anomalous nodes that deviate from the typical distribution (in ANY direction).

    Args:
        features: Array of shape (num_nodes, num_features)
        method: Detection method
                - 'zscore': Absolute z-score (bi-directional)
                - 'isolation_forest': Multivariate anomaly detection
                - 'lof': Local Outlier Factor
                - 'combined': Combination of multiple features
        threshold: Threshold value (for zscore: 2.5 = ~1.2% outliers on each tail)

    Returns:
        anomaly_mask: Boolean array of shape (num_nodes, num_features) - per-feature anomalies
        node_anomaly_scores: Array of shape (num_nodes,) - overall anomaly score per node
        statistics: Dictionary with statistics
    """
    num_nodes, num_features = features.shape
    anomaly_mask = np.zeros((num_nodes, num_features), dtype=bool)

    if method == 'zscore':
        # Z-score method: detect nodes with |z-score| > threshold (bi-directional)
        node_anomaly_scores = np.zeros(num_nodes)
        statistics = {}

        for feat_idx in range(num_features):
            feature_data = features[:, feat_idx]

            # Calculate z-scores
            mean = np.mean(feature_data)
            std = np.std(feature_data)

            if std > 1e-8:  # Avoid division by zero
                z_scores = np.abs((feature_data - mean) / std)
                mask = z_scores > threshold

                # Accumulate anomaly scores
                node_anomaly_scores += z_scores
            else:
                z_scores = np.zeros(num_nodes)
                mask = np.zeros(num_nodes, dtype=bool)

            anomaly_mask[:, feat_idx] = mask

            statistics[feat_idx] = {
                'method': 'zscore',
                'mean': float(mean),
                'std': float(std),
                'threshold': float(threshold),
                'num_anomalous': int(np.sum(mask)),
                'percentage': float(100 * np.sum(mask) / num_nodes),
                'max_zscore': float(np.max(z_scores)),
                'anomalous_node_ids': np.where(mask)[0].tolist()[:20]  # First 20 for brevity
            }

        # Normalize overall anomaly scores
        node_anomaly_scores = node_anomaly_scores / num_features

    elif method == 'combined':
        # Combined method: nodes anomalous in multiple features are more suspicious
        node_anomaly_scores = np.zeros(num_nodes)
        statistics = {}

        for feat_idx in range(num_features):
            feature_data = features[:, feat_idx]

            mean = np.mean(feature_data)
            std = np.std(feature_data)

            if std > 1e-8:
                z_scores = np.abs((feature_data - mean) / std)
                mask = z_scores > threshold

                # Weight by z-score magnitude
                node_anomaly_scores += z_scores * mask
            else:
                mask = np.zeros(num_nodes, dtype=bool)

            anomaly_mask[:, feat_idx] = mask

            statistics[feat_idx] = {
                'method': 'combined',
                'mean': float(mean),
                'std': float(std),
                'threshold': float(threshold),
                'num_anomalous': int(np.sum(mask)),
                'percentage': float(100 * np.sum(mask) / num_nodes)
            }

    elif method == 'mad':
        # MAD (Median Absolute Deviation) - more robust to outliers
        node_anomaly_scores = np.zeros(num_nodes)
        statistics = {}

        for feat_idx in range(num_features):
            feature_data = features[:, feat_idx]

            median = np.median(feature_data)
            mad = np.median(np.abs(feature_data - median))

            if mad > 1e-8:
                # Modified z-score using MAD
                mad_scores = np.abs(0.6745 * (feature_data - median) / mad)
                mask = mad_scores > threshold

                node_anomaly_scores += mad_scores
            else:
                mad_scores = np.zeros(num_nodes)
                mask = np.zeros(num_nodes, dtype=bool)

            anomaly_mask[:, feat_idx] = mask

            statistics[feat_idx] = {
                'method': 'mad',
                'median': float(median),
                'mad': float(mad),
                'threshold': float(threshold),
                'num_anomalous': int(np.sum(mask)),
                'percentage': float(100 * np.sum(mask) / num_nodes)
            }

        node_anomaly_scores = node_anomaly_scores / num_features

    else:
        raise ValueError(f"Unknown method: {method}")

    return anomaly_mask, node_anomaly_scores, statistics


def identify_multi_feature_anomalies(anomaly_mask):
    """
    Identify nodes that are anomalous in multiple features.

    Args:
        anomaly_mask: Boolean array of shape (num_nodes, num_features)

    Returns:
        multi_anomaly_counts: Array of shape (num_nodes,) with count of anomalous features per node
        highly_anomalous_nodes: Indices of nodes anomalous in 2+ features
    """
    # Count how many features each node is anomalous in
    multi_anomaly_counts = np.sum(anomaly_mask, axis=1)

    # Find nodes anomalous in 2 or more features
    highly_anomalous_nodes = np.where(multi_anomaly_counts >= 2)[0]

    return multi_anomaly_counts, highly_anomalous_nodes


def visualize_anomaly_detection(features, anomaly_mask, feature_names, graph_name, statistics):
    """
    Visualize anomaly detection for a single graph.
    """
    num_features = features.shape[1]

    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(f'Anomaly Detection - {graph_name}\nMethod: {statistics[0]["method"]} (bi-directional)',
                 fontsize=16, fontweight='bold')

    axes = axes.flatten()

    for feat_idx, (ax, name) in enumerate(zip(axes, feature_names)):
        feature_data = features[:, feat_idx]
        mask = anomaly_mask[:, feat_idx]

        # Plot histogram with anomalous nodes highlighted
        ax.hist(feature_data[~mask], bins=40, alpha=0.6, color='steelblue',
               label='Normal', edgecolor='black', linewidth=0.5)
        ax.hist(feature_data[mask], bins=40, alpha=0.9, color='red',
               label='Anomalous', edgecolor='darkred', linewidth=1.0)

        # Draw mean line
        mean = statistics[feat_idx].get('mean', statistics[feat_idx].get('median'))
        ax.axvline(mean, color='green', linestyle='-', linewidth=2,
                  label=f'Mean/Median: {mean:.3f}', alpha=0.7)

        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Node Count', fontsize=10)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

        # Add statistics
        num_anom = statistics[feat_idx]['num_anomalous']
        percentage = statistics[feat_idx]['percentage']

        stats_text = f'Anomalous:\n{num_anom} nodes\n({percentage:.1f}%)'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.7),
                fontsize=9, fontweight='bold')

    plt.tight_layout()
    filename = f'anomaly_detection_{graph_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Anomaly visualization saved to '{filename}'")
    plt.close()


def visualize_multi_feature_anomalies(features, anomaly_mask, multi_anomaly_counts,
                                     feature_names, graph_name, node_anomaly_scores):
    """
    Visualize nodes that are anomalous in multiple features.
    """
    num_nodes = features.shape[0]

    # Create figure with multiple subplots using constrained_layout
    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    fig.suptitle(f'Multi-Feature Anomaly Analysis - {graph_name}',
                 fontsize=16, fontweight='bold')

    # 1. Histogram of anomaly counts per node
    ax1 = fig.add_subplot(gs[0, 0])
    counts = np.bincount(multi_anomaly_counts.astype(int), minlength=9)
    bars = ax1.bar(range(9), counts, color='steelblue', edgecolor='black')
    bars[0].set_color('lightgreen')  # Normal nodes
    if len(bars) > 1:
        for i in range(2, len(bars)):
            bars[i].set_color('red')  # Multi-feature anomalies
    ax1.set_xlabel('Number of Anomalous Features', fontsize=11)
    ax1.set_ylabel('Number of Nodes', fontsize=11)
    ax1.set_title('Distribution of Anomaly Counts', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Node anomaly score distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(node_anomaly_scores, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    threshold_line = np.percentile(node_anomaly_scores, 95)
    ax2.axvline(threshold_line, color='red', linestyle='--', linewidth=2,
               label=f'95th percentile: {threshold_line:.2f}')
    ax2.set_xlabel('Anomaly Score', fontsize=11)
    ax2.set_ylabel('Number of Nodes', fontsize=11)
    ax2.set_title('Node Anomaly Score Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Per-feature anomaly percentage
    ax3 = fig.add_subplot(gs[0, 2])
    anomaly_percentages = 100 * np.sum(anomaly_mask, axis=0) / num_nodes
    bars = ax3.barh(feature_names, anomaly_percentages, color='coral', edgecolor='black')
    ax3.set_xlabel('% Anomalous Nodes', fontsize=11)
    ax3.set_title('Anomaly Rate by Feature', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. Heatmap of anomalies for top anomalous nodes
    ax4 = fig.add_subplot(gs[1, :])
    top_n = min(50, np.sum(multi_anomaly_counts > 0))
    if top_n > 0:
        top_nodes = np.argsort(node_anomaly_scores)[-top_n:][::-1]
        heatmap_data = anomaly_mask[top_nodes, :].astype(float)

        im = ax4.imshow(heatmap_data.T, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
        ax4.set_yticks(range(len(feature_names)))
        ax4.set_yticklabels(feature_names)
        ax4.set_xlabel('Node Index (sorted by anomaly score)', fontsize=11)
        ax4.set_ylabel('Features', fontsize=11)
        ax4.set_title(f'Top {top_n} Most Anomalous Nodes (Red = Anomalous)',
                     fontsize=12, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4, orientation='horizontal', pad=0.1, aspect=30)
        cbar.set_label('Anomalous (1) vs Normal (0)', fontsize=10)

    filename = f'multi_feature_anomaly_{graph_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Multi-feature anomaly visualization saved to '{filename}'")
    plt.close()


def analyze_anomalies_across_graphs(list_of_features, graph_names, feature_names,
                                    method='zscore', threshold=2.5):
    """
    Analyze anomalous nodes across multiple graphs.
    """
    all_results = {}

    print(f"\nDetecting anomalous nodes using method='{method}', threshold={threshold}")
    print("=" * 90)

    for graph_idx, (features, graph_name) in enumerate(zip(list_of_features, graph_names)):
        # Detect anomalous nodes
        anomaly_mask, node_anomaly_scores, statistics = detect_anomalous_nodes(
            features, method=method, threshold=threshold
        )

        # Identify multi-feature anomalies
        multi_anomaly_counts, highly_anomalous_nodes = identify_multi_feature_anomalies(anomaly_mask)

        # Store results
        all_results[graph_name] = {
            'features': features,
            'anomaly_mask': anomaly_mask,
            'node_anomaly_scores': node_anomaly_scores,
            'multi_anomaly_counts': multi_anomaly_counts,
            'highly_anomalous_nodes': highly_anomalous_nodes,
            'statistics': statistics
        }

        # Print summary
        print(f"\nGraph {graph_idx+1}: {graph_name}")
        print("-" * 90)
        print(f"  Total nodes: {features.shape[0]}")
        print(f"  Nodes with any anomaly: {np.sum(multi_anomaly_counts > 0)} ({100*np.sum(multi_anomaly_counts > 0)/features.shape[0]:.1f}%)")
        print(f"  Nodes anomalous in 2+ features: {len(highly_anomalous_nodes)} ({100*len(highly_anomalous_nodes)/features.shape[0]:.1f}%)")
        print(f"  Nodes anomalous in 3+ features: {np.sum(multi_anomaly_counts >= 3)} ({100*np.sum(multi_anomaly_counts >= 3)/features.shape[0]:.1f}%)")
        print(f"\n  Per-feature anomaly breakdown:")
        for feat_idx, name in enumerate(feature_names):
            stats = statistics[feat_idx]
            print(f"    {name:25s} | Anomalous: {stats['num_anomalous']:4d} nodes ({stats['percentage']:5.2f}%)")

    return all_results


def create_summary_visualizations(all_results, feature_names, graph_names):
    """
    Create summary visualizations across all graphs.
    """
    num_graphs = len(graph_names)
    num_features = len(feature_names)

    # Create matrix of anomaly percentages
    anomaly_matrix = np.zeros((num_graphs, num_features))
    multi_feature_counts = []

    for graph_idx, graph_name in enumerate(graph_names):
        stats = all_results[graph_name]['statistics']
        for feat_idx in range(num_features):
            anomaly_matrix[graph_idx, feat_idx] = stats[feat_idx]['percentage']

        multi_count = np.sum(all_results[graph_name]['multi_anomaly_counts'] >= 2)
        total_nodes = all_results[graph_name]['features'].shape[0]
        multi_feature_counts.append(100 * multi_count / total_nodes)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, max(8, num_graphs * 0.5)))
    fig.suptitle('Anomaly Detection Summary Across All Graphs', fontsize=16, fontweight='bold')

    # Heatmap 1: Per-feature anomaly rates
    im1 = ax1.imshow(anomaly_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=10)
    ax1.set_xticks(np.arange(num_features))
    ax1.set_yticks(np.arange(num_graphs))
    ax1.set_xticklabels(feature_names, rotation=45, ha='right')
    ax1.set_yticklabels([f"G{i+1}: {name[:20]}" for i, name in enumerate(graph_names)])

    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('% Anomalous Nodes', rotation=270, labelpad=20)

    for i in range(num_graphs):
        for j in range(num_features):
            text = ax1.text(j, i, f'{anomaly_matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=7)

    ax1.set_title('Per-Feature Anomaly Rates', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Features', fontsize=12)
    ax1.set_ylabel('Graphs', fontsize=12)

    # Bar chart 2: Multi-feature anomalies
    bars = ax2.barh(range(num_graphs), multi_feature_counts, color='coral', edgecolor='black')
    ax2.set_yticks(range(num_graphs))
    ax2.set_yticklabels([f"G{i+1}: {name[:25]}" for i, name in enumerate(graph_names)])
    ax2.set_xlabel('% Nodes Anomalous in 2+ Features', fontsize=12)
    ax2.set_title('Multi-Feature Anomaly Rates', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, multi_feature_counts)):
        ax2.text(pct + 0.1, i, f'{pct:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('anomaly_detection_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Anomaly detection summary saved to 'anomaly_detection_summary.png'")
    plt.close()


def save_anomaly_results(all_results, feature_names, output_file='anomaly_results.json'):
    """
    Save anomaly detection results to JSON file.
    """
    json_results = {}

    for graph_name, result in all_results.items():
        graph_data = {
            'per_feature_stats': {},
            'total_nodes': int(result['features'].shape[0]),
            'nodes_with_any_anomaly': int(np.sum(result['multi_anomaly_counts'] > 0)),
            'nodes_multi_feature_anomaly': int(len(result['highly_anomalous_nodes'])),
            'highly_anomalous_node_ids': result['highly_anomalous_nodes'].tolist()[:50]
        }

        for feat_idx, name in enumerate(feature_names):
            graph_data['per_feature_stats'][name] = result['statistics'][feat_idx]

        json_results[graph_name] = graph_data

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"✓ Anomaly results saved to '{output_file}'")


def main():
    # Dataset configuration
    root_dir = "/home/ogharbi/datasets/SHREC25/graphs/normalized_8feat_1k"
    num_samples = 10

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
    method = 'zscore'  # 'zscore', 'mad', 'combined'
    threshold = 2.5    # For zscore: 2.5 = ~1.2% on each tail

    print("=" * 90)
    print("Anomalous Node Detection Analysis")
    print("=" * 90)
    print(f"Dataset: {root_dir}")
    print(f"Method: {method} (bi-directional - detects both high and low outliers)")
    print(f"Threshold: {threshold}")
    print(f"Sampling {num_samples} graphs...")
    print()

    # Load graphs
    list_of_graphs, list_of_features, graph_names = load_graphs(root_dir, num_samples)

    print(f"\nLoaded {len(list_of_features)} graphs")

    # Analyze anomalies across all graphs
    all_results = analyze_anomalies_across_graphs(
        list_of_features, graph_names, feature_names,
        method=method, threshold=threshold
    )

    # Create summary visualizations
    print("\n" + "=" * 90)
    print("Creating visualizations...")
    print("=" * 90)
    create_summary_visualizations(all_results, feature_names, graph_names)

    # Visualize first 3 graphs in detail
    print("\nCreating detailed visualizations for first 3 graphs...")
    for graph_idx in range(min(3, len(graph_names))):
        graph_name = graph_names[graph_idx]
        result = all_results[graph_name]

        # Basic anomaly visualization
        visualize_anomaly_detection(
            result['features'],
            result['anomaly_mask'],
            feature_names,
            graph_name,
            result['statistics']
        )

        # Multi-feature anomaly visualization
        visualize_multi_feature_anomalies(
            result['features'],
            result['anomaly_mask'],
            result['multi_anomaly_counts'],
            feature_names,
            graph_name,
            result['node_anomaly_scores']
        )

    # Save results
    save_anomaly_results(all_results, feature_names)

    print("\n" + "=" * 90)
    print("Analysis complete!")
    print("=" * 90)
    print("\nGenerated files:")
    print("  - anomaly_detection_summary.png: Summary across all graphs")
    print("  - anomaly_detection_<graph>.png: Per-feature anomalies (first 3 graphs)")
    print("  - multi_feature_anomaly_<graph>.png: Multi-feature analysis (first 3 graphs)")
    print("  - anomaly_results.json: Complete results in JSON format")


if __name__ == "__main__":
    main()
