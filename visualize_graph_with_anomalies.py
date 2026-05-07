#!/usr/bin/env python
"""
Visualize protein graphs in 3D with anomalous nodes colored differently.
Shows the spatial distribution of anomalous nodes on the protein surface.
"""

import torch
import dgl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from datetime import datetime
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


def load_graph(graph_path):
    """Load a single graph from file."""
    graphs, _ = dgl.load_graphs(str(graph_path))
    return graphs[0]


def detect_anomalous_nodes(features, method='zscore', threshold=2.5):
    """
    Detect anomalous nodes (simplified version for visualization).

    Returns:
        anomaly_mask: Boolean array of shape (num_nodes, num_features)
        overall_anomaly: Boolean array of shape (num_nodes,) - True if anomalous in any feature
        anomaly_scores: Array of shape (num_nodes,) - overall anomaly score
    """
    num_nodes, num_features = features.shape
    anomaly_mask = np.zeros((num_nodes, num_features), dtype=bool)
    anomaly_scores = np.zeros(num_nodes)

    for feat_idx in range(num_features):
        feature_data = features[:, feat_idx]
        mean = np.mean(feature_data)
        std = np.std(feature_data)

        if std > 1e-8:
            z_scores = np.abs((feature_data - mean) / std)
            mask = z_scores > threshold
            anomaly_mask[:, feat_idx] = mask
            anomaly_scores += z_scores
        else:
            anomaly_mask[:, feat_idx] = False

    # Normalize anomaly scores
    anomaly_scores = anomaly_scores / num_features

    # Overall anomaly: True if anomalous in at least one feature
    overall_anomaly = np.any(anomaly_mask, axis=1)

    return anomaly_mask, overall_anomaly, anomaly_scores


def visualize_graph_3d_with_anomalies(graph, features, anomaly_mask, overall_anomaly,
                                      anomaly_scores, graph_name, feature_names, output_dir, vtk_file=None,
                                      skip_per_feature=False):
    """
    Create 3D visualization of the graph with anomalous nodes highlighted.

    Args:
        graph: DGL graph
        features: Node features array (num_nodes, 8)
        anomaly_mask: Boolean mask per feature (num_nodes, num_features)
        overall_anomaly: Boolean mask for any anomaly (num_nodes,)
        anomaly_scores: Anomaly scores (num_nodes,)
        graph_name: Name of the graph
        feature_names: List of feature names
    """
    # Extract positions (first 3 features are x, y, z)
    positions = features[:, :3]

    # Get edges
    src, dst = graph.edges()
    src = src.numpy()
    dst = dst.numpy()

    # Create figure with multiple views
    fig = plt.figure(figsize=(24, 8))
    fig.suptitle(f'3D Graph Visualization with Anomalies - {graph_name}',
                 fontsize=16, fontweight='bold')

    # View 1: Overall anomalies (any feature)
    ax1 = fig.add_subplot(131, projection='3d')
    plot_3d_graph_with_anomalies(ax1, positions, src, dst, overall_anomaly,
                                  title='All Anomalies Combined',
                                  normal_color='lightblue', anomaly_color='red')

    # View 2: Anomaly intensity (color by score)
    ax2 = fig.add_subplot(132, projection='3d')
    plot_3d_graph_colored_by_score(ax2, positions, src, dst, anomaly_scores,
                                   title='Anomaly Score Intensity')

    # View 3: Multi-feature anomalies
    num_anomalies_per_node = np.sum(anomaly_mask, axis=1)
    ax3 = fig.add_subplot(133, projection='3d')
    plot_3d_graph_multi_feature(ax3, positions, src, dst, num_anomalies_per_node,
                                title='Multi-Feature Anomalies')

    plt.tight_layout()
    filename = output_dir / f'graph_3d_anomalies_{graph_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ 3D graph visualization saved to '{filename}'")
    plt.close()

    # Create per-feature visualizations (unless skipped)
    if not skip_per_feature:
        visualize_per_feature_anomalies(graph, features, anomaly_mask, graph_name, feature_names, output_dir, vtk_path=vtk_file)


def plot_3d_graph_with_anomalies(ax, positions, src, dst, anomaly_mask,
                                 title='', normal_color='lightblue', anomaly_color='red'):
    """Plot 3D graph with binary anomaly highlighting."""
    # Draw edges
    for i in range(len(src)):
        s, d = src[i], dst[i]
        xs = [positions[s, 0], positions[d, 0]]
        ys = [positions[s, 1], positions[d, 1]]
        zs = [positions[s, 2], positions[d, 2]]
        ax.plot(xs, ys, zs, color='dimgray', alpha=0.25, linewidth=0.4)

    # Draw normal nodes
    normal_mask = ~anomaly_mask
    if np.any(normal_mask):
        ax.scatter(positions[normal_mask, 0],
                  positions[normal_mask, 1],
                  positions[normal_mask, 2],
                  c=normal_color, s=20, alpha=0.6, label='Normal', edgecolors='none')

    # Draw anomalous nodes
    if np.any(anomaly_mask):
        ax.scatter(positions[anomaly_mask, 0],
                  positions[anomaly_mask, 1],
                  positions[anomaly_mask, 2],
                  c=anomaly_color, s=50, alpha=0.9, label='Anomalous',
                  edgecolors='darkred', linewidths=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')

    # Add statistics
    num_anomalous = np.sum(anomaly_mask)
    num_total = len(anomaly_mask)
    percentage = 100 * num_anomalous / num_total
    ax.text2D(0.05, 0.95, f'Anomalous: {num_anomalous}/{num_total} ({percentage:.1f}%)',
             transform=ax.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))


def plot_3d_graph_colored_by_score(ax, positions, src, dst, scores, title=''):
    """Plot 3D graph with nodes colored by anomaly score intensity."""
    # Draw edges
    for i in range(len(src)):
        s, d = src[i], dst[i]
        xs = [positions[s, 0], positions[d, 0]]
        ys = [positions[s, 1], positions[d, 1]]
        zs = [positions[s, 2], positions[d, 2]]
        ax.plot(xs, ys, zs, color='dimgray', alpha=0.25, linewidth=0.4)

    # Draw nodes colored by score
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                        c=scores, s=30, alpha=0.8, cmap='YlOrRd',
                        edgecolors='black', linewidths=0.3, vmin=0, vmax=np.percentile(scores, 95))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Add colorbar
    plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='Anomaly Score')


def plot_3d_graph_multi_feature(ax, positions, src, dst, num_anomalies, title=''):
    """Plot 3D graph with nodes colored by number of anomalous features."""
    # Draw edges
    for i in range(len(src)):
        s, d = src[i], dst[i]
        xs = [positions[s, 0], positions[d, 0]]
        ys = [positions[s, 1], positions[d, 1]]
        zs = [positions[s, 2], positions[d, 2]]
        ax.plot(xs, ys, zs, color='dimgray', alpha=0.25, linewidth=0.4)

    # Color scheme for different anomaly counts
    colors = ['lightblue', 'yellow', 'orange', 'red', 'darkred']
    labels = ['Normal', '1 feature', '2 features', '3+ features', '4+ features']

    for anom_count in range(5):
        if anom_count == 0:
            mask = num_anomalies == 0
            size = 15
            alpha = 0.5
        elif anom_count < 3:
            mask = num_anomalies == anom_count
            size = 30 + anom_count * 10
            alpha = 0.7 + anom_count * 0.1
        elif anom_count == 3:
            mask = num_anomalies == 3
            size = 60
            alpha = 0.9
        else:  # 4+
            mask = num_anomalies >= 4
            size = 80
            alpha = 1.0

        if np.any(mask):
            ax.scatter(positions[mask, 0], positions[mask, 1], positions[mask, 2],
                      c=colors[anom_count], s=size, alpha=alpha,
                      label=labels[anom_count], edgecolors='black', linewidths=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)


def load_vtk_mesh(vtk_path):
    """Load mesh from VTK file."""
    if not PYVISTA_AVAILABLE:
        return None
    try:
        mesh = pv.read(str(vtk_path))
        return mesh
    except Exception as e:
        print(f"  Warning: Could not load VTK file: {e}")
        return None


def plot_vtk_mesh(ax, mesh, title='Original Protein Mesh'):
    """Plot VTK mesh on matplotlib 3D axis."""
    if mesh is None:
        ax.text(0.5, 0.5, 0.5, 'VTK mesh not available', 
                ha='center', va='center', fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        return
    
    # Extract points and faces
    points = mesh.points
    
    # Plot the surface points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
              c='lightcoral', s=1, alpha=0.3, edgecolors='none')
    
    # Try to plot faces if available
    if mesh.faces is not None and len(mesh.faces) > 0:
        # PyVista faces format: [n_points, p0, p1, p2, ...]
        faces_array = mesh.faces.reshape(-1, 4)[:, 1:4]  # Get triangles
        
        # Plot a subset of faces for performance
        max_faces = min(1000, len(faces_array))
        for i in range(0, max_faces, 10):  # Plot every 10th face
            face = faces_array[i]
            face_points = points[face]
            # Close the triangle
            face_points = np.vstack([face_points, face_points[0]])
            ax.plot(face_points[:, 0], face_points[:, 1], face_points[:, 2],
                   'gray', alpha=0.1, linewidth=0.3)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=12, fontweight='bold')


def visualize_per_feature_anomalies(graph, features, anomaly_mask, graph_name, feature_names, output_dir, vtk_path=None):
    """
    Create separate 3D visualizations for each feature's anomalies.
    """
    positions = features[:, :3]
    src, dst = graph.edges()
    src = src.numpy()
    dst = dst.numpy()

    # Always show all 8 features in a 3x3 grid
    num_features = len(feature_names)
    nrows = 3
    ncols = 3

    fig = plt.figure(figsize=(24, 24))
    fig.suptitle(f'Per-Feature Anomalies - {graph_name}', fontsize=16, fontweight='bold')

    # Plot each feature
    for feat_idx in range(num_features):
        ax = fig.add_subplot(nrows, ncols, feat_idx + 1, projection='3d')
        feature_anomaly = anomaly_mask[:, feat_idx]

        plot_3d_graph_with_anomalies(
            ax, positions, src, dst, feature_anomaly,
            title=feature_names[feat_idx],
            normal_color='lightblue',
            anomaly_color='red'
        )
    
    # Add VTK mesh in the 9th position
    ax9 = fig.add_subplot(nrows, ncols, 9, projection='3d')
    
    # Debug: print vtk_path status
    print(f"    VTK path for visualization: {vtk_path}")
    print(f"    VTK path exists: {vtk_path.exists() if vtk_path else 'N/A'}")
    print(f"    PyVista available: {PYVISTA_AVAILABLE}")
    
    if vtk_path and vtk_path.exists() and PYVISTA_AVAILABLE:
        print(f"    Loading VTK mesh from: {vtk_path}")
        mesh = load_vtk_mesh(vtk_path)
        plot_vtk_mesh(ax9, mesh, 'Original Protein Mesh')
    else:
        # Display a clear message about why VTK is not available
        if not PYVISTA_AVAILABLE:
            reason = 'PyVista not installed\nRun: pip install pyvista'
        elif not vtk_path:
            reason = 'VTK file path not provided'
        else:
            reason = f'VTK file not found\nSearched: {vtk_path}'
        
        print(f"    VTK mesh not loaded. Reason: {reason}")
        
        # Remove axis for cleaner look
        ax9.set_xlim([0, 1])
        ax9.set_ylim([0, 1])
        ax9.set_zlim([0, 1])
        ax9.set_xticks([])
        ax9.set_yticks([])
        ax9.set_zticks([])
        ax9.grid(False)
        ax9.axis('off')
        
        # Add centered text with better visibility
        ax9.text2D(0.5, 0.5, f'VTK Mesh Not Available\n\n{reason}', 
                  ha='center', va='center', fontsize=14, 
                  transform=ax9.transAxes,
                  bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', 
                           edgecolor='orange', linewidth=2, alpha=0.8))
        ax9.set_title('Original Protein Mesh', fontsize=12, fontweight='bold')

    plt.tight_layout()
    filename = output_dir / f'graph_3d_per_feature_{graph_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Per-feature 3D visualization saved to '{filename}'")
    plt.close()


def create_interactive_view(graph, features, anomaly_mask, overall_anomaly,
                           anomaly_scores, graph_name, output_dir):
    """
    Create a simple rotating view animation (multiple angles).
    """
    positions = features[:, :3]
    src, dst = graph.edges()
    src = src.numpy()
    dst = dst.numpy()

    # Create figure with 4 different viewing angles
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f'Multi-Angle View - {graph_name}', fontsize=16, fontweight='bold')

    angles = [(30, 45), (30, 135), (30, 225), (30, 315)]
    titles = ['Front-Right', 'Front-Left', 'Back-Left', 'Back-Right']

    for idx, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')

        # Draw edges
        for i in range(len(src)):
            s, d = src[i], dst[i]
            xs = [positions[s, 0], positions[d, 0]]
            ys = [positions[s, 1], positions[d, 1]]
            zs = [positions[s, 2], positions[d, 2]]
            ax.plot(xs, ys, zs, color='dimgray', alpha=0.25, linewidth=0.4)

        # Draw normal nodes
        normal_mask = ~overall_anomaly
        if np.any(normal_mask):
            ax.scatter(positions[normal_mask, 0],
                      positions[normal_mask, 1],
                      positions[normal_mask, 2],
                      c='lightblue', s=20, alpha=0.6, edgecolors='none')

        # Draw anomalous nodes
        if np.any(overall_anomaly):
            ax.scatter(positions[overall_anomaly, 0],
                      positions[overall_anomaly, 1],
                      positions[overall_anomaly, 2],
                      c='red', s=50, alpha=0.9, edgecolors='darkred', linewidths=1)

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(titles[idx], fontsize=12, fontweight='bold')

    plt.tight_layout()
    filename = output_dir / f'graph_3d_multiangle_{graph_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Multi-angle visualization saved to '{filename}'")
    plt.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize protein graphs in 3D with anomalous nodes highlighted',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset parameters
    parser.add_argument('--root_dir', type=str,
                       default="/home/ogharbi/datasets/SHREC25/graphs/normalized_8feat_1k",
                       help='Root directory containing graph data')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'test'],
                       help='Which split to visualize')

    # Visualization parameters
    parser.add_argument('--num_graphs', type=int, default=3,
                       help='Number of graphs to visualize')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for graph sampling')

    # Anomaly detection parameters
    parser.add_argument('--method', type=str, default='zscore',
                       choices=['zscore', 'mad', 'combined'],
                       help='Anomaly detection method')
    parser.add_argument('--threshold', type=float, default=2.5,
                       help='Threshold for anomaly detection (for zscore: number of std)')

    # Visualization options
    parser.add_argument('--skip_per_feature', action='store_true',
                       help='Skip per-feature visualizations (faster)')
    parser.add_argument('--skip_multiangle', action='store_true',
                       help='Skip multi-angle visualizations (faster)')

    # Graph selection
    parser.add_argument('--graph_names', type=str, nargs='+', default=None,
                       help='Specific graph names to visualize (e.g., 8kd6_5_F_R_model1)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for experiment results (default: auto-generated with timestamp)')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for the experiment (default: auto-generated)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Feature names
    feature_names = [
        "pos_x", "pos_y", "pos_z",
        "normal_x", "normal_y", "normal_z",
        "potential_norm", "normal_potential_norm"
    ]

    # Create experiment directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.experiment_name:
            exp_name = f"{args.experiment_name}_{timestamp}"
        else:
            exp_name = f"graph_viz_{args.method}_t{args.threshold}_{timestamp}"
        output_dir = Path(f"experiments/{exp_name}")
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("3D Graph Visualization with Anomalies")
    print("=" * 80)
    print(f"Dataset: {args.root_dir}")
    print(f"Split: {args.split}")
    print(f"Method: {args.method}, Threshold: {args.threshold}")
    print(f"Num graphs: {args.num_graphs}")
    print(f"Output directory: {output_dir}")
    print()

    # Get graph files
    root_path = Path(args.root_dir)
    split_dir = root_path / args.split

    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")

    graph_files = sorted(split_dir.glob("*.bin"))

    if len(graph_files) == 0:
        raise ValueError(f"No graph files found in {split_dir}")

    # Select graphs to visualize
    if args.graph_names is not None:
        # Visualize specific graphs by name
        sampled_files = []
        for name in args.graph_names:
            matching = [f for f in graph_files if f.stem == name]
            if matching:
                sampled_files.append(matching[0])
            else:
                print(f"Warning: Graph '{name}' not found, skipping...")

        if len(sampled_files) == 0:
            raise ValueError("None of the specified graph names were found")
    else:
        # Sample random graphs
        np.random.seed(args.seed)
        sampled_files = np.random.choice(graph_files, min(args.num_graphs, len(graph_files)),
                                         replace=False)

    print(f"Processing {len(sampled_files)} graphs...")
    
    # Build VTK file mapping by normalized names (strip underscores and colons)
    vtk_file_map = {}
    if PYVISTA_AVAILABLE:
        vtk_dir = root_path.parent / 'vtk' / f'{args.split}_set'
        if vtk_dir.exists():
            for vtk_file in vtk_dir.glob('*.vtk'):
                # Normalize: remove underscores and colons from stem
                normalized_name = vtk_file.stem.replace('_', '').replace(':', '')
                vtk_file_map[normalized_name] = vtk_file
            print(f"Found {len(vtk_file_map)} VTK files in {vtk_dir}")
    
    # Store graph info for config
    processed_graphs = []
    
    for graph_file in tqdm(sampled_files):
        graph_name = graph_file.stem
        print(f"\n{'='*80}")
        print(f"Processing: {graph_name}")
        print(f"{'='*80}")

        # Find corresponding VTK file by normalized name matching
        vtk_file = None
        if PYVISTA_AVAILABLE and vtk_file_map:
            # Normalize graph name: remove underscores
            normalized_graph_name = graph_name.replace('_', '')
            if normalized_graph_name in vtk_file_map:
                vtk_file = vtk_file_map[normalized_graph_name]
                print(f"  ✓ Found VTK file: {vtk_file.name}")
            else:
                print(f"  ✗ VTK file not found (normalized name: {normalized_graph_name})")

        # Load graph
        graph = load_graph(graph_file)
        features = graph.ndata['attr'].numpy()

        # Detect anomalies
        anomaly_mask, overall_anomaly, anomaly_scores = detect_anomalous_nodes(
            features, method=args.method, threshold=args.threshold
        )

        num_anomalous = np.sum(overall_anomaly)
        print(f"  Found {num_anomalous} anomalous nodes ({100*num_anomalous/len(overall_anomaly):.1f}%)")

        # Store graph info
        graph_info = {
            'name': graph_name,
            'file_path': str(graph_file),
            'num_nodes': graph.num_nodes(),
            'num_edges': graph.num_edges(),
            'num_anomalous_nodes': int(num_anomalous),
            'anomaly_percentage': float(100*num_anomalous/len(overall_anomaly))
        }
        processed_graphs.append(graph_info)

        # Create visualizations
        print("  Creating 3D visualizations...")

        # Main visualization (always created, includes per-feature if not skipped)
        visualize_graph_3d_with_anomalies(
            graph, features, anomaly_mask, overall_anomaly,
            anomaly_scores, graph_name, feature_names, output_dir, vtk_file,
            skip_per_feature=args.skip_per_feature
        )

        # Optional multi-angle visualization
        if not args.skip_multiangle:
            create_interactive_view(
                graph, features, anomaly_mask, overall_anomaly,
                anomaly_scores, graph_name, output_dir
            )

    # Save experiment config
    config = {
        'experiment': {
            'name': args.experiment_name if args.experiment_name else output_dir.name,
            'timestamp': datetime.now().isoformat(),
            'output_dir': str(output_dir)
        },
        'parameters': {
            'root_dir': args.root_dir,
            'split': args.split,
            'method': args.method,
            'threshold': args.threshold,
            'num_graphs': args.num_graphs,
            'seed': args.seed,
            'skip_per_feature': args.skip_per_feature,
            'skip_multiangle': args.skip_multiangle
        },
        'proteins': processed_graphs,
        'feature_names': feature_names,
        'summary': {
            'total_graphs_processed': len(processed_graphs),
            'total_nodes': sum(g['num_nodes'] for g in processed_graphs),
            'total_edges': sum(g['num_edges'] for g in processed_graphs),
            'total_anomalous_nodes': sum(g['num_anomalous_nodes'] for g in processed_graphs)
        }
    }
    
    config_file = output_dir / 'experiment_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"Config file: {config_file}")
    print("\nGenerated files per graph:")
    print("  - graph_3d_anomalies_<name>.png: Three views (combined, score, multi-feature)")
    if not args.skip_per_feature:
        print("  - graph_3d_per_feature_<name>.png: Separate view for each feature")
    if not args.skip_multiangle:
        print("  - graph_3d_multiangle_<name>.png: Four viewing angles")
    print(f"\nProcessed {len(processed_graphs)} proteins:")
    for g in processed_graphs:
        print(f"  - {g['name']}: {g['num_nodes']} nodes, {g['num_anomalous_nodes']} anomalous ({g['anomaly_percentage']:.1f}%)")


if __name__ == "__main__":
    main()
