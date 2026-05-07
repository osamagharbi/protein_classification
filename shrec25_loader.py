"""
Custom loader for SHREC25 protein mesh graphs.
Loads pre-converted DGL graphs from .bin files with on-demand loading.
"""

import torch
import dgl
from pathlib import Path
import json


def _resolve_split_dir(root_path: Path, split: str) -> Path:
    """Resolve split folder names across dataset variants.

    Supports both train/test and train_set/test_set naming.
    """
    candidates = [root_path / split]
    if split in ("train", "test"):
        candidates.append(root_path / f"{split}_set")
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _list_graph_files(split_dir: Path):
    """List graph files for supported formats in a split directory."""
    files = sorted(split_dir.glob("*.bin"))
    if files:
        return files
    return sorted(split_dir.glob("*.pt"))


def _load_graph_any(graph_file: Path):
    """Load either a DGL .bin graph or a PyTorch .pt graph dict into DGL."""
    if graph_file.suffix == ".bin":
        graphs, _ = dgl.load_graphs(str(graph_file))
        return graphs[0]

    if graph_file.suffix == ".pt":
        data = torch.load(graph_file, map_location="cpu", weights_only=False)
        if not isinstance(data, dict):
            raise TypeError(f"Unsupported .pt content in {graph_file}")

        edge_index = data.get("edge_index_mesh", data.get("edge_index"))
        feat = data.get("x", data.get("pos"))
        if edge_index is None or feat is None:
            raise KeyError(
                f"{graph_file} missing required keys. Expected edge_index_mesh/edge_index and x/pos"
            )

        if not torch.is_tensor(edge_index):
            edge_index = torch.as_tensor(edge_index)
        if not torch.is_tensor(feat):
            feat = torch.as_tensor(feat)

        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=int(feat.shape[0]))
        g.ndata["attr"] = feat.float()
        if "pos" in data and torch.is_tensor(data["pos"]):
            g.ndata["pos"] = data["pos"].float()
        return g

    raise ValueError(f"Unsupported graph format: {graph_file}")


class SHREC25Dataset:
    """
    On-demand dataset loader for SHREC25.
    Loads graphs from .bin files only when accessed (lazy loading).
    This prevents loading all 11,555 graphs into RAM at once.
    """
    def __init__(self, root_dir="/home/ogharbi/datasets/SHREC25/graphs/normalized_8feat_1k"):
        self.root_path = Path(root_dir)

        # Get all graph file paths
        train_dir = _resolve_split_dir(self.root_path, "train")
        test_dir = _resolve_split_dir(self.root_path, "test")

        self.train_files = _list_graph_files(train_dir)
        self.test_files = _list_graph_files(test_dir)
        self.all_files = self.train_files + self.test_files

        print(f"SHREC25 Dataset initialized:")
        print(f"  Train graphs: {len(self.train_files)}")
        print(f"  Test graphs: {len(self.test_files)}")
        print(f"  Total: {len(self.all_files)}")

        # Load metadata
        metadata_file = train_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                self.num_features = metadata.get('num_features', metadata.get('feature_dim', 11))
        else:
            self.num_features = 11

        self.num_classes = 1  # Dummy for unsupervised

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        """Load a single graph on-demand when accessed."""
        graph_file = self.all_files[idx]

        # Load DGL graph from .bin or convert .pt dict to DGL
        dgl_graph = _load_graph_any(graph_file)

        # Dummy label for unsupervised learning
        label = torch.tensor([0])

        return dgl_graph, label


def load_shrec25_dataset(root_dir="/home/ogharbi/datasets/SHREC25/graphs/normalized_8feat_1k"):
    """
    Load SHREC25 protein mesh graphs in DGL format with on-demand loading.

    Args:
        root_dir: Root directory containing train/ and test/ subdirectories

    Returns:
        dataset: SHREC25Dataset object (lazy-loading, no memory overhead)
        (num_features, num_classes): Feature and class dimensions
    """
    # Create lazy-loading dataset
    dataset = SHREC25Dataset(root_dir)

    print(f"✓ SHREC25 dataset ready (lazy-loading mode)")
    print(f"  Num features: {dataset.num_features}")
    print(f"  Graphs will be loaded on-demand during training")

    return dataset, (dataset.num_features, dataset.num_classes)


class SHREC25SplitDataset:
    """
    On-demand dataset loader for single split (train or test).
    """
    def __init__(self, root_dir="/home/ogharbi/datasets/SHREC25/graphs/normalized_8feat_1k", split="train"):
        self.root_path = Path(root_dir)
        split_dir = _resolve_split_dir(self.root_path, split)

        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        self.graph_files = _list_graph_files(split_dir)

        print(f"SHREC25 {split} split initialized:")
        print(f"  Graphs: {len(self.graph_files)}")

        # Load metadata
        metadata_file = split_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                self.num_features = metadata.get('num_features', metadata.get('feature_dim', 11))
        else:
            self.num_features = 11

        self.num_classes = 1

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        """Load a single graph on-demand when accessed."""
        graph_file = self.graph_files[idx]
        dgl_graph = _load_graph_any(graph_file)
        label = torch.tensor([0])
        return dgl_graph, label


def load_shrec25_split(root_dir="/home/ogharbi/datasets/SHREC25/graphs/normalized_8feat_1k", split="train"):
    """
    Load only train or test split with on-demand loading.

    Args:
        root_dir: Root directory
        split: "train" or "test"

    Returns:
        dataset: SHREC25SplitDataset object (lazy-loading)
        (num_features, num_classes): Feature and class dimensions
    """
    dataset = SHREC25SplitDataset(root_dir, split)
    print(f"✓ SHREC25 {split} split ready (lazy-loading mode)")
    return dataset, (dataset.num_features, dataset.num_classes)


if __name__ == "__main__":
    # Test the loader
    print("Testing SHREC25 lazy-loading dataset...\n")
    dataset, (num_features, num_classes) = load_shrec25_dataset()

    print("\nDataset info:")
    print(f"  Total graphs: {len(dataset)}")
    print(f"  Num features: {num_features}")

    print("\nLoading first graph on-demand...")
    graph, label = dataset[0]
    print(f"  Sample graph: {graph}")
    print(f"  Node features shape: {graph.ndata['attr'].shape}")
    print(f"  Num nodes: {graph.num_nodes()}")
    print(f"  Num edges: {graph.num_edges()}")

    print("\n✓ Lazy loading working correctly!")
