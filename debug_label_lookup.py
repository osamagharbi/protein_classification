#!/usr/bin/env python3
"""Debug label lookup issue."""

import argparse
import os
from classify_gnn import LabeledSHREC25Dataset, load_labels

def main():
    parser = argparse.ArgumentParser(description="Debug label resolution for SHREC25 graphs")
    parser.add_argument(
        "--data-dir",
        default=os.getenv("SHREC25_GRAPH_DATA_DIR", "datasets/SHREC25/graphs/normalized_8feat_1k"),
        help="Dataset root containing train split and labels",
    )
    args = parser.parse_args()

    train_labels, _ = load_labels(args.data_dir, "train")

    print(f"Label dict size: {len(train_labels)}")
    print(f"\nFirst 5 label dict entries:")
    for _, (k, v) in enumerate(list(train_labels.items())[:5]):
        print(f"  '{k}' -> {v}")

    dataset = LabeledSHREC25Dataset(args.data_dir, "train", train_labels)

    print(f"\nDataset size: {len(dataset)}")
    print(f"\nFirst 5 graph files:")
    for i in range(min(5, len(dataset))):
        graph_file = dataset.graph_files[i]
        print(f"  {graph_file.name} (stem: '{graph_file.stem}')")

    print(f"\nTrying to get first 5 items:")
    for i in range(min(5, len(dataset))):
        graph, label = dataset[i]
        graph_file = dataset.graph_files[i]
        graph_name = graph_file.stem
        expected_label = train_labels.get(graph_name, "NOT_FOUND")
        print(f"  Index {i}: file='{graph_name}', got label={label.item()}, expected={expected_label}")


if __name__ == "__main__":
    main()
