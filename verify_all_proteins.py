#!/usr/bin/env python3
"""
Verify that all 9244 proteins in train_labels.csv can be loaded successfully.
"""

import pandas as pd
from pathlib import Path
import dgl

def verify_proteins(data_dir, split='train'):
    """Verify all proteins can be loaded."""
    labels_file = Path(data_dir) / f'{split}_labels.csv'
    split_dir = Path(data_dir) / split

    print(f"{'='*80}")
    print(f"VERIFYING {split.upper()} SET")
    print(f"{'='*80}")

    # Load labels
    df = pd.read_csv(labels_file)
    print(f"\nLabels CSV: {len(df)} entries")

    # Check each protein
    missing = []
    loadable = []
    errors = []

    print(f"Checking if all {len(df)} proteins can be loaded...")

    for idx, row in df.iterrows():
        protein_id = str(row['protein_id'])
        class_id = int(row['class_id'])

        # Try to find the file
        graph_file = split_dir / f'{protein_id}.bin'

        if not graph_file.exists():
            missing.append((protein_id, class_id))
        else:
            # Try to load it
            try:
                graphs, _ = dgl.load_graphs(str(graph_file))
                g = graphs[0]
                loadable.append((protein_id, class_id, g.num_nodes(), g.num_edges()))
            except Exception as e:
                errors.append((protein_id, class_id, str(e)))

        # Progress
        if (idx + 1) % 1000 == 0:
            print(f"  Checked {idx + 1}/{len(df)}...")

    # Report
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"\n✅ Successfully loadable: {len(loadable)}/{len(df)} ({len(loadable)/len(df)*100:.2f}%)")

    if missing:
        print(f"\n❌ Missing files: {len(missing)}")
        print(f"   First 10 examples:")
        for protein_id, class_id in missing[:10]:
            print(f"     {protein_id} (class {class_id})")

    if errors:
        print(f"\n⚠️  Load errors: {len(errors)}")
        print(f"   First 5 examples:")
        for protein_id, class_id, err in errors[:5]:
            print(f"     {protein_id} (class {class_id}): {err}")

    if len(loadable) == len(df):
        print(f"\n🎉 ALL {len(df)} PROTEINS VERIFIED SUCCESSFULLY!")

        # Stats
        nodes = [x[2] for x in loadable]
        edges = [x[3] for x in loadable]
        print(f"\nGraph statistics:")
        print(f"  Nodes: all graphs have {nodes[0]} nodes" if len(set(nodes)) == 1 else f"  Nodes: vary from {min(nodes)} to {max(nodes)}")
        print(f"  Edges: all graphs have {edges[0]} edges" if len(set(edges)) == 1 else f"  Edges: vary from {min(edges)} to {max(edges)}")

    return loadable, missing, errors

def main():
    data_dir = '/home/ogharbi/datasets/SHREC25/graphs/normalized_8feat_1k/'

    # Verify train set
    train_loadable, train_missing, train_errors = verify_proteins(data_dir, 'train')

    print(f"\n\n")

    # Verify test set
    test_loadable, test_missing, test_errors = verify_proteins(data_dir, 'test')

    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\nTrain: {len(train_loadable)}/{9244} verified ✅" if len(train_loadable) == 9244 else f"Train: {len(train_loadable)}/{9244} verified ⚠️")
    print(f"Test:  {len(test_loadable)}/{2308} verified ✅" if len(test_loadable) == 2308 else f"Test:  {len(test_loadable)}/{2308} verified ⚠️")

    if len(train_loadable) == 9244 and len(test_loadable) == 2308:
        print(f"\n🎉 DATASET READY FOR TRAINING!")
    else:
        print(f"\n⚠️  Some issues found - see details above")

if __name__ == '__main__':
    main()
