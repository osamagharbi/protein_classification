#!/usr/bin/env python3
"""
Re-evaluate the best HierGIN-14D checkpoint on the REAL 90-class SHREC25 test
set (pc/test_labels.csv) instead of the stale graphs_old 88-class set, to
measure whether dropping real classes 97 & 98 affected the headline number.
"""
import sys, warnings; sys.path.insert(0, "."); warnings.filterwarnings("ignore")
import numpy as np, torch, pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, accuracy_score

from classify_gnn import canonical_graph_id
from classify_hierarchical_gnn import HierPatchDataset, collate, HierarchicalGNN

GRAPH = "/home/ogharbi/datasets/SHREC25/ssl_graphs_1k_geom_enriched"
ASSIGN = "/home/ogharbi/datasets/SHREC25/ssl_graphs_1k_patchassign_k64"
CKPT = "runs/hier_gin_h512_geom14d_raw/best_model.pt"

def label_dict_from_csv(csv):
    df = pd.read_csv(csv)
    d = {}
    for _, r in df.iterrows():
        pid = str(r["protein_id"])
        if pid.endswith("_pc"):
            pid = pid[:-3]
        d[canonical_graph_id(pid)] = int(r["class_id"])
    return d

def evaluate(label_dict, tag):
    ds = HierPatchDataset(GRAPH, ASSIGN, "test", label_dict, feature_mode="geom_14d_raw")
    ld = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate, num_workers=4)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HierarchicalGNN(14, 512, 99, node_layers=3, patch_layers=2,
                            pool="attn", readout="max", node_backbone="gin",
                            patch_backbone="gin").to(dev)
    model.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False))
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for bn, bp, gpid, y in ld:
            bn, bp, gpid = bn.to(dev), bp.to(dev), gpid.to(dev)
            preds.extend(model(bn, bp, gpid).argmax(1).cpu().numpy())
            labs.extend(y.numpy())
    preds, labs = np.array(preds), np.array(labs)
    print(f"\n=== {tag}: {len(ds)} test proteins, {len(set(labs))} distinct true classes ===")

    valid = sorted(set(labs.tolist()))           # classes actually present in test
    for denom, lbls in [("range(99)", list(range(99))), (f"{len(valid)} present", valid)]:
        mf1 = f1_score(labs, preds, average="macro", zero_division=0, labels=lbls)
        bac = recall_score(labs, preds, average="macro", zero_division=0, labels=lbls)
        print(f"   macro-F1 [{denom:14s}] = {mf1:.4f}   bal-acc = {bac:.4f}")
    print(f"   accuracy = {accuracy_score(labs, preds):.4f}   "
          f"wtd-F1 = {f1_score(labs, preds, average='weighted', zero_division=0, labels=valid):.4f}")

    # per-class F1 for the two classes that graphs_old dropped
    per = f1_score(labs, preds, average=None, zero_division=0, labels=valid)
    for c in (97, 98):
        if c in valid:
            print(f"   >>> class {c}: F1 = {per[valid.index(c)]:.4f}  (support={int((labs==c).sum())})")
    return labs, preds

if __name__ == "__main__":
    print("################ graphs_old 88-class (reproduce report 0.808) ################")
    evaluate(label_dict_from_csv("/home/ogharbi/datasets/SHREC25/graphs_old/test_labels.csv"),
             "graphs_old (88-class)")
    print("\n################ pc 90-class (real SHREC25 test) ################")
    evaluate(label_dict_from_csv("/home/ogharbi/datasets/SHREC25/pc/test_labels.csv"),
             "pc (90-class)")
