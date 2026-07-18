#!/usr/bin/env python3
"""
Ensemble inference: average logits from multiple trained hierarchical GNN checkpoints.

Usage:
  python ensemble_predict.py \
    --run-dirs runs/hier_k64_coords_pot_attn runs/hier_attn2 \
    --graph-dir /home/ogharbi/datasets/SHREC25/ssl_graphs_1k \
    --assign-dir /home/ogharbi/datasets/SHREC25/ssl_graphs_1k_patchassign_k64
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from classify_hierarchical_gnn import HierPatchDataset, HierarchicalGNN, collate
from classify_gnn import load_labels, compute_metrics
from fix_training import compute_class_weights


def load_model(run_dir: Path, device):
    cfg_path = run_dir / "final_results.json"
    data = json.load(open(cfg_path))
    cfg = data["config"]

    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best_model.pt in {run_dir}. "
                                "Re-run with --save-model flag (see below).")
    state = torch.load(ckpt_path, map_location=device)

    model = HierarchicalGNN(
        in_dim=state["node_convs.0.apply_func.0.weight"].shape[1]
               if "node_convs.0.apply_func.0.weight" in state
               else cfg.get("in_dim", 4),
        hidden_dim=cfg["hidden_dim"],
        num_classes=cfg.get("num_classes", 99),
        node_layers=cfg["node_layers"],
        patch_layers=cfg["patch_layers"],
        pool=cfg["pool"],
        readout=cfg.get("readout", "max"),
        dropout=cfg["dropout"],
        node_readout=cfg.get("node_readout", False),
        node_backbone=cfg.get("node_backbone", "gin"),
        patch_backbone=cfg.get("patch_backbone", "gin"),
        pna_delta=cfg.get("pna_delta") or 6.0,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def get_logits(model, loader, device):
    all_logits, all_labels = [], []
    with torch.no_grad():
        for bn, bp, gpid, y in loader:
            bn = bn.to(device); bp = bp.to(device); gpid = gpid.to(device)
            logits = model(bn, bp, gpid)
            all_logits.append(logits.cpu())
            all_labels.extend(y.numpy())
    return torch.cat(all_logits, dim=0), np.array(all_labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--graph-dir", default="/home/ogharbi/datasets/SHREC25/ssl_graphs_1k")
    ap.add_argument("--assign-dir", default="/home/ogharbi/datasets/SHREC25/ssl_graphs_1k_patchassign_k64")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    run_dirs = [Path(d) for d in args.run_dirs]

    # Use config from first model for dataset setup
    cfg0 = json.load(open(run_dirs[0] / "final_results.json"))["config"]
    test_labels, num_classes = load_labels(args.graph_dir, "test", cfg0.get("test_labels_csv"))

    test_ds = HierPatchDataset(
        args.graph_dir, args.assign_dir, "test", test_labels,
        feature_mode=cfg0.get("feature_mode", "all"),
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate)

    print(f"Ensembling {len(run_dirs)} models on {len(test_ds)} test samples...")
    sum_logits = None
    labels = None

    for rd in run_dirs:
        print(f"  Loading {rd.name} ...")
        model, cfg = load_model(rd, device)
        logits, labs = get_logits(model, test_loader, device)
        if sum_logits is None:
            sum_logits = logits
            labels = labs
        else:
            sum_logits = sum_logits + logits

    avg_logits = sum_logits / len(run_dirs)
    preds = avg_logits.argmax(1).numpy()
    m = compute_metrics(labels.tolist(), preds.tolist(), num_classes=num_classes)
    print("\n=== ENSEMBLE TEST RESULTS ===")
    for k in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "f1_micro"]:
        print(f"  {k:20s}: {m[k]:.4f}")


if __name__ == "__main__":
    main()
