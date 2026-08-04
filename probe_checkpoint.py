#!/usr/bin/env python3
"""
Re-score ONE already-trained HierarchicalGNN checkpoint against a (corrected)
test split -- inference only, no training. Rebuilds the exact architecture
from the run's own final_results.json["config"] (this is why grid_hiergin.py
runs are reusable: classify_hierarchical_gnn.py already saves both
best_model.pt and the full config needed to reconstruct the model).

Use case: the SHREC19 .cla parser bug (papers/paper_hiergin_shrec19/TODO.md,
P3) dropped one surface per class from BOTH the train and test splits. Since
the recovered surfaces never existed in either split before, scoring an
EXISTING checkpoint against the corrected test set is a valid, retrain-free
way to measure how much the corrected test set changes the paper's reported
test metrics -- it does not tell you whether retraining on the corrected
(slightly larger) train set would additionally help; that is what
run_shrec19_recheck.sh (full retrain) is for.

Usage:
  python probe_checkpoint.py \
      --run-dir  $ORIG_STUDY_ROOT/shrec19_protein_geom_k64/runs/shrec19_protein_geom_k64_gvp_h256_s1 \
      --test-labels-csv /path/to/corrected/graphs_pyg_shared_proteins/test_labels.csv \
      [--graph-dir ... --assign-dir ...]   # default: reuse the run's own saved paths
      --out corrected_test_shrec19_protein_geom_k64_gvp_h256_s1.json
"""
from __future__ import annotations
import sys, argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from classify_gnn import load_labels
from classify_hierarchical_gnn import (
    HierarchicalGNN, HierPatchDataset, collate, collate_three,
    run_epoch, compute_map,
)


def build_model_from_cfg(cfg: dict, device) -> nn.Module:
    xyz_slice = slice(5, 8)
    if cfg["feature_mode"] in ("coords", "coords_normals", "coords_potential"):
        xyz_slice = slice(0, 3)
    model = HierarchicalGNN(
        cfg["in_dim"], cfg["hidden_dim"], cfg["num_classes"],
        cfg["node_layers"], cfg["patch_layers"], cfg["pool"], cfg["readout"],
        cfg["dropout"],
        node_readout=cfg["node_readout"],
        node_backbone=cfg["node_backbone"],
        patch_backbone=cfg["patch_backbone"],
        pna_delta=cfg["pna_delta"],
        scalar_inject=cfg["scalar_inject"],
        patch_scalar_inject=cfg["patch_scalar_inject"],
        patch_inject_all=cfg["patch_inject_all"],
        three_level=(cfg.get("assign_dir2") is not None),
        mid_layers=cfg["mid_layers"],
        xyz_slice=xyz_slice,
        proj_dim=cfg["proj_dim"],
    ).to(device)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                     help="original run dir containing best_model.pt + final_results.json")
    ap.add_argument("--test-labels-csv", default=None,
                     help="corrected test_labels.csv; default = the run's own saved path")
    ap.add_argument("--graph-dir", default=None, help="default = the run's own saved graph_dir")
    ap.add_argument("--assign-dir", default=None, help="default = the run's own saved assign_dir")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=None, help="default = the run's own batch_size")
    ap.add_argument("--out", default=None,
                     help="default = <run-dir>/final_results_corrected_test.json")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    fr_path = run_dir / "final_results.json"
    ckpt_path = run_dir / "best_model.pt"
    if not fr_path.is_file():
        sys.exit(f"[probe] missing {fr_path}")
    if not ckpt_path.is_file():
        sys.exit(f"[probe] missing {ckpt_path}")

    orig = json.load(open(fr_path))
    cfg = orig["config"]
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    graph_dir = args.graph_dir or cfg["graph_dir"]
    assign_dir = args.assign_dir or cfg["assign_dir"]
    test_labels_csv = args.test_labels_csv or cfg["test_labels_csv"]
    batch_size = args.batch_size or cfg["batch_size"]

    print(f"[probe] run_dir         = {run_dir}")
    print(f"[probe] graph_dir       = {graph_dir}")
    print(f"[probe] assign_dir      = {assign_dir}")
    print(f"[probe] test_labels_csv = {test_labels_csv}  (orig was {cfg['test_labels_csv']})")

    model = build_model_from_cfg(cfg, device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    use_lrf = (cfg["node_backbone"] == "lrf_gin")
    three_level = (cfg.get("assign_dir2") is not None)
    test_labels, _ = load_labels(graph_dir, "test", test_labels_csv)
    test_ds = HierPatchDataset(
        graph_dir, assign_dir, "test", test_labels,
        ssl_emb_dir=cfg.get("ssl_emb_dir"), feature_mode=cfg["feature_mode"],
        rwpe_dim=cfg.get("rwpe_dim", 0), use_lrf_feat=use_lrf,
        assign_dir2=cfg.get("assign_dir2"),
    )
    col = collate_three if three_level else collate
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=col)

    criterion = nn.CrossEntropyLoss()  # loss value unused; run_epoch needs one
    test_m, test_labs, test_scores = run_epoch(
        model, test_loader, criterion, device, None, cfg["num_classes"],
        "corrected-test", collect_scores=True,
    )
    test_m["mAP"] = compute_map(test_labs, test_scores, cfg["num_classes"])
    del test_m["loss"]

    print(f"[probe] n_test (corrected) = {len(test_ds)}")
    print("=== CORRECTED-TEST metrics (same checkpoint, no retraining) ===")
    for k in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
              "tpr_macro", "tnr_macro", "mAP"]:
        old = orig["final_test_metrics"].get(k)
        new = test_m.get(k)
        old_s = f"{old:.4f}" if isinstance(old, (int, float)) else str(old)
        print(f"  {k:18s}: orig={old_s}  corrected={new:.4f}")

    out_path = Path(args.out) if args.out else run_dir / "final_results_corrected_test.json"
    json.dump({
        "run_dir": str(run_dir),
        "orig_final_test_metrics": orig["final_test_metrics"],
        "corrected_test_metrics": test_m,
        "n_test_corrected": len(test_ds),
        "graph_dir": graph_dir, "assign_dir": assign_dir,
        "test_labels_csv": test_labels_csv,
        "config": cfg,
    }, open(out_path, "w"), indent=2, default=float)
    print(f"[probe] saved -> {out_path}")


if __name__ == "__main__":
    main()
