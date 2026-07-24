#!/usr/bin/env python3
"""
SHREC21 retrieval evaluation: extract HierGIN embeddings for the 10 official
query surfaces and the 554 database surfaces, build the query-to-database
distance matrix, and (once query ground truth is available) compute standard
retrieval metrics (NN, first-tier, second-tier, mAP).

Uses the EXACT SHREC19-winning HierGIN recipe/checkpoint for each modality
(no SHREC21-specific tuning) so the method is identical across both datasets
for the paper. Does not modify anything used by SHREC19 training/eval.

Usage:
    python shrec21_retrieval_eval.py --modality shape
    python shrec21_retrieval_eval.py --modality electro
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

import classify_hierarchical_gnn as CH

RUN_DIRS = {
    "shape": "runs_shrec21/shrec21_shape_hier_gin_coords_ls",
    "electro": "runs_shrec21/shrec21_electro_hier_gin_coordspot_ls",
}

# Working hypothesis (NOT yet confirmed from an official source): query_N
# corresponds to the Nth class in the .cla file's declared order.
CLA_CLASS_ORDER = ["SH2", "SH3", "PDZ", "SH3_2", "m50", "PDZ_6", "Bromodomain", "PHD", "ZF", "Stat"]


def load_model_and_cfg(modality):
    run_dir = Path(RUN_DIRS[modality])
    cfg = json.load(open(run_dir / "final_results.json"))["config"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = cfg["num_classes"]

    model = CH.HierarchicalGNN(
        cfg["in_dim"], cfg["hidden_dim"], num_classes, cfg["node_layers"], cfg["patch_layers"],
        cfg["pool"], cfg["readout"], cfg["dropout"],
        node_backbone=cfg["node_backbone"], patch_backbone=cfg["patch_backbone"],
        pna_delta=cfg.get("pna_delta", 6.0), proj_dim=cfg.get("proj_dim", 128),
    ).to(device)
    state = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg, device


def extract_embeddings(model, cfg, device, graph_dir, assign_dir, split, label_dict):
    ds_kw = dict(ssl_emb_dir=cfg.get("ssl_emb_dir"), feature_mode=cfg["feature_mode"],
                 rwpe_dim=cfg.get("rwpe_dim", 0), use_lrf_feat=False, assign_dir2=None)
    ds = CH.HierPatchDataset(graph_dir, assign_dir, split, label_dict, **ds_kw)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=CH.collate, num_workers=2)

    ids = [gid for gid, _, _ in ds.items]
    embs = []
    with torch.no_grad():
        for bn, bp, gpid, y in loader:
            bn, bp, gpid = bn.to(device), bp.to(device), gpid.to(device)
            _, hg = model(bn, bp, gpid, return_embedding=True)
            embs.append(hg.cpu().numpy())
    return ids, np.concatenate(embs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["shape", "electro"], required=True)
    ap.add_argument("--out-dir", default="runs_shrec21/retrieval")
    args = ap.parse_args()

    D = Path("/home/ogharbi/datasets/SHREC21")
    graph_dir = D / f"ssl_graphs_{args.modality}"
    assign_dir = D / f"ssl_graphs_{args.modality}_patchassign_k64"
    label_csv = D / f"graphs_pyg_{args.modality}" / "train_labels.csv"

    model, cfg, device = load_model_and_cfg(args.modality)
    print(f"loaded {args.modality} model: {RUN_DIRS[args.modality]}")

    # database (554 surfaces) -- use "train" split resolution (train_set symlink -> all_set)
    db_labels, num_classes = CH.load_labels(str(graph_dir), "train", str(label_csv))
    db_ids, db_emb = extract_embeddings(model, cfg, device, str(graph_dir), str(assign_dir),
                                         "train", db_labels)
    print(f"database embeddings: {db_emb.shape}")

    # queries (10 surfaces) -- no labels needed for embedding extraction itself.
    # NOTE: pass split="query_set" literally -- resolve_split_dir() only has a
    # train/test -> train_set/test_set fallback, not query -> query_set, so the
    # literal folder name must be given directly.
    query_dummy_labels = {f"query_{i}": 0 for i in range(1, 11)}
    q_ids, q_emb = extract_embeddings(model, cfg, device, str(graph_dir), str(assign_dir),
                                       "query_set", query_dummy_labels)
    print(f"query embeddings: {q_emb.shape}  ids={q_ids}")

    # cosine distance matrix: queries x database
    q_norm = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    db_norm = db_emb / np.linalg.norm(db_emb, axis=1, keepdims=True)
    cos_sim = q_norm @ db_norm.T
    dist = 1.0 - cos_sim  # [10, 554]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{args.modality}_distance_matrix.npz",
             dist=dist, query_ids=np.array(q_ids), db_ids=np.array(db_ids),
             db_labels=np.array([db_labels[i] for i in db_ids]))
    print(f"saved -> {out_dir}/{args.modality}_distance_matrix.npz  dist shape={dist.shape}")
    print(f"\nNOTE: query ground truth not yet confirmed. Working hypothesis "
          f"(query_N -> {CLA_CLASS_ORDER}) NOT applied here -- only the raw "
          f"distance matrix + database labels are saved. Retrieval metrics "
          f"(NN/first-tier/second-tier/mAP) still need the confirmed query labels.")


if __name__ == "__main__":
    main()
