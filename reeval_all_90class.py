#!/usr/bin/env python3
"""
Re-evaluate every table-row checkpoint on the REAL 90-class SHREC25 test set
(pc/test_labels.csv). Config-driven: each run's own final_results.json supplies
feature_mode, dataset paths, backbone, and special-architecture flags. Always
loads best_model.pt (best-val checkpoint), which is the principled selection and
also recovers runs whose final epoch diverged (e.g. the z-scored 14D run).
"""
import sys, json, warnings; sys.path.insert(0, "."); warnings.filterwarnings("ignore")
import numpy as np, torch, pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, accuracy_score

from classify_gnn import canonical_graph_id
import classify_hierarchical_gnn as H

DS = Path("/home/ogharbi/datasets/SHREC25")
PC_TEST = DS / "pc/test_labels.csv"

# table_row -> run_dir
ROWS = {
    "6D (xyz+normals)":      "hier_gin_h512_coords_normals",
    "7D (no Pot)":           "hier_gin_h512_7d_no_pot",
    "4D (xyz+Pot)":          "hier_bb_gin_cosine",
    "7D (no NormPot)":       "hier_gin_h512_7d_no_normpot",
    "8D (all)":              "hier_gin_h512_all8d",
    "14D (z-scored)":        "hier_gin_h512_geom14d",
    "14D (global norm) BEST":"hier_gin_h512_geom14d_raw",
    "8D + RI-GIN":           "hier_ri_gin_h512_8d",
    "14D + RI-GIN":          "hier_ri_gin_h512_14d",
    "8D + LRF-GIN":          "hier_lrf_gin_h512_8d",
    "14D + 3-level":         "hier_threelevel_h512_14d",
    "+ inject 2-feat":       "hier_gin_h512_geom14d_patch_scalar_inject_v2",
    "+ inject 8-feat":       "hier_gin_h512_geom14d_patch_inject_all_v2",
}

def pc_label_dict():
    df = pd.read_csv(PC_TEST)
    d = {}
    for _, r in df.iterrows():
        pid = str(r["protein_id"])
        pid = pid[:-3] if pid.endswith("_pc") else pid
        d[canonical_graph_id(pid)] = int(r["class_id"])
    return d

def resolve(p):
    """Resolve a possibly-stale dataset path to the local datasets dir."""
    if p is None: return None
    name = str(p).rstrip("/").split("/")[-1]
    cand = DS / name
    return str(cand if cand.exists() else p)

def eval_run(tag, run, labeld, dev):
    fr = Path("runs")/run/"final_results.json"
    ck = Path("runs")/run/"best_model.pt"
    if not fr.exists() or not ck.exists():
        return tag, None, "missing ckpt/json"
    cfg = json.loads(fr.read_text())["config"]
    bb = cfg["node_backbone"]
    three = bool(cfg.get("assign_dir2"))
    ds_kw = dict(feature_mode=cfg["feature_mode"], rwpe_dim=cfg.get("rwpe_dim", 0) or 0,
                 use_lrf_feat=(bb == "lrf_gin"), assign_dir2=resolve(cfg.get("assign_dir2")))
    try:
        ds = H.HierPatchDataset(resolve(cfg["graph_dir"]), resolve(cfg["assign_dir"]),
                                "test", labeld, **ds_kw)
        col = H.collate_three if three else H.collate
        ld = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=col, num_workers=4)
        model = H.HierarchicalGNN(
            cfg["in_dim"], cfg["hidden_dim"], cfg["num_classes"],
            node_layers=cfg["node_layers"], patch_layers=cfg["patch_layers"],
            pool=cfg.get("pool", "attn"), readout=cfg.get("readout", "max"),
            dropout=cfg.get("dropout", 0.3), node_backbone=bb,
            patch_backbone=cfg.get("patch_backbone", "gin"),
            scalar_inject=cfg.get("scalar_inject", False),
            patch_scalar_inject=cfg.get("patch_scalar_inject", False),
            patch_inject_all=cfg.get("patch_inject_all", False),
            three_level=three, mid_layers=cfg.get("mid_layers", 2),
            pna_delta=cfg.get("pna_delta") or 6.0,
        ).to(dev)
        model.load_state_dict(torch.load(ck, map_location=dev, weights_only=False))
        model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for batch in ld:
                if len(batch) == 6:
                    bn, bm, bp, mg, gp, y = [b.to(dev) if hasattr(b, "to") else b for b in batch]
                    logits = model(bn, bp, gp, bm, mg)
                else:
                    bn, bp, gp, y = batch
                    bn, bp, gp = bn.to(dev), bp.to(dev), gp.to(dev)
                    logits = model(bn, bp, gp)
                preds.extend(logits.argmax(1).cpu().numpy()); labs.extend(y.cpu().numpy() if hasattr(y,'cpu') else np.array(y))
        preds, labs = np.array(preds), np.array(labs)
        valid = sorted(set(labs.tolist()))
        mf1 = f1_score(labs, preds, average="macro", zero_division=0, labels=valid)
        bac = recall_score(labs, preds, average="macro", zero_division=0, labels=valid)
        acc = accuracy_score(labs, preds)
        wf1 = f1_score(labs, preds, average="weighted", zero_division=0, labels=valid)
        return tag, (acc, bac, mf1, wf1, len(valid), len(ds)), None
    except Exception as e:
        import traceback; traceback.print_exc()
        return tag, None, str(e)

if __name__ == "__main__":
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    labeld = pc_label_dict()
    print(f"{'row':26s} {'acc':6s} {'bacc':6s} {'mF1':6s} {'wF1':6s}  nclass nprot")
    print("-"*72)
    results = {}
    for tag, run in ROWS.items():
        t, vals, err = eval_run(tag, run, labeld, dev)
        if vals is None:
            print(f"{tag:26s} ERROR: {err}")
        else:
            acc, bac, mf1, wf1, nc, npr = vals
            results[tag] = dict(acc=acc, bacc=bac, mf1=mf1, wf1=wf1)
            print(f"{tag:26s} {acc:.4f} {bac:.4f} {mf1:.4f} {wf1:.4f}  {nc:5d}  {npr}")
    Path("reeval_90class_results.json").write_text(json.dumps(results, indent=2))
    print("\nsaved -> reeval_90class_results.json")
