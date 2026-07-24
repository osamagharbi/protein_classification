import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

import classify_hierarchical_gnn as chg


def compute_for_run(run_dir: Path) -> dict:
    with open(run_dir / "final_results.json", "r", encoding="utf-8") as f:
        fr = json.load(f)
    cfg = fr["config"]

    test_labels, num_classes = chg.load_labels(
        cfg["graph_dir"], "test", cfg.get("test_labels_csv")
    )

    ds = chg.HierPatchDataset(
        cfg["graph_dir"],
        cfg["assign_dir"],
        "test",
        test_labels,
        ssl_emb_dir=cfg.get("ssl_emb_dir"),
        feature_mode=cfg.get("feature_mode", "all"),
        rwpe_dim=cfg.get("rwpe_dim", 0),
        use_lrf_feat=(cfg.get("node_backbone") == "lrf_gin"),
        assign_dir2=cfg.get("assign_dir2"),
    )
    collate_fn = chg.collate_three if cfg.get("assign_dir2") else chg.collate
    loader = DataLoader(
        ds,
        batch_size=cfg.get("batch_size", 64),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    sample = ds[0][0]
    in_dim = sample.ndata["attr"].shape[1]
    xyz_slice = slice(5, 8)
    if cfg.get("feature_mode") in ("coords", "coords_normals", "coords_potential"):
        xyz_slice = slice(0, 3)

    model = chg.HierarchicalGNN(
        in_dim=in_dim,
        hidden_dim=cfg["hidden_dim"],
        num_classes=cfg.get("num_classes", num_classes),
        node_layers=cfg["node_layers"],
        patch_layers=cfg["patch_layers"],
        pool=cfg["pool"],
        readout=cfg["readout"],
        dropout=cfg.get("dropout", 0.3),
        node_readout=cfg.get("node_readout", False),
        node_backbone=cfg.get("node_backbone", "gin"),
        patch_backbone=cfg.get("patch_backbone", "gin"),
        pna_delta=cfg.get("pna_delta", 6.0),
        scalar_inject=cfg.get("scalar_inject", False),
        patch_scalar_inject=cfg.get("patch_scalar_inject", False),
        patch_inject_all=cfg.get("patch_inject_all", False),
        three_level=bool(cfg.get("assign_dir2")),
        mid_layers=cfg.get("mid_layers", 2),
        xyz_slice=xyz_slice,
        proj_dim=cfg.get("proj_dim", 128),
    )

    state = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()

    y_true, y_score = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 6:
                bn, bm, bp, mid_gpid, gpid, y = batch
                logits = model(bn, bp, gpid, bm, mid_gpid)
            else:
                bn, bp, gpid, y = batch
                logits = model(bn, bp, gpid)
            probs = torch.softmax(logits, dim=1).numpy()
            y_score.append(probs)
            y_true.append(y.numpy())

    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    y_bin = label_binarize(y_true, classes=np.arange(num_classes))

    ap_per_class = average_precision_score(y_bin, y_score, average=None)
    map_macro = float(np.nanmean(ap_per_class))
    map_micro = float(average_precision_score(y_bin, y_score, average="micro"))

    return {
        "num_samples": int(y_true.shape[0]),
        "num_classes": int(num_classes),
        "mAP_macro": map_macro,
        "mAP_micro": map_micro,
        "accuracy": fr["final_test_metrics"].get("accuracy"),
        "balanced_accuracy": fr["final_test_metrics"].get("balanced_accuracy"),
        "f1_macro": fr["final_test_metrics"].get("f1_macro"),
        "loss": fr["final_test_metrics"].get("loss"),
    }


def main() -> None:
    base = Path("/home/ogharbi/projects/protein_classification/runs_shrec19")
    runs = {
        "k16": base / "shrec19_protein_hier_gin_coords_ls_k16",
        "k32": base / "shrec19_protein_hier_gin_coords_ls_k32",
        "k64": base / "shrec19_protein_hier_gin_coords_ls",
        "k128": base / "shrec19_protein_hier_gin_coords_ls_k128",
    }

    out = {}
    for name, run_dir in runs.items():
        print(f"Computing mAP for {name} from {run_dir}...")
        out[name] = compute_for_run(run_dir)

    out_path = Path("/tmp/shrec19_protein_map_metrics.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
