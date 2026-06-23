#!/usr/bin/env python3
"""Evaluate a GeoChem-MSRINet checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from geochem_msrinet.data import BatchAugment, ProteinPointDataset, make_collate
from geochem_msrinet.metrics import classification_metrics
from geochem_msrinet.model import GeoChemMSRINet
from geochem_msrinet.ri_hybrid_model import RIConv2GeoChemHybrid


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--graph-dir", default=None)
    p.add_argument("--labels-csv", default=None)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--tta-views", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="Evaluate only the first N samples.")
    p.add_argument("--no-chem", action="store_true", help="Force chemistry channels to zero at evaluation time.")
    p.add_argument("--output-json", default=None)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def build_model(config: dict) -> GeoChemMSRINet:
    if config.get("model_kind") == "ri_hybrid":
        return RIConv2GeoChemHybrid(
            num_classes=99,
            chem_dim=2,
            ri_width_mult=int(config.get("ri_width_mult", 1)),
            ri_token_dim=int(config.get("ri_token_dim", 512)),
            patch_knn=int(config.get("patch_knn", 8)),
            patch_gin_layers=int(config.get("patch_gin_layers", 1)),
            patch_transformer_layers=int(config.get("patch_transformer_layers", 1)),
            dropout=float(config.get("dropout", 0.30)),
            logit_blend=float(config.get("logit_blend", 1.0)),
        )
    return GeoChemMSRINet(
        num_classes=99,
        points=int(config.get("points", 1000)),
        chem_dim=2,
        geom_dim=6 if config.get("use_geom_raw", False) else 0,
        patch_knn=int(config.get("patch_knn", 8)),
        patch_gin_layers=int(config.get("patch_gin_layers", 2)),
        patch_transformer_layers=int(config.get("patch_transformer_layers", 2)),
        dropout=float(config.get("dropout", 0.30)),
        logit_blend=float(config.get("logit_blend", 0.65)),
    )


@torch.no_grad()
def evaluate(model, loader, device, tta_views: int = 1, use_chem: bool = True):
    model.eval()
    preds = []
    labels_all = []
    ids_all = []
    for batch in loader:
        pos = batch["pos"].to(device)
        normal = batch["normal"].to(device)
        chem = batch["chem"].to(device)
        geom = batch["geom"].to(device)
        logits_sum = None
        for _ in range(max(1, tta_views)):
            out = model(pos, normal, chem, geom=geom if geom.shape[-1] else None, use_chem=use_chem)
            logits_sum = out["logits"] if logits_sum is None else logits_sum + out["logits"]
        pred = (logits_sum / max(1, tta_views)).argmax(dim=1)
        preds.extend(pred.cpu().tolist())
        labels_all.extend(batch["label"].cpu().tolist())
        ids_all.extend(batch["id"])
    return classification_metrics(labels_all, preds), ids_all, labels_all, preds


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    graph_dir = args.graph_dir or config.get("graph_dir")
    if graph_dir is None:
        raise ValueError("--graph-dir is required when checkpoint config lacks graph_dir")

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    ds = ProteinPointDataset(
        graph_dir,
        args.split,
        args.labels_csv,
        int(config.get("points", 1000)),
        bool(config.get("use_geom_raw", False)),
    )
    if args.limit and args.limit > 0:
        ds = Subset(ds, list(range(min(args.limit, len(ds)))))
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate(BatchAugment(rotate=False)),
        pin_memory=device.type == "cuda",
    )
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    use_chem = bool(ckpt.get("use_chem", config.get("phase", "chem") != "geom")) and not args.no_chem
    metrics, ids, labels, preds = evaluate(model, loader, device, args.tta_views, use_chem=use_chem)
    metrics["use_chem"] = use_chem
    payload = {"metrics": metrics, "predictions": [{"id": i, "label": y, "pred": p} for i, y, p in zip(ids, labels, preds)]}
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
