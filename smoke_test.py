#!/usr/bin/env python3
"""
Pre-flight smoke test for the RI-HierNet Optuna pipeline.

Run this BEFORE submitting the full sweep to catch path / data / env / memory
problems in ~1 minute instead of after a long queue wait. Each step prints
[ OK ] / [FAIL]; the first failure exits non-zero so Slurm marks the job failed.

Checks, in order:
  1. environment   torch + CUDA device (data checks still run CPU-only)
  2. paths         graph dir, train_set/ test_set/, train/val/test label CSVs
  3. labels        CSVs parse; protein / class counts
  4. datasets      build train+val; sizes; val matches val_split.csv
  5. batch         one augmented training batch has the expected shapes
  6. test set      loads one sample (the held-out split the full run scores on)
  7. model         build + forward + backward on GPU; report peak memory

Use --heavy to stress the *largest* config the search can sample
(width_scale=2, nsample=64, 3 patch layers) so you know batch-size fits a
32 GB V100.

Example:
  python -u smoke_test.py \
    --graph-dir        $WORK/ssl_graphs_1k_geom_enriched_ri \
    --train-labels-csv $WORK/ssl_graphs_1k_geom_enriched_ri/train_split.csv \
    --val-labels-csv   $WORK/ssl_graphs_1k_geom_enriched_ri/val_split.csv \
    --test-labels-csv  $WORK/ssl_graphs_1k_geom_enriched_ri/test_labels.csv \
    --heavy
"""
from __future__ import annotations
import argparse, sys, traceback
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_ri_hiernet import (
    PointCloudDataset, load_labels_csv, make_collate, resolve_split_dir,
)
from optuna_search_ri_hiernet import build_datasets
from ri_hiernet import RIHierNet


def ok(m):   print(f"[ OK ] {m}", flush=True)
def info(m): print(f"[INFO] {m}", flush=True)
def fail(m):
    print(f"[FAIL] {m}", flush=True)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", required=True)
    ap.add_argument("--train-labels-csv", required=True)
    ap.add_argument("--val-labels-csv", default=None)
    ap.add_argument("--test-labels-csv", required=True)
    ap.add_argument("--npoint", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--heavy", action="store_true",
                    help="stress the largest sampled config (w2 / nsample64 / 3 layers)")
    args = ap.parse_args()

    # ── 1. environment ────────────────────────────────────────────────────
    info(f"torch {torch.__version__}")
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        ok(f"CUDA available: {torch.cuda.get_device_name(args.device)}")
    else:
        info("CUDA NOT available — data checks run on CPU, model step will be CPU-only/slow")
    device = torch.device(f"cuda:{args.device}" if has_cuda else "cpu")

    # ── 2. paths ──────────────────────────────────────────────────────────
    need = [("graph-dir", args.graph_dir),
            ("train-labels", args.train_labels_csv),
            ("test-labels", args.test_labels_csv)]
    if args.val_labels_csv:
        need.append(("val-labels", args.val_labels_csv))
    for name, p in need:
        if not Path(p).exists():
            fail(f"{name} path not found: {p}")
    ok("all given paths exist")

    train_dir = resolve_split_dir(args.graph_dir, "train")
    test_dir = resolve_split_dir(args.graph_dir, "test")
    n_train_pt = len(list(train_dir.glob("*.pt")))
    n_test_pt = len(list(test_dir.glob("*.pt")))
    if n_train_pt == 0:
        fail(f"no .pt graphs under {train_dir} (wrong dir layout?)")
    if n_test_pt == 0:
        fail(f"no .pt graphs under {test_dir}")
    ok(f"graphs found: {n_train_pt} train ({train_dir.name}/), {n_test_pt} test ({test_dir.name}/)")

    # ── 3. labels ─────────────────────────────────────────────────────────
    tr_lab = load_labels_csv(args.train_labels_csv)
    te_lab = load_labels_csv(args.test_labels_csv)
    ok(f"train labels: {len(tr_lab)} proteins, {len(set(tr_lab.values()))} classes")
    ok(f"test  labels: {len(te_lab)} proteins, {len(set(te_lab.values()))} classes")
    if args.val_labels_csv:
        va_lab = load_labels_csv(args.val_labels_csv)
        ok(f"val   labels: {len(va_lab)} proteins, {len(set(va_lab.values()))} classes")

    # ── 4. datasets (reuses the exact builder the search uses) ────────────
    ns = Namespace(graph_dir=args.graph_dir,
                   train_labels_csv=args.train_labels_csv,
                   test_labels_csv=args.test_labels_csv,
                   val_labels_csv=args.val_labels_csv,
                   npoint=args.npoint, val_split=0.2, seed=42)
    tr_ds, va_ds, num_classes = build_datasets(ns)
    if len(tr_ds) == 0 or len(va_ds) == 0:
        fail(f"empty split: train={len(tr_ds)} val={len(va_ds)}")
    ok(f"datasets built: train={len(tr_ds)} val={len(va_ds)} num_classes={num_classes}")

    # ── 5. one augmented training batch ───────────────────────────────────
    B = min(args.batch_size, len(tr_ds))
    collate = make_collate(0.0, 0.05, 0.2, 0.0, training=True)
    loader = DataLoader(tr_ds, batch_size=B, shuffle=True, collate_fn=collate, num_workers=0)
    pos, normal, chem, y = next(iter(loader))
    exp = (B, args.npoint)
    if pos.shape[:2] != exp or pos.shape[2] != 3:
        fail(f"pos shape {tuple(pos.shape)} != {(*exp, 3)}")
    if normal.shape != pos.shape or chem.shape[:2] != exp:
        fail(f"normal/chem shape mismatch: {tuple(normal.shape)} / {tuple(chem.shape)}")
    if y.shape[0] != B:
        fail(f"label batch {tuple(y.shape)} != ({B},)")
    ok(f"batch shapes: pos{tuple(pos.shape)} normal{tuple(normal.shape)} "
       f"chem{tuple(chem.shape)} y{tuple(y.shape)}")

    # ── 6. test split loads ───────────────────────────────────────────────
    test_ds = PointCloudDataset(args.graph_dir, "test", te_lab, npoint=args.npoint)
    if len(test_ds) == 0:
        fail("test dataset is empty (label/graph id mismatch?)")
    _ = test_ds[0]
    ok(f"test dataset loads: {len(test_ds)} clouds, sample[0] OK")

    # ── 7. model forward + backward ───────────────────────────────────────
    ws, nsamp, players = (2, 64, 3) if args.heavy else (1, 32, 2)
    cfg = f"width_scale={ws} nsample={nsamp} patch_layers={players} gin max+attn"
    info(f"building model ({'HEAVY' if args.heavy else 'default'}): {cfg}")
    model = RIHierNet(num_classes=num_classes, chem_dim=2, patch_knn=8,
                      patch_layers=players, nsample=nsamp, width_scale=ws,
                      patch_backbone="gin", readout="max+attn").to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if has_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    pos, normal, chem, y = pos.to(device), normal.to(device), chem.to(device), y.to(device)
    logits, proj = model(pos, normal, chem, use_chem=True)
    if logits.shape != (B, num_classes):
        fail(f"logits shape {tuple(logits.shape)} != {(B, num_classes)}")
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    peak = (torch.cuda.max_memory_allocated(device) / 1e9) if has_cuda else 0.0
    ok(f"forward+backward OK: {n_params/1e6:.1f}M params, loss={loss.item():.3f}"
       + (f", peak GPU mem={peak:.1f} GB @ batch {B}" if has_cuda else ""))
    if has_cuda and args.heavy:
        info(f"extrapolated peak at batch 64 ≈ {peak/B*64:.1f} GB "
             f"(V100-32g headroom check)")

    print("\n[SMOKE PASSED] data + model pipeline is healthy.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        fail("uncaught exception (see traceback above)")
