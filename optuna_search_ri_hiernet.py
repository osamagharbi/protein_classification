#!/usr/bin/env python3
"""
Optuna hyper-parameter search for RI-HierNet on SHREC25.

Reuses the dataset / training building blocks from train_ri_hiernet.py and
searches the *critical* knobs only, with a TPE sampler + Hyperband pruner so the
grid stays small and bad trials die early.

Design choices that keep the search cheap
------------------------------------------
* Datasets are built ONCE and wrapped in an in-RAM cache, and the search runs
  single-process (num-workers 0) so every protein .pt is read from disk at most
  once across the *entire* study — subsequent epochs/trials are pure GPU compute.
* Each trial trains for --search-epochs (default 40, < the 120 of a full run) and
  reports val accuracy every epoch; the Hyperband pruner kills under-performers.
* Conditional sampling: patch_backbone / patch_knn are only drawn when they
  matter (patch_layers>0; knn only for the local backbones gin/gat/sage).

Searched parameters
-------------------
  patch_layers      {0,1,2,3}      0 = RIConv2-style (no patch graph)
  patch_backbone    {gin,gat,sage,transformer,graphormer}   (if patch_layers>0)
  patch_knn         {4,8,16}       (if backbone in gin/gat/sage)
  readout           {max, max+attn, mean+max, mean+max+attn}
  width_scale       {1,2}          512-D vs 1024-D SA encoder
  lr                [1e-4, 2e-3]   log
  weight_decay      [1e-5, 1e-3]   log
  dropout           [0.1, 0.4]
  aug_scale         {0.0, 0.2}
  aug_dropout       {0.0, 0.05}
  label_smoothing   {0.0, 0.1}

Everything else (radii, nsample, phase=chem, npoint) is held at the best-run
values to keep the grid focused. The study is stored in SQLite so it is
resumable and can be run from several processes against the same DB.

Example
-------
  conda run --no-capture-output -n graphmae python3 -u optuna_search_ri_hiernet.py \
      --data-dir /home/ogharbi/datasets/SHREC25 \
      --graph-subdir ssl_graphs_1k_geom_enriched_ri \
      --n-trials 60 --search-epochs 40 --batch-size 128 \
      --output-dir runs/optuna_ri --study-name ri_hiernet_v1
"""
from __future__ import annotations
import argparse, json, math, time, sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from train_ri_hiernet import (
    PointCloudDataset, load_labels_csv, grouped_train_val_split,
    make_collate, run_epoch,
)
from ri_hiernet import RIHierNet

RADII = (0.10, 0.20, 0.40, 0.80)
LOCAL_BACKBONES = ("gin", "gat", "sage")        # use patch_knn; the rest are full-attention


# ─────────────────────────────────────────────────────────────────────────────
# In-RAM caching wrapper (augmentation lives in collate, so caching __getitem__
# output is safe — every epoch still gets freshly-augmented batches).
# ─────────────────────────────────────────────────────────────────────────────
class CachedDataset:
    def __init__(self, base):
        self.base = base
        self._cache: dict[int, tuple] = {}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        v = self._cache.get(i)
        if v is None:
            v = self.base[i]
            self._cache[i] = v
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def build_datasets(args):
    train_labels = load_labels_csv(args.train_labels_csv)
    test_labels  = load_labels_csv(args.test_labels_csv)
    num_classes  = 99
    print(f"labels: train {len(train_labels)} / test {len(test_labels)}")

    if args.val_labels_csv:
        val_labels = load_labels_csv(args.val_labels_csv)
        val_ids = set(val_labels.keys())
        combined = {**train_labels, **val_labels}
        full_train = PointCloudDataset(args.graph_dir, "train", combined, npoint=args.npoint)
        tr_idx = [i for i, (gid, _, _) in enumerate(full_train.items) if gid not in val_ids]
        va_idx = [i for i, (gid, _, _) in enumerate(full_train.items) if gid in val_ids]
        print(f"  predefined val split: {len(tr_idx)} train / {len(va_idx)} val")
    else:
        full_train = PointCloudDataset(args.graph_dir, "train", train_labels, npoint=args.npoint)
        tr_idx, va_idx = grouped_train_val_split(full_train, val_split=args.val_split, seed=args.seed)

    cached = CachedDataset(full_train)
    tr_ds = Subset(cached, tr_idx)
    va_ds = Subset(cached, va_idx)
    return tr_ds, va_ds, num_classes


# ─────────────────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────────────────
def make_objective(tr_ds, va_ds, num_classes, device, args):
    eval_collate = make_collate(training=False)
    val_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=eval_collate,
                            pin_memory=True)

    def objective(trial: optuna.Trial) -> float:
        # ── architecture ──
        nsample     = trial.suggest_categorical("nsample", [16, 32, 64])
        patch_layers = trial.suggest_categorical("patch_layers", [0, 1, 2, 3])
        if patch_layers == 0:
            backbone, patch_knn = "gin", 8                    # unused when 0 layers
        else:
            backbone = trial.suggest_categorical(
                "patch_backbone", ["gin", "gat", "sage", "transformer", "graphormer"])
            patch_knn = (trial.suggest_categorical("patch_knn", [4, 8, 16])
                         if backbone in LOCAL_BACKBONES else 8)
        readout     = trial.suggest_categorical(
            "readout", ["max", "max+attn", "mean+max", "mean+max+attn"])
        width_scale = trial.suggest_categorical("width_scale", [1, 2])
        # ── optim / reg ──
        lr        = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
        wd        = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        dropout   = trial.suggest_float("dropout", 0.1, 0.4)
        aug_scale = trial.suggest_categorical("aug_scale", [0.0, 0.2])
        aug_drop  = trial.suggest_categorical("aug_dropout", [0.0, 0.05])
        ls        = trial.suggest_categorical("label_smoothing", [0.0, 0.1])

        train_collate = make_collate(0.0, aug_drop, aug_scale, 0.0, training=True)
        train_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=train_collate,
                                  drop_last=True, pin_memory=True)

        try:
            model = RIHierNet(num_classes=num_classes, chem_dim=2, patch_knn=patch_knn,
                              patch_layers=patch_layers, dropout=dropout, radii=RADII,
                              nsample=nsample, width_scale=width_scale,
                              patch_backbone=backbone, readout=readout).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            crit = nn.CrossEntropyLoss(label_smoothing=ls)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            warmup, total = args.warmup_epochs, args.search_epochs
            def lr_lambda(ep):
                if ep < warmup:
                    return (ep + 1) / max(1, warmup)
                prog = (ep - warmup) / max(1, total - warmup)
                return 0.5 * (1 + math.cos(math.pi * prog))
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

            t0, best = time.time(), -1.0
            for ep in range(args.search_epochs):
                run_epoch(model, train_loader, crit, device, opt, use_chem=True, desc="")
                va = run_epoch(model, val_loader, crit, device, None, use_chem=True, desc="")
                sched.step()
                sel = va[args.select_metric]
                best = max(best, sel)
                trial.report(sel, ep)
                if trial.should_prune():
                    print(f"  trial {trial.number} pruned @ep{ep} "
                          f"({args.select_metric}={sel:.4f}, best={best:.4f})")
                    raise optuna.TrialPruned()
            trial.set_user_attr("params", int(n_params))
            print(f"  trial {trial.number} done: best {args.select_metric}={best:.4f} "
                  f"({time.time()-t0:.0f}s, {n_params/1e6:.1f}M, "
                  f"L{patch_layers}/{backbone}/{readout}/w{width_scale})")
            return best
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  trial {trial.number} OOM -> pruned")
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()
            raise
        finally:
            torch.cuda.empty_cache()

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    # data (same derivation as train_ri_hiernet.py)
    ap.add_argument("--data-dir", default="/home/ogharbi/datasets/SHREC25")
    ap.add_argument("--graph-subdir", default="ssl_graphs_1k_geom_enriched")
    ap.add_argument("--graph-dir", default=None)
    ap.add_argument("--train-labels-csv", default=None)
    ap.add_argument("--test-labels-csv", default=None)
    ap.add_argument("--val-labels-csv", default=None)
    ap.add_argument("--npoint", type=int, default=1000)
    ap.add_argument("--nsample", type=int, default=32)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    # search
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--search-epochs", type=int, default=40,
                    help="epochs per trial (< full-run epochs; pruner trims further)")
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="keep 0 so the in-RAM cache persists across epochs/trials")
    ap.add_argument("--select-metric", default="accuracy",
                    choices=["accuracy", "f1_weighted", "f1_macro", "balanced_accuracy"])
    ap.add_argument("--timeout", type=int, default=None, help="optional wall-clock budget (s)")
    ap.add_argument("--device", type=int, default=0)
    # bookkeeping
    ap.add_argument("--output-dir", default="/home/ogharbi/projects/protein_classification/runs")
    ap.add_argument("--study-name", default=None)
    args = ap.parse_args()

    root = Path(args.data_dir)
    if args.graph_dir is None:        args.graph_dir = str(root / args.graph_subdir)
    if args.train_labels_csv is None: args.train_labels_csv = str(root / "pc" / "train_labels.csv")
    if args.test_labels_csv is None:  args.test_labels_csv = str(root / "pc" / "test_labels.csv")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    study_name = args.study_name or f"ri_hiernet_optuna_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = Path(args.output_dir) / study_name
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{out_dir / 'study.db'}"
    print(f"study={study_name}  device={device}  storage={storage}")
    print(f"graph-dir={args.graph_dir}")
    print(f"search: {args.n_trials} trials x {args.search_epochs} ep, "
          f"select by val '{args.select_metric}', batch={args.batch_size}")

    tr_ds, va_ds, num_classes = build_datasets(args)

    sampler = TPESampler(seed=args.seed, multivariate=True, group=True)
    pruner = HyperbandPruner(min_resource=8, max_resource=args.search_epochs, reduction_factor=3)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner,
                                study_name=study_name, storage=storage, load_if_exists=True)

    objective = make_objective(tr_ds, va_ds, num_classes, device, args)
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   gc_after_trial=True, show_progress_bar=False)

    # ── report ──
    best = study.best_trial
    print("\n" + "=" * 70)
    print(f"BEST trial #{best.number}: val {args.select_metric} = {best.value:.4f}")
    print("params:")
    for k, v in best.params.items():
        print(f"  {k:16s} {v}")
    if "params" in best.user_attrs:
        print(f"  model size: {best.user_attrs['params']/1e6:.1f}M")

    # full param dict including conditionals not drawn (so the command is complete)
    p = dict(best.params)
    pl = p.get("patch_layers", 2)
    backbone = p.get("patch_backbone", "gin")
    knn = p.get("patch_knn", 8)
    full = {
        "nsample": p.get("nsample", 32),
        "patch_layers": pl,
        "patch_backbone": backbone,
        "patch_knn": knn,
        "readout": p.get("readout", "max+attn"),
        "width_scale": p.get("width_scale", 1),
        "lr": p.get("lr", 5e-4),
        "weight_decay": p.get("weight_decay", 1e-4),
        "dropout": p.get("dropout", 0.3),
        "aug_scale": p.get("aug_scale", 0.0),
        "aug_dropout": p.get("aug_dropout", 0.0),
        "label_smoothing": p.get("label_smoothing", 0.0),
        "val_best": best.value,
        "select_metric": args.select_metric,
        "trial_number": best.number,
    }
    (out_dir / "best_config.json").write_text(json.dumps(full, indent=2, default=float))

    # ready-to-run full-training command
    cmd = (
        "python3 -u train_ri_hiernet.py \\\n"
        f"  --graph-dir {args.graph_dir} \\\n"
        f"  --train-labels-csv {args.train_labels_csv} \\\n"
        f"  --test-labels-csv {args.test_labels_csv} \\\n"
        + (f"  --val-labels-csv {args.val_labels_csv} \\\n" if args.val_labels_csv else "")
        + f"  --output-dir {args.output_dir} \\\n"
        "  --phase chem --epochs 120 --patience 30 \\\n"
        f"  --batch-size {args.batch_size} --num-workers 8 \\\n"
        f"  --lr {full['lr']:.2e} --weight-decay {full['weight_decay']:.2e} --warmup-epochs 5 \\\n"
        f"  --label-smoothing {full['label_smoothing']} "
        f"--aug-scale {full['aug_scale']} --aug-dropout {full['aug_dropout']} \\\n"
        f"  --dropout {full['dropout']:.3f} --width-scale {full['width_scale']} "
        f"--nsample {full['nsample']} \\\n"
        f"  --patch-layers {pl} --patch-knn {knn} "
        f"--patch-backbone {backbone} --readout {full['readout']} \\\n"
        f"  --run-name ri_hiernet_optuna_best"
    )
    (out_dir / "best_command.sh").write_text("#!/bin/bash\n" + cmd + "\n")
    print("\nfull-training command for the best config:")
    print(cmd)
    print(f"\nsaved -> {out_dir}/best_config.json  +  best_command.sh  +  study.db")

    # quick study summary
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\ntrials: {len(done)} complete, {len(pruned)} pruned, {len(study.trials)} total")


if __name__ == "__main__":
    main()
