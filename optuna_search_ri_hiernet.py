#!/usr/bin/env python3
"""
Optuna hyper-parameter exploration for RI-HierNet on SHREC25.

Two ways to run, selected by --mode:

  search  (single/few GPUs, adaptive)
      Classic Optuna study: TPE sampler + Hyperband pruner, optionally shared
      across a small Slurm array through a journal file. Each worker runs
      several trials sequentially so the sampler actually learns.

  gen / train / agg  (massively parallel, seed-averaged — one training per GPU)
      For a big array (e.g. 100 configs x 5 seeds = 500 V100s, all at once).
      Because every task starts simultaneously, adaptive TPE can't learn, so we
      split the work:
        gen   : sample N configs ONCE (Optuna sampler) -> configs.json
        train : array task i trains config (i // n_seeds) with seed (i % n_seeds),
                writes one result JSON. No shared Optuna storage => no lock
                contention with 500 concurrent writers.
        agg   : read all result JSONs, group by config, report mean +/- std,
                emit a ready-to-run full-training command for the winner.

Searched parameters
-------------------
  nsample           {16,32,64}     ball-query neighbours per SA layer
  patch_layers      {0,1,2,3}      0 = RIConv2-style (no patch graph)
  patch_backbone    {gin,gat,sage,transformer,graphormer}   (if patch_layers>0)
  patch_knn         {4,8,16}       (if backbone in gin/gat/sage)
  readout           {max, max+attn, mean+max, mean+max+attn, attn}
  width_scale       {1,2}          512-D vs 1024-D SA encoder
  lr                [1e-4, 2e-3]   log
  weight_decay      [1e-5, 1e-3]   log
  dropout           [0.1, 0.4]
  aug_scale         {0.0, 0.2}
  aug_dropout       {0.0, 0.05}
  label_smoothing   {0.0, 0.1}
"""
from __future__ import annotations
import argparse, json, math, time, sys, os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

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
    return Subset(cached, tr_idx), Subset(cached, va_idx), num_classes


def build_test_dataset(args):
    test_labels = load_labels_csv(args.test_labels_csv)
    return PointCloudDataset(args.graph_dir, "test", test_labels, npoint=args.npoint)


# ─────────────────────────────────────────────────────────────────────────────
# Search space + model + training (shared by every mode)
# ─────────────────────────────────────────────────────────────────────────────
def suggest_params(trial: optuna.Trial) -> dict:
    """Draw one full config. Conditional knobs (backbone/knn) are resolved to
    fixed defaults when inapplicable so the returned dict is always complete."""
    nsample = trial.suggest_categorical("nsample", [16, 32, 64])
    patch_layers = trial.suggest_categorical("patch_layers", [0, 1, 2, 3])
    if patch_layers == 0:
        backbone, patch_knn = "gin", 8                       # unused when 0 layers
    else:
        backbone = trial.suggest_categorical(
            "patch_backbone", ["gin", "gat", "sage", "transformer", "graphormer"])
        patch_knn = (trial.suggest_categorical("patch_knn", [4, 8, 16])
                     if backbone in LOCAL_BACKBONES else 8)
    return {
        "nsample": nsample,
        "patch_layers": patch_layers,
        "patch_backbone": backbone,
        "patch_knn": patch_knn,
        "readout": trial.suggest_categorical(
            "readout", ["max", "max+attn", "mean+max", "mean+max+attn", "attn"]),
        "width_scale": trial.suggest_categorical("width_scale", [1, 2]),
        "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "aug_scale": trial.suggest_categorical("aug_scale", [0.0, 0.2]),
        "aug_dropout": trial.suggest_categorical("aug_dropout", [0.0, 0.05]),
        "label_smoothing": trial.suggest_categorical("label_smoothing", [0.0, 0.1]),
    }


def build_model(params: dict, num_classes: int) -> RIHierNet:
    return RIHierNet(num_classes=num_classes, chem_dim=2,
                     patch_knn=params["patch_knn"], patch_layers=params["patch_layers"],
                     dropout=params["dropout"], radii=RADII, nsample=params["nsample"],
                     width_scale=params["width_scale"], patch_backbone=params["patch_backbone"],
                     readout=params["readout"])


def train_model(params, train_seed, tr_ds, va_ds, num_classes, device, args,
                test_ds=None, on_epoch=None, tag=""):
    """Train one config for args.search_epochs and return its metrics dict.
    on_epoch(ep, val_sel) is called each epoch (search mode uses it to prune).
    If test_ds is given, evaluate the best-val checkpoint on test once."""
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    train_collate = make_collate(0.0, params["aug_dropout"], params["aug_scale"], 0.0, training=True)
    eval_collate = make_collate(training=False)
    train_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=train_collate,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=eval_collate, pin_memory=True)

    model = build_model(params, num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    crit = nn.CrossEntropyLoss(label_smoothing=params["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    warmup, total = args.warmup_epochs, args.search_epochs
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / max(1, warmup)
        prog = (ep - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    t0, best, best_ep, best_state = time.time(), -1.0, -1, None
    for ep in range(args.search_epochs):
        run_epoch(model, train_loader, crit, device, opt, use_chem=True, desc="")
        va = run_epoch(model, val_loader, crit, device, None, use_chem=True, desc="")
        sched.step()
        sel = va[args.select_metric]
        if sel > best:
            best, best_ep = sel, ep
            if test_ds is not None:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if on_epoch is not None:
            on_epoch(ep, sel)                                # may raise optuna.TrialPruned

    out = {"val_best": best, "val_best_epoch": best_ep,
           "n_params": int(n_params), "seconds": round(time.time() - t0, 1)}
    if test_ds is not None:
        if best_state is not None:
            model.load_state_dict(best_state)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=eval_collate, pin_memory=True)
        te = run_epoch(model, test_loader, crit, device, None, use_chem=True, desc="")
        out.update({"test_acc": te["accuracy"], "test_wf1": te["f1_weighted"],
                    "test_mf1": te["f1_macro"], "test_balacc": te["balanced_accuracy"]})
    return out


def full_train_command(args, params, run_name="ri_hiernet_best"):
    """A copy-paste train_ri_hiernet.py command for one config (120 epochs)."""
    return (
        "python3 -u train_ri_hiernet.py \\\n"
        f"  --graph-dir {args.graph_dir} \\\n"
        f"  --train-labels-csv {args.train_labels_csv} \\\n"
        f"  --test-labels-csv {args.test_labels_csv} \\\n"
        + (f"  --val-labels-csv {args.val_labels_csv} \\\n" if args.val_labels_csv else "")
        + "  --phase chem --epochs 120 --patience 30 \\\n"
        f"  --batch-size {args.batch_size} --num-workers 8 \\\n"
        f"  --lr {params['lr']:.3e} --weight-decay {params['weight_decay']:.3e} --warmup-epochs 5 \\\n"
        f"  --label-smoothing {params['label_smoothing']} "
        f"--aug-scale {params['aug_scale']} --aug-dropout {params['aug_dropout']} \\\n"
        f"  --dropout {params['dropout']:.3f} --width-scale {params['width_scale']} "
        f"--nsample {params['nsample']} \\\n"
        f"  --patch-layers {params['patch_layers']} --patch-knn {params['patch_knn']} "
        f"--patch-backbone {params['patch_backbone']} --readout {params['readout']} \\\n"
        f"  --run-name {run_name}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mode: gen  (sample N configs once → configs.json)
# ─────────────────────────────────────────────────────────────────────────────
def cmd_gen(args):
    sampler = TPESampler(seed=args.seed, multivariate=True, group=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    configs = []
    for _ in range(args.n_configs):
        trial = study.ask()
        configs.append(suggest_params(trial))
        study.tell(trial, 0.0)                               # constant value → space-filling draw
    out = Path(args.configs_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(configs, indent=2, default=float))
    # quick distribution summary
    bb = defaultdict(int); pl = defaultdict(int)
    for c in configs:
        pl[c["patch_layers"]] += 1
        bb[c["patch_backbone"] if c["patch_layers"] > 0 else "none(0-layer)"] += 1
    print(f"generated {len(configs)} configs -> {out}")
    print(f"  patch_layers: {dict(sorted(pl.items()))}")
    print(f"  backbone:     {dict(sorted(bb.items()))}")
    print(f"  total array tasks for {args.n_seeds} seeds = {len(configs) * args.n_seeds}")


# ─────────────────────────────────────────────────────────────────────────────
# Mode: train  (one config + one seed = one GPU task → result JSON)
# ─────────────────────────────────────────────────────────────────────────────
def cmd_train(args, device):
    configs = json.loads(Path(args.configs_file).read_text())
    if not (0 <= args.config_id < len(configs)):
        sys.exit(f"config-id {args.config_id} out of range (0..{len(configs)-1})")
    params = configs[args.config_id]
    print(f"config {args.config_id}/{len(configs)} seed {args.seed_id}: {params}")

    tr_ds, va_ds, num_classes = build_datasets(args)
    test_ds = build_test_dataset(args) if args.eval_test else None

    result = {"config_id": args.config_id, "seed": args.seed_id, "params": params}
    try:
        res = train_model(params, args.seed_id, tr_ds, va_ds, num_classes, device, args,
                          test_ds=test_ds)
        result.update(res); result["status"] = "ok"
    except RuntimeError as e:
        torch.cuda.empty_cache()
        result["status"] = "oom" if "out of memory" in str(e).lower() else "error"
        result["error"] = str(e)[:300]

    rdir = Path(args.result_dir); rdir.mkdir(parents=True, exist_ok=True)
    rfile = rdir / f"cfg{args.config_id:03d}_seed{args.seed_id}.json"
    rfile.write_text(json.dumps(result, indent=2, default=float))
    print(f"wrote {rfile}  status={result['status']} "
          f"val_best={result.get('val_best')} test_acc={result.get('test_acc')}")


# ─────────────────────────────────────────────────────────────────────────────
# Mode: agg  (group result JSONs by config → mean ± std, rank, best command)
# ─────────────────────────────────────────────────────────────────────────────
def cmd_agg(args):
    rdir = Path(args.result_dir)
    files = sorted(rdir.glob("cfg*_seed*.json"))
    if not files:
        sys.exit(f"no result files under {rdir}")
    by_cfg = defaultdict(list)
    statuses = defaultdict(int)
    for f in files:
        r = json.loads(f.read_text())
        statuses[r.get("status", "?")] += 1
        if r.get("status") == "ok":
            by_cfg[r["config_id"]].append(r)

    has_test = any("test_acc" in r for runs in by_cfg.values() for r in runs)
    rank_key = "test_acc" if has_test else "val_best"
    rows = []
    for cid, runs in by_cfg.items():
        row = {"config_id": cid, "n_seeds": len(runs), "params": runs[0]["params"]}
        for m in (["val_best", "test_acc", "test_wf1", "test_mf1", "test_balacc"]
                  if has_test else ["val_best"]):
            arr = np.array([r[m] for r in runs if m in r], dtype=float)
            row[f"{m}_mean"] = float(arr.mean()) if arr.size else None
            row[f"{m}_std"] = float(arr.std()) if arr.size else None
        rows.append(row)
    rows.sort(key=lambda r: (r.get(f"{rank_key}_mean") or -1), reverse=True)

    print(f"\nresults: {dict(statuses)}  |  {len(rows)} configs with >=1 ok run  "
          f"| ranked by {rank_key}_mean\n")
    header = f"{'rank':>4} {'cfg':>4} {'seeds':>5} {'val_acc':>16} "
    header += f"{'test_acc':>16} {'test_wf1':>16} " if has_test else ""
    header += " summary"
    print(header)
    for i, r in enumerate(rows[:20]):
        line = (f"{i:>4} {r['config_id']:>4} {r['n_seeds']:>5} "
                f"{r['val_best_mean']:.4f}±{r['val_best_std']:.3f} ")
        if has_test:
            line += (f"{r['test_acc_mean']:.4f}±{r['test_acc_std']:.3f} "
                     f"{r['test_wf1_mean']:.4f}±{r['test_wf1_std']:.3f} ")
        p = r["params"]
        line += (f" L{p['patch_layers']}/"
                 f"{p['patch_backbone'] if p['patch_layers'] else 'none'}/"
                 f"{p['readout']}/w{p['width_scale']}/ns{p['nsample']}")
        print(line)

    summary = {"ranked_by": rank_key, "n_configs": len(rows), "statuses": dict(statuses),
               "rows": rows}
    (rdir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    if rows:
        best = rows[0]
        cmd = full_train_command(args, best["params"], run_name="ri_hiernet_sweep_best")
        (rdir / "best_command.sh").write_text("#!/bin/bash\n" + cmd + "\n")
        print(f"\nbest config #{best['config_id']} "
              f"({rank_key}={best[f'{rank_key}_mean']:.4f}±{best[f'{rank_key}_std']:.3f}):")
        print(cmd)
        print(f"\nsaved -> {rdir}/summary.json  +  best_command.sh")


# ─────────────────────────────────────────────────────────────────────────────
# Mode: search  (adaptive Optuna study; optional shared journal)
# ─────────────────────────────────────────────────────────────────────────────
def cmd_search(args, device):
    worker_id = (args.worker_id if args.worker_id is not None
                 else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    worker_seed = args.seed + 1000 * worker_id
    torch.manual_seed(worker_seed); np.random.seed(worker_seed)

    study_name = args.study_name or f"ri_hiernet_optuna_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = Path(args.output_dir) / study_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.journal_file:
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend, JournalFileSymlinkLock
        Path(args.journal_file).parent.mkdir(parents=True, exist_ok=True)
        storage = JournalStorage(JournalFileBackend(
            args.journal_file, lock_obj=JournalFileSymlinkLock(args.journal_file)))
        storage_desc = f"journal:{args.journal_file}"
    else:
        storage = f"sqlite:///{out_dir / 'study.db'}"
        storage_desc = storage
    print(f"study={study_name}  worker={worker_id}(seed={worker_seed})  storage={storage_desc}")

    tr_ds, va_ds, num_classes = build_datasets(args)
    sampler = TPESampler(seed=worker_seed, multivariate=True, group=True)
    pruner = HyperbandPruner(min_resource=8, max_resource=args.search_epochs, reduction_factor=3)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner,
                                study_name=study_name, storage=storage, load_if_exists=True)

    def objective(trial):
        params = suggest_params(trial)
        def on_epoch(ep, sel):
            trial.report(sel, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
        try:
            res = train_model(params, args.seed + trial.number, tr_ds, va_ds,
                              num_classes, device, args, on_epoch=on_epoch)
        except optuna.TrialPruned:
            raise
        except RuntimeError as e:
            torch.cuda.empty_cache()
            if "out of memory" in str(e).lower():
                raise optuna.TrialPruned()
            raise
        finally:
            torch.cuda.empty_cache()
        trial.set_user_attr("params_count", res["n_params"])
        print(f"  trial {trial.number}: val={res['val_best']:.4f} "
              f"({res['seconds']:.0f}s, {res['n_params']/1e6:.1f}M) {params['patch_backbone']}")
        return res["val_best"]

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   gc_after_trial=True, show_progress_bar=False)
    try:
        best = study.best_trial
    except ValueError:
        print("no completed trial visible from this worker; skipping summary."); return
    full = {**suggest_params_defaults(best.params), "val_best": best.value}
    (out_dir / "best_config.json").write_text(json.dumps(full, indent=2, default=float))
    cmd = full_train_command(args, full, run_name="ri_hiernet_optuna_best")
    (out_dir / "best_command.sh").write_text("#!/bin/bash\n" + cmd + "\n")
    print(f"\nBEST trial #{best.number}: val {args.select_metric} = {best.value:.4f}\n{cmd}")


def suggest_params_defaults(p: dict) -> dict:
    """Fill conditional knobs not present in a study's best_params."""
    pl = p.get("patch_layers", 2)
    return {
        "nsample": p.get("nsample", 32), "patch_layers": pl,
        "patch_backbone": p.get("patch_backbone", "gin"),
        "patch_knn": p.get("patch_knn", 8), "readout": p.get("readout", "max+attn"),
        "width_scale": p.get("width_scale", 1), "lr": p.get("lr", 5e-4),
        "weight_decay": p.get("weight_decay", 1e-4), "dropout": p.get("dropout", 0.3),
        "aug_scale": p.get("aug_scale", 0.0), "aug_dropout": p.get("aug_dropout", 0.0),
        "label_smoothing": p.get("label_smoothing", 0.0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["search", "gen", "train", "agg"], default="search")
    # data (same derivation as train_ri_hiernet.py)
    ap.add_argument("--data-dir", default="/home/ogharbi/datasets/SHREC25")
    ap.add_argument("--graph-subdir", default="ssl_graphs_1k_geom_enriched")
    ap.add_argument("--graph-dir", default=None)
    ap.add_argument("--train-labels-csv", default=None)
    ap.add_argument("--test-labels-csv", default=None)
    ap.add_argument("--val-labels-csv", default=None)
    ap.add_argument("--npoint", type=int, default=1000)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    # training budget (shared)
    ap.add_argument("--search-epochs", type=int, default=40,
                    help="epochs per training run (search mode also prunes earlier)")
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="keep 0 so the in-RAM cache persists across epochs")
    ap.add_argument("--select-metric", default="accuracy",
                    choices=["accuracy", "f1_weighted", "f1_macro", "balanced_accuracy"])
    ap.add_argument("--device", type=int, default=0)
    # parallel sweep (gen/train/agg)
    ap.add_argument("--n-configs", type=int, default=100, help="[gen] configs to sample")
    ap.add_argument("--n-seeds", type=int, default=5, help="[gen] seeds/config (for task-count math)")
    ap.add_argument("--configs-file", default=None, help="[gen out / train in] configs.json")
    ap.add_argument("--config-id", type=int, default=None, help="[train] index into configs.json")
    ap.add_argument("--seed-id", type=int, default=0, help="[train] training seed for this run")
    ap.add_argument("--result-dir", default=None, help="[train out / agg in] per-run JSON dir")
    ap.add_argument("--eval-test", action="store_true", default=True,
                    help="[train] evaluate best-val checkpoint on the test split")
    ap.add_argument("--no-eval-test", dest="eval_test", action="store_false")
    # search mode (adaptive study)
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--output-dir", default="/home/ogharbi/projects/protein_classification/runs")
    ap.add_argument("--study-name", default=None)
    ap.add_argument("--journal-file", default=None,
                    help="[search] shared Optuna journal (Lustre-safe symlink lock)")
    ap.add_argument("--worker-id", type=int, default=None,
                    help="[search] distinct TPE seed per worker; default $SLURM_ARRAY_TASK_ID")
    args = ap.parse_args()

    root = Path(args.data_dir)
    if args.graph_dir is None:        args.graph_dir = str(root / args.graph_subdir)
    if args.train_labels_csv is None: args.train_labels_csv = str(root / "pc" / "train_labels.csv")
    if args.test_labels_csv is None:  args.test_labels_csv = str(root / "pc" / "test_labels.csv")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"mode={args.mode}  device={device}  graph-dir={args.graph_dir}")

    if args.mode == "gen":
        if not args.configs_file:
            sys.exit("--configs-file is required for --mode gen")
        cmd_gen(args)
    elif args.mode == "train":
        if args.configs_file is None or args.config_id is None or args.result_dir is None:
            sys.exit("--mode train needs --configs-file, --config-id and --result-dir")
        cmd_train(args, device)
    elif args.mode == "agg":
        if not args.result_dir:
            sys.exit("--mode agg needs --result-dir")
        cmd_agg(args)
    else:
        cmd_search(args, device)


if __name__ == "__main__":
    main()
