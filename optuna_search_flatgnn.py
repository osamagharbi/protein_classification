#!/usr/bin/env python3
"""Per-backbone distributed Optuna search for the FLAT GNN (classify_gnn.py).

Mirrors the cluster/optuna-search branch's `optuna_search_ri_hiernet.py` pattern
(shared JournalStorage across SLURM-array workers, TPE sampler) but:
  * targets the flat GNN in `classify_gnn.py`,
  * runs ONE independent study PER backbone (identical protocol, per-backbone
    optimum), and
  * launches each trial as a subprocess (so trials are exactly reproducible real
    classify_gnn.py runs).

See paper_gnn_shrec25/planning/09_flatgnn_search_protocol.md for the design.

Modes
-----
  search        one Optuna worker for --backbone; joins the shared per-backbone
                study via --journal-file (or SQLite). Each trial trains
                classify_gnn.py at --search-epochs and returns best_val f1_macro.
                On finish, writes best_config.json + best_command.sh.
  retrain-best  read best_config.json for each --backbones and run the full-epoch
                (--retrain-epochs) training across --seeds.
  agg           group the retrain runs per backbone -> mean +/- std -> summary json.
  dry-run       print the trial command for one sampled config WITHOUT training
                (used to validate command construction off-cluster).

NOTHING here is meant to run the search locally; --mode search/retrain-best launch
real trainings and are intended for the SLURM scripts. `--mode dry-run` is safe.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev

import optuna
from optuna.samplers import TPESampler

PROJECT_ROOT = Path(__file__).resolve().parent
CLASSIFY = PROJECT_ROOT / "classify_gnn.py"

BACKBONES = ["gin", "gcn", "gat", "sage", "sgformer"]

# ---------------------------------------------------------------------------- #
# Shared search space (identical for every backbone) + fixed scaffold
# ---------------------------------------------------------------------------- #
def suggest_params(trial: optuna.Trial) -> dict:
    """The one search space used for ALL backbones (the comparability guarantee).
    Only these knobs vary; everything else is the fixed scaffold below."""
    return {
        "lr": trial.suggest_float("lr", 3e-4, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        "num_layers": trial.suggest_int("num_layers", 2, 6),
        "pooling": trial.suggest_categorical("pooling", ["mean", "max", "sum", "attention"]),
        "grad_clip": trial.suggest_categorical("grad_clip", [0.0, 1.0]),
    }


def scaffold_args(args, backbone: str, params: dict, epochs: int,
                  run_name: str, seed: int) -> list[str]:
    """Build the classify_gnn.py argv for one config. The FIXED scaffold is applied
    identically to every backbone; `params` are the searched hyperparameters."""
    # Resolve label CSVs explicitly (default: inside the dataset dir) so the run is
    # deterministic and never falls back to sibling-directory auto-resolution — a
    # real trap on the cluster, where only <data-dir> is copied. ssl_graphs_1k is
    # self-contained: train_split.csv / val_split.csv / test_labels.csv all inside.
    data = Path(args.data_dir)
    train_csv = args.train_labels_csv or str(data / "train_split.csv")
    val_csv = args.val_labels_csv or str(data / "val_split.csv")
    test_csv = args.test_labels_csv or str(data / "test_labels.csv")
    cmd = [
        args.python_executable, str(args.classify_script),
        "--data-dir", args.data_dir,
        "--model", backbone,
        # --- explicit labels (no sibling-dir fallback) ---
        "--train-labels-csv", train_csv,
        "--val-labels-csv", val_csv,
        "--test-labels-csv", test_csv,
        # --- fixed scaffold (identical across backbones) ---
        "--class-weight-method", "inverse_sqrt",
        "--loss-type", "ce",
        "--warmup-epochs", str(args.warmup_epochs),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--epochs", str(epochs),
        "--device", str(args.device),
        "--seed", str(seed),
        "--output-dir", str(args.trial_output_dir),
        "--run-name", run_name,
        # --- searched hyperparameters ---
        "--lr", f"{params['lr']:.6g}",
        "--weight-decay", f"{params['weight_decay']:.6g}",
        "--dropout", f"{params['dropout']:.4g}",
        "--hidden-dim", str(params["hidden_dim"]),
        "--num-layers", str(params["num_layers"]),
        "--pooling", params["pooling"],
        "--grad-clip", f"{params['grad_clip']:.4g}",
    ]
    # Fixed feature scaffold: 8D (all) = no --keep-features. If a subset is given,
    # it is applied identically to every backbone.
    if args.keep_features:
        cmd += ["--keep-features", *[str(i) for i in args.keep_features]]
    return cmd


def read_val_f1(run_dir: Path) -> float | None:
    fr = run_dir / "final_results.json"
    if not fr.exists():
        return None
    try:
        d = json.loads(fr.read_text())
        return float(d["best_val_metrics"]["f1_macro"])
    except Exception:
        return None


# ---------------------------------------------------------------------------- #
# Storage (shared across array workers) — mirrors the branch
# ---------------------------------------------------------------------------- #
def make_storage(args, backbone: str):
    if args.journal_file:
        from optuna.storages import JournalStorage
        try:
            from optuna.storages.journal import JournalFileBackend  # optuna >= 3.6
        except Exception:
            from optuna.storages import JournalFileStorage as _JFS  # older API
            return _JFS(str(args.journal_file)), f"journal:{args.journal_file}"
        jf = Path(args.journal_file)
        jf.parent.mkdir(parents=True, exist_ok=True)
        return JournalStorage(JournalFileBackend(str(jf))), f"journal:{jf}"
    db = Path(args.output_dir) / f"flatgnn_{backbone}.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db}", f"sqlite:{db}"


# ---------------------------------------------------------------------------- #
# Modes
# ---------------------------------------------------------------------------- #
def cmd_search(args):
    backbone = args.backbone
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage, storage_desc = make_storage(args, backbone)
    worker_seed = args.sampler_seed + args.worker_id
    study_name = f"flatgnn_{backbone}"
    print(f"study={study_name}  worker={args.worker_id}(seed={worker_seed})  storage={storage_desc}")

    sampler = TPESampler(seed=worker_seed, multivariate=True, group=True)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=study_name, storage=storage,
                                load_if_exists=True)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        if backbone == "gat" and params["hidden_dim"] % 4 != 0:
            raise optuna.exceptions.TrialPruned("gat hidden_dim not divisible by num_heads(4)")
        run_name = f"opt_{backbone}_t{trial.number}"
        run_dir = Path(args.trial_output_dir) / run_name
        cmd = scaffold_args(args, backbone, params, args.search_epochs, run_name,
                            seed=args.search_seed)
        trial.set_user_attr("command", " ".join(cmd))
        log = run_dir.with_suffix(".log")
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as fh:
            try:
                subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT,
                               timeout=args.max_trial_seconds)
            except subprocess.TimeoutExpired:
                raise optuna.exceptions.TrialPruned(f"timeout; see {log}")
            except subprocess.CalledProcessError:
                raise optuna.exceptions.TrialPruned(f"classify_gnn.py failed; see {log}")
        val = read_val_f1(run_dir)
        if val is None:
            raise optuna.exceptions.TrialPruned(f"no metric; see {log}")
        return val

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   catch=(RuntimeError,))

    # Only the worker that observes the best writes the artifacts; harmless if >1 do.
    try:
        best = study.best_trial
    except ValueError:
        print("no completed trials yet"); return
    best_cfg = dict(best.params); best_cfg["val_best_f1_macro"] = best.value
    (out_dir / f"best_config_{backbone}.json").write_text(json.dumps(best_cfg, indent=2))
    # best_command.sh = full-epoch retrain command (seed placeholder $SEED)
    retrain_cmd = scaffold_args(args, backbone, best.params, args.retrain_epochs,
                                run_name=f"flatgnn_best_{backbone}_seed$SEED", seed=0)
    # replace the fixed seed token with the shell variable
    retrain_cmd = [a if a != "--seed" else a for a in retrain_cmd]
    joined = " ".join(retrain_cmd).replace("--seed 0", "--seed $SEED")
    (out_dir / f"best_command_{backbone}.sh").write_text(
        "#!/bin/bash\n# full-epoch retrain of the best %s config; set SEED before running\n%s\n"
        % (backbone, joined))
    print(f"best {backbone}: val f1_macro={best.value:.4f}  params={best.params}")
    print(f"wrote {out_dir}/best_config_{backbone}.json + best_command_{backbone}.sh")


def cmd_retrain_best(args):
    out_dir = Path(args.output_dir)
    for bb in args.backbones:
        cfg_path = out_dir / f"best_config_{bb}.json"
        if not cfg_path.exists():
            print(f"[skip] {bb}: {cfg_path} not found (run --mode search first)"); continue
        cfg = json.loads(cfg_path.read_text())
        params = {k: cfg[k] for k in ("lr", "weight_decay", "dropout", "hidden_dim",
                                      "num_layers", "pooling", "grad_clip")}
        for seed in args.seeds:
            run_name = f"flatgnn_best_{bb}_seed{seed}"
            cmd = scaffold_args(args, bb, params, args.retrain_epochs, run_name, seed=seed)
            print("RUN", " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=False)


def cmd_agg(args):
    out_dir = Path(args.output_dir)
    rows = []
    for bb in args.backbones:
        per_seed = []
        for seed in args.seeds:
            fr = Path(args.trial_output_dir) / f"flatgnn_best_{bb}_seed{seed}" / "final_results.json"
            if fr.exists():
                d = json.loads(fr.read_text())["final_test_metrics"]
                per_seed.append(d)
        if not per_seed:
            rows.append({"backbone": bb, "n_seeds": 0}); continue
        def ms(key):
            xs = [r[key] for r in per_seed]
            return round(mean(xs), 4), round(pstdev(xs) if len(xs) > 1 else 0.0, 4)
        acc_m, acc_s = ms("accuracy"); f1_m, f1_s = ms("f1_macro")
        wf_m, wf_s = ms("f1_weighted"); ba_m, ba_s = ms("balanced_accuracy")
        rows.append({"backbone": bb, "n_seeds": len(per_seed),
                     "acc_mean": acc_m, "acc_std": acc_s,
                     "f1_macro_mean": f1_m, "f1_macro_std": f1_s,
                     "wf1_mean": wf_m, "wf1_std": wf_s,
                     "balacc_mean": ba_m, "balacc_std": ba_s})
    rows.sort(key=lambda r: r.get("acc_mean", -1), reverse=True)
    summary = {"ranked_by": "acc_mean", "seeds": args.seeds, "rows": rows}
    (out_dir / "summary_flatgnn.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir}/summary_flatgnn.json")


def cmd_dry_run(args):
    """Validate command construction without training."""
    sampler = TPESampler(seed=args.sampler_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    trial = study.ask()
    params = suggest_params(trial)
    bb = args.backbone or BACKBONES[0]
    cmd = scaffold_args(args, bb, params, args.search_epochs, f"opt_{bb}_dryrun",
                        seed=args.search_seed)
    print("sampled params:", json.dumps(params, indent=2))
    print("\ntrial command:\n ", " ".join(cmd))


# ---------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["search", "retrain-best", "agg", "dry-run"],
                    default="dry-run")
    ap.add_argument("--backbone", choices=BACKBONES, default=None,
                    help="single backbone for --mode search / dry-run")
    ap.add_argument("--backbones", nargs="+", default=BACKBONES,
                    help="backbones for --mode retrain-best / agg")
    # data / scaffold
    ap.add_argument("--data-dir",
                    default="/home/ogharbi/datasets/SHREC25/ssl_graphs_1k_geom_enriched_ri",
                    help="cluster dataset (self-contained; 21D features: 8D base + 13 geom)")
    ap.add_argument("--train-labels-csv", default=None,
                    help="default: <data-dir>/train_split.csv (7395 train; disjoint from val)")
    ap.add_argument("--val-labels-csv", default=None,
                    help="default: <data-dir>/val_split.csv (1849 val)")
    ap.add_argument("--test-labels-csv", default=None,
                    help="default: <data-dir>/test_labels.csv (2311 test, 88 classes)")
    # NOTE: this dataset stores 21 features. "all features" would be 21D, so we pin
    # the feature scaffold EXPLICITLY and identically for every backbone. Default =
    # base 8D [Potential_norm, NormalPotential_norm, nx,ny,nz, x,y,z] (indices 0-7).
    ap.add_argument("--keep-features", type=int, nargs="*",
                    default=[0, 1, 2, 3, 4, 5, 6, 7],
                    help="fixed feature indices used identically for all backbones. "
                         "Default 0-7 = base 8D. Full order (21D): 0 Potential_norm, "
                         "1 NormalPotential_norm, 2-4 nx/ny/nz, 5-7 x/y/z, 8 dist_to_centroid, "
                         "9-12 dist_to_anchor_0..3, 13 mean_neighbor_dist, 14 std_neighbor_dist, "
                         "15 inv_mean_neighbor_dist, 16 laplacian_curvature_mag, "
                         "17 laplacian_curvature_signed, 18 normal_variation, 19 planarity, "
                         "20 surface_variation. Pass e.g. `--keep-features $(seq -s' ' 0 20)` for 21D.")
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--search-epochs", type=int, default=60)
    ap.add_argument("--retrain-epochs", type=int, default=120)
    ap.add_argument("--search-seed", type=int, default=42,
                    help="training seed used during the search phase (fixed)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                    help="retrain seeds for --mode retrain-best / agg")
    # optuna / distributed
    ap.add_argument("--study-name", default=None, help="(informational; auto per backbone)")
    ap.add_argument("--journal-file", default=None,
                    help="shared JournalStorage file for cluster-parallel workers; "
                         "omit for a local SQLite study")
    ap.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs_optuna_flatgnn"))
    ap.add_argument("--trial-output-dir", default=str(PROJECT_ROOT / "runs_optuna_flatgnn" / "trials"))
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--sampler-seed", type=int, default=42)
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=None, help="worker wall-clock seconds")
    ap.add_argument("--max-trial-seconds", type=int, default=None)
    # execution
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--python-executable", default=sys.executable)
    ap.add_argument("--classify-script", default=str(CLASSIFY))
    ap.add_argument("--dry-run", action="store_true",
                    help="for --mode retrain-best: print commands without running")
    args = ap.parse_args()

    if args.mode == "search":
        if not args.backbone:
            ap.error("--mode search requires --backbone")
        cmd_search(args)
    elif args.mode == "retrain-best":
        cmd_retrain_best(args)
    elif args.mode == "agg":
        cmd_agg(args)
    else:
        cmd_dry_run(args)


if __name__ == "__main__":
    main()
