#!/usr/bin/env python3
"""
grid_hiergin.py — deterministic GRID launcher for the hierarchical patch-pooling
GNN (HSNet / classify_hierarchical_gnn.py), designed for max-GPU-parallel SLURM
job arrays on Jean-Zay (one independent training run per GPU, no DDP).

Why a grid and not an Optuna/TPE search: the study is an *exhaustive* sweep over
{backbone} x {hidden-dim} x {seed}. Every cell is independent, so we enumerate the
Cartesian product and map SLURM_ARRAY_TASK_ID -> one config. This needs no shared
journal / storage coordination (unlike the TPE search in optuna_search_flatgnn.py),
which is exactly what makes it embarrassingly parallel across all allotted GPUs.

The dataset, feature-group and patch-count are chosen ONCE per submission (they
define the "cell" of the study); the array grids over backbone x hidden x seed
inside that cell. Re-run with a different --dataset / --feature-group / --patch-k
to cover another cell.

Modes
-----
  plan : print the resolved grid and its size (the submit script reads N from here)
  run  : run ONE grid point (given --task-id) by invoking classify_hierarchical_gnn.py
  agg  : collect every run's final_results.json into ONE summary.csv + summary.json
         (mean +/- std over seeds per backbone x hidden), so you download few files.

Each training run writes a single self-contained final_results.json that already
contains: full config, n_params, train_time_sec, sec_per_epoch, epochs_run,
per-epoch history (the training log), and final_test_metrics including
mAP / tpr_macro / tnr_macro / f1_weighted. That is the "verbose + concentrated"
artifact — the SLURM .out logs are redundant and can be discarded.
"""
from __future__ import annotations
import argparse, csv, itertools, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "classify_hierarchical_gnn.py"

# ----------------------------------------------------------------------------
# Dataset + feature-group registry.
#
# DATA_ROOT is the directory that CONTAINS the graph dirs (on Jean-Zay this is
# $WORK, the level of ssl_graphs_1k_geom_enriched_ri). Everything below is
# resolved relative to it, so the same code runs locally and on the cluster by
# only changing --data-root.
#
# Each (dataset, feature_group) resolves to:
#   graph_subdir  : node-graph directory (relative to DATA_ROOT)
#   feature_mode  : --feature-mode passed to the trainer
#   labels        : "internal" (train_split.csv/val_split.csv/test_labels.csv live
#                    inside graph_subdir, auto-detected) OR explicit CSV subpaths.
#   assign_tmpl   : patch-assignment dir template, {k} filled from --patch-k
# ----------------------------------------------------------------------------
DATASETS = {
    # SHREC19 protein-level (17 classes). Base graphs are pure-geometry (xyz).
    "shrec19_protein": {
        "assign_tmpl": "ssl_graphs_shared_proteinsplit_patchassign_k{k}",
        "labels": {
            "train": "graphs_pyg_shared_proteins/train_labels.csv",
            "test":  "graphs_pyg_shared_proteins/test_labels.csv",
        },
        "features": {
            "geom":     {"graph_subdir": "ssl_graphs_shared_proteinsplit",              "feature_mode": "coords"},  # 3D
            "enriched": {"graph_subdir": "ssl_graphs_shared_proteinsplit_geom_enriched", "feature_mode": "all"},     # 16D
        },
    },
    # SHREC19 species-level (54/53 classes). Same graphs, different label CSVs.
    "shrec19_species": {
        "assign_tmpl": "ssl_graphs_shared_proteinsplit_patchassign_k{k}",
        "labels": {
            "train": "graphs_pyg_shared_species/train_labels.csv",
            "test":  "graphs_pyg_shared_species/test_labels.csv",
        },
        "features": {
            "geom":     {"graph_subdir": "ssl_graphs_shared_proteinsplit",              "feature_mode": "coords"},
            "enriched": {"graph_subdir": "ssl_graphs_shared_proteinsplit_geom_enriched", "feature_mode": "all"},
        },
    },
    # SHREC25 (99 classes). Splits live inside the graph dir (auto-detected).
    # enriched uses geom_14d_raw (14D = 8D base + 6 geom); labels reused from base.
    "shrec25": {
        # ssl_graphs_1k_geom_enriched_ri is a strict superset of both
        # ssl_graphs_1k (plain, x=8D) and ssl_graphs_1k_geom_enriched (x=21D =
        # 8D base + 13D geom_features): verified byte-identical on the shared
        # columns (x, pos, geom_features_*, edge_index). It additionally
        # carries ri_feat/ri_lrf_feat (unused by "coords"/"geom_14d_raw"), and
        # it is the only one of the three actually deployed on Jean-Zay, so
        # both feature groups point at it -- no duplicate upload needed.
        "assign_tmpl": "ssl_graphs_1k_patchassign_k{k}",
        "labels": {"train": None, "test": None},  # internal auto-detect for base dir
        "features": {
            "geom":     {"graph_subdir": "ssl_graphs_1k_geom_enriched_ri", "feature_mode": "coords",       "labels_from": "ssl_graphs_1k_geom_enriched_ri"},
            "enriched": {"graph_subdir": "ssl_graphs_1k_geom_enriched_ri", "feature_mode": "geom_14d_raw", "labels_from": "ssl_graphs_1k_geom_enriched_ri"},
        },
    },
}

# 11 dataset-agnostic node backbones (exclude ri_gin/lrf_gin: they need RI edge
# features that SHREC19 graphs do not carry). gvp = equivariant GVP conv reading
# xyz from the feature slice. patch-backbone stays gin throughout.
DEFAULT_BACKBONES = ["gin", "sage", "gcn", "gat", "gatv2", "pna", "gt", "gps", "sgformer", "msg_gin", "gvp"]
DEFAULT_HIDDEN    = [128, 256, 512]
DEFAULT_SEEDS     = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Fixed headline recipe (everything the grid does NOT vary). Mirrors the SHREC19
# winning config: 3 node layers, 2 patch layers, attn pool, max readout, inverse-sqrt
# class weights, cosine LR with warmup, label smoothing 0.1, no augmentation.
FIXED = dict(
    node_layers=3, patch_layers=2, pool="attn", readout="max",
    class_weight_method="inverse_sqrt", loss_type="ce", label_smoothing=0.1,
    batch_size=64, epochs=150, patience=30, lr=5e-4, lr_schedule="cosine",
    warmup_epochs=5, dropout=0.3, patch_backbone="gin",
)


# Architecture variants for the no-hierarchy ablation. "hier" is the paper
# model; "flatN" drops patch pooling + the patch GNN (--flat) and reads out over
# all nodes with N node layers (flat5 matches hier's depth and parameter count).
ARCHS = {"hier": None, "flat3": 3, "flat5": 5}


def resolve_paths(data_root: str, dataset: str, feature_group: str, patch_k: int):
    """Return concrete (graph_dir, assign_dir, train_csv, test_csv, feature_mode)."""
    if dataset not in DATASETS:
        sys.exit(f"unknown --dataset {dataset!r}; choices: {list(DATASETS)}")
    ds = DATASETS[dataset]
    if feature_group not in ds["features"]:
        sys.exit(f"unknown --feature-group {feature_group!r} for {dataset}; "
                 f"choices: {list(ds['features'])}")
    fg = ds["features"][feature_group]
    root = Path(data_root)
    graph_dir  = root / fg["graph_subdir"]
    assign_dir = root / ds["assign_tmpl"].format(k=patch_k)
    # labels: explicit CSVs if given, else internal auto-detect (return None)
    labels_base = fg.get("labels_from")  # SHREC25: CSVs live in a different dir
    def _csv(split):
        rel = ds["labels"].get(split)
        if rel is not None:
            return str(root / rel)
        if labels_base is not None:                       # SHREC25 explicit
            fname = "train_split.csv" if split == "train" else "test_labels.csv"
            return str(root / labels_base / fname)
        return None                                       # fully internal auto-detect
    return (str(graph_dir), str(assign_dir), _csv("train"), _csv("test"), fg["feature_mode"])


def build_grid(backbones, hidden_dims, seeds):
    """Cartesian product -> ordered list of (backbone, hidden, seed)."""
    return list(itertools.product(backbones, hidden_dims, seeds))


def run_tag(dataset, feature_group, patch_k, backbone, hidden, seed, arch="hier"):
    arch_tag = "" if arch == "hier" else f"{arch}_"
    return f"{dataset}_{feature_group}_k{patch_k}_{arch_tag}{backbone}_h{hidden}_s{seed}"


# ----------------------------------------------------------------------------
def cmd_plan(args):
    grid = build_grid(args.backbones, args.hidden_dims, args.seeds)
    graph_dir, assign_dir, tr, te, fmode = resolve_paths(
        args.data_root, args.dataset, args.feature_group, args.patch_k)
    print(f"# dataset={args.dataset} feature_group={args.feature_group} "
          f"(feature_mode={fmode}) patch_k={args.patch_k}")
    print(f"# graph_dir  = {graph_dir}")
    print(f"# assign_dir = {assign_dir}")
    print(f"# labels     = train:{tr or 'INTERNAL'}  test:{te or 'INTERNAL'}")
    print(f"# backbones({len(args.backbones)}) = {args.backbones}")
    print(f"# hidden({len(args.hidden_dims)})   = {args.hidden_dims}")
    print(f"# seeds({len(args.seeds)})     = {args.seeds}")
    print(f"GRID_SIZE={len(grid)}")
    if args.verbose:
        for i, (bb, h, s) in enumerate(grid):
            print(f"  [{i:3d}] {run_tag(args.dataset, args.feature_group, args.patch_k, bb, h, s, args.arch)}")


def cmd_check(args):
    """Validate that a cell's data (graph dir, patch dir, label CSVs) exists."""
    graph_dir, assign_dir, tr, te, fmode = resolve_paths(
        args.data_root, args.dataset, args.feature_group, args.patch_k)
    missing = [p for p in (graph_dir, assign_dir) if not os.path.isdir(p)]
    missing += [c for c in (tr, te) if c is not None and not os.path.isfile(c)]
    if missing:
        print(f"MISSING for {args.dataset}/{args.feature_group}/k{args.patch_k}:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)
    print(f"OK {args.dataset}/{args.feature_group}/k{args.patch_k} "
          f"(feature_mode={fmode}, graph_dir={os.path.basename(graph_dir)})")


def cmd_run(args):
    grid = build_grid(args.backbones, args.hidden_dims, args.seeds)
    if not (0 <= args.task_id < len(grid)):
        sys.exit(f"--task-id {args.task_id} out of range [0,{len(grid)})")
    backbone, hidden, seed = grid[args.task_id]
    graph_dir, assign_dir, tr, te, fmode = resolve_paths(
        args.data_root, args.dataset, args.feature_group, args.patch_k)

    # ---- preflight: fail loud & early if data is missing ----
    for p in (graph_dir, assign_dir):
        if not os.path.isdir(p):
            sys.exit(f"[preflight] missing directory: {p}\n"
                     f"  -> upload it to {args.data_root} (same level as the SHREC25 ri dataset).")
    for c in (tr, te):
        if c is not None and not os.path.isfile(c):
            sys.exit(f"[preflight] missing label CSV: {c}")

    tag = run_tag(args.dataset, args.feature_group, args.patch_k, backbone, hidden, seed, args.arch)
    out_dir = Path(args.grid_root) / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-u", str(TRAIN),
           "--graph-dir", graph_dir,
           "--assign-dir", assign_dir,
           "--feature-mode", fmode,
           "--node-backbone", backbone,
           "--hidden-dim", str(hidden),
           "--seed", str(seed),
           "--device", str(args.device),
           "--num-workers", str(args.num_workers),
           "--output-dir", str(out_dir),
           "--run-name", tag]
    if tr: cmd += ["--train-labels-csv", tr]
    if te: cmd += ["--test-labels-csv", te]
    fixed = dict(FIXED)
    if ARCHS[args.arch] is not None:
        fixed["node_layers"] = ARCHS[args.arch]
        cmd += ["--flat"]
    for k, v in fixed.items():
        flag = "--" + k.replace("_", "-")
        cmd += [flag, str(v)]

    print(f"[grid] task {args.task_id}/{len(grid)-1}  tag={tag}")
    print("[grid] " + " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = time.time() - t0
    print(f"[grid] task {args.task_id} finished rc={rc} wall={dt/60:.1f} min", flush=True)
    sys.exit(rc)


ROW_COLS = ["tag", "dataset", "feature_group", "patch_k", "backbone", "hidden",
            "seed", "in_dim", "num_classes", "n_params", "n_params_trainable",
            "epochs_run", "best_epoch", "train_time_sec", "sec_per_epoch",
            "acc", "bal_acc", "f1_macro", "f1_weighted", "tpr_macro", "tnr_macro", "mAP"]
SUMMARY_METRICS = ["mAP", "f1_weighted", "f1_macro", "bal_acc", "acc",
                   "tpr_macro", "tnr_macro", "train_time_sec", "sec_per_epoch"]


def _collect_rows(runs_dir, dataset, feature_group, patch_k):
    """One flat row per completed run under runs_dir."""
    rows = []
    for fr in sorted(Path(runs_dir).glob("*/final_results.json")):
        d = json.load(open(fr)); c = d.get("config", {}); m = d.get("final_test_metrics", {})
        rows.append({
            "tag": fr.parent.name, "dataset": dataset, "feature_group": feature_group,
            "patch_k": patch_k if patch_k is not None else c.get("assign_dir", "").split("_k")[-1],
            "backbone": c.get("node_backbone") + (f"+flat{c.get('node_layers')}" if c.get("flat") else ""),
            "hidden": c.get("hidden_dim"),
            "seed": c.get("seed"), "in_dim": c.get("in_dim"), "num_classes": c.get("num_classes"),
            "n_params": d.get("n_params"), "n_params_trainable": d.get("n_params_trainable"),
            "epochs_run": d.get("epochs_run"), "best_epoch": d.get("best_epoch"),
            "train_time_sec": d.get("train_time_sec"), "sec_per_epoch": d.get("sec_per_epoch"),
            "acc": m.get("accuracy"), "bal_acc": m.get("balanced_accuracy"),
            "f1_macro": m.get("f1_macro"), "f1_weighted": m.get("f1_weighted"),
            "tpr_macro": m.get("tpr_macro"), "tnr_macro": m.get("tnr_macro"), "mAP": m.get("mAP"),
        })
    return rows


def _summarize(rows):
    """mean +/- std over seeds, grouped by (dataset, feature, patch_k, backbone, hidden)."""
    import statistics as st
    groups = {}
    for r in rows:
        key = (r["dataset"], r["feature_group"], str(r["patch_k"]), r["backbone"], r["hidden"])
        groups.setdefault(key, []).append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], int(kv[0][2] or 0),
                                                          str(kv[0][3]), kv[0][4] or 0)):
        ds, fg, pk, bb, h = key
        e = {"dataset": ds, "feature_group": fg, "patch_k": pk, "backbone": bb, "hidden": h,
             "n_seeds": len(rs), "n_params": rs[0]["n_params"]}
        for k in SUMMARY_METRICS:
            vals = [x[k] for x in rs if isinstance(x[k], (int, float))]
            e[f"{k}_mean"] = st.mean(vals) if vals else None
            e[f"{k}_std"] = (st.pstdev(vals) if len(vals) > 1 else 0.0) if vals else None
        out.append(e)
    return out


def _write_tables(rows, out_dir, prefix, meta):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{prefix}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_COLS); w.writeheader()
        for r in rows: w.writerow({k: r.get(k) for k in ROW_COLS})
    summary = _summarize(rows)
    json.dump({**meta, "per_config": summary, "n_runs": len(rows)},
              open(out_dir / f"{prefix}.json", "w"), indent=2, default=float)
    return csv_path, summary


def _print_summary(summary, title):
    print(f"\n{title}")
    print(f"{'dataset':16s} {'feat':9s} {'k':>4s} {'backbone':9s} {'hid':>4s} "
          f"{'seeds':>5s} {'mAP':>16s} {'F1w':>16s} {'min/run':>8s}")
    for e in summary:
        def f(k):
            mn, sd = e[f'{k}_mean'], e[f'{k}_std']
            return f"{mn:.4f}+/-{sd:.4f}" if mn is not None else "     n/a      "
        t = e['train_time_sec_mean']
        print(f"{str(e['dataset']):16s} {str(e['feature_group']):9s} {str(e['patch_k']):>4s} "
              f"{str(e['backbone']):9s} {str(e['hidden']):>4s} {e['n_seeds']:>5d} "
              f"{f('mAP'):>16s} {f('f1_weighted'):>16s} {(t/60 if t else 0):>8.1f}")


def cmd_agg(args):
    """Aggregate ONE cell (grid-root)."""
    rows = _collect_rows(Path(args.grid_root) / "runs", args.dataset, args.feature_group, args.patch_k)
    if not rows:
        sys.exit(f"[agg] no final_results.json under {args.grid_root}/runs")
    csv_path, summary = _write_tables(
        rows, args.grid_root, "summary",
        {"cell": {"dataset": args.dataset, "feature_group": args.feature_group, "patch_k": args.patch_k}})
    print(f"[agg] {len(rows)} runs -> {csv_path}  (+ summary.json)")
    _print_summary(summary, f"cell {args.dataset}/{args.feature_group}/k{args.patch_k}")


def cmd_agg_study(args):
    """Master aggregation across ALL cells under a study root.

    Expects cell dirs named <dataset>_<feature>_k<K>/ (as run_study.sh creates them);
    feature is a single token (geom|enriched), dataset may contain underscores.
    """
    study_root = Path(args.study_root)
    cell_dirs = sorted(d for d in study_root.iterdir() if d.is_dir() and (d / "runs").is_dir())
    all_rows = []
    for cd in cell_dirs:
        name = cd.name
        if "_k" not in name:
            continue
        head, k = name.rsplit("_k", 1)          # head = <dataset>_<feature>
        dataset, _, feature = head.rpartition("_")
        rows = _collect_rows(cd / "runs", dataset, feature, k)
        all_rows.extend(rows)
        # also refresh the per-cell tables
        if rows:
            _write_tables(rows, cd, "summary",
                          {"cell": {"dataset": dataset, "feature_group": feature, "patch_k": k}})
    if not all_rows:
        sys.exit(f"[agg-study] no completed runs under {study_root}")
    csv_path, summary = _write_tables(all_rows, study_root, "master_summary",
                                      {"study": str(study_root), "cells": len(cell_dirs)})
    print(f"[agg-study] {len(all_rows)} runs across {len(cell_dirs)} cells -> {csv_path}")
    print(f"[agg-study] master table -> {study_root/'master_summary.json'}")
    _print_summary(summary, "MASTER (mean +/- std over seeds)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["plan", "check", "run", "agg", "agg-study"])
    ap.add_argument("--dataset", default="shrec19_protein", choices=list(DATASETS))
    ap.add_argument("--feature-group", default="geom", help="geom | enriched")
    ap.add_argument("--patch-k", type=int, default=64)
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/home/ogharbi/datasets/SHREC19"),
                    help="dir CONTAINING the graph dirs (Jean-Zay: $WORK)")
    ap.add_argument("--grid-root", default=os.environ.get("GRID_ROOT", "runs/grid"),
                    help="where a cell's run outputs + summary live")
    ap.add_argument("--study-root", default=None,
                    help="study dir containing many cell dirs (agg-study mode)")
    ap.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES)
    ap.add_argument("--hidden-dims", nargs="+", type=int, default=DEFAULT_HIDDEN)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--arch", default=os.environ.get("ARCH", "hier"), choices=list(ARCHS),
                    help="hier (paper model) | flat3 | flat5 (no-hierarchy ablation)")
    ap.add_argument("--task-id", type=int, default=-1, help="grid index (run mode)")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.mode == "agg-study" and not args.study_root:
        sys.exit("--study-root is required for --mode agg-study")
    {"plan": cmd_plan, "check": cmd_check, "run": cmd_run,
     "agg": cmd_agg, "agg-study": cmd_agg_study}[args.mode](args)


if __name__ == "__main__":
    main()
