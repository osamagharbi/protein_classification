#!/usr/bin/env python3
"""Future-work training launcher for Jean-Zay Slurm arrays (points 3 and 4).

One array task = one (arm, seed) = one single-GPU training of
classify_hierarchical_gnn.py with the paper's FIXED recipe (grid_hiergin.FIXED:
3 node + 2 patch layers, attention pooling, max readout, label smoothing 0.1,
cosine LR, 150 epochs, patience 30), HiPatchNet (gvp) node encoder, hidden 256.

Studies
  rotation      point 3: exactly rotation-invariant HiPatchNet (--invariant), every
                run also evaluated under 5 fixed random rotations (--test-rotations 5).
                On the SHREC2025 FINAL labels (rotated release) three arms are paired:
                original model / original + SO(3) augmentation / invariant.
  multiscale19  point 4: multi-scale model (k = 16, 64, 256 on one shared encoder),
                gate fusion, shared (param-matched) vs separate patch GNNs, SHREC2019.
  multiscale25  same on SHREC2025 (needs k16/k256 sidecars: jeanzay/build_patchassign.slurm).
  edges         reference model with bidirectional message passing (the reported models
                use each stored edge in one direction only), + the invariant FINAL arm.

Modes
  plan  print the task list and TASKS=<n> (read by submit_fw.sh)
  run   run task --task-id
  agg   collect every finished run of a study into summary.md / summary.csv

Seeds 1-10 match the clean recheck runs (runs/shrec19_recheck), so single-scale
baselines are paired by seed with existing results.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from grid_hiergin import FIXED, DATASETS, TRAIN, resolve_paths  # noqa: E402

LABELS = REPO / "data_labels"
ROT = ["--test-rotations", "5"]
SO3_AUG = ["--augment", "--aug-scale-std", "0", "--aug-jitter-std", "0"]

ARMS = {
    # ---------------- point 3: rotation invariance ----------------
    "s19p_inv":        dict(ds="shrec19_protein", fg="geom", k=64, flags=["--invariant"] + ROT),
    "s19s_inv":        dict(ds="shrec19_species", fg="geom", k=64, flags=["--invariant"] + ROT),
    "s25_inv":         dict(ds="shrec25", fg="enriched", k=64, flags=["--invariant"] + ROT),
    "s25final_base":   dict(ds="shrec25", fg="enriched", k=64, labels="final", flags=ROT),
    "s25final_so3":    dict(ds="shrec25", fg="enriched", k=64, labels="final", flags=SO3_AUG + ROT),
    "s25final_inv":    dict(ds="shrec25", fg="enriched", k=64, labels="final", flags=["--invariant"] + ROT),
    # ---------------- message-passing direction (found 2026-11-26) ----------------
    # The reported models pass messages along each stored edge in ONE direction
    # (edge_index_mesh stores every undirected edge once, low -> high node index):
    # 9.4% (SHREC2019) / 24.3% (SHREC2025) of nodes get no neighbour message.
    # These arms rerun the reference with both directions (--use-bidirectional-edges).
    "s19p_bidir":        dict(ds="shrec19_protein", fg="geom", k=64, flags=["--use-bidirectional-edges"]),
    "s19s_bidir":        dict(ds="shrec19_species", fg="geom", k=64, flags=["--use-bidirectional-edges"]),
    "s25_bidir":         dict(ds="shrec25", fg="enriched", k=64, flags=["--use-bidirectional-edges"]),
    "s25final_inv_bidir": dict(ds="shrec25", fg="enriched", k=64, labels="final",
                               flags=["--invariant", "--use-bidirectional-edges"] + ROT),
    # invariant + bidirectional on the PAPER labels: directly comparable with the HiPatch reference
    "s25_inv_bidir":     dict(ds="shrec25", fg="enriched", k=64,
                              flags=["--invariant", "--use-bidirectional-edges"] + ROT),
    # ---------------- point 4: multi-scale ----------------
    "ms19p_gate_shared": dict(ds="shrec19_protein", fg="geom", ks=[16, 64, 256],
                              flags=["--fusion", "gate", "--share-patch-gnn"]),
    "ms19p_gate_sep":    dict(ds="shrec19_protein", fg="geom", ks=[16, 64, 256], flags=["--fusion", "gate"]),
    "ms19s_gate_shared": dict(ds="shrec19_species", fg="geom", ks=[16, 64, 256],
                              flags=["--fusion", "gate", "--share-patch-gnn"]),
    "ms19s_gate_sep":    dict(ds="shrec19_species", fg="geom", ks=[16, 64, 256], flags=["--fusion", "gate"]),
    "ms25_gate_shared":  dict(ds="shrec25", fg="enriched", ks=[16, 64, 256],
                              flags=["--fusion", "gate", "--share-patch-gnn"]),
    "ms25_gate_sep":     dict(ds="shrec25", fg="enriched", ks=[16, 64, 256], flags=["--fusion", "gate"]),
    # single-resolution references for the multi-scale arms (SHREC2025 has only k=64 runs)
    "s25_k16":           dict(ds="shrec25", fg="enriched", k=16, flags=[]),
    "s25_k256":          dict(ds="shrec25", fg="enriched", k=256, flags=[]),
}
STUDIES = {
    "rotation": ["s19p_inv", "s19s_inv", "s25_inv", "s25final_base", "s25final_so3", "s25final_inv"],
    "multiscale19": ["ms19p_gate_shared", "ms19p_gate_sep", "ms19s_gate_shared", "ms19s_gate_sep"],
    "multiscale25": ["ms25_gate_shared", "ms25_gate_sep", "s25_k16", "s25_k256"],
    "edges": ["s19p_bidir", "s19s_bidir", "s25_bidir", "s25final_inv_bidir", "s25_inv_bidir"],
}


def tasks(study, arms, seeds):
    names = arms or STUDIES[study]
    for a in names:
        if a not in ARMS:
            sys.exit(f"unknown arm {a!r}; choices: {list(ARMS)}")
    return [(a, s) for a in names for s in seeds]


def tag(arm, seed):
    c = ARMS[arm]
    if "ks" in c:
        return f"{c['ds']}_{c['fg']}_ms{'-'.join(map(str, c['ks']))}_{arm}_gvp_h256_s{seed}"
    return f"{c['ds']}_{c['fg']}_k{c['k']}_{arm}_gvp_h256_s{seed}"


def command(arm, seed, args):
    c = ARMS[arm]
    k0 = c["ks"][0] if "ks" in c else c["k"]
    graph_dir, assign_dir, tr, te, fmode = resolve_paths(args.data_root, c["ds"], c["fg"], k0)
    va = None
    if c.get("labels") == "final":
        tr, va, te = (str(LABELS / "shrec25_final" / f) for f in
                      ("train_split.csv", "val_split.csv", "test_labels.csv"))
    need = [graph_dir] + ([str(Path(args.data_root) / DATASETS[c["ds"]]["assign_tmpl"].format(k=k))
                           for k in c["ks"]] if "ks" in c else [assign_dir])
    missing = [p for p in need if not os.path.isdir(p)] + \
              [f for f in (tr, va, te) if f and not os.path.isfile(f)]
    if missing:
        sys.exit("[preflight] missing:\n  " + "\n  ".join(missing) +
                 "\n  -> run ./jeanzay/restore_data.sh (and sbatch jeanzay/build_patchassign.slurm for k16/k256);"
                 "\n     or submit only the arms whose data is present: ARMS=\"...\"")
    out_dir = Path(args.study_root) / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(TRAIN), "--graph-dir", graph_dir, "--feature-mode", fmode,
           "--node-backbone", "gvp", "--hidden-dim", "256", "--seed", str(seed),
           "--device", "0", "--num-workers", str(args.num_workers),
           "--output-dir", str(out_dir), "--run-name", tag(arm, seed)]
    if "ks" in c:
        cmd += ["--multi-assign-dirs"] + need[1:]
    else:
        cmd += ["--assign-dir", assign_dir]
    if tr:
        cmd += ["--train-labels-csv", tr]
    if va:
        cmd += ["--val-labels-csv", va]
    if te:
        cmd += ["--test-labels-csv", te]
    fixed = dict(FIXED)
    if args.smoke:
        fixed.update(epochs=1, patience=1)
    for k, v in fixed.items():
        cmd += ["--" + k.replace("_", "-"), str(v)]
    return cmd + c["flags"]


def cmd_plan(args):
    t = tasks(args.study, args.arms, args.seeds)
    for arm in dict.fromkeys(a for a, _ in t):   # preflight: refuse to submit if any arm lacks data
        command(arm, t[0][1], args)
    for i, (a, s) in enumerate(t):
        print(f"  [{i:3d}] {tag(a, s)}")
    print(f"TASKS={len(t)}")


def cmd_run(args):
    t = tasks(args.study, args.arms, args.seeds)
    if not 0 <= args.task_id < len(t):
        sys.exit(f"--task-id {args.task_id} out of range [0, {len(t)})")
    arm, seed = t[args.task_id]
    out = Path(args.study_root) / "runs" / tag(arm, seed) / "final_results.json"
    if out.exists() and not args.smoke:
        print(f"[fw] {tag(arm, seed)} already finished, skipping")
        return
    cmd = command(arm, seed, args)
    print("[fw] " + " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    print(f"[fw] task {args.task_id} {tag(arm, seed)} rc={rc} wall={(time.time() - t0) / 60:.1f} min", flush=True)
    sys.exit(rc)


def cmd_agg(args):
    rows, by_arm = [], defaultdict(list)
    for fr in sorted((Path(args.study_root) / "runs").glob("*/final_results.json")):
        d = json.load(open(fr))
        m, c = d["final_test_metrics"], d["config"]
        arm = next((a for a in ARMS if f"_{a}_gvp_" in fr.parent.name), "?")
        rot = (d.get("rotation_test") or {}).get("mean", {})
        r = {"run": fr.parent.name, "arm": arm, "seed": c.get("seed"), "n_params": d.get("n_params"),
             "epochs_run": d.get("epochs_run"), "acc": m.get("accuracy"), "f1w": m.get("f1_weighted"),
             "f1m_present": m.get("f1_macro_present"), "bal_present": m.get("balanced_accuracy_present"),
             "mAP": m.get("mAP"), "rot_acc": rot.get("accuracy"), "rot_f1w": rot.get("f1_weighted"),
             "min_per_run": (d.get("train_time_sec") or 0) / 60}
        rows.append(r)
        by_arm[arm].append(r)
    if not rows:
        sys.exit(f"no finished runs under {args.study_root}/runs")
    keys = list(rows[0])
    with open(Path(args.study_root) / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def ms(v):
        v = [x for x in v if x is not None]
        return f"{np.mean(v):.4f} ± {np.std(v):.4f}" if v else "-"
    L = [f"# {Path(args.study_root).name}\n",
         "| arm | runs | params | F1w | acc | macro-F1 (present) | bal. acc (present) | mAP | F1w rotated | acc rotated | min/run |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, rs in sorted(by_arm.items()):
        L.append(f"| {arm} | {len(rs)} | {rs[0]['n_params']:,} | {ms([r['f1w'] for r in rs])} | "
                 f"{ms([r['acc'] for r in rs])} | {ms([r['f1m_present'] for r in rs])} | "
                 f"{ms([r['bal_present'] for r in rs])} | {ms([r['mAP'] for r in rs])} | "
                 f"{ms([r['rot_f1w'] for r in rs])} | {ms([r['rot_acc'] for r in rs])} | "
                 f"{np.mean([r['min_per_run'] for r in rs]):.0f} |")
    (Path(args.study_root) / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["plan", "run", "agg"], required=True)
    ap.add_argument("--study", choices=list(STUDIES), required=True)
    ap.add_argument("--arms", nargs="*", default=None, help="subset of the study's arms")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    ap.add_argument("--task-id", type=int, default=-1)
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", os.environ.get("WORK", ".")))
    ap.add_argument("--study-root", default=None, help="default: runs/fw_<study>")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="1 epoch, for a quick end-to-end check")
    args = ap.parse_args()
    args.study_root = args.study_root or str(REPO / "runs" / f"fw_{args.study}{'_smoke' if args.smoke else ''}")
    {"plan": cmd_plan, "run": cmd_run, "agg": cmd_agg}[args.mode](args)


if __name__ == "__main__":
    main()
