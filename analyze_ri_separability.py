#!/usr/bin/env python3
"""
analyze_ri_separability.py

Separability analysis for the 5 precomputed SO(3)-invariant RI edge features.
Mirrors the methodology used for the 8D node features:

  1. KS test        — marginal distribution per RI dimension (pooled over edges)
  2. Sliced Wasserstein — 5D distribution distance between class pairs
  3. SVM            — protein-level statistics (10D: mean+std per dim)

Results are printed as a comparison table against the existing node-feature
separability numbers and saved as a PDF figure.

Confusable pairs studied:
  (61, 90) — near-inseparable
  (70, 56) — moderately confusable
  (86, 14) — moderately confusable
  (8,  5)  — control (should be easily separable)

Usage:
    python3 analyze_ri_separability.py \\
        --dataset-dir /path/to/ssl_graphs_1k_geom_enriched_ri \\
        --labels-csv  /path/to/ssl_graphs_1k/train_split.csv \\
        [--out-dir    report/figures]
"""

import argparse
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import torch
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Baseline numbers from existing analysis (node features) ───────────────────
BASELINE_SW_FEAT = {(61,90): 0.0272, (70,56): 0.1041, (86,14): 0.1397, (8,5): 0.0613}
BASELINE_SW_XYZ  = {(61,90): 0.0416, (70,56): 0.0433, (86,14): 0.0291, (8,5): 0.0629}
BASELINE_SVM     = {(61,90): 0.543,  (70,56): 0.573,  (86,14): 0.665,  (8,5): 0.660}

PAIRS       = [(61,90), (70,56), (86,14), (8,5)]
PAIR_LABELS = ["(61,90)\nconfusable", "(70,56)\nconfusable",
               "(86,14)\nconfusable", "(8,5)\ncontrol"]

RI_DIM_NAMES = [
    "dist  ‖pⱼ−pᵢ‖",
    "⟨nᵢ, ê⟩",
    "⟨nⱼ, ê⟩",
    "⟨nᵢ, nⱼ⟩",
    "⟨nⱼ×nᵢ, ê⟩",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def sliced_wasserstein(X: np.ndarray, Y: np.ndarray,
                       n_projections: int = 100, seed: int = 42,
                       max_pts: int = 30_000) -> float:
    """Sliced Wasserstein distance between two point clouds (vectorised)."""
    rng = np.random.default_rng(seed)
    # Subsample for speed
    if len(X) > max_pts:
        X = X[rng.choice(len(X), max_pts, replace=False)]
    if len(Y) > max_pts:
        Y = Y[rng.choice(len(Y), max_pts, replace=False)]
    d = X.shape[1]
    dirs = rng.standard_normal((n_projections, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    # Vectorised: project + sort in one shot
    px = np.sort(X @ dirs.T, axis=0)   # [N, P]
    py = np.sort(Y @ dirs.T, axis=0)   # [M, P]
    # Interpolate to same length for 1D Wasserstein
    n, m = len(px), len(py)
    t = np.linspace(0, 1, max(n, m))
    px_i = np.array([np.interp(t, np.linspace(0, 1, n), px[:, i])
                     for i in range(n_projections)])   # [P, T]
    py_i = np.array([np.interp(t, np.linspace(0, 1, m), py[:, i])
                     for i in range(n_projections)])
    return float(np.abs(px_i - py_i).mean())


def protein_stats(ri: np.ndarray) -> np.ndarray:
    """10D protein-level descriptor: mean + std of each RI dimension."""
    return np.concatenate([ri.mean(axis=0), ri.std(axis=0)])


def svm_balanced_acc(X_A: np.ndarray, X_B: np.ndarray, seed: int = 0,
                     cv_folds: int = 5) -> float:
    """Binary SVM balanced accuracy via stratified k-fold cross-validation."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    X = np.vstack([X_A, X_B])
    y = np.array([0]*len(X_A) + [1]*len(X_B))
    pipe = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced",
            random_state=seed),
    )
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = []
    for tr, te in skf.split(X, y):
        pipe.fit(X[tr], y[tr])
        scores.append(balanced_accuracy_score(y[te], pipe.predict(X[te])))
    return float(np.mean(scores))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_class_data(dataset_dir: Path, labels_csv: Path,
                    classes: set, max_proteins: int = 200) -> dict:
    """
    Returns {class_id: list of ri_feat np.ndarray [E, 5]} for the given classes.
    Loads from <dataset_dir>/train_set/*.pt  (capped at max_proteins per class).
    """
    df = pd.read_csv(labels_csv)
    df = df[df["class_id"].isin(classes)]

    train_dir = dataset_dir / "train_set"
    avail = {p.stem: p for p in train_dir.glob("*.pt")}

    data = {c: [] for c in classes}
    missing = 0
    for _, row in df.iterrows():
        pid = str(row["protein_id"])
        cls = int(row["class_id"])
        if len(data[cls]) >= max_proteins:
            continue
        stem = pid.replace(":", "_")
        path = avail.get(stem) or avail.get(pid)
        if path is None:
            missing += 1
            continue
        d  = torch.load(str(path), map_location="cpu", weights_only=False)
        ri = d.get("ri_feat")
        if ri is not None:
            data[cls].append(ri.numpy().astype(np.float32))
    if missing:
        print(f"  [warn] {missing} proteins not found in dataset")
    return data


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse_pair(ri_A: list, ri_B: list, label: str, pair: tuple):
    """Run KS, SW, and SVM for one pair. Returns dict of results."""
    # Pool edges across all proteins
    pool_A = np.vstack(ri_A)   # [E_total_A, 5]
    pool_B = np.vstack(ri_B)   # [E_total_B, 5]

    print(f"\n{'─'*60}")
    print(f"  Pair {label}   |  {len(ri_A)} proteins (A), {len(ri_B)} proteins (B)")
    print(f"  Edges: {len(pool_A):,} (A)  {len(pool_B):,} (B)")

    # 1. KS test per dimension
    ks_stats = []
    print(f"\n  KS statistics (pooled edge distributions):")
    for dim, name in enumerate(RI_DIM_NAMES):
        stat, pval = ks_2samp(pool_A[:, dim], pool_B[:, dim])
        ks_stats.append(stat)
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        print(f"    [{dim}] {name:<22}  KS={stat:.4f}  p={pval:.2e}  {sig}")

    # 2. Sliced Wasserstein (5D joint)
    sw_ri   = sliced_wasserstein(pool_A, pool_B)
    sw_dist = sliced_wasserstein(pool_A[:, :1], pool_B[:, :1])   # dist only
    sw_ang  = sliced_wasserstein(pool_A[:, 1:], pool_B[:, 1:])   # 4 angle dims
    print(f"\n  Sliced Wasserstein:")
    print(f"    SW(all 5D):  {sw_ri:.4f}")
    print(f"    SW(dist):    {sw_dist:.4f}")
    print(f"    SW(4 angles):{sw_ang:.4f}")

    # 3. SVM on protein-level statistics (cross-validated)
    stats_A = np.vstack([protein_stats(ri) for ri in ri_A])
    stats_B = np.vstack([protein_stats(ri) for ri in ri_B])
    svm_acc = svm_balanced_acc(stats_A, stats_B)
    print(f"\n  SVM balanced accuracy (10D protein stats, 5-fold CV): {svm_acc:.3f}")

    # Baselines — use the pair tuple directly (avoid sorted-key bug)
    bl_sw   = BASELINE_SW_FEAT.get(pair, BASELINE_SW_FEAT.get(pair[::-1], None))
    bl_svm  = BASELINE_SVM.get(pair, BASELINE_SVM.get(pair[::-1], None))
    print(f"\n  ── Baseline (8D node features) ──")
    if bl_sw is not None:
        delta_sw = sw_ri - bl_sw
        print(f"    SW(8D feat):  {bl_sw:.4f}   →  RI: {sw_ri:.4f}  (Δ{delta_sw:+.4f})")
    if bl_svm is not None:
        delta_svm = svm_acc - bl_svm
        print(f"    SVM(8D):      {bl_svm:.3f}    →  RI: {svm_acc:.3f}  (Δ{delta_svm:+.3f})")

    return {
        "pair": label,
        "pair_key": pair,
        "ks": ks_stats,
        "sw_5d": sw_ri,
        "sw_dist": sw_dist,
        "sw_ang": sw_ang,
        "svm": svm_acc,
        "n_A": len(ri_A),
        "n_B": len(ri_B),
    }


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_results(results: list, out_dir: Path):
    C = plt.cm.tab10.colors
    pair_labels_short = ["(61,90)", "(70,56)", "(86,14)", "(8,5)"]
    n_pairs = len(results)
    x = np.arange(n_pairs)
    w = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # ── Plot 1: KS per RI dimension ──────────────────────────────────────────
    ax = axes[0]
    ks_vals = np.array([r["ks"] for r in results])  # [n_pairs, 5]
    w_ks = 0.14
    for i, (name, col) in enumerate(zip(RI_DIM_NAMES, C)):
        ax.bar(x + (i - 2) * w_ks, ks_vals[:, i], w_ks,
               color=col, alpha=0.85, label=name)
    ax.set_xticks(x); ax.set_xticklabels(pair_labels_short, fontsize=9)
    ax.set_ylabel("KS statistic")
    ax.set_title("KS test per RI dimension\n(pooled edge distributions)")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # ── Plot 2: SW comparison ─────────────────────────────────────────────────
    ax = axes[1]
    sw_ri   = [r["sw_5d"]   for r in results]
    sw_ang  = [r["sw_ang"]  for r in results]
    bl_sw   = [BASELINE_SW_FEAT[p] for p in PAIRS]
    bl_xyz  = [BASELINE_SW_XYZ[p]  for p in PAIRS]

    ax.bar(x - 1.5*w, bl_xyz,  w, color=C[2], alpha=0.85, label="SW(xyz) — node coords")
    ax.bar(x - 0.5*w, bl_sw,   w, color=C[0], alpha=0.85, label="SW(8D feat) — node features")
    ax.bar(x + 0.5*w, sw_ang,  w, color=C[3], alpha=0.85, label="SW(RI angles) — 4 angle dims")
    ax.bar(x + 1.5*w, sw_ri,   w, color=C[1], alpha=0.85, label="SW(RI 5D) — all RI dims")

    ax.set_xticks(x); ax.set_xticklabels(pair_labels_short, fontsize=9)
    ax.set_ylabel("Sliced Wasserstein distance")
    ax.set_title("SW comparison: node features vs RI edge features")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # ── Plot 3: SVM comparison ────────────────────────────────────────────────
    ax = axes[2]
    svm_ri  = [r["svm"] for r in results]
    bl_svm  = [BASELINE_SVM[p] for p in PAIRS]

    ax.bar(x - 0.5*w, bl_svm, w*1.5, color=C[0], alpha=0.85,
           label="SVM(8D node stats) — 32D descriptor")
    ax.bar(x + 0.5*w, svm_ri, w*1.5, color=C[1], alpha=0.85,
           label="SVM(RI stats) — 10D descriptor")
    ax.axhline(0.5, color="gray", lw=1, ls="--", label="chance")
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks(x); ax.set_xticklabels(pair_labels_short, fontsize=9)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("SVM balanced accuracy\n(protein-level statistics)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("RI edge feature separability vs 8D node feature baseline", fontsize=11)
    plt.tight_layout()
    out_path = out_dir / "separability_ri_features.pdf"
    fig.savefig(str(out_path))
    plt.close(fig)
    print(f"\n  Figure saved → {out_path}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(results: list):
    print(f"\n{'═'*72}")
    print("SUMMARY TABLE — RI edge features vs 8D node features")
    print(f"{'═'*72}")
    hdr = f"{'Pair':<12}{'SW(8D)':>9}{'SW(RI)':>9}{'Δ SW':>8} │ {'SVM(8D)':>9}{'SVM(RI)':>9}{'Δ SVM':>8}"
    print(hdr)
    print("─" * 72)
    for r in results:
        pk = r["pair_key"]
        bl_sw  = BASELINE_SW_FEAT.get(pk, BASELINE_SW_FEAT.get(pk[::-1], float("nan")))
        bl_svm = BASELINE_SVM.get(pk, BASELINE_SVM.get(pk[::-1], float("nan")))
        dsw  = r["sw_5d"] - bl_sw  if not np.isnan(bl_sw)  else float("nan")
        dsvm = r["svm"]   - bl_svm if not np.isnan(bl_svm) else float("nan")
        label = r["pair"]
        dsw_str  = f"{dsw:>+8.4f}"  if not np.isnan(dsw)  else f"{'n/a':>8}"
        dsvm_str = f"{dsvm:>+8.3f}" if not np.isnan(dsvm) else f"{'n/a':>8}"
        bl_sw_str  = f"{bl_sw:>8.4f}"  if not np.isnan(bl_sw)  else f"{'n/a':>8}"
        bl_svm_str = f"{bl_svm:>8.3f}" if not np.isnan(bl_svm) else f"{'n/a':>8}"
        print(f"  {label:<10} {bl_sw_str} {r['sw_5d']:>8.4f} {dsw_str} │ "
              f"{bl_svm_str} {r['svm']:>8.3f} {dsvm_str}")
    print(f"{'═'*72}")
    print("SW: higher = more separable.  SVM: higher = more separable.")
    print("Δ positive = RI features are MORE separating than 8D node features.")
    print("SVM uses 5-fold stratified CV (protein-level 10D stats).")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir",
                    default="/home/ogharbi/datasets/SHREC25/ssl_graphs_1k_geom_enriched_ri")
    ap.add_argument("--labels-csv",
                    default="/home/ogharbi/datasets/SHREC25/ssl_graphs_1k/train_split.csv")
    ap.add_argument("--out-dir",
                    default="/home/ogharbi/projects/protein_classification/report/figures")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    labels_csv  = Path(args.labels_csv)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Classes needed for all 4 pairs
    all_classes = {c for pair in PAIRS for c in pair}
    print(f"Loading RI features for classes {sorted(all_classes)} …")
    class_data = load_class_data(dataset_dir, labels_csv, all_classes)
    for cls, items in class_data.items():
        print(f"  class {cls:3d}: {len(items)} proteins")

    # Run analysis for each pair
    results = []
    for pair, label in zip(PAIRS, PAIR_LABELS):
        a, b = pair
        results.append(analyse_pair(class_data[a], class_data[b], f"({a},{b})", pair=pair))

    print_summary(results)
    plot_results(results, out_dir)


if __name__ == "__main__":
    main()
