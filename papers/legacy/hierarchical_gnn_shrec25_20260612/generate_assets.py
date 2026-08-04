#!/usr/bin/env python3
"""Generate all figures and tables for the hierarchical GNN SHREC25 report."""
import json, pathlib, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT   = pathlib.Path(__file__).parent
FIG    = ROOT / "figures"
TAB    = ROOT / "tables"
RUNS   = pathlib.Path("/home/ogharbi/projects/protein_classification/runs")
RUNS2  = pathlib.Path("/home/ogharbi/runs")

FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

DEEP_BLUE  = "#17324D"
SIGNAL     = "#B94A3A"
LEAF       = "#2E6F5E"
GRAY       = "#9AA7B4"
WARM       = "#E8B87A"
LIGHT_BLUE = "#4A90C4"

# ── load results ──────────────────────────────────────────────────────────────
def load(path):
    return json.load(open(path))

patch_runs = {
    "k=8":   load(RUNS / "hier_ab_k8_fast/final_results.json"),
    "k=32":  load(RUNS / "hier_ab_k32_fast/final_results.json"),
    "k=64":  load(RUNS / "hier_ab_k64_fast/final_results.json"),
    "k=128": load(RUNS / "hier_ab_k128_fast2/final_results.json"),
}

ablation_runs = {
    "k64 baseline\n(3L / 256h / 2patch / 40ep)": load(RUNS / "hier_ab_k64_fast/final_results.json"),
    "patch-layers=3\n(oversmoothing)":            load(RUNS2 / "hier_k64_patchdepth3/final_results.json"),
    "hidden=512\n(overfits)":                     load(RUNS2 / "hier_k64_wide512/final_results.json"),
    "80 epochs\n(converged)":                     load(RUNS2 / "hier_k64_long80/final_results.json"),
}

def metrics(d):
    m = d["final_test_metrics"]
    return dict(
        acc=m["accuracy"],
        bal=m["balanced_accuracy"],
        macro=m["f1_macro"],
        weighted=m["f1_weighted"],
    )


# ── Figure 1: patch count sweep ───────────────────────────────────────────────
def fig_patch_sweep():
    labels = list(patch_runs.keys())
    data   = [metrics(v) for v in patch_runs.values()]

    x = np.arange(len(labels))
    w = 0.20
    metric_keys   = ["acc", "bal", "macro", "weighted"]
    metric_labels = ["Accuracy", "Bal. Accuracy", "F1-Macro", "F1-Weighted"]
    colors = [LIGHT_BLUE, WARM, SIGNAL, LEAF]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, colors)):
        vals = [d[key] for d in data]
        bars = ax.bar(x + i * w, vals, w, label=label, color=color, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, color=DEEP_BLUE)

    # highlight k=64 best
    ax.axvspan(2 - 0.5*w - 0.03, 2 + 3.5*w + 0.03, alpha=0.08, color=SIGNAL, zorder=0)
    ax.text(2 + 1.5*w, 0.62, "★ best", ha="center", fontsize=9, color=SIGNAL, fontweight="bold")

    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0.35, 0.68)
    ax.set_title("Patch Count Sweep — Hierarchical GNN on SHREC25\n"
                 "(node-layers=3, hidden=256, patch-layers=2, 40 epochs, CE+inverse_sqrt)",
                 fontsize=11, color=DEEP_BLUE, pad=8)
    ax.legend(fontsize=9, loc="upper left")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG / "patch_sweep.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


# ── Figure 2: ablation comparison ────────────────────────────────────────────
def fig_ablations():
    labels = list(ablation_runs.keys())
    data   = [metrics(v) for v in ablation_runs.values()]

    metric_keys   = ["acc", "bal", "macro", "weighted"]
    metric_labels = ["Accuracy", "Bal. Accuracy", "F1-Macro", "F1-Weighted"]
    colors = [LIGHT_BLUE, WARM, SIGNAL, LEAF]

    x = np.arange(len(labels))
    w = 0.18

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, colors)):
        vals = [d[key] for d in data]
        alphas = [1.0 if j == 0 else 0.75 for j in range(len(vals))]
        bars = ax.bar(x + i * w, vals, w, label=label, color=color, zorder=3)
        for bar, a in zip(bars, alphas):
            bar.set_alpha(a)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, color=DEEP_BLUE)

    ax.axvspan(-0.5*w - 0.05, 3.5*w + 0.05, alpha=0.08, color=LEAF, zorder=0)
    ax.text(1.5*w, 0.835, "baseline", ha="center", fontsize=9, color=LEAF, fontweight="bold")

    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0.40, 0.87)
    ax.set_title("Architecture Ablations at k=64 — SHREC25 Hierarchical GNN",
                 fontsize=11, color=DEEP_BLUE, pad=8)
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG / "ablations.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


# ── Figure 3: training curves (k64 best model) ───────────────────────────────
def fig_training_curve():
    d = load(RUNS / "hier_ab_k64_fast/final_results.json")
    hist = d["history"]
    epochs      = [h["epoch"]               for h in hist]
    train_f1    = [h["train"]["f1_macro"]   for h in hist]
    val_f1      = [h["val"]["f1_macro"]     for h in hist]
    train_acc   = [h["train"]["accuracy"]   for h in hist]
    val_acc     = [h["val"]["accuracy"]     for h in hist]
    val_bal     = [h["val"]["balanced_accuracy"] for h in hist]
    best_ep     = max(range(len(val_f1)), key=lambda i: val_f1[i])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # F1
    ax1.plot(epochs, train_f1, color=DEEP_BLUE, lw=1.8, label="Train F1-Macro")
    ax1.plot(epochs, val_f1,   color=SIGNAL,    lw=1.8, label="Val F1-Macro")
    ax1.plot(epochs, val_bal,  color=LEAF,      lw=1.4, linestyle="--", label="Val Bal-Acc")
    ax1.axvline(epochs[best_ep], color=SIGNAL, linestyle=":", alpha=0.7)
    ax1.annotate(f"best val F1\n{val_f1[best_ep]:.3f} @ ep {epochs[best_ep]}",
                 xy=(epochs[best_ep], val_f1[best_ep]),
                 xytext=(epochs[best_ep]+3, val_f1[best_ep]-0.07),
                 fontsize=8.5, color=SIGNAL,
                 arrowprops=dict(arrowstyle="->", color=SIGNAL, lw=1.2))
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Score")
    ax1.set_title("F1 & Balanced Accuracy — k64 best model", fontsize=10, color=DEEP_BLUE)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    ax1.spines[["top","right"]].set_visible(False)

    # Accuracy
    ax2.plot(epochs, train_acc, color=DEEP_BLUE, lw=1.8, label="Train Accuracy")
    ax2.plot(epochs, val_acc,   color=SIGNAL,    lw=1.8, label="Val Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy — k64 best model", fontsize=10, color=DEEP_BLUE)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.spines[["top","right"]].set_visible(False)

    fig.suptitle("Hierarchical GNN (k=64) — Training Dynamics\n"
                 "node-layers=3, hidden=256, patch-layers=2, CE+inverse_sqrt",
                 fontsize=11, color=DEEP_BLUE, y=1.01)
    fig.tight_layout()
    out = FIG / "training_curve_k64.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ── Figure 4: macro-F1 trend line across patch counts ────────────────────────
def fig_patch_trend():
    ks   = [8, 32, 64, 128]
    f1s  = [metrics(v)["macro"] for v in patch_runs.values()]
    bals = [metrics(v)["bal"]   for v in patch_runs.values()]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, f1s,  "o-", color=SIGNAL,    lw=2.2, ms=8, label="F1-Macro",     zorder=3)
    ax.plot(ks, bals, "s--",color=DEEP_BLUE, lw=1.8, ms=7, label="Bal. Accuracy", zorder=3)
    ax.fill_between(ks, f1s, alpha=0.08, color=SIGNAL)

    ax.scatter([64], [metrics(patch_runs["k=64"])["macro"]], s=120, color=SIGNAL,
               zorder=5, edgecolors="white", linewidths=1.5)
    ax.annotate("k=64\n★ best", xy=(64, metrics(patch_runs["k=64"])["macro"]),
                xytext=(72, 0.555), fontsize=9, color=SIGNAL, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=SIGNAL, lw=1.2))

    ax.set_xscale("log", base=2)
    ax.set_xticks(ks); ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_xlabel("Number of patches (log₂ scale)", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0.40, 0.62)
    ax.set_title("Patch Count vs. Performance\nHierarchical GNN on SHREC25", fontsize=10, color=DEEP_BLUE)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    out = FIG / "patch_trend.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


# ── Table 1: patch sweep ──────────────────────────────────────────────────────
def table_patch_sweep():
    rows = []
    for name, d in patch_runs.items():
        m = metrics(d)
        rows.append((name, m["acc"], m["bal"], m["macro"], m["weighted"]))

    best_macro = max(r[3] for r in rows)
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Patches & Accuracy & Bal.\ Acc. & F1-Macro & F1-Weighted \\",
        r"\midrule",
    ]
    for k, acc, bal, macro, wt in rows:
        bf = lambda v, best: fr"\textbf{{{v:.4f}}}" if abs(v - best) < 1e-6 else f"{v:.4f}"
        line = fr"{k} & {acc:.4f} & {bal:.4f} & {bf(macro, best_macro)} & {wt:.4f} \\"
        lines.append(line)
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = TAB / "patch_sweep.tex"
    out.write_text("\n".join(lines))
    print(f"saved {out}")


# ── Table 2: ablations ────────────────────────────────────────────────────────
def table_ablations():
    display_names = {
        "k64 baseline\n(3L / 256h / 2patch / 40ep)": r"k64 baseline (3L / 256h / 2patch / 40ep)",
        "patch-layers=3\n(oversmoothing)":            r"patch-layers=3 (oversmoothing)",
        "hidden=512\n(overfits)":                     r"hidden=512 (overfits)",
        "80 epochs\n(converged)":                     r"80 epochs (converged)",
    }
    rows = []
    for name, d in ablation_runs.items():
        m = metrics(d)
        rows.append((display_names[name], m["acc"], m["bal"], m["macro"], m["weighted"]))

    best_macro = max(r[3] for r in rows)
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Configuration & Accuracy & Bal.\ Acc. & F1-Macro & F1-Weighted \\",
        r"\midrule",
    ]
    for i, (name, acc, bal, macro, wt) in enumerate(rows):
        bf = lambda v, best: fr"\textbf{{{v:.4f}}}" if abs(v - best) < 1e-6 else f"{v:.4f}"
        suffix = r" \\ \midrule" if i == 0 else r" \\"
        line = fr"{name} & {acc:.4f} & {bal:.4f} & {bf(macro, best_macro)} & {wt:.4f}{suffix}"
        lines.append(line)
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = TAB / "ablations.tex"
    out.write_text("\n".join(lines))
    print(f"saved {out}")


# ── Figure 5: architecture diagram ───────────────────────────────────────────
def _draw_graph(ax, cx, cy, pts, edges, node_colors, node_sizes, edge_alpha=0.35, zorder=3):
    """Draw a small graph centred at (cx, cy)."""
    for i, j in edges:
        ax.plot([pts[i,0]+cx, pts[j,0]+cx], [pts[i,1]+cy, pts[j,1]+cy],
                color=GRAY, lw=0.7, alpha=edge_alpha, zorder=zorder)
    for i, (px, py) in enumerate(pts):
        c = plt.Circle((px+cx, py+cy), node_sizes[i],
                       color=node_colors[i], ec="white", lw=0.4, zorder=zorder+1)
        ax.add_patch(c)

def _rbox(ax, lx, by, w, h, fc, ec, lw=1.5, zorder=2, alpha=1.0):
    p = mpatches.FancyBboxPatch((lx, by), w, h,
        boxstyle="round,pad=0.06", facecolor=fc, edgecolor=ec,
        linewidth=lw, zorder=zorder, alpha=alpha)
    ax.add_patch(p)

def _arrow(ax, x0, y0, x1, y1, color=None, lw=2.0, ms=16, zorder=5):
    color = color or DEEP_BLUE
    ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=ms),
        zorder=zorder)

def fig_architecture():
    """Two-row hierarchical layout: node level (top) and patch level (bottom)."""
    np.random.seed(3)
    W, H = 15, 8.0
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis("off"); fig.patch.set_facecolor("white")

    # ── level band backgrounds ────────────────────────────────────────────────
    # Level 1 (node): y 4.35 – 7.55
    # Coarsening zone: y 2.85 – 4.35
    # Level 2 (patch): y 0.55 – 2.85
    _rbox(ax, 0.15, 4.35, W-0.3, 3.2,  "#EAF0F7", LIGHT_BLUE, lw=1.2, alpha=0.55)
    _rbox(ax, 0.15, 0.55, W-0.3, 2.3,  "#EAF7F0", LEAF,       lw=1.2, alpha=0.55)

    # level labels
    ax.text(0.42, 7.45, "LEVEL 1 — Node Graph  (1,000 nodes)",
            fontsize=9, fontweight="bold", color=LIGHT_BLUE, va="top")
    ax.text(0.42, 2.75, "LEVEL 2 — Patch Graph  (k = 64 super-nodes)",
            fontsize=9, fontweight="bold", color=LEAF, va="top")

    # ── dense mesh graph (level 1 left) ──────────────────────────────────────
    n1 = 38
    ang1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
    r1   = 0.55 + 0.18*np.random.randn(n1)
    pts1 = np.column_stack([r1*np.cos(ang1)*0.85, r1*np.sin(ang1)*0.65])
    # random triangulation-like edges (each node to 2 nearest)
    edges1 = set()
    for i in range(n1):
        d = np.linalg.norm(pts1 - pts1[i], axis=1); d[i] = np.inf
        for j in np.argsort(d)[:3]:
            edges1.add((min(i,j), max(i,j)))
    _draw_graph(ax, 1.35, 5.95, pts1, list(edges1),
                [DEEP_BLUE]*n1, [0.055]*n1, edge_alpha=0.3)
    ax.text(1.35, 4.55, "Input Mesh\n1,000 nodes", ha="center", fontsize=8.5,
            color=DEEP_BLUE, fontweight="bold")

    # ── node GIN box ──────────────────────────────────────────────────────────
    gx, gy0, gw, gh = 3.05, 5.28, 2.55, 1.35
    _rbox(ax, gx, gy0, gw, gh, "#D6E8F7", LIGHT_BLUE)
    ax.text(gx+gw/2, gy0+gh-0.22, "Node GIN",
            ha="center", va="center", fontsize=9, fontweight="bold", color=LIGHT_BLUE)
    layer_ys = [gy0+0.82, gy0+0.50, gy0+0.18]
    layer_labels = ["Layer 1 — BN + ReLU", "Layer 2 — BN + ReLU", "Layer 3 — BN + ReLU"]
    for ly, ll in zip(layer_ys, layer_labels):
        _rbox(ax, gx+0.12, ly-0.13, gw-0.24, 0.26, LIGHT_BLUE, LIGHT_BLUE, lw=0, alpha=0.8)
        ax.text(gx+gw/2, ly, ll, ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold")
    ax.text(gx+gw/2, 4.58,
            r"$h=256$   ·   $L_n=3$   ·   dropout=0.3",
            ha="center", va="center", fontsize=8, color=DEEP_BLUE)

    # ── per-node embedding text ───────────────────────────────────────────────
    _rbox(ax, 6.3, 5.50, 2.0, 0.90, "#EAF0F7", LIGHT_BLUE, lw=1.2)
    ax.text(7.3, 6.1, r"$\mathbf{h}_v \in \mathbb{R}^{256}$",
            ha="center", va="center", fontsize=10, color=DEEP_BLUE)
    ax.text(7.3, 5.65, "per-node embedding\n(1,000 vectors)",
            ha="center", va="center", fontsize=7.8, color="#555")

    # ── level-1 arrows ────────────────────────────────────────────────────────
    _arrow(ax, 1.95, 5.95, 3.02, 5.95,  color=LIGHT_BLUE)   # mesh → node GIN
    _arrow(ax, 5.60, 5.95, 6.27, 5.95,  color=LIGHT_BLUE)   # node GIN → h_v

    # ── coarsening block (centre) ─────────────────────────────────────────────
    px0, py0 = 7.3, 4.35
    # downward arrow from h_v to coarsening box
    _arrow(ax, 7.3, 5.48, 7.3, 4.28,  color=WARM, lw=2.2, ms=18)
    _rbox(ax, 5.85, 3.00, 2.9, 1.22, "#FFF4EC", WARM, lw=1.5)
    ax.text(7.3, 4.08, "Geodesic Patch Assignment",
            ha="center", va="center", fontsize=9, fontweight="bold", color=WARM)
    ax.text(7.3, 3.70, "k=64 Voronoi seeds  ·  assign each node to nearest seed",
            ha="center", va="center", fontsize=7.8, color="#555")
    ax.text(7.3, 3.35,
            "mean-pool  +  max-pool  within each patch  →  64 super-node embeddings",
            ha="center", va="center", fontsize=7.8, color="#555")
    ax.text(7.3, 3.02, r"$\mathbf{h}_v\ (1000\!\times\!256)\ \longrightarrow\ \mathbf{h}_p\ (64\!\times\!256)$",
            ha="center", va="center", fontsize=8, color=WARM, fontweight="bold")

    # ── patch graph (level 2 left) ────────────────────────────────────────────
    n2 = 9
    patch_cols = [SIGNAL, LEAF, WARM, LIGHT_BLUE, "#8E6BBF", SIGNAL, LEAF, DEEP_BLUE, WARM]
    ang2  = np.linspace(0, 2*np.pi, n2, endpoint=False) + 0.2
    r2    = 0.52 + 0.1*np.random.randn(n2)
    pts2  = np.column_stack([r2*np.cos(ang2)*0.82, r2*np.sin(ang2)*0.65])
    edges2 = set()
    for i in range(n2):
        d = np.linalg.norm(pts2 - pts2[i], axis=1); d[i] = np.inf
        for j in np.argsort(d)[:2]:
            edges2.add((min(i,j), max(i,j)))
    _draw_graph(ax, 2.0, 1.65, pts2, list(edges2),
                patch_cols, [0.12]*n2, edge_alpha=0.45)
    ax.text(2.0, 0.68, "Patch Graph\n64 super-nodes", ha="center", fontsize=8.5,
            color=LEAF, fontweight="bold")

    # downward arrow from coarsening to patch graph
    _arrow(ax, 5.85, 3.00, 3.05, 2.25,  color=WARM, lw=2.0, ms=16)

    # ── patch GIN box ─────────────────────────────────────────────────────────
    px, py0, pw, ph = 3.35, 0.82, 2.55, 1.35
    _rbox(ax, px, py0, pw, ph, "#D5EEE5", LEAF)
    ax.text(px+pw/2, py0+ph-0.22, "Patch GIN",
            ha="center", va="center", fontsize=9, fontweight="bold", color=LEAF)
    player_ys  = [py0+0.82, py0+0.45]
    player_lbl = ["Layer 1 — BN + ReLU", "Layer 2 — BN + ReLU"]
    for ly, ll in zip(player_ys, player_lbl):
        _rbox(ax, px+0.12, ly-0.16, pw-0.24, 0.30, LEAF, LEAF, lw=0, alpha=0.85)
        ax.text(px+pw/2, ly, ll, ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold")

    # ── max readout ───────────────────────────────────────────────────────────
    _rbox(ax, 6.6, 0.92, 1.85, 0.70, "#EAF7F0", LEAF, lw=1.2)
    ax.text(7.525, 1.34, "Max Readout", ha="center", fontsize=8.5,
            fontweight="bold", color=LEAF)
    ax.text(7.525, 1.02,
            r"$\mathbf{h}_G \in \mathbb{R}^{256}$",
            ha="center", fontsize=9, color=DEEP_BLUE)

    # ── MLP classifier ────────────────────────────────────────────────────────
    _rbox(ax, 9.2, 0.82, 2.3, 1.35, "#FDECEA", SIGNAL)
    ax.text(10.35, py0+ph-0.22, "MLP Classifier",
            ha="center", va="center", fontsize=9, fontweight="bold", color=SIGNAL)
    mlp_ys  = [0.82+0.95, 0.82+0.58, 0.82+0.22]
    mlp_lbl = ["Linear(256, 256)  +  ReLU", "Dropout(0.3)", "Linear(256, 99)  →  logits"]
    for ly, ll in zip(mlp_ys, mlp_lbl):
        ax.text(10.35, ly, ll, ha="center", va="center", fontsize=7.8, color="#333")

    # ── output ────────────────────────────────────────────────────────────────
    _rbox(ax, 12.2, 1.00, 1.9, 1.00, "#F5EAF7", "#6B4F8E", lw=1.8)
    ax.text(13.15, 1.72, "Class Label",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#6B4F8E")
    ax.text(13.15, 1.42, "99 protein families",
            ha="center", va="center", fontsize=8, color="#555")
    ax.text(13.15, 1.12, "CE loss  ·  inv-sqrt weights",
            ha="center", va="center", fontsize=7.5, color="#888")

    # ── level-2 arrows ────────────────────────────────────────────────────────
    _arrow(ax, 2.72, 1.65, 3.32, 1.65,  color=LEAF)            # patch graph → patch GIN
    _arrow(ax, 5.90, 1.50, 6.57, 1.50,  color=LEAF)            # patch GIN → readout
    _arrow(ax, 8.45, 1.50, 9.17, 1.50,  color=SIGNAL)          # readout → MLP
    _arrow(ax, 11.50, 1.50, 12.17, 1.50, color="#6B4F8E")      # MLP → output

    # ── coarsening zoom annotation ────────────────────────────────────────────
    ax.annotate("", xy=(1.35, 4.35), xytext=(7.3, 3.00-0.01),
        arrowprops=dict(arrowstyle="-", color=WARM, lw=1.0,
                        linestyle="dashed", connectionstyle="arc3,rad=0.3"),
        zorder=1)

    # ── title ─────────────────────────────────────────────────────────────────
    ax.text(W/2, 7.78,
            "Hierarchical GNN — Two-Level Architecture",
            ha="center", va="center", fontsize=13,
            fontweight="bold", color=DEEP_BLUE)

    # ── bottom training config bar ────────────────────────────────────────────
    _rbox(ax, 0.15, 0.02, W-0.3, 0.46, "#F6F8FA", GRAY, lw=0.8, alpha=0.7)
    ax.text(W/2, 0.25,
            "Training: Adam lr=5e-4  ·  batch=96  ·  dropout=0.3  ·  "
            "inverse_sqrt class weights (29× ratio)  ·  40 epochs  ·  seed=42",
            ha="center", va="center", fontsize=7.5, color="#555")

    out = FIG / "architecture.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    fig_architecture()
    fig_patch_sweep()
    fig_ablations()
    fig_training_curve()
    fig_patch_trend()
    table_patch_sweep()
    table_ablations()
    print("\nAll assets generated.")
