#!/usr/bin/env python3
"""Generate all figures for the protein classification report (SHREC25 study)."""

import json, csv, collections, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ      = Path("/home/ogharbi/projects/protein_classification")
RUNS      = PROJ / "runs"
OUT       = PROJ / "report/figures"
GRAPHS    = Path("/home/ogharbi/datasets/SHREC25/graphs")
OUT.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':          10,
    'axes.titlesize':     11,
    'axes.labelsize':     10,
    'legend.fontsize':     8,
    'xtick.labelsize':     8,
    'ytick.labelsize':     8,
    'figure.dpi':         150,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

C = plt.get_cmap('tab10').colors
FEAT_COLORS = {
    '8D (all)':        C[0],
    '7D (-NormPot)':   C[1],
    '4D (xyz+Pot)':    C[2],
    '7D (-Pot)':       C[3],
    '6D (xyz+Nrm)':    C[4],
}
FEAT_ORDER = ['8D (all)', '7D (-NormPot)', '4D (xyz+Pot)', '7D (-Pot)', '6D (xyz+Nrm)']

# ── Hardcoded analysis data ───────────────────────────────────────────────────

# VTK vertex-level KS (10k subsample) — all 8 features incl. normalised xyz
KS = {
    'Potential':        {(61,90):0.0313, (70,56):0.1915, (86,14):0.2503, (8,5):0.1903},
    'NormPotential':    {(61,90):0.0092, (70,56):0.0291, (86,14):0.0405, (8,5):0.0253},
    'n_x':              {(61,90):0.0080, (70,56):0.0130, (86,14):0.0107, (8,5):0.0098},
    'n_y':              {(61,90):0.0115, (70,56):0.0183, (86,14):0.0150, (8,5):0.0176},
    'n_z':              {(61,90):0.0136, (70,56):0.0187, (86,14):0.0093, (8,5):0.0120},
    'x_norm':           {(61,90):0.0262, (70,56):0.0154, (86,14):0.0108, (8,5):0.0462},
    'y_norm':           {(61,90):0.0172, (70,56):0.0288, (86,14):0.0255, (8,5):0.0313},
    'z_norm':           {(61,90):0.0342, (70,56):0.0434, (86,14):0.0283, (8,5):0.0398},
}

# VTK Sliced Wasserstein (full-res, 5k subsample) — added Coords and All-8D
SW_VTK = {
    (61,90): {'Potential':0.0882, 'NormPot':0.0272, 'Normals':0.0234, 'Coords':0.0151, 'All-5D':0.0418},
    (70,56): {'Potential':0.4298, 'NormPot':0.0497, 'Normals':0.0320, 'Coords':0.0283, 'All-5D':0.1702},
    (86,14): {'Potential':0.4827, 'NormPot':0.0464, 'Normals':0.0169, 'Coords':0.0223, 'All-5D':0.1893},
    (8,5):   {'Potential':0.3615, 'NormPot':0.0573, 'Normals':0.0271, 'Coords':0.0322, 'All-5D':0.1182},
}
SW = SW_VTK  # backwards-compat alias

# Graph-level SW (1k-node SSL graphs): shape vs features vs joint
SW_GRAPH = {
    (61,90): {'xyz':0.0416, 'feat':0.0272, 'joint':0.0303},
    (70,56): {'xyz':0.0433, 'feat':0.1041, 'joint':0.1082},
    (86,14): {'xyz':0.0291, 'feat':0.1397, 'joint':0.1439},
    (8,5):   {'xyz':0.0629, 'feat':0.0613, 'joint':0.0669},
}

# SVM balanced accuracy (protein-level 30D stats)
SVM_GRAPH = {(61,90):0.543, (70,56):0.573, (86,14):0.665, (8,5):0.660, (74,25):0.829, (16,62):0.713}
SVM_VTK   = {(61,90):0.563, (70,56):0.598, (86,14):0.699, (8,5):0.725, (74,25):0.853, (16,62):0.728}

# Patch-SVM (512D) vs Global-SVM (32D) on graphs
SVM_PATCH512 = {(61,90):0.508, (70,56):0.570, (86,14):0.550, (8,5):0.524}

# Moran's I |Delta I| per feature per pair (graph-level)
MORAN_DI = {
    'Potential':    {(61,90):0.048, (70,56):0.020, (86,14):0.112, (8,5):0.073},
    'NormPotential':{(61,90):0.032, (70,56):0.007, (86,14):0.024, (8,5):0.035},
    'Normals (ny)': {(61,90):0.035, (70,56):0.018, (86,14):0.038, (8,5):0.107},
    'Coords':       {(61,90):0.003, (70,56):0.003, (86,14):0.009, (8,5):0.008},
}

# Feature ablation test metrics (proper eval over 88 classes)
ABLATION = {
    '8D (all)':      {'acc':0.8533,'bacc':0.7997,'mf1':0.7983,'wf1':0.8532},
    '7D (-NormPot)': {'acc':0.8486,'bacc':0.7966,'mf1':0.7896,'wf1':0.8479},
    '4D (xyz+Pot)':  {'acc':0.8447,'bacc':0.7955,'mf1':0.7839,'wf1':0.8434},
    '7D (-Pot)':     {'acc':0.8079,'bacc':0.6813,'mf1':0.6802,'wf1':0.7993},
    '6D (xyz+Nrm)':  {'acc':0.7828,'bacc':0.6123,'mf1':0.6190,'wf1':0.7602},
}

# Per-class F1 on hard classes (from eval_ablation.py)
HARD_CLASSES = [61, 90, 70, 56, 86, 14, 80]
PER_CLASS_F1 = {
    '8D (all)':      {61:0.576, 90:0.850, 70:0.532, 56:0.775, 86:0.667, 14:0.729, 80:0.174},
    '7D (-NormPot)': {61:0.474, 90:0.850, 70:0.571, 56:0.760, 86:0.690, 14:0.690, 80:0.333},
    '4D (xyz+Pot)':  {61:0.435, 90:0.832, 70:0.509, 56:0.726, 86:0.643, 14:0.754, 80:0.091},
    '7D (-Pot)':     {61:0.321, 90:0.870, 70:0.392, 56:0.708, 86:0.529, 14:0.634, 80:0.174},
    '6D (xyz+Nrm)':  {61:0.300, 90:0.860, 70:0.203, 56:0.743, 86:0.486, 14:0.560, 80:0.000},
}

# Backbone ablation metrics (proper eval, 4D features)
BACKBONE = {
    'GIN':       {'acc':0.8447,'bacc':0.7955,'mf1':0.7839,'wf1':0.8434},
    'SGFormer':  {'acc':0.8295,'bacc':0.7826,'mf1':0.7627,'wf1':0.8310},
    'GAT':       {'acc':0.8390,'bacc':0.7833,'mf1':0.7739,'wf1':0.8343},
    'GATv2':     {'acc':0.8360,'bacc':0.7865,'mf1':0.7741,'wf1':0.8327},
    'GCN':       {'acc':0.8291,'bacc':0.7679,'mf1':0.7543,'wf1':0.8260},
    'GPS':       {'acc':0.8265,'bacc':0.7719,'mf1':0.7615,'wf1':0.8232},
    'GraphSAGE': {'acc':0.8230,'bacc':0.7750,'mf1':0.7519,'wf1':0.8216},
    'PNA':       {'acc':0.7884,'bacc':0.5640,'mf1':0.5530,'wf1':0.7708},
}

# External comparison methods
EXTERN = {'PointMamba': 0.720, 'Mamba3D': 0.672, 'Flat GIN h512': 0.640}

# ── Geometric feature study (from ssl_graphs_1k_geom_enriched, 13 new feats) ─
GEOM_NAMES_SHORT = [
    'dist_cen', 'dist_anc0', 'dist_anc1', 'dist_anc2', 'dist_anc3',
    'mean_nbr', 'std_nbr', 'inv_nbr', 'lap_mag',
    'lap_sign', 'norm_var', 'planar', 'surf_var'
]
GEOM_NAMES_FULL = [
    'dist_centroid', 'dist_anchor_0', 'dist_anchor_1', 'dist_anchor_2', 'dist_anchor_3',
    'mean_nbr_dist', 'std_nbr_dist', 'inv_mean_nbr', 'lap_curv_mag',
    'lap_curv_sign', 'norm_variation', 'planarity', 'surf_variation'
]
KS_GEOM = {
    'dist_centroid':  {(61,90):0.0106,(70,56):0.0090,(86,14):0.0126,(8,5):0.0357},
    'dist_anchor_0':  {(61,90):0.0037,(70,56):0.0045,(86,14):0.0164,(8,5):0.0235},
    'dist_anchor_1':  {(61,90):0.0051,(70,56):0.0071,(86,14):0.0105,(8,5):0.0322},
    'dist_anchor_2':  {(61,90):0.0069,(70,56):0.0050,(86,14):0.0107,(8,5):0.0080},
    'dist_anchor_3':  {(61,90):0.0046,(70,56):0.0094,(86,14):0.0088,(8,5):0.0173},
    'mean_nbr_dist':  {(61,90):0.0037,(70,56):0.0123,(86,14):0.0038,(8,5):0.0128},
    'std_nbr_dist':   {(61,90):0.0057,(70,56):0.0112,(86,14):0.0057,(8,5):0.0212},
    'inv_mean_nbr':   {(61,90):0.0026,(70,56):0.0069,(86,14):0.0046,(8,5):0.0044},
    'lap_curv_mag':   {(61,90):0.0034,(70,56):0.0052,(86,14):0.0035,(8,5):0.0083},
    'lap_curv_sign':  {(61,90):0.0039,(70,56):0.0044,(86,14):0.0062,(8,5):0.0082},
    'norm_variation': {(61,90):0.0018,(70,56):0.0039,(86,14):0.0047,(8,5):0.0125},
    'planarity':      {(61,90):0.0012,(70,56):0.0015,(86,14):0.0059,(8,5):0.0056},
    'surf_variation': {(61,90):0.0078,(70,56):0.0088,(86,14):0.0221,(8,5):0.0268},
}
SW_GEOM = {
    (61,90): {'8D (exist.)':0.0118,'13D (geom)':0.0156,'21D (all)':0.0160},
    (70,56): {'8D (exist.)':0.1364,'13D (geom)':0.0223,'21D (all)':0.0832},
    (86,14): {'8D (exist.)':0.1879,'13D (geom)':0.0201,'21D (all)':0.1157},
    (8,5):   {'8D (exist.)':0.0548,'13D (geom)':0.0408,'21D (all)':0.0369},
}
MORAN_GEOM_MEAN = {
    'dist_centroid':0.9659,'dist_anchor_0':0.9926,'dist_anchor_1':0.9914,
    'dist_anchor_2':0.9856,'dist_anchor_3':0.9830,'mean_nbr_dist':0.3152,
    'std_nbr_dist':0.1491,'inv_mean_nbr':0.3082,'lap_curv_mag':0.1229,
    'lap_curv_sign':0.3089,'norm_variation':0.4659,'planarity':0.1451,
    'surf_variation':0.2311,'Potential (ref)':0.7661,
}
MORAN_DI_GEOM = {
    'dist_centroid':  {(61,90):0.0159,(70,56):0.0033,(86,14):0.0021,(8,5):0.0117},
    'dist_anchor_0':  {(61,90):0.0053,(70,56):0.0024,(86,14):0.0010,(8,5):0.0035},
    'dist_anchor_1':  {(61,90):0.0081,(70,56):0.0039,(86,14):0.0028,(8,5):0.0026},
    'dist_anchor_2':  {(61,90):0.0004,(70,56):0.0054,(86,14):0.0027,(8,5):0.0032},
    'dist_anchor_3':  {(61,90):0.0025,(70,56):0.0030,(86,14):0.0049,(8,5):0.0014},
    'mean_nbr_dist':  {(61,90):0.0416,(70,56):0.0106,(86,14):0.0098,(8,5):0.0134},
    'std_nbr_dist':   {(61,90):0.0319,(70,56):0.1166,(86,14):0.0126,(8,5):0.0373},
    'inv_mean_nbr':   {(61,90):0.0415,(70,56):0.0075,(86,14):0.0138,(8,5):0.0072},
    'lap_curv_mag':   {(61,90):0.1157,(70,56):0.1045,(86,14):0.0323,(8,5):0.0210},
    'lap_curv_sign':  {(61,90):0.0887,(70,56):0.1097,(86,14):0.0158,(8,5):0.0068},
    'norm_variation': {(61,90):0.0087,(70,56):0.0321,(86,14):0.0301,(8,5):0.0053},
    'planarity':      {(61,90):0.1024,(70,56):0.1172,(86,14):0.0177,(8,5):0.0026},
    'surf_variation': {(61,90):0.0728,(70,56):0.1075,(86,14):0.0140,(8,5):0.0219},
}
SVM_GEOM = {
    (61,90): {'8D (exist.)':0.5507,'13D (geom)':0.5731,'21D (all)':0.6360},
    (70,56): {'8D (exist.)':0.6166,'13D (geom)':0.5290,'21D (all)':0.5933},
    (86,14): {'8D (exist.)':0.6686,'13D (geom)':0.6706,'21D (all)':0.7309},
    (8,5):   {'8D (exist.)':0.6921,'13D (geom)':0.6382,'21D (all)':0.7575},
}

# ── Curvature feature study (from ssl_graphs_1k_curvature_enriched) ─────────
# KS test vertex-level on the 4 new curvature features
KS_CURV = {
    'k_min':        {(61,90):0.0052, (70,56):0.0084, (86,14):0.0134, (8,5):0.0134},
    'k_max':        {(61,90):0.0062, (70,56):0.0118, (86,14):0.0080, (8,5):0.0171},
    'H (mean)':     {(61,90):0.0025, (70,56):0.0051, (86,14):0.0083, (8,5):0.0122},
    'K (gaussian)': {(61,90):0.0264, (70,56):0.0318, (86,14):0.0429, (8,5):0.0342},
}
# Sliced Wasserstein: existing 8D vs 4D curvature vs 12D joint
SW_CURV = {
    (61,90): {'8D (exist.)':0.0118, '4D (curv)':0.0178, '12D (all)':0.0140},
    (70,56): {'8D (exist.)':0.1359, '4D (curv)':0.0251, '12D (all)':0.1043},
    (86,14): {'8D (exist.)':0.1715, '4D (curv)':0.0311, '12D (all)':0.1370},
    (8,5):   {'8D (exist.)':0.0604, '4D (curv)':0.0411, '12D (all)':0.0411},
}
# Moran's I mean over all study classes
MORAN_CURV_MEAN = {'k_min':0.2518, 'k_max':0.2669, 'H (mean)':0.3182, 'K (gaussian)':0.0875,
                   'Potential (ref)':0.7661}
# |Delta I| per curvature feature per pair
MORAN_DI_CURV = {
    'k_min':        {(61,90):0.1016, (70,56):0.1131, (86,14):0.0133, (8,5):0.0270},
    'k_max':        {(61,90):0.1208, (70,56):0.0732, (86,14):0.0178, (8,5):0.0052},
    'H (mean)':     {(61,90):0.1238, (70,56):0.0939, (86,14):0.0132, (8,5):0.0070},
    'K (gaussian)': {(61,90):0.1234, (70,56):0.0961, (86,14):0.0292, (8,5):0.0132},
}
# Protein-level SVM balanced accuracy: 8D vs 4D curvature vs 12D
SVM_CURV = {
    (61,90): {'8D (exist.)':0.5507, '4D (curv)':0.5042, '12D (all)':0.5662},
    (70,56): {'8D (exist.)':0.6166, '4D (curv)':0.5081, '12D (all)':0.6149},
    (86,14): {'8D (exist.)':0.6686, '4D (curv)':0.5427, '12D (all)':0.6884},
    (8,5):   {'8D (exist.)':0.6921, '4D (curv)':0.5187, '12D (all)':0.7122},
}


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Class Distribution
# ═══════════════════════════════════════════════════════════════════════════════
def fig_class_distribution():
    # Read all 99 classes from the full graphs label file
    train_counts = {}
    with open(GRAPHS / "train_labels.csv") as f:
        for row in csv.DictReader(f):
            c = int(row['class_id'])
            train_counts[c] = train_counts.get(c,0) + 1

    # Sort by count
    pairs = sorted(train_counts.items(), key=lambda x: -x[1])
    classes, counts = zip(*pairs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                              gridspec_kw={'width_ratios':[3,1]})

    # Left: all classes sorted by count
    ax = axes[0]
    x = np.arange(len(classes))
    bars = ax.bar(x, counts, width=0.8, color=C[0], alpha=0.75, linewidth=0)
    # Highlight top-5 and hard confusable classes
    hard = {61, 90, 70, 56, 86, 14, 8}
    for i, cls in enumerate(classes):
        if cls in hard:
            bars[i].set_color(C[3])
            bars[i].set_alpha(0.9)
    ax.set_xlabel('Class rank (sorted by training count)')
    ax.set_ylabel('Training samples')
    ax.set_title('Class distribution (99 classes, 9244 proteins)')
    ax.set_yscale('log')
    ax.axhline(counts[-1], color='grey', ls='--', lw=0.8)
    ax.axhline(counts[0],  color='black', ls='--', lw=0.8)
    ax.text(0.98, 0.60, f'Imbalance ratio\n{counts[0]:,} : {counts[-1]} = {counts[0]//counts[-1]:,}×',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    legend_handles = [
        mpatches.Patch(facecolor=C[0], alpha=0.75, label='Regular class'),
        mpatches.Patch(facecolor=C[3], alpha=0.90, label='Hard/confusable class'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8)

    # Right: CDF of counts
    ax2 = axes[1]
    sorted_c = sorted(counts)
    cdf = np.arange(1, len(sorted_c)+1) / len(sorted_c)
    ax2.step(sorted_c, cdf, color=C[0], lw=1.5)
    ax2.set_xlabel('Training samples per class')
    ax2.set_ylabel('Cumulative fraction')
    ax2.set_title('Sample CDF')
    ax2.set_xscale('log')
    ax2.axvline(10, color='grey', ls='--', lw=0.8, label='n=10')
    # Find fraction of classes with ≤N samples
    for th in [5, 10, 20]:
        frac = sum(1 for c in counts if c <= th) / len(counts)
        ax2.annotate(f'{100*frac:.0f}% ≤ {th}',
                     xy=(th, frac), xytext=(th*2.5, frac-0.08),
                     fontsize=7, arrowprops=dict(arrowstyle='->', lw=0.8))
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT / 'class_distribution.pdf')
    plt.close(fig)
    print('  [1] class_distribution.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Training curves — backbone comparison
# ═══════════════════════════════════════════════════════════════════════════════
def fig_training_curves_all():
    bb_runs = {
        'GIN':       'hier_bb_gin_cosine',
        'SGFormer':  'hier_bb_sgformer_cosine',
        'GATv2':     'hier_bb_gatv2_cosine',
        'GAT':       'hier_bb_gat_cosine',
        'GCN':       'hier_bb_gcn_cosine',
        'GraphSAGE': 'hier_bb_sage_cosine',
        'GPS':       'hier_bb_gps_cosine',
        'PNA':       'hier_bb_pna_cosine',
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for i, (label, rname) in enumerate(bb_runs.items()):
        f = RUNS / rname / 'final_results.json'
        if not f.exists(): continue
        d = json.load(open(f))
        hist = d['history']
        epochs = [h['epoch'] for h in hist]
        val_mf1  = [h['val']['f1_macro'] for h in hist]
        train_mf1= [h['train']['f1_macro'] for h in hist]
        col = C[i % 10]
        axes[0].plot(epochs, val_mf1,   color=col, lw=1.5, label=label)
        axes[1].plot(epochs, train_mf1, color=col, lw=1.5, label=label)

    for ax, title in zip(axes, ['Validation macro-F1', 'Train macro-F1']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Macro-F1 (99-class eval)')
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 80)
        ax.set_ylim(0, 0.75)

    plt.suptitle('Training dynamics — node backbone ablation (4D features, h=512)', fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / 'training_curves_all.pdf')
    plt.close(fig)
    print('  [2] training_curves_all.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Training curves — feature ablation
# ═══════════════════════════════════════════════════════════════════════════════
def fig_training_curves_features():
    feat_runs = [
        ('8D (all)',      'hier_gin_h512_all8d'),
        ('7D (-NormPot)', 'hier_gin_h512_7d_no_normpot'),
        ('4D (xyz+Pot)',  'hier_bb_gin_cosine'),
        ('7D (-Pot)',     'hier_gin_h512_7d_no_pot'),
        ('6D (xyz+Nrm)',  'hier_gin_h512_coords_normals'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for label, rname in feat_runs:
        f = RUNS / rname / 'final_results.json'
        if not f.exists(): continue
        d = json.load(open(f))
        hist = d['history']
        epochs   = [h['epoch'] for h in hist]
        val_mf1  = [h['val']['f1_macro'] for h in hist]
        val_acc  = [h['val']['accuracy'] for h in hist]
        col = FEAT_COLORS[label]
        ls  = '--' if 'no Pot' in label or '-Pot' in label else '-'
        axes[0].plot(epochs, val_mf1, color=col, lw=1.8, ls=ls, label=label)
        axes[1].plot(epochs, val_acc, color=col, lw=1.8, ls=ls, label=label)

    for ax, metric in zip(axes, ['Validation macro-F1 (99-class)', 'Validation accuracy']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 80)

    plt.suptitle('Training dynamics — feature ablation (HierGIN h=512, GIN backbone)', fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / 'training_curves_features.pdf')
    plt.close(fig)
    print('  [3] training_curves_features.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Architecture diagram
# ═══════════════════════════════════════════════════════════════════════════════
def fig_architecture_diagram():
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)
    ax.axis('off')

    def box(x, y, w, h, label, sublabel='', fc='#dce8f5', ec='#3a6ea5', fs=9, subfs=7.5):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                        boxstyle='round,pad=0.05',
                                        fc=fc, ec=ec, lw=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.1 if sublabel else 0), label,
                ha='center', va='center', fontsize=fs, fontweight='bold', wrap=True)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.25, sublabel,
                    ha='center', va='center', fontsize=subfs, color='#555')

    def arrow(x1, x2, y=2.0):
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.3))

    # Data pipeline row (top)
    ax.text(0.15, 3.55, 'Input', fontsize=8, color='grey')
    box(0.1, 2.8, 1.0, 0.65, 'VTK Mesh', '~10,676 verts', fc='#fff2cc', ec='#c4a000', fs=8.5)
    arrow(1.1, 1.6, 3.12)
    box(1.6, 2.8, 1.3, 0.65, 'FPS 1k', '+kNN-6 edges', fc='#fff2cc', ec='#c4a000', fs=8.5)
    arrow(2.9, 3.4, 3.12)
    box(3.4, 2.8, 1.5, 0.65, '8D Node feat.', '[Pot, NPot, nx,ny,nz, x,y,z]', fc='#fff2cc', ec='#c4a000', fs=8.5, subfs=7)
    ax.text(0.15, 2.72, 'Pre-processing', fontsize=7, color='grey')

    # Model row (middle)
    ax.text(0.15, 2.3, 'Model', fontsize=8, color='grey')
    box(0.1, 1.45, 1.3, 0.75, 'Node GIN', '×3 layers\nh=512', fc='#dce8f5', ec='#3a6ea5')
    arrow(1.4, 1.9, 1.82)
    box(1.9, 1.45, 1.6, 0.75, 'Geodesic\nPatch Pool', 'k=64 patches\nattn weights', fc='#e8d5f5', ec='#7030a0', fs=8.5)
    arrow(3.5, 4.0, 1.82)
    box(4.0, 1.45, 1.3, 0.75, 'Patch GIN', '×2 layers\nh=512', fc='#dce8f5', ec='#3a6ea5')
    arrow(5.3, 5.8, 1.82)
    box(5.8, 1.45, 1.2, 0.75, 'Max\nReadout', 'global pool\n512D', fc='#d5e8d4', ec='#2d7a2d')
    arrow(7.0, 7.5, 1.82)
    box(7.5, 1.45, 1.1, 0.75, 'MLP\nHead', '512→256→99', fc='#d5e8d4', ec='#2d7a2d')
    arrow(8.6, 9.1, 1.82)
    box(9.1, 1.45, 1.0, 0.75, 'Softmax\n99 classes', '', fc='#ffe6cc', ec='#d67900')

    # Feature vector annotation
    ax.annotate('', xy=(0.1, 1.82), xytext=(4.9, 2.8),
                arrowprops=dict(arrowstyle='->', color='#555', lw=0.9, ls='dashed'))
    ax.text(1.9, 2.52, '1000 nodes × 8D', fontsize=7.5, color='#555', rotation=22)

    # Patch assign path
    ax.text(3.4, 3.55, 'Voronoi', fontsize=7.5, color='#7030a0')
    ax.annotate('', xy=(2.7, 2.2), xytext=(4.9, 2.8),
                arrowprops=dict(arrowstyle='->', color='#7030a0', lw=0.9, ls='dotted'))

    # Loss annotation
    ax.text(9.5, 0.85, 'Cross-entropy loss\n(inverse-sqrt class weights)',
            fontsize=7.5, ha='center', color='#aa0000',
            bbox=dict(fc='#fff0f0', ec='#cc0000', pad=3, boxstyle='round'))
    ax.annotate('', xy=(9.6, 1.45), xytext=(9.6, 1.1),
                arrowprops=dict(arrowstyle='<->', color='#aa0000', lw=0.9))

    ax.set_title('HierGIN Architecture: Node GIN → Geodesic Voronoi Patch Pooling → Patch GIN → Classification',
                  fontsize=10, pad=8)
    plt.tight_layout()
    fig.savefig(OUT / 'architecture_diagram.pdf')
    plt.close(fig)
    print('  [4] architecture_diagram.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Separability — KS test
# ═══════════════════════════════════════════════════════════════════════════════
def fig_separability_ks():
    pairs       = [(61,90), (70,56), (86,14), (8,5)]
    pair_labels = ['(61,90)\nconfusable', '(70,56)\nconfusable', '(86,14)\nconfusable', '(8,5)\ncontrol']
    # All 8 features, split into two rows for clarity
    feat_groups = [
        ('Electrostatics', ['Potential', 'NormPotential']),
        ('Normals',        ['n_x', 'n_y', 'n_z']),
        ('Coords (norm.)', ['x_norm', 'y_norm', 'z_norm']),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    for ax, (grp_name, feats) in zip(axes, feat_groups):
        x = np.arange(len(pairs))
        n_f = len(feats)
        w = 0.22
        for i, feat in enumerate(feats):
            vals = [KS[feat].get(p, 0) for p in pairs]
            offset = (i - n_f/2 + 0.5) * w
            ax.bar(x + offset, vals, w, color=C[i], alpha=0.85, label=feat)
        ax.set_xticks(x); ax.set_xticklabels(pair_labels, fontsize=7.5)
        ax.set_title(grp_name, fontsize=10)
        ax.set_ylabel('KS statistic' if ax is axes[0] else '')
        ax.legend(fontsize=7.5); ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylim(0, 0.32)
    axes[1].set_ylim(0, 0.032)
    axes[2].set_ylim(0, 0.055)
    fig.suptitle('KS separability on VTK full-resolution meshes (all 8 features)\n'
                 'Potential dominates; normalised coordinates are nearly uniform across classes',
                 fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT / 'separability_ks.pdf')
    plt.close(fig)
    print('  [5] separability_ks.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6: Separability — SVM balanced accuracy
# ═══════════════════════════════════════════════════════════════════════════════
def fig_separability_svm():
    pairs       = [(61,90), (70,56), (86,14), (8,5), (74,25), (16,62)]
    pair_labels = ['(61,90)\nconfusable', '(70,56)\nconfusable', '(86,14)\nconfusable',
                   '(8,5)\ncontrol',      '(74,25)\ncontrol',    '(16,62)\ncontrol']
    pair_types  = ['confusable','confusable','confusable','control','control','control']

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(pairs))
    w = 0.35

    bar1 = ax.bar(x - w/2, [SVM_GRAPH.get(p,0) for p in pairs], w,
                  color=C[0], alpha=0.8, label='1k graph (FPS subsampled)')
    bar2 = ax.bar(x + w/2, [SVM_VTK.get(p,0)   for p in pairs], w,
                  color=C[1], alpha=0.8, label='Full-res VTK mesh')

    ax.axhline(0.5, color='red', ls='--', lw=1.2, label='Random baseline (0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=8)
    ax.set_ylabel('SVM balanced accuracy (protein-level, 5-fold CV)')
    ax.set_title('Pairwise separability: SVM on global statistics (30D per protein)\n'
                 'Confusable pairs remain hard even at full VTK resolution')
    ax.legend(fontsize=8)
    ax.set_ylim(0.4, 0.95)
    ax.grid(axis='y', alpha=0.3)

    # Shade confusable vs control
    ax.axvspan(-0.5, 2.5, alpha=0.06, color='red')
    ax.axvspan(2.5, 5.5, alpha=0.06, color='green')
    ax.text(1.0,  0.42, 'Confusable pairs', ha='center', fontsize=8, color='darkred')
    ax.text(4.0,  0.42, 'Control pairs',    ha='center', fontsize=8, color='darkgreen')

    plt.tight_layout()
    fig.savefig(OUT / 'separability_svm.pdf')
    plt.close(fig)
    print('  [6] separability_svm.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7: Separability — Sliced Wasserstein
# ═══════════════════════════════════════════════════════════════════════════════
def fig_separability_sw():
    pairs       = [(61,90), (70,56), (86,14), (8,5)]
    pair_labels = ['(61,90)\nconfusable', '(70,56)\nconfusable', '(86,14)\nconfusable', '(8,5)\ncontrol']
    components  = ['Potential', 'NormPot', 'Normals', 'Coords', 'All-5D']
    n_comp      = len(components)
    x           = np.arange(len(pairs))
    w           = 0.15

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (comp, col) in enumerate(zip(components, C)):
        vals = [SW_VTK[p][comp] for p in pairs]
        offset = (i - n_comp/2 + 0.5) * w
        ax.bar(x + offset, vals, w, color=col, alpha=0.85, label=comp)

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=8)
    ax.set_ylabel('Sliced Wasserstein distance')
    ax.set_title('Sliced Wasserstein distance per feature component — VTK full-resolution\n'
                 'Coords (normalised xyz) SW is low for all pairs, confirming shape alone does not discriminate')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.02, 0.97, 'SW(Pot) for (61,90) = 0.088\n<< controls (0.36-0.72)\n=> near-inseparable pair',
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(fc='#fff0e0', ec='orange', pad=3, boxstyle='round', alpha=0.85))

    plt.tight_layout()
    fig.savefig(OUT / 'separability_sw.pdf')
    plt.close(fig)
    print('  [7] separability_sw.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 8: Feature ablation — radar chart
# ═══════════════════════════════════════════════════════════════════════════════
def fig_feature_ablation_radar():
    metrics    = ['Accuracy', 'Bal.Acc', 'Macro-F1', 'Wtd-F1']
    metric_keys= ['acc', 'bacc', 'mf1', 'wf1']
    n_metrics  = len(metrics)
    angles     = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
    angles    += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for label in FEAT_ORDER:
        vals = [ABLATION[label][k] for k in metric_keys]
        vals += vals[:1]
        ax.plot(angles, vals, color=FEAT_COLORS[label], lw=2, label=label)
        ax.fill(angles, vals, color=FEAT_COLORS[label], alpha=0.07)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0.55, 0.90)
    ax.set_yticks([0.60, 0.70, 0.80])
    ax.set_yticklabels(['0.60','0.70','0.80'], fontsize=7)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    ax.set_title('Feature ablation: test metrics\n(HierGIN h=512, 88 evaluated classes)',
                  pad=20, fontsize=10)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    fig.savefig(OUT / 'feature_ablation_radar.pdf')
    plt.close(fig)
    print('  [8] feature_ablation_radar.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 9: Backbone ablation bar chart
# ═══════════════════════════════════════════════════════════════════════════════
def fig_backbone_ablation():
    bb_order = sorted(BACKBONE.keys(), key=lambda k: -BACKBONE[k]['mf1'])
    labels   = bb_order
    mf1_vals = [BACKBONE[k]['mf1']  for k in bb_order]
    bacc_vals= [BACKBONE[k]['bacc'] for k in bb_order]
    acc_vals = [BACKBONE[k]['acc']  for k in bb_order]

    fig, ax = plt.subplots(figsize=(9, 4))
    x  = np.arange(len(labels))
    w  = 0.25

    ax.bar(x - w,   acc_vals,  w, color=C[0], alpha=0.85, label='Accuracy')
    ax.bar(x,       bacc_vals, w, color=C[1], alpha=0.85, label='Bal. Accuracy')
    ax.bar(x + w,   mf1_vals,  w, color=C[2], alpha=0.85, label='Macro-F1')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('Node backbone ablation (4D features, h=512, cosine LR, 80 epochs)\n'
                 'GIN achieves best macro-F1; PNA underfits')
    ax.legend(fontsize=8)
    ax.set_ylim(0.45, 0.90)
    ax.grid(axis='y', alpha=0.3)

    # Annotate best
    best_i = mf1_vals.index(max(mf1_vals))
    ax.annotate(f'Best\n{max(mf1_vals):.3f}',
                xy=(x[best_i]+w, max(mf1_vals)),
                xytext=(x[best_i]+w+0.4, max(mf1_vals)+0.02),
                fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.8))

    plt.tight_layout()
    fig.savefig(OUT / 'backbone_ablation.pdf')
    plt.close(fig)
    print('  [9] backbone_ablation.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 10: Per-class F1 on hard classes
# ═══════════════════════════════════════════════════════════════════════════════
def fig_per_class_f1_hard():
    hard = HARD_CLASSES
    n_cls = len(hard)
    n_cfg = len(FEAT_ORDER)
    x = np.arange(n_cls)
    w = 0.13
    offsets = np.linspace(-(n_cfg-1)/2, (n_cfg-1)/2, n_cfg) * w

    fig, ax = plt.subplots(figsize=(12, 5))

    for j, label in enumerate(FEAT_ORDER):
        vals = [PER_CLASS_F1[label].get(c, 0) for c in hard]
        ax.bar(x + offsets[j], vals, w, color=FEAT_COLORS[label], alpha=0.85, label=label)

    # Add PointMamba and Mamba3D macro-F1 as reference dashed lines (not per-class, just overall)
    ax.axhline(0.0, color='white', lw=0)  # placeholder

    ax.set_xticks(x)
    ax.set_xticklabels([f'cls {c}' for c in hard], fontsize=9)
    ax.set_ylabel('F1 score')
    ax.set_ylim(0, 1.0)
    ax.set_title('Per-class F1 for hard/confusable classes across feature configurations\n'
                 '(cls 61/90 = near-inseparable pair; cls 70/56, 86/14 = partial confusion)')
    ax.legend(fontsize=8, ncol=3, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Pair brackets
    pair_span = [(0, 1, '61↔90\nnear-ins.'), (2, 3, '70↔56'), (4, 5, '86↔14')]
    y_br = 0.96
    for i1, i2, txt in pair_span:
        ax.annotate('', xy=(x[i2]+w*1.2, y_br), xytext=(x[i1]-w*1.2, y_br),
                    arrowprops=dict(arrowstyle='<->', color='#555', lw=1.0))
        ax.text((x[i1]+x[i2])/2, y_br+0.02, txt,
                ha='center', fontsize=7.5, color='#555')

    plt.tight_layout()
    fig.savefig(OUT / 'per_class_f1_hard.pdf')
    plt.close(fig)
    print('  [10] per_class_f1_hard.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 11: Feature ablation macro-F1 summary bar chart
# ═══════════════════════════════════════════════════════════════════════════════
def fig_feature_ablation_summary():
    """Horizontal bar chart showing all key metrics for 5 feature configs + external."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: macro-F1 comparison (all methods)
    all_methods = {
        '6D (xyz+Nrm)':  0.6190,
        'Flat GIN h512': 0.6400,
        'Mamba3D':       0.6720,
        '7D (-Pot)':     0.6802,
        'PointMamba':    0.7200,
        '4D (xyz+Pot)':  0.7839,
        '7D (-NormPot)': 0.7896,
        '8D (all)':      0.7983,
    }
    methods = list(all_methods.keys())
    vals    = list(all_methods.values())
    colors  = []
    for m in methods:
        if m in FEAT_COLORS:      colors.append(FEAT_COLORS[m])
        elif m == 'PointMamba':   colors.append('#888888')
        elif m == 'Mamba3D':      colors.append('#aaaaaa')
        else:                     colors.append('#cccccc')

    ax = axes[0]
    bars = ax.barh(methods, vals, color=colors, alpha=0.85)
    ax.set_xlabel('Macro-F1')
    ax.set_title('Macro-F1 comparison (all methods)')
    ax.set_xlim(0.5, 0.85)
    ax.axvline(0.720, color='grey', ls='--', lw=0.8)
    ax.text(0.723, 0.5, 'PointMamba', fontsize=7, color='grey', rotation=90, va='bottom')
    for i, (bar, val) in enumerate(zip(bars, vals)):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8)
    ax.grid(axis='x', alpha=0.3)

    # Right: Feature ablation — all 4 metrics
    ax2 = axes[1]
    metric_labels = ['Accuracy', 'Bal.Acc', 'Macro-F1', 'Wtd-F1']
    metric_keys   = ['acc', 'bacc', 'mf1', 'wf1']
    x = np.arange(len(metric_labels))
    w = 0.15
    offsets = np.linspace(-(len(FEAT_ORDER)-1)/2, (len(FEAT_ORDER)-1)/2, len(FEAT_ORDER)) * w

    for j, label in enumerate(FEAT_ORDER):
        vals2 = [ABLATION[label][k] for k in metric_keys]
        ax2.bar(x + offsets[j], vals2, w, color=FEAT_COLORS[label], alpha=0.85, label=label)

    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_labels, fontsize=9)
    ax2.set_ylabel('Score')
    ax2.set_title('Feature ablation — all metrics (88-class proper eval)')
    ax2.legend(fontsize=7.5, ncol=2)
    ax2.set_ylim(0.55, 0.92)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT / 'feature_ablation_summary.pdf')
    plt.close(fig)
    print('  [11] feature_ablation_summary.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 12: Graph-level SW (xyz vs feat vs joint)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_separability_graph_sw():
    pairs       = [(61,90), (70,56), (86,14), (8,5)]
    pair_labels = ['(61,90)\nconfusable', '(70,56)\nconfusable', '(86,14)\nconfusable', '(8,5)\ncontrol']
    components  = ['xyz', 'feat', 'joint']
    comp_labels = ['SW(xyz)\nshape only', 'SW(feat)\nelectrostatics+normals', 'SW(joint)\nall 8D']
    comp_colors = [C[2], C[0], C[4]]
    x = np.arange(len(pairs))
    w = 0.22

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (comp, lbl, col) in enumerate(zip(components, comp_labels, comp_colors)):
        vals = [SW_GRAPH[p][comp] for p in pairs]
        offset = (i - 1) * w
        ax.bar(x + offset, vals, w, color=col, alpha=0.85, label=lbl)

    ax.set_xticks(x); ax.set_xticklabels(pair_labels, fontsize=8)
    ax.set_ylabel('Sliced Wasserstein distance')
    ax.set_title('Graph-level SW: 3D shape vs electrostatic features vs joint (1k-node SSL graphs)\n'
                 'SW(xyz) is uniformly low for all pairs — 3D shape alone carries no class signal')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    # Annotate key insight
    ax.text(0.98, 0.97, 'SW(xyz) low even for\neasily-separated controls\n=> shape alone insufficient',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(fc='#e8f4e8', ec='green', pad=3, boxstyle='round', alpha=0.85))

    plt.tight_layout()
    fig.savefig(OUT / 'separability_graph_sw.pdf')
    plt.close(fig)
    print('  [12] separability_graph_sw.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 13: Moran's I |Delta I| per feature
# ═══════════════════════════════════════════════════════════════════════════════
def fig_separability_moran():
    pairs       = [(61,90), (70,56), (86,14), (8,5)]
    pair_labels = ['(61,90)\nconfusable', '(70,56)\nconfusable', '(86,14)\nconfusable', '(8,5)\ncontrol']
    features    = list(MORAN_DI.keys())
    n_feat      = len(features)
    x           = np.arange(len(pairs))
    w           = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: |Delta I| bar chart
    ax = axes[0]
    for i, (feat, col) in enumerate(zip(features, C)):
        vals = [MORAN_DI[feat].get(p, 0) for p in pairs]
        offset = (i - n_feat/2 + 0.5) * w
        ax.bar(x + offset, vals, w, color=col, alpha=0.85, label=feat)
    ax.set_xticks(x); ax.set_xticklabels(pair_labels, fontsize=8)
    ax.set_ylabel('|Delta I| (Moran\'s I difference between classes)')
    ax.set_title('Spatial autocorrelation difference |Delta I|\n(higher = different spatial patterning)')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    # Right: mean Moran's I per feature (to show absolute autocorrelation level)
    ax2 = axes[1]
    mean_I = {
        'Potential':     0.630,
        'NormPotential': 0.071,
        'Normals (ny)':  0.390,
        'Coords':        0.962,
    }
    feat_names = list(mean_I.keys())
    vals2 = list(mean_I.values())
    colors2 = [C[i] for i in range(len(feat_names))]
    bars = ax2.barh(feat_names, vals2, color=colors2, alpha=0.85)
    ax2.set_xlabel("Mean Moran's I (across 12 classes)")
    ax2.set_title("Mean spatial autocorrelation per feature\n(high I = smooth spatial patterns on surface)")
    ax2.set_xlim(0, 1.05)
    for bar, val in zip(bars, vals2):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=8)
    ax2.grid(axis='x', alpha=0.3)
    ax2.text(0.98, 0.05, 'NormPotential is spatially\nrandom (I~0.07);\nexplains low SW(NormPot)',
             transform=ax2.transAxes, ha='right', va='bottom', fontsize=8,
             bbox=dict(fc='#f0f0ff', ec='blue', pad=3, boxstyle='round', alpha=0.8))

    plt.suptitle("Moran's I spatial autocorrelation analysis (graph-level)", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / 'separability_moran.pdf')
    plt.close(fig)
    print('  [13] separability_moran.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 14: Graph-level SVM: Global-32D vs Patch-512D
# ═══════════════════════════════════════════════════════════════════════════════
def fig_separability_graph_svm():
    pairs       = [(61,90), (70,56), (86,14), (8,5)]
    pair_labels = ['(61,90)\nconfusable', '(70,56)\nconfusable', '(86,14)\nconfusable', '(8,5)\ncontrol']
    x = np.arange(len(pairs))
    w = 0.30

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, [SVM_GRAPH[p] for p in pairs], w,
           color=C[0], alpha=0.85, label='Global stats SVM (32D: mean/std/etc per feature)')
    ax.bar(x + w/2, [SVM_PATCH512[p] for p in pairs], w,
           color=C[3], alpha=0.85, label='Patch histogram SVM (512D: spatial layout)')

    ax.axhline(0.5, color='red', ls='--', lw=1.2, label='Random baseline')
    ax.set_xticks(x); ax.set_xticklabels(pair_labels, fontsize=8)
    ax.set_ylabel('SVM balanced accuracy (5-fold CV)')
    ax.set_title('Graph-level SVM: global statistics vs spatial patch layout\n'
                 'Adding spatial information (Patch-512D) does not improve — and sometimes hurts')
    ax.legend(fontsize=8); ax.set_ylim(0.4, 0.75); ax.grid(axis='y', alpha=0.3)

    # Annotate delta
    for i, p in enumerate(pairs):
        delta = SVM_PATCH512[p] - SVM_GRAPH[p]
        ax.text(x[i], max(SVM_GRAPH[p], SVM_PATCH512[p]) + 0.012,
                f'{delta:+.3f}', ha='center', fontsize=7.5,
                color='darkred' if delta < 0 else 'darkgreen')

    plt.tight_layout()
    fig.savefig(OUT / 'separability_graph_svm.pdf')
    plt.close(fig)
    print('  [14] separability_graph_svm.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 15: Resolution comparison — graph vs VTK
# ═══════════════════════════════════════════════════════════════════════════════
def fig_separability_resolution():
    pairs       = [(61,90), (70,56), (86,14), (8,5), (74,25), (16,62)]
    pair_labels = ['(61,90)', '(70,56)', '(86,14)', '(8,5)', '(74,25)', '(16,62)']
    pair_types  = ['conf.','conf.','conf.','ctrl','ctrl','ctrl']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: SVM comparison
    ax = axes[0]
    x = np.arange(len(pairs)); w = 0.35
    ax.bar(x - w/2, [SVM_GRAPH.get(p, 0) for p in pairs], w,
           color=C[0], alpha=0.85, label='1k-node graph (FPS subsampled)')
    ax.bar(x + w/2, [SVM_VTK.get(p, 0)   for p in pairs], w,
           color=C[1], alpha=0.85, label='Full-res VTK mesh')
    ax.axhline(0.5, color='red', ls='--', lw=1.0, label='Chance')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}\n{t}' for l,t in zip(pair_labels, pair_types)], fontsize=7.5)
    ax.set_ylabel('SVM balanced accuracy')
    ax.set_title('Protein-level SVM:\n1k graph vs full-res VTK')
    ax.legend(fontsize=7.5); ax.set_ylim(0.4, 0.92); ax.grid(axis='y', alpha=0.3)
    # Delta annotations
    for i, p in enumerate(pairs):
        if p not in SVM_VTK or p not in SVM_GRAPH: continue
        delta = SVM_VTK[p] - SVM_GRAPH[p]
        ax.text(x[i], max(SVM_GRAPH[p], SVM_VTK[p]) + 0.01,
                f'+{delta:.2f}', ha='center', fontsize=7, color='grey')

    # Right: SW(feat) graph vs VTK
    ax2 = axes[1]
    pairs4 = [(61,90), (70,56), (86,14), (8,5)]
    pl4    = ['(61,90)\nconf.', '(70,56)\nconf.', '(86,14)\nconf.', '(8,5)\nctrl']
    x2 = np.arange(len(pairs4))
    ax2.bar(x2 - w/2, [SW_GRAPH[p]['feat'] for p in pairs4], w,
            color=C[0], alpha=0.85, label='Graph SW(feat) — 1k nodes')
    ax2.bar(x2 + w/2, [SW_VTK[p]['All-5D'] for p in pairs4], w,
            color=C[1], alpha=0.85, label='VTK SW(All-5D) — full res')
    ax2.set_xticks(x2); ax2.set_xticklabels(pl4, fontsize=8)
    ax2.set_ylabel('Sliced Wasserstein distance')
    ax2.set_title('SW(features) comparison:\n1k graph vs full-res VTK')
    ax2.legend(fontsize=7.5); ax2.grid(axis='y', alpha=0.3)
    ax2.text(0.02, 0.97, 'Both agree on pair ranking;\nVTK slightly higher (more vertices)',
             transform=ax2.transAxes, va='top', fontsize=8,
             bbox=dict(fc='#e8f4e8', ec='green', pad=3, boxstyle='round', alpha=0.85))

    plt.suptitle('Resolution comparison: 1k-node SSL graph preserves separability of full-resolution VTK',
                 fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT / 'separability_resolution.pdf')
    plt.close(fig)
    print('  [15] separability_resolution.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 16: Geometric feature separability study (21D dataset)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_geom_study():
    pairs      = [(61,90), (70,56), (86,14), (8,5)]
    plabels    = ['(61,90)\nconf.', '(70,56)\nconf.', '(86,14)\nconf.', '(8,5)\nctrl']

    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    # ── Panel 1 (top-left): KS heatmap — 13 features × 4 pairs ───────────────
    ax1 = fig.add_subplot(gs[0, 0])
    mat = np.array([[KS_GEOM[fn][p] for p in pairs] for fn in GEOM_NAMES_FULL])
    im  = ax1.imshow(mat, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.04)
    ax1.set_xticks(range(len(pairs))); ax1.set_xticklabels(plabels, fontsize=8)
    ax1.set_yticks(range(len(GEOM_NAMES_SHORT)))
    ax1.set_yticklabels(GEOM_NAMES_SHORT, fontsize=7.5)
    ax1.set_title('KS statistic per\ngeometric feature (vertex-level)', fontsize=9)
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    for i in range(len(GEOM_NAMES_FULL)):
        for j in range(len(pairs)):
            v = mat[i, j]
            ax1.text(j, i, f'{v:.3f}', ha='center', va='center',
                     fontsize=6.5, color='black' if v < 0.025 else 'white')
    # Potential reference annotation
    pot_ref = [0.031, 0.192, 0.250, 0.190]
    ax1.text(1.38, -0.7, f'Potential (ref): {pot_ref}', transform=ax1.transAxes,
             fontsize=6.5, color='grey', ha='right')

    # ── Panel 2 (top-right): Moran's I mean per feature ──────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    feat_order = ['Potential (ref)'] + GEOM_NAMES_FULL
    I_vals = [MORAN_GEOM_MEAN[fn] for fn in feat_order]
    cols   = ['#d62728'] + ['#aec7e8']*5 + ['#1f77b4']*3 + ['#ff7f0e']*5
    bars   = ax2.barh(range(len(feat_order)), I_vals, color=cols, alpha=0.85)
    ax2.set_yticks(range(len(feat_order)))
    ax2.set_yticklabels(['Potential (ref)']+GEOM_NAMES_SHORT, fontsize=7.5)
    ax2.set_xlabel("Mean Moran's I"); ax2.invert_yaxis()
    ax2.set_title("Mean Moran's I per feature\n(high=spatially smooth)", fontsize=9)
    ax2.axvline(0.77, color='#d62728', ls='--', lw=1.0, alpha=0.6)
    ax2.grid(axis='x', alpha=0.3)
    # Legend
    from matplotlib.patches import Patch
    ax2.legend(handles=[
        Patch(fc='#aec7e8', label='distance feats'),
        Patch(fc='#1f77b4', label='mesh spacing'),
        Patch(fc='#ff7f0e', label='curvature/shape'),
        Patch(fc='#d62728', label='Potential (ref)'),
    ], fontsize=7, loc='lower right')

    # ── Panel 3 (bottom-left): SW comparison ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    sw_labels  = ['8D (exist.)', '13D (geom)', '21D (all)']
    sw_cols    = [C[0], C[2], C[1]]
    x3 = np.arange(len(pairs)); w3 = 0.26
    off3 = [-1, 0, 1]
    for i, (sl, sc) in enumerate(zip(sw_labels, sw_cols)):
        ax3.bar(x3 + off3[i]*w3, [SW_GEOM[p][sl] for p in pairs], w3,
                color=sc, alpha=0.85, label=sl)
    ax3.set_xticks(x3); ax3.set_xticklabels(plabels, fontsize=8)
    ax3.set_ylabel('Sliced Wasserstein')
    ax3.set_title('SW: 8D existing vs 13D geometric\nvs 21D joint (graph-level)', fontsize=9)
    ax3.legend(fontsize=7.5); ax3.grid(axis='y', alpha=0.3)

    # ── Panel 4 (bottom-right): SVM balanced accuracy ────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    svm_labels = ['8D (exist.)', '13D (geom)', '21D (all)']
    svm_cols   = [C[0], C[2], C[1]]
    x4 = np.arange(len(pairs)); w4 = 0.26
    for i, (sl, sc) in enumerate(zip(svm_labels, svm_cols)):
        ax4.bar(x4 + off3[i]*w4, [SVM_GEOM[p][sl] for p in pairs], w4,
                color=sc, alpha=0.85, label=sl)
    ax4.axhline(0.5, color='red', ls='--', lw=1.0, label='Chance')
    for i, p in enumerate(pairs):
        delta = SVM_GEOM[p]['21D (all)'] - SVM_GEOM[p]['8D (exist.)']
        top   = max(SVM_GEOM[p].values())
        sign  = '+' if delta >= 0 else ''
        ax4.text(x4[i] + off3[2]*w4, top + 0.012,
                 f'{sign}{delta:.3f}', ha='center', fontsize=8,
                 color='#2d7a2d' if delta >= 0 else '#c0392b',
                 fontweight='bold')
    ax4.set_xticks(x4); ax4.set_xticklabels(plabels, fontsize=8)
    ax4.set_ylabel('SVM balanced accuracy')
    ax4.set_title('Protein-level SVM:\n21D adds geometric context', fontsize=9)
    ax4.legend(fontsize=7.5); ax4.set_ylim(0.44, 0.86); ax4.grid(axis='y', alpha=0.3)

    plt.suptitle(
        '21D geometric-enriched dataset study: distances, mesh spacing, curvature/shape features',
        fontsize=11, y=1.01)
    fig.savefig(OUT / 'geom_study.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  [16] geom_study.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 17: Curvature feature separability study
# ═══════════════════════════════════════════════════════════════════════════════
def fig_curvature_study():
    pairs       = [(61,90), (70,56), (86,14), (8,5)]
    plabels     = ['(61,90)\nconf.', '(70,56)\nconf.', '(86,14)\nconf.', '(8,5)\nctrl']
    curv_names  = ['k_min', 'k_max', 'H (mean)', 'K (gaussian)']
    curv_cols   = [C[2], C[3], C[4], C[6]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # ── Panel 1: KS test per curvature feature vs Potential reference ──────────
    ax = axes[0]
    x  = np.arange(len(pairs)); w = 0.17
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for i, (fn, col) in enumerate(zip(curv_names, curv_cols)):
        ks_vals = [KS_CURV[fn][p] for p in pairs]
        ax.bar(x + offsets[i]*w, ks_vals, w, color=col, alpha=0.85, label=fn)
    # Potential reference line per pair
    ax.plot(x, [KS['Potential'][p] for p in pairs], 'k^--', lw=1.2, ms=6,
            label='Potential (ref)')
    ax.set_xticks(x); ax.set_xticklabels(plabels, fontsize=8)
    ax.set_ylabel('KS statistic (vertex-level)')
    ax.set_title('KS Test: Curvature features\nvs Potential reference')
    ax.legend(fontsize=7, ncol=2); ax.grid(axis='y', alpha=0.3)
    ax.text(0.03, 0.96, 'All curvature KS < 0.05\n(Potential: up to 0.250)',
            transform=ax.transAxes, va='top', fontsize=7.5,
            bbox=dict(fc='#fff8e1', ec='orange', pad=2.5, boxstyle='round', alpha=0.9))

    # ── Panel 2: SW comparison — 8D existing vs 4D curvature vs 12D joint ──────
    ax2 = axes[1]
    sw_labels = ['8D (exist.)', '4D (curv)', '12D (all)']
    sw_cols   = [C[0], C[4], C[1]]
    x2 = np.arange(len(pairs)); w2 = 0.25
    off2 = [-1, 0, 1]
    for i, (sl, sc) in enumerate(zip(sw_labels, sw_cols)):
        ax2.bar(x2 + off2[i]*w2, [SW_CURV[p][sl] for p in pairs], w2,
                color=sc, alpha=0.85, label=sl)
    ax2.set_xticks(x2); ax2.set_xticklabels(plabels, fontsize=8)
    ax2.set_ylabel('Sliced Wasserstein')
    ax2.set_title('SW: Existing 8D vs 4D Curvature\nvs 12D Joint (graph-level)')
    ax2.legend(fontsize=7.5); ax2.grid(axis='y', alpha=0.3)

    # ── Panel 3: Protein-level SVM — does adding curvature help? ──────────────
    ax3 = axes[2]
    svm_labels = ['8D (exist.)', '4D (curv)', '12D (all)']
    svm_cols   = [C[0], C[4], C[1]]
    x3 = np.arange(len(pairs)); w3 = 0.25
    for i, (sl, sc) in enumerate(zip(svm_labels, svm_cols)):
        ax3.bar(x3 + off2[i]*w3, [SVM_CURV[p][sl] for p in pairs], w3,
                color=sc, alpha=0.85, label=sl)
    ax3.axhline(0.5, color='red', ls='--', lw=1.0, label='Chance')
    # Annotate delta: 12D vs 8D
    for i, p in enumerate(pairs):
        delta = SVM_CURV[p]['12D (all)'] - SVM_CURV[p]['8D (exist.)']
        top = max(SVM_CURV[p]['8D (exist.)'], SVM_CURV[p]['12D (all)'])
        sign = '+' if delta >= 0 else ''
        ax3.text(x3[i] + off2[2]*w3, top + 0.012,
                 f'{sign}{delta:.3f}', ha='center', fontsize=7.5,
                 color='#2d7a2d' if delta >= 0 else '#c0392b')
    ax3.set_xticks(x3); ax3.set_xticklabels(plabels, fontsize=8)
    ax3.set_ylabel('SVM balanced accuracy')
    ax3.set_title('Protein-level SVM:\nDoes 12D improve over 8D?')
    ax3.legend(fontsize=7.5); ax3.set_ylim(0.44, 0.82); ax3.grid(axis='y', alpha=0.3)

    plt.suptitle('Curvature feature analysis: principal curvature, mean curvature, Gaussian curvature',
                 fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT / 'curvature_study.pdf')
    plt.close(fig)
    print('  [17] curvature_study.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 18: RI-HierNet training dynamics
# ═══════════════════════════════════════════════════════════════════════════════
def fig_ri_hiernet_curves():
    f = RUNS / 'ri_hiernet_acc_v4' / 'final_results.json'
    if not f.exists():
        print('  [18] ri_hiernet_curves.pdf  SKIPPED (no data)')
        return
    d = json.load(open(f))
    hist = d['history']
    epochs   = [h['epoch'] for h in hist]
    val_acc  = [h['val']['accuracy']    for h in hist]
    val_wf1  = [h['val']['f1_weighted'] for h in hist]
    test_acc = [h['test']['accuracy']    if h['test'] else None for h in hist]
    test_wf1 = [h['test']['f1_weighted'] if h['test'] else None for h in hist]

    best_ep = d.get('best_val_epoch', None)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (val_m, test_m, ylabel, title) in zip(axes, [
        (val_acc,  test_acc, 'Accuracy',    'Accuracy over training'),
        (val_wf1,  test_wf1, 'Weighted-F1', 'Weighted-F1 over training'),
    ]):
        ax.plot(epochs, val_m,  color=C[0], lw=1.8, label='Val')
        te_epochs = [e for e, v in zip(epochs, test_m) if v is not None]
        te_vals   = [v for v in test_m if v is not None]
        ax.plot(te_epochs, te_vals, color=C[1], lw=1.8, ls='--', label='Test')
        ax.axhline(0.867, color='grey', lw=1, ls=':', alpha=0.7, label='HierGIN-14D best')
        ax.axhline(0.929, color='black', lw=1, ls=':', alpha=0.5, label='RIConv2 RUN2')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, max(epochs))
        ax.set_ylim(0.5, 1.0)

    # mark best-checkpoint epoch
    if best_ep is not None:
        for ax in axes:
            ax.axvline(best_ep, color=C[2], lw=1, ls=':', alpha=0.8)

    plt.suptitle('RI-HierNet training dynamics (ri_hiernet_acc_v4, 120 epochs)', fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / 'ri_hiernet_curves.pdf')
    plt.close(fig)
    print('  [18] ri_hiernet_curves.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'Generating figures → {OUT}')
    fig_class_distribution()
    fig_training_curves_all()
    fig_training_curves_features()
    fig_architecture_diagram()
    fig_separability_ks()
    fig_separability_svm()
    fig_separability_sw()
    fig_feature_ablation_radar()
    fig_backbone_ablation()
    fig_per_class_f1_hard()
    fig_feature_ablation_summary()
    fig_separability_graph_sw()
    fig_separability_moran()
    fig_separability_graph_svm()
    fig_separability_resolution()
    fig_geom_study()
    fig_curvature_study()
    fig_ri_hiernet_curves()
    print('\nAll figures done.')
    import os
    pdfs = sorted(OUT.glob('*.pdf'))
    print(f'{len(pdfs)} PDFs in {OUT}:')
    for p in pdfs:
        kb = os.path.getsize(p) // 1024
        print(f'  {p.name:<45} {kb} KB')
