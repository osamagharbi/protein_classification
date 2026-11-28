#!/usr/bin/env python3
"""
Hierarchical patch-pooling GNN for SHREC25.

Idea (vs the coarsened distribution patch-graph, which loses the fine mesh):
  1. message-pass GIN layers on the FULL 1000-node mesh  (this is where the
     ~81% class signal lives),
  2. pool node embeddings into geodesic-patch super-nodes (precomputed
     assignment; build_patchassign_dataset.py),
  3. optionally message-pass GIN layers on the patch graph (multi-scale),
  4. read out over patches -> MLP head.

This preserves BOTH the original per-node features AND the fine topology, using
patches only as an internal pooling layer.

Usage:
  python classify_hierarchical_gnn.py \
    --graph-dir /home/ogharbi/datasets/SHREC25/ssl_graphs_1k \
    --assign-dir /home/ogharbi/datasets/SHREC25/ssl_graphs_1k_patchassign_k32 \
    --hidden-dim 512 --node-layers 6 --patch-layers 2 \
    --pool meanmax --readout max --batch-size 64 --device 0
"""
from __future__ import annotations
import sys, argparse, os, json, math, time
from pathlib import Path
from datetime import datetime

# Force line-buffered output so log files update in real-time when redirected
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

import dgl
import dgl.function as dgl_fn
from dgl.nn.pytorch import GINConv, SAGEConv, GraphConv, GATConv, GATv2Conv, PNAConv

from classify_gnn import (
    load_labels, compute_metrics, canonical_graph_id, resolve_split_dir,
    select_topk_populated_classes, FocalLoss,
)
from fix_training import compute_class_weights, create_robust_train_val_split


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def compute_rwpe(edge_index, num_nodes: int, k: int = 16) -> torch.Tensor:
    """Random Walk Positional Encoding: k-step landing probabilities per node.
    pe[v, i] = probability of returning to v after i steps (diagonal of (D^{-1}A)^i).
    Returns [num_nodes, k] tensor encoding global structural position.
    """
    row, col = edge_index[0].long(), edge_index[1].long()
    deg = torch.zeros(num_nodes).scatter_add_(0, row, torch.ones(row.shape[0]))
    deg_inv = 1.0 / deg.clamp(min=1.0)
    vals = deg_inv[row]
    A = torch.sparse_coo_tensor(
        torch.stack([row, col]), vals, (num_nodes, num_nodes)
    ).coalesce()
    X = torch.eye(num_nodes)   # [N, N] — each col = walk starting from one node
    pe = []
    for _ in range(k):
        X = torch.sparse.mm(A, X)
        pe.append(X.diagonal().clone())
    return torch.stack(pe, dim=1)   # [N, k]


class HierPatchDataset(Dataset):
    """Loads (full mesh graph, patch graph, label) triples."""

    def __init__(self, graph_dir, assign_dir, split, label_dict, ssl_emb_dir=None,
                 feature_mode="all", rwpe_dim: int = 0,
                 use_lrf_feat: bool = False, assign_dir2=None,
                 use_bidirectional_edges: bool = False):
        self.gdir = resolve_split_dir(graph_dir, split)
        self.adir = resolve_split_dir(assign_dir, split)
        self.use_lrf_feat = use_lrf_feat
        # The legacy SHREC19/25 graphs use edge_index_mesh by default.  The
        # OFF converter also stores a genuinely bidirectional edge_index, but
        # we select it only when explicitly requested so existing protocols
        # remain unchanged.
        self.use_bidirectional_edges = use_bidirectional_edges
        self.adir2 = resolve_split_dir(assign_dir2, split) if assign_dir2 else None
        self.label_dict = label_dict
        self.feature_mode = feature_mode
        self.rwpe_dim = rwpe_dim
        # Build a normalized index for SSL embeddings. SSL files use ':' as a
        # separator (e.g. "105m_1:A:A_model1.pt") while graph files use '_'.
        # Normalize both sides by replacing ':' with '_'.
        self._ssl_index: dict = {}
        if ssl_emb_dir:
            sdir = resolve_split_dir(ssl_emb_dir, split)
            for p in sdir.glob("*.pt"):
                self._ssl_index[p.stem.replace(":", "_")] = p

        # index available files by canonical id
        avail = {}
        for p in self.gdir.glob("*.pt"):
            avail[canonical_graph_id(p.stem)] = p
        self.items = []
        missing = 0
        for gid in label_dict.keys():
            gp = avail.get(gid)
            ap = self.adir / f"{gp.name}" if gp is not None else None
            ssl_ok = (not self._ssl_index) or (gp is not None and gp.stem in self._ssl_index)
            if gp is not None and ap is not None and ap.exists() and ssl_ok:
                self.items.append((gid, gp, ap))
            else:
                missing += 1
        print(f"  [{split}] {len(self.items)} graphs ({missing} missing)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        gid, gp, ap = self.items[idx]
        d = torch.load(gp, map_location="cpu", weights_only=False)
        a = torch.load(ap, map_location="cpu", weights_only=False)

        if self.use_bidirectional_edges:
            ei = d.get("edge_index", d.get("edge_index_mesh"))
        else:
            ei = d.get("edge_index_mesh", d.get("edge_index"))

        if self._ssl_index:
            ssl_path = self._ssl_index.get(gp.stem)
            if ssl_path is None:
                raise FileNotFoundError(f"No SSL embedding for {gp.stem!r}")
            sd = torch.load(ssl_path, map_location="cpu", weights_only=False)
            x = sd["ssl_emb"].float()
        elif self.feature_mode == "hks":
            # Intrinsic heat-kernel signatures produced by prepare_off_hierarchy.py.
            x = d["hks"].float()
        elif self.feature_mode == "coords_hks":
            # Intrinsic HKS plus XYZ for a controlled hybrid experiment.
            x = torch.cat([d["pos"].float(), d["hks"].float()], dim=1)
        elif self.feature_mode == "coords_normals_hks":
            # xyz (0-2) + normals (3-5) + 16D HKS (6-21): with --invariant the GVP-inspired
            # encoder sees only the HKS (+ distance to centre) as scalars, xyz/normals as vectors.
            x = torch.cat([d["pos"].float(), d["normals"].float(), d["hks"].float()], dim=1)
        elif self.feature_mode == "coords":
            # x_ssl, y_ssl, z_ssl — normalized coordinates only (3-dim)
            x = d["pos"].float()
        elif self.feature_mode == "coords_potential":
            # xyz + Potential_norm (4-dim) — same as FAG-MAE feature spec
            x = torch.cat([d["pos"].float(), d["scalars_norm"][:, 0:1].float()], dim=1)
        elif self.feature_mode == "coords_normals":
            # normalized xyz + surface normals (6-dim)
            x = torch.cat([d["pos"].float(), d["normals"].float()], dim=1)
        elif self.feature_mode == "all_no_normpot":
            # 8D all features minus NormalPotential_norm (idx 1) → 7D
            x = d["x"].float()[:, [0, 2, 3, 4, 5, 6, 7]]
            x[:, 0:1] = x[:, 0:1].clamp(-5.0, 5.0)
        elif self.feature_mode == "all_no_pot":
            # 8D all features minus Potential_norm (idx 0) → 7D: NormalPotential + normals + coords
            x = d["x"].float()[:, [1, 2, 3, 4, 5, 6, 7]]
            x[:, 0:1] = x[:, 0:1].clamp(-5.0, 5.0)
        elif self.feature_mode == "geom_14d":
            # 14D: 8D base + dist_centroid + dist_anchor_0..3 + surf_variation
            # Loaded from ssl_graphs_1k_geom_enriched (21D x vector)
            # indices: 0-7 (base) + 8(dist_cen) + 9,10,11,12(anchors) + 20(surf_var)
            x = d["x"].float()[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20]]
            x[:, :2] = x[:, :2].clamp(-5.0, 5.0)
        elif self.feature_mode == "geom_14d_raw":
            # 14D: 8D base + 6 geom features using RAW values with global normalisation.
            # Fixes the per-graph z-score issue in geom_14d: global stats computed over
            # all 9244 train proteins preserve cross-protein scale differences.
            # geom_features_raw indices: 0=dist_cen, 1=anc0, 2=anc1, 3=anc2, 4=anc3, 12=surf_var
            base = d["x"].float()[:, :8]
            base[:, :2] = base[:, :2].clamp(-5.0, 5.0)
            raw = d["geom_features_raw"].float()[:, [0, 1, 2, 3, 4, 12]]
            _G_MEAN = torch.tensor([0.530406, 1.068526, 0.948508, 0.792245, 0.763201, 0.064826],
                                   dtype=torch.float32)
            _G_STD  = torch.tensor([0.187909, 0.417554, 0.416769, 0.320249, 0.304622, 0.049166],
                                   dtype=torch.float32)
            raw = (raw - _G_MEAN) / _G_STD
            x = torch.cat([base, raw], dim=1)  # [N, 14]
        else:
            x = d["x"].float()
            # clamp z-scored scalars (dims 0,1) at +/-5 (same as classify_gnn)
            x[:, :2] = x[:, :2].clamp(-5.0, 5.0)

        if self.rwpe_dim > 0:
            # remove self-loops before RWPE so diagonal reflects true random walk
            mask = ei[0] != ei[1]
            ei_rw = (ei[0][mask], ei[1][mask])
            pe = compute_rwpe(ei_rw, x.shape[0], k=self.rwpe_dim)
            x = torch.cat([x, pe], dim=1)

        g_node = dgl.graph((ei[0], ei[1]), num_nodes=x.shape[0])
        g_node = g_node.remove_self_loop().add_self_loop()
        g_node.ndata["attr"] = x
        g_node.ndata["pid"] = a["patch_ids"].long()

        # Option 1: LRF edge features — set AFTER add_self_loop so shapes align.
        # edge_index_mesh has no self-loops; add_self_loop appends N zero-feature edges.
        if self.use_lrf_feat:
            lrf = d.get("ri_lrf_feat")
            if lrf is not None:
                N = x.shape[0]
                zeros = torch.zeros(N, 8, dtype=torch.float32)
                g_node.edata["lrf_feat"] = torch.cat([lrf.float(), zeros], dim=0)
            else:
                E = g_node.num_edges()
                g_node.edata["lrf_feat"] = torch.zeros(E, 8, dtype=torch.float32)

        P = int(a["num_patches"])
        pei = a["patch_edge_index"]
        g_patch = dgl.graph((pei[0], pei[1]), num_nodes=P)
        g_patch = g_patch.remove_self_loop().add_self_loop()

        label = torch.tensor(self.label_dict[gid], dtype=torch.long)

        # Option 3: three-level hierarchy — load intermediate assignment
        if self.adir2 is not None:
            ap2 = self.adir2 / gp.name
            a2 = torch.load(ap2, map_location="cpu", weights_only=False)
            g_node.ndata["mid_pid"] = a2["patch_ids"].long()   # node → mid (k128)
            M = int(a2["num_patches"])
            mei = a2["patch_edge_index"]
            g_mid = dgl.graph((mei[0], mei[1]), num_nodes=M)
            g_mid = g_mid.remove_self_loop().add_self_loop()
            # mid → final assignment: majority vote of final patch_ids over mid-patch nodes
            mid_ids = a2["patch_ids"].long()    # [N] node → mid
            fin_ids = a["patch_ids"].long()     # [N] node → final
            mid_to_final = torch.zeros(M, dtype=torch.long)
            for p in range(M):
                mask = (mid_ids == p)
                if mask.any():
                    mid_to_final[p] = torch.mode(fin_ids[mask])[0]
            return g_node, g_mid, g_patch, mid_to_final, label

        return g_node, g_patch, label


def _random_so3(generator=None):
    """Sample a uniformly random 3×3 rotation matrix via QR decomposition.
    `generator` (optional torch.Generator) makes the draw reproducible."""
    Q, R = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    Q = Q * torch.sign(torch.diag(R))  # ensure det = +1
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# Per feature mode: (xyz slice, normal slice or None, indices of channels that are
# already rotation-invariant scalars). Used by --invariant (which feeds only the
# invariant channels to the node encoder; xyz and normals enter HiPatchNet only as
# relative geometry / GVP vectors) and by the rotation test (which rotates the xyz
# and normal channels). Potentials and the geom_14d* descriptors (distances to the
# centroid and to FPS anchors, surface variation) are invariant scalars.
FEATURE_LAYOUT = {
    "coords":           (slice(0, 3), None,         []),
    "coords_potential": (slice(0, 3), None,         [3]),
    "coords_normals":   (slice(0, 3), slice(3, 6),  []),
    "geom_14d":         (slice(5, 8), slice(2, 5),  [0, 1, 8, 9, 10, 11, 12, 13]),
    "geom_14d_raw":     (slice(5, 8), slice(2, 5),  [0, 1, 8, 9, 10, 11, 12, 13]),
    # OFF meshes (prepare_off_hierarchy.py): HKS is intrinsic, so rotation- and pose-invariant
    "coords_hks":         (slice(0, 3), None,        list(range(3, 19))),
    "coords_normals_hks": (slice(0, 3), slice(3, 6), list(range(6, 22))),
}


def rotate_attr(g, feature_mode, R):
    """Rotate the xyz (and normal) channels of g.ndata['attr'] by R, in place."""
    xyz_sl, nor_sl, _ = FEATURE_LAYOUT[feature_mode]
    attr = g.ndata["attr"].clone()
    attr[:, xyz_sl] = attr[:, xyz_sl] @ R.T
    if nor_sl is not None:
        attr[:, nor_sl] = attr[:, nor_sl] @ R.T
    g.ndata["attr"] = attr
    return g


class RotatedView(Dataset):
    """Deterministic random SO(3) rotation of every test graph (rotation test).

    The rotation of item `idx` depends only on (seed, idx), so the result is the
    same for any number of DataLoader workers. Patch assignments are untouched:
    they are built from FPS seeds and geodesic distances, which do not change
    under a rigid rotation, so rotating the features is equivalent to rotating
    the surface."""

    def __init__(self, base, feature_mode, seed):
        if feature_mode not in FEATURE_LAYOUT:
            raise ValueError(f"rotation test not supported for feature mode {feature_mode!r}")
        self.base, self.feature_mode, self.seed = base, feature_mode, int(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + int(idx))
        rotate_attr(item[0], self.feature_mode, _random_so3(gen))
        return item


def present_class_metrics(labs, preds):
    """Macro metrics over the classes that occur in `labs` only (the convention
    used in the papers). compute_metrics() averages over the full class list,
    which counts every class absent from the test set as 0."""
    from sklearn.metrics import f1_score, recall_score
    present = sorted(set(int(v) for v in labs))
    return {
        "f1_macro_present": float(f1_score(labs, preds, labels=present, average="macro", zero_division=0)),
        "balanced_accuracy_present": float(recall_score(labs, preds, labels=present, average="macro", zero_division=0)),
        "n_classes_present": len(present),
    }


_FEATURE_MODE_AUG_INDICES = {
    # coords_potential: [x, y, z, Potential_norm]  -> coords 0:3, no normals
    "coords_potential": (slice(0, 3), None),
    "coords":           (slice(0, 3), None),
    # coords_normals: [x, y, z, nx, ny, nz]
    "coords_normals":   (slice(0, 3), slice(3, 6)),
    # all (8-dim): [Pot, NormPot, nx, ny, nz, x, y, z]
    "all":              (slice(5, 8), slice(2, 5)),
    # geom_14d / geom_14d_raw: same coord/normal layout as "all" (positions 5-7 and 2-4)
    "geom_14d":         (slice(5, 8), slice(2, 5)),
    "geom_14d_raw":     (slice(5, 8), slice(2, 5)),
    # OFF meshes: xyz first; HKS channels are intrinsic and never rotated
    "coords_hks":         (slice(0, 3), None),
    "coords_normals_hks": (slice(0, 3), slice(3, 6)),
}


def augment_graph(g, scale_std=0.1, jitter_std=0.005,
                  coord_sl=slice(0, 3), normal_sl=None):
    """Apply SO(3) rotation + random scale + jitter to node features (CPU, in-place)."""
    attr = g.ndata["attr"].clone()

    R = _random_so3()
    attr[:, coord_sl] = attr[:, coord_sl] @ R.T
    if normal_sl is not None:
        attr[:, normal_sl] = attr[:, normal_sl] @ R.T

    s = 1.0 + torch.empty(1).uniform_(-scale_std, scale_std).item()
    attr[:, coord_sl] = attr[:, coord_sl] * s

    attr[:, coord_sl] = attr[:, coord_sl] + torch.randn_like(attr[:, coord_sl]) * jitter_std

    g.ndata["attr"] = attr
    return g


def collate_aug(scale_std=0.1, jitter_std=0.005, feature_mode="coords_potential"):
    """Return a collate function that applies augmentation to node graphs."""
    coord_sl, normal_sl = _FEATURE_MODE_AUG_INDICES.get(feature_mode, (slice(0, 3), None))

    def _collate(batch):
        gnodes, gpatches, labels = zip(*batch)
        gnodes = [augment_graph(g, scale_std, jitter_std, coord_sl, normal_sl)
                  for g in gnodes]

        patch_counts = [gp.num_nodes() for gp in gpatches]
        offsets = np.cumsum([0] + patch_counts[:-1])
        gpids = []
        for gn, off in zip(gnodes, offsets):
            gpids.append(gn.ndata["pid"] + int(off))
        global_pid = torch.cat(gpids, dim=0)

        bn = dgl.batch(gnodes)
        bp = dgl.batch(gpatches)
        y = torch.stack(labels)
        return bn, bp, global_pid, y
    return _collate


def collate(batch):
    gnodes, gpatches, labels = zip(*batch)
    patch_counts = [gp.num_nodes() for gp in gpatches]
    offsets = np.cumsum([0] + patch_counts[:-1])
    gpids = []
    for gn, off in zip(gnodes, offsets):
        gpids.append(gn.ndata["pid"] + int(off))
    global_pid = torch.cat(gpids, dim=0)
    bn = dgl.batch(gnodes)
    bp = dgl.batch(gpatches)
    y = torch.stack(labels)
    return bn, bp, global_pid, y


class MultiScalePatchDataset(HierPatchDataset):
    """Same graphs, several patch resolutions at once (multi-scale model).

    assign_dirs[0] plays the role of the usual --assign-dir; every further dir
    adds one resolution. Item = (g_node, [g_patch_s for each scale], label), with
    the node->patch ids of scale s stored in g_node.ndata[f"pid{s}"]."""

    def __init__(self, graph_dir, assign_dirs, split, label_dict, **kw):
        super().__init__(graph_dir, assign_dirs[0], split, label_dict, **kw)
        self.extra_adirs = [resolve_split_dir(a, split) for a in assign_dirs[1:]]
        keep = [it for it in self.items
                if all((ad / it[1].name).exists() for ad in self.extra_adirs)]
        if len(keep) != len(self.items):
            print(f"  [{split}] multi-scale: dropped {len(self.items) - len(keep)} graphs "
                  f"missing a patch sidecar at some resolution")
        self.items = keep

    def __getitem__(self, idx):
        g_node, g_patch, label = super().__getitem__(idx)
        g_node.ndata["pid0"] = g_node.ndata["pid"]
        patches = [g_patch]
        gp_name = self.items[idx][1].name
        for s, ad in enumerate(self.extra_adirs, start=1):
            a = torch.load(ad / gp_name, map_location="cpu", weights_only=False)
            g_node.ndata[f"pid{s}"] = a["patch_ids"].long()
            pei = a["patch_edge_index"]
            gp = dgl.graph((pei[0], pei[1]), num_nodes=int(a["num_patches"]))
            patches.append(gp.remove_self_loop().add_self_loop())
        return g_node, patches, label


def collate_multi(batch):
    """Collate for MultiScalePatchDataset -> (bn, [bp_s], [gpid_s], y)."""
    gnodes, patch_lists, labels = zip(*batch)
    bps, gpids = [], []
    for s in range(len(patch_lists[0])):
        gps = [pl[s] for pl in patch_lists]
        offsets = np.cumsum([0] + [gp.num_nodes() for gp in gps[:-1]])
        gpids.append(torch.cat([gn.ndata[f"pid{s}"] + int(o) for gn, o in zip(gnodes, offsets)]))
        bps.append(dgl.batch(gps))
    return dgl.batch(gnodes), bps, gpids, torch.stack(labels)


def collate_three(batch):
    """Collate for three-level hierarchy: returns (bn, bm, bp, mid_gpid, gpid, y)."""
    gnodes, gmids, gpatches, m2fs, labels = zip(*batch)
    mid_counts = [gm.num_nodes() for gm in gmids]
    mid_offsets = np.cumsum([0] + mid_counts[:-1])
    fin_counts = [gp.num_nodes() for gp in gpatches]
    fin_offsets = np.cumsum([0] + fin_counts[:-1])
    node_mid_pids, mid_fin_pids = [], []
    for gn, m_off in zip(gnodes, mid_offsets):
        node_mid_pids.append(gn.ndata["mid_pid"] + int(m_off))
    for m2f, f_off in zip(m2fs, fin_offsets):
        mid_fin_pids.append(m2f + int(f_off))
    return (dgl.batch(gnodes), dgl.batch(gmids), dgl.batch(gpatches),
            torch.cat(node_mid_pids, dim=0), torch.cat(mid_fin_pids, dim=0),
            torch.stack(labels))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def gin_mlp(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))


class GATLayer(nn.Module):
    """Multi-head GAT that flattens heads → (N, out_dim)."""
    def __init__(self, in_dim, out_dim, num_heads=4):
        super().__init__()
        assert out_dim % num_heads == 0
        self.conv = GATConv(in_dim, out_dim // num_heads, num_heads,
                            feat_drop=0.0, attn_drop=0.0, allow_zero_in_degree=True)

    def forward(self, g, h):
        return self.conv(g, h).flatten(1)


class GATv2Layer(nn.Module):
    """GATv2: dynamic attention — fixes rank-1 attention issue of original GAT."""
    def __init__(self, in_dim, out_dim, num_heads=4):
        super().__init__()
        assert out_dim % num_heads == 0
        self.conv = GATv2Conv(in_dim, out_dim // num_heads, num_heads,
                              feat_drop=0.0, attn_drop=0.0, allow_zero_in_degree=True)

    def forward(self, g, h):
        return self.conv(g, h).flatten(1)


class PNALayer(nn.Module):
    """PNA: multiple aggregators (mean/max/sum/std) + degree scalers."""
    def __init__(self, in_dim, out_dim, delta=6.0):
        super().__init__()
        self.conv = PNAConv(
            in_size=in_dim, out_size=out_dim,
            aggregators=["mean", "max", "sum", "std"],
            scalers=["identity", "amplification", "attenuation"],
            delta=delta,
            dropout=0.0,
        )

    def forward(self, g, h):
        return self.conv(g, h)


class SGFormerLayer(nn.Module):
    """SGFormer node encoder: O(N) linearized global attention + local GIN.

    Ported from Wu et al. 2023 (https://github.com/qitianwu/SGFormer).
    Key trick: Q(K^T V) / (Q K^T 1) avoids materialising the N×N matrix,
    so memory is O(N·D) not O(N²). Applied per-graph inside each DGL batch.

    alpha controls local (GIN) vs global (transformer) mix.
    """
    def __init__(self, in_dim, out_dim, num_heads=4, alpha=0.5, pna_delta=6.0):
        super().__init__()
        assert out_dim % num_heads == 0
        self.H = num_heads
        self.D = out_dim // num_heads   # head dim; mean over heads → out_dim/H
        # project to num_heads * D = out_dim each
        self.Wq = nn.Linear(in_dim, out_dim)
        self.Wk = nn.Linear(in_dim, out_dim)
        self.Wv = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(self.D, out_dim)   # head-dim → out_dim
        self.local_conv = make_conv("gin", in_dim, out_dim, pna_delta)
        self.alpha = alpha
        self.norm = nn.LayerNorm(out_dim)

    def _linear_attn_per_graph(self, h, g):
        """SGFormer full_attention_conv applied per graph in the batch."""
        N = h.shape[0]
        q = self.Wq(h).view(N, self.H, self.D)
        k = self.Wk(h).view(N, self.H, self.D)
        v = self.Wv(h).view(N, self.H, self.D)
        # L2-normalise keys and queries (SGFormer eq.)
        q = q / q.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        k = k / k.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)

        out = torch.zeros(N, self.H, self.D, device=h.device, dtype=h.dtype)
        offset = 0
        for n in g.batch_num_nodes().tolist():
            qi, ki, vi = q[offset:offset+n], k[offset:offset+n], v[offset:offset+n]
            # numerator: Q (K^T V) + N·V   [n, H, D]
            kvs = torch.einsum("nhm,nhd->hmd", ki, vi)       # [H, D, D]
            num = torch.einsum("nhm,hmd->nhd", qi, kvs) + n * vi
            # denominator: Q (K^T 1) + N   [n, H]
            ks_sum = ki.sum(dim=0)                            # [H, D]
            denom = (torch.einsum("nhm,hm->nh", qi, ks_sum) + n).unsqueeze(-1)
            out[offset:offset+n] = num / denom.clamp(min=1e-8)
            offset += n
        # mean over heads → [N, D], then project → [N, out_dim]
        return self.out_proj(out.mean(dim=1))

    def forward(self, g, h):
        x_global = self._linear_attn_per_graph(h, g)
        x_local  = self.local_conv(g, h)
        return self.norm(self.alpha * x_local + (1 - self.alpha) * x_global)


class GTLayer(nn.Module):
    """Graph Transformer layer: dot-product multi-head attention over neighbours."""
    def __init__(self, in_dim, out_dim, num_heads=4):
        super().__init__()
        assert out_dim % num_heads == 0
        self.H = num_heads
        self.D = out_dim // num_heads
        self.q = nn.Linear(in_dim, out_dim, bias=False)
        self.k = nn.Linear(in_dim, out_dim, bias=False)
        self.v = nn.Linear(in_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.scale = self.D ** -0.5

    def forward(self, g, h):
        N = h.shape[0]
        with g.local_scope():
            g.ndata["q"] = self.q(h).view(N, self.H, self.D)
            g.ndata["k"] = self.k(h).view(N, self.H, self.D)
            g.ndata["v"] = self.v(h).view(N, self.H, self.D)
            g.apply_edges(lambda e: {
                "a": (e.dst["q"] * e.src["k"]).sum(-1, keepdim=True) * self.scale
            })
            g.edata["a"] = dgl.ops.edge_softmax(g, g.edata["a"])
            g.update_all(dgl_fn.u_mul_e("v", "a", "m"), dgl_fn.sum("m", "agg"))
            return self.out_proj(g.ndata["agg"].view(N, -1))


class GPSLayer(nn.Module):
    """GPS layer: local MPNN + global multi-head self-attention (per graph in batch).
    Combines local inductive bias with long-range attention — no PE required.
    Rampasek et al. 2022 (simplified: no edge attr, no RRWP).
    """
    def __init__(self, in_dim, out_dim, local_backbone="gin", num_heads=4, pna_delta=6.0):
        super().__init__()
        self.local_conv = make_conv(local_backbone, in_dim, out_dim, pna_delta)
        self.proj_in = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        assert out_dim % num_heads == 0
        self.attn = nn.MultiheadAttention(out_dim, num_heads, batch_first=True, dropout=0.0)
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.ff = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2), nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, g, h):
        h_local = self.local_conv(g, h)
        h_res = self.proj_in(h)

        # global attention within each graph (extract per-graph node subsets)
        h_global = torch.zeros_like(h_local)
        offset = 0
        for n in g.batch_num_nodes().tolist():
            nodes = h_res[offset:offset + n].unsqueeze(0)   # [1, n, d]
            out, _ = self.attn(nodes, nodes, nodes)
            h_global[offset:offset + n] = out.squeeze(0)
            offset += n

        h_out = self.norm1(h_local + h_global)
        h_out = h_out + self.ff(self.norm2(h_out))
        return h_out


class LRFConvLayer(nn.Module):
    """GNN layer using precomputed 8D LRF edge features (ri_lrf_feat key in graph).

    Requires dataset built with add_lrf_features.py (ssl_graphs_1k_geom_enriched_ri).
    Edge features [E, 8] must be stored in g.edata['lrf_feat'] by the dataloader.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim + 8, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        with g.local_scope():
            g.ndata["_h"] = h
            g.apply_edges(self._msg)
            g.update_all(dgl_fn.copy_e("_m", "_agg_m"), dgl_fn.sum("_agg_m", "_out"))
            return g.ndata["_out"]

    def _msg(self, edges) -> dict:
        hi = edges.src["_h"]            # [E, in_dim]
        ef = edges.data["lrf_feat"]     # [E, 8]
        return {"_m": self.edge_mlp(torch.cat([hi, ef], dim=1))}


class MSGINLayer(nn.Module):
    """Multi-Scale GIN: aggregates node features at kNN-6/16/32 and concatenates.

    Builds kNN graphs on-the-fly from node xyz positions (no extra data files).
    Requires a feature mode that includes xyz (all, geom_14d_raw, coords_normals).
    """
    def __init__(self, in_dim: int, out_dim: int,
                 ks=(6, 16, 32), xyz_slice: slice = slice(5, 8)):
        super().__init__()
        self.ks = list(ks)
        self.xyz_slice = xyz_slice
        self.mlps = nn.ModuleList([gin_mlp(in_dim, out_dim) for _ in ks])
        self.out_proj = nn.Linear(out_dim * len(ks), out_dim)

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        pos = g.ndata["attr"][:, self.xyz_slice]  # [total_N, 3]
        # Process each graph in the batch separately to avoid cross-graph distances
        counts = g.batch_num_nodes()              # [B]
        per_scale: list = [[] for _ in self.ks]
        offset = 0
        with torch.no_grad():
            for n in counts:
                n = n.item()
                pi = pos[offset: offset + n]      # [n, 3]
                D = torch.cdist(pi, pi)
                D.fill_diagonal_(float("inf"))
                hi = h[offset: offset + n]        # [n, d]
                for ki, k in enumerate(self.ks):
                    k_eff = min(k, n - 1)
                    nbrs = D.topk(k_eff, dim=1, largest=False).indices  # [n, k]
                    src = torch.arange(n, device=h.device).repeat_interleave(k_eff)
                    dst = nbrs.reshape(-1)
                    agg = hi.clone()              # self-term (ε=1)
                    agg.index_add_(0, src, hi[dst])
                    per_scale[ki].append(agg)
                offset += n
        out = [mlp(torch.cat(parts, dim=0)) for mlp, parts in zip(self.mlps, per_scale)]
        return self.out_proj(torch.cat(out, dim=1))


class RIConvLayer(nn.Module):
    """Rotation-invariant GNN layer (RIConv2-style).

    For each directed edge i→j computes 5 SO(3)-invariant scalars from the
    positions and normals of i and j, concatenates them with the source
    hidden state h_i, passes through a 2-layer MLP, and scatter-sums into
    destination nodes.  Self-information enters through the self-loop edge.

    Reads xyz from g.ndata['attr'][:, xyz_slice] and normals from
    g.ndata['attr'][:, nor_slice].  Defaults match the 8D/14D feature layout
    used in this project: normals at dims [2:5], xyz at dims [5:8].

    NOTE: requires feature modes that include surface normals (all, all_no_pot,
    all_no_normpot, geom_14d, geom_14d_raw). Not compatible with coords/4D modes.
    """
    def __init__(self, in_dim: int, out_dim: int,
                 xyz_slice: slice = slice(5, 8),
                 nor_slice: slice = slice(2, 5)):
        super().__init__()
        self.xyz_slice = xyz_slice
        self.nor_slice = nor_slice
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim + 5, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        with g.local_scope():
            attr = g.ndata["attr"]                         # always the original features
            g.ndata["_pos"] = attr[:, self.xyz_slice]      # [N, 3] xyz
            g.ndata["_nor"] = attr[:, self.nor_slice]      # [N, 3] normals
            g.ndata["_h"]   = h
            g.apply_edges(self._ri_msg)
            g.update_all(dgl_fn.copy_e("_m", "_agg_m"), dgl_fn.sum("_agg_m", "_out"))
            # Self-info enters via the self-loop edge (dist≈0, ⟨n,n⟩=1).
            # Outer HierarchicalGNN residual block adds h when in_dim==out_dim.
            return g.ndata["_out"]

    def _ri_msg(self, edges) -> dict:
        pi = edges.src["_pos"]                             # [E, 3]
        pj = edges.dst["_pos"]
        ni = edges.src["_nor"]                             # [E, 3]
        nj = edges.dst["_nor"]
        hi = edges.src["_h"]                               # [E, in_dim]

        d    = pj - pi                                     # [E, 3]
        dist = d.norm(dim=1, keepdim=True).clamp(min=1e-8)
        dn   = d / dist                                    # unit edge direction

        ri = torch.cat([
            dist,                                                         # ||p_j − p_i||
            (ni * dn).sum(1, keepdim=True),                               # ⟨n_i, e_ij⟩
            (nj * dn).sum(1, keepdim=True),                               # ⟨n_j, e_ij⟩
            (ni * nj).sum(1, keepdim=True),                               # ⟨n_i, n_j⟩
            torch.linalg.cross(nj, ni).mul(dn).sum(1, keepdim=True),     # ⟨n_j×n_i, e_ij⟩
        ], dim=1)                                                         # [E, 5]

        return {"_m": self.edge_mlp(torch.cat([hi, ri], dim=1))}


class GVP(nn.Module):
    """Geometric Vector Perceptron (Jing et al., ICLR 2021).

    Maps a (scalar, vector) tuple to a new (scalar, vector) tuple while staying
    rotation-equivariant: scalars are built only from vector *norms* (invariants),
    output vectors are the input vectors linearly recombined and gated by a scalar
    sigmoid. Vectors have shape [..., n_vec, 3]; scalars [..., n_scalar].
    """
    def __init__(self, in_s, in_v, out_s, out_v, h_v=None):
        super().__init__()
        h_v = h_v if h_v is not None else max(in_v, out_v)
        self.in_v = in_v
        self.out_v = out_v
        self.wh = nn.Linear(in_v, h_v, bias=False)        # mix vector channels (equivariant)
        self.ws = nn.Linear(in_s + h_v, out_s)            # scalars <- [scalars, vec norms]
        self.wv = nn.Linear(h_v, out_v, bias=False) if out_v > 0 else None
        self.wg = nn.Linear(out_s, out_v) if out_v > 0 else None  # vector gate from scalars

    def forward(self, s, V):
        # V: [..., in_v, 3] ; s: [..., in_s]
        Vh = self.wh(V.transpose(-1, -2)).transpose(-1, -2)       # [..., h_v, 3]
        vn = torch.norm(Vh, dim=-1)                               # [..., h_v]  (rotation-invariant)
        s_out = self.ws(torch.cat([s, vn], dim=-1))              # [..., out_s]
        if self.out_v > 0:
            Vout = self.wv(Vh.transpose(-1, -2)).transpose(-1, -2)  # [..., out_v, 3]
            gate = torch.sigmoid(self.wg(s_out)).unsqueeze(-1)       # [..., out_v, 1]
            Vout = Vout * gate
        else:
            Vout = None
        return F.relu(s_out), Vout


class GVPConvLayer(nn.Module):
    """GVP message-passing layer with the standard scalar-in / scalar-out interface.

    Per directed edge i->j it forms a rotation-invariant scalar part (source hidden
    state + RBF-expanded distance) and a vector part (the unit edge direction),
    runs a GVP, sum-aggregates scalars AND vectors into destination nodes, then a
    node-update GVP returns the output scalars. Vectors are recomputed from geometry
    each layer (they are not threaded through the scalar pipeline), so the layer
    drops into the existing HierarchicalGNN unchanged. Equivariant: only relative
    directions and distances are used, so the returned scalars are SO(3)-invariant.

    Reads xyz from g.ndata['attr'][:, xyz_slice] (0:3 for coords modes, 5:8 for
    the 8D/14D/16D layouts) — works on any feature mode that carries coordinates.
    """
    def __init__(self, in_dim, out_dim, xyz_slice=slice(5, 8), n_rbf=8, vec_dim=4,
                 nor_slice=None):
        super().__init__()
        self.xyz_slice = xyz_slice
        # nor_slice (opt-in, used by --invariant): surface normals of both edge
        # endpoints enter the message GVP as two extra *vector* channels, so the
        # layer uses normal information while staying exactly rotation-invariant.
        # Default None keeps the original layer (and checkpoint shapes) unchanged.
        self.nor_slice = nor_slice
        self.n_rbf = n_rbf
        # RBF centres over the unit-ball distance range (~[0, 2] after normalisation)
        centers = torch.linspace(0.0, 2.0, n_rbf)
        self.register_buffer("rbf_centers", centers)
        self.rbf_gamma = float((centers[1] - centers[0]) ** 2 + 1e-6)
        self.msg_gvp = GVP(in_dim + n_rbf, 1 if nor_slice is None else 3, out_dim, vec_dim)
        self.upd_gvp = GVP(out_dim, vec_dim, out_dim, 0)

    def _rbf(self, dist):
        return torch.exp(-((dist - self.rbf_centers) ** 2) / self.rbf_gamma)  # [E, n_rbf]

    def _msg(self, edges):
        pi = edges.src["_pos"]; pj = edges.dst["_pos"]
        d = pj - pi
        dist = d.norm(dim=1, keepdim=True).clamp(min=1e-8)
        u = d / dist                                        # [E, 3] unit direction
        s_in = torch.cat([edges.src["_h"], self._rbf(dist)], dim=1)
        if self.nor_slice is None:
            V_in = u.unsqueeze(1)                           # [E, 1, 3]
        else:
            V_in = torch.stack([u, edges.src["_nor"], edges.dst["_nor"]], dim=1)  # [E, 3, 3]
        s_m, V_m = self.msg_gvp(s_in, V_in)                 # [E, out], [E, vec, 3]
        return {"_ms": s_m, "_mv": V_m}

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        with g.local_scope():
            g.ndata["_pos"] = g.ndata["attr"][:, self.xyz_slice]
            if self.nor_slice is not None:
                g.ndata["_nor"] = g.ndata["attr"][:, self.nor_slice]
            g.ndata["_h"] = h
            g.apply_edges(self._msg)
            g.update_all(dgl_fn.copy_e("_ms", "m"), dgl_fn.sum("m", "_sagg"))
            g.update_all(dgl_fn.copy_e("_mv", "m"), dgl_fn.sum("m", "_vagg"))
            s_upd, _ = self.upd_gvp(g.ndata["_sagg"], g.ndata["_vagg"])
            return s_upd


def make_conv(backbone, in_dim, out_dim, pna_delta=6.0,
              xyz_slice=slice(5, 8), nor_slice=slice(2, 5), gvp_nor_slice=None):
    if backbone == "gin":
        return GINConv(gin_mlp(in_dim, out_dim), "sum", learn_eps=True)
    elif backbone == "lrf_gin":
        return LRFConvLayer(in_dim, out_dim)
    elif backbone == "msg_gin":
        return MSGINLayer(in_dim, out_dim, xyz_slice=xyz_slice)
    elif backbone == "ri_gin":
        return RIConvLayer(in_dim, out_dim, xyz_slice=xyz_slice, nor_slice=nor_slice)
    elif backbone == "gvp":
        return GVPConvLayer(in_dim, out_dim, xyz_slice=xyz_slice, nor_slice=gvp_nor_slice)
    elif backbone == "sage":
        return SAGEConv(in_dim, out_dim, aggregator_type="mean")
    elif backbone == "gcn":
        return GraphConv(in_dim, out_dim, norm="both", allow_zero_in_degree=True)
    elif backbone == "gat":
        return GATLayer(in_dim, out_dim, num_heads=4)
    elif backbone == "gatv2":
        return GATv2Layer(in_dim, out_dim, num_heads=4)
    elif backbone == "pna":
        return PNALayer(in_dim, out_dim, delta=pna_delta)
    elif backbone == "gt":
        return GTLayer(in_dim, out_dim, num_heads=4)
    elif backbone == "gps":
        return GPSLayer(in_dim, out_dim, local_backbone="gin", num_heads=4, pna_delta=pna_delta)
    elif backbone == "sgformer":
        return SGFormerLayer(in_dim, out_dim, num_heads=4, alpha=0.5, pna_delta=pna_delta)
    raise ValueError(f"Unknown backbone: {backbone!r}")


class HierarchicalGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, node_layers=6,
                 patch_layers=2, pool="meanmax", readout="max", dropout=0.3,
                 node_readout=False, node_backbone="gin", patch_backbone="gin",
                 pna_delta=6.0, scalar_inject=False, patch_scalar_inject=False,
                 patch_inject_all=False, three_level=False, mid_layers=2,
                 xyz_slice=slice(5, 8), proj_dim=128,
                 nor_slice=slice(2, 5), flat=False,
                 invariant=False, inv_idx=None, gvp_normals=False):
        super().__init__()
        # flat=True is the no-hierarchy ablation: node GNN -> graph readout over
        # all nodes -> head. No patch pooling and no patch GNN are built.
        self.flat = flat
        # invariant=True (HiPatchNet, exactly rotation-invariant): the node encoder
        # receives only the rotation-invariant channels inv_idx plus the distance of
        # each node to the surface centre; xyz (and normals, if gvp_normals) reach
        # the GVP layers only as relative geometry / vectors. `in_dim` must then be
        # len(inv_idx) + 1. Default False leaves the model exactly as before.
        self.invariant = invariant
        self.inv_idx = list(inv_idx or [])
        self.xyz_slice = xyz_slice
        gvp_nor = nor_slice if gvp_normals else None
        if flat:
            patch_layers = 0
        self.pool = pool
        self.readout = readout
        self.node_readout = node_readout
        self.node_layers = node_layers
        self.scalar_inject = scalar_inject
        self.patch_scalar_inject = patch_scalar_inject
        self.patch_inject_all = patch_inject_all
        self.three_level = three_level
        self._in_dim = in_dim  # needed at forward time for patch_inject_all

        # node-level GNN on the full mesh (node_layers=0 bypasses entirely)
        self.node_convs = nn.ModuleList()
        self.node_norms = nn.ModuleList()
        d = in_dim
        for _ in range(node_layers):
            self.node_convs.append(make_conv(node_backbone, d, hidden_dim, pna_delta,
                                             xyz_slice, nor_slice, gvp_nor))
            self.node_norms.append(nn.LayerNorm(hidden_dim))
            d = hidden_dim
        # when node_layers=0, project raw features to hidden_dim for pooling
        self.node_proj = nn.Linear(in_dim, hidden_dim) if node_layers == 0 else nn.Identity()

        # node-level scalar re-injection (failed experiment — left for reproducibility)
        self.scalar_projs = nn.ModuleList(
            [nn.Linear(hidden_dim + 2, hidden_dim) for _ in range(max(node_layers - 1, 0))]
        ) if scalar_inject else nn.ModuleList()

        # learned attention scorer for attention pooling
        self.attn_pool = nn.Linear(hidden_dim, 1) if pool == "attn" and not flat else None

        # projection after node->patch pooling (meanmax doubles width)
        pooled_dim = hidden_dim * (2 if pool == "meanmax" else 1)
        self.pool_proj = (nn.Linear(pooled_dim, hidden_dim)
                          if pooled_dim != hidden_dim and not flat else nn.Identity())

        # Option 3: three-level hierarchy — intermediate (mid) pooling stage
        if three_level:
            self.mid_pool_proj = nn.Linear(pooled_dim, hidden_dim) if pooled_dim != hidden_dim else nn.Identity()
            self.mid_convs = nn.ModuleList()
            self.mid_norms = nn.ModuleList()
            for _ in range(mid_layers):
                self.mid_convs.append(make_conv(patch_backbone, hidden_dim, hidden_dim, pna_delta))
                self.mid_norms.append(nn.LayerNorm(hidden_dim))

        # patch-level GNN on the patch graph
        self.patch_convs = nn.ModuleList()
        self.patch_norms = nn.ModuleList()
        for _ in range(patch_layers):
            self.patch_convs.append(make_conv(patch_backbone, hidden_dim, hidden_dim, pna_delta))
            self.patch_norms.append(nn.LayerNorm(hidden_dim))

        # patch-level injection between patch GNN layers (not after final layer):
        #   patch_scalar_inject : inject patch-mean of dims [0,1] (Potential, NormalPotential)
        #   patch_inject_all    : inject patch-mean of dims [0,1] + [8:] — all signal scalars,
        #                         skipping normals [2:5] and xyz [5:8]. For 8D this equals [0,1];
        #                         for 14D this adds the 6 geometric features.
        # (mutually exclusive; patch_inject_all takes precedence if both set)
        n_signal = (2 + max(in_dim - 8, 0)) if patch_inject_all else (2 if patch_scalar_inject else 0)
        self.patch_scalar_projs = nn.ModuleList(
            [nn.Linear(hidden_dim + n_signal, hidden_dim) for _ in range(max(patch_layers - 1, 0))]
        ) if n_signal > 0 else nn.ModuleList()

        # attention-based readout over patch super-nodes (patch→graph)
        self.readout_attn = nn.Linear(hidden_dim, 1) if readout == "attn" else None

        self.dropout = nn.Dropout(dropout)
        # head sees patch readout (hidden) [+ node mean & max readout (2*hidden)]
        head_in = hidden_dim + (2 * hidden_dim if node_readout else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes),
        )
        # SupCon projection head (opt-in via forward(..., return_proj=True); unused
        # and untouched by gradients when supcon_weight=0, so existing checkpoints
        # remain loadable and every non-SupCon call site is unaffected).
        self.proj_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def _scatter(self, h, gpid, P):
        if self.pool == "attn":
            # learned per-node attention weight, softmax within each patch
            scores = self.attn_pool(h).squeeze(-1)                          # [N]
            # per-patch max subtraction for numerical stability
            patch_max = torch.full((P,), float("-inf"), device=h.device, dtype=h.dtype)
            patch_max = patch_max.scatter_reduce(0, gpid, scores, reduce="amax", include_self=True)
            scores = scores - patch_max[gpid]
            w = scores.exp()
            denom = torch.zeros(P, device=h.device, dtype=h.dtype).index_add_(0, gpid, w)
            w = w / denom[gpid].clamp(min=1e-8)
            return torch.zeros(P, h.shape[1], device=h.device, dtype=h.dtype).index_add_(
                0, gpid, h * w.unsqueeze(1))
        if self.pool in ("mean", "meanmax"):
            s = torch.zeros(P, h.shape[1], device=h.device, dtype=h.dtype).index_add_(0, gpid, h)
            cnt = torch.zeros(P, device=h.device, dtype=h.dtype).index_add_(
                0, gpid, torch.ones(h.shape[0], device=h.device, dtype=h.dtype))
            mean = s / cnt.clamp(min=1.0).unsqueeze(1)
        if self.pool in ("max", "meanmax"):
            mx = torch.full((P, h.shape[1]), float("-inf"), device=h.device, dtype=h.dtype)
            mx = mx.scatter_reduce(0, gpid.unsqueeze(1).expand(-1, h.shape[1]), h, reduce="amax",
                                   include_self=True)
            mx = torch.nan_to_num(mx, neginf=0.0)
        if self.pool == "mean":
            return mean
        if self.pool == "max":
            return mx
        return torch.cat([mean, mx], dim=1)

    def node_input(self, attr):
        """Features fed to the first node layer (all channels unless invariant)."""
        if not self.invariant:
            return attr
        radial = attr[:, self.xyz_slice].norm(dim=1, keepdim=True)   # distance to centre
        return torch.cat([attr[:, self.inv_idx], radial], dim=1) if self.inv_idx else radial

    def forward(self, bn, bp, gpid, bm=None, mid_gpid=None, return_proj=False, return_embedding=False,
                return_patch=False):
        """Forward pass.

        Standard mode: forward(bn, bp, gpid)
        Three-level:   forward(bn, bp, gpid, bm=mid_graph, mid_gpid=node_to_mid)
          where gpid = mid_to_final assignment.
        return_proj: when True, also return the L2-normalised SupCon projection
          (logits, proj); default False preserves the original single-tensor
          return for every existing caller.
        return_embedding: when True, also return the pre-head pooled graph
          representation `hg` (used for retrieval/similarity, e.g. SHREC21) --
          default False preserves the original return signature for every
          existing caller (SHREC19/25 training and eval are unaffected).
        return_patch: when True, return (logits, ph) where ph holds the final
          patch embeddings (input of the graph readout), for patch attribution.
        """
        h = self.node_input(bn.ndata["attr"])
        # cache the 2 scalar channels (Potential_norm, NormalPotential_norm) at
        # input resolution for re-injection between node layers
        scalars = h[:, :2].clamp(-5.0, 5.0) if self.scalar_inject else None

        if self.node_layers == 0:
            # bypass node GNN: project raw/ssl features directly to hidden_dim
            h = self.node_proj(h)
        else:
            n = len(self.node_convs)
            for i, (conv, norm) in enumerate(zip(self.node_convs, self.node_norms)):
                h_in = h
                h = norm(conv(bn, h))
                if h_in.shape[-1] == h.shape[-1]:
                    h = h + h_in
                if i < n - 1:
                    h = self.dropout(F.elu(h))
                    if self.scalar_inject:
                        # re-inject raw scalar features so each layer can
                        # reason about electrostatics at its own resolution
                        h = self.scalar_projs[i](torch.cat([h, scalars], dim=1))

        if self.flat:
            # no-hierarchy ablation: read out directly over node embeddings
            P = 0
            ph = h
        # Three-level path: pool nodes → mid-patches → final-patches
        elif self.three_level and bm is not None:
            M = bm.num_nodes()
            mh = self._scatter(h, mid_gpid, M)
            mh = self.mid_pool_proj(mh)
            m_len = len(self.mid_convs)
            for i, (conv, norm) in enumerate(zip(self.mid_convs, self.mid_norms)):
                mh_in = mh
                mh = norm(conv(bm, mh))
                if mh_in.shape[-1] == mh.shape[-1]:
                    mh = mh + mh_in
                if i < m_len - 1:
                    mh = self.dropout(F.elu(mh))
            # use mh as input to final pooling
            P = bp.num_nodes()
            ph = self._scatter(mh, gpid, P)
            ph = self.pool_proj(ph)
        else:
            P = bp.num_nodes()
            ph = self._scatter(h, gpid, P)
            ph = self.pool_proj(ph)

        # compute patch-mean features for patch-level injection
        if self.patch_inject_all or self.patch_scalar_inject:
            x_raw = bn.ndata["attr"]                                  # [N, in_dim]
            if self.patch_inject_all:
                # signal scalars only: [Pot, NormPot] + geom features beyond 8D
                # skips normals [2:5] and xyz [5:8]
                parts = [x_raw[:, :2].clamp(-5.0, 5.0)]
                if x_raw.shape[1] > 8:
                    parts.append(x_raw[:, 8:])
                raw = torch.cat(parts, dim=1)
            else:
                raw = x_raw[:, :2].clamp(-5.0, 5.0)
            n_feat = raw.shape[1]
            patch_feats = torch.zeros(P, n_feat, device=ph.device, dtype=ph.dtype)
            patch_count = torch.zeros(P,         device=ph.device, dtype=ph.dtype)
            patch_feats.index_add_(0, gpid, raw)
            patch_count.index_add_(0, gpid, torch.ones(gpid.shape[0], device=ph.device))
            patch_feats = patch_feats / patch_count.clamp(min=1.0).unsqueeze(1)  # [P, n_feat]

        m = len(self.patch_convs)
        for i, (conv, norm) in enumerate(zip(self.patch_convs, self.patch_norms)):
            ph_in = ph
            ph = norm(conv(bp, ph))
            if ph_in.shape[-1] == ph.shape[-1]:
                ph = ph + ph_in
            if i < m - 1:
                ph = self.dropout(F.elu(ph))
                if self.patch_inject_all or self.patch_scalar_inject:
                    ph = self.patch_scalar_projs[i](torch.cat([ph, patch_feats], dim=1))

        # graph-level readout over patch super-nodes (or over nodes when flat)
        g_out = bn if self.flat else bp
        with g_out.local_scope():
            g_out.ndata["h"] = ph
            if self.readout == "attn":
                # attention-weighted sum over patches within each graph
                scores = self.readout_attn(ph).squeeze(-1)  # [total_patches]
                # softmax within each graph using batch node counts
                batch_nids = torch.zeros(g_out.num_nodes(), dtype=torch.long, device=ph.device)
                offset = 0
                for i, n in enumerate(g_out.batch_num_nodes()):
                    batch_nids[offset:offset + n] = i
                    offset += n
                B = g_out.batch_size
                patch_max = torch.full((B,), float("-inf"), device=ph.device)
                patch_max = patch_max.scatter_reduce(0, batch_nids, scores, reduce="amax", include_self=True)
                scores = scores - patch_max[batch_nids]
                w = scores.exp()
                denom = torch.zeros(B, device=ph.device).index_add_(0, batch_nids, w)
                w = w / denom[batch_nids].clamp(min=1e-8)
                hg = torch.zeros(B, ph.shape[1], device=ph.device).index_add_(0, batch_nids, ph * w.unsqueeze(1))
            elif self.readout == "max":
                hg = dgl.max_nodes(g_out, "h")
            else:
                hg = dgl.mean_nodes(g_out, "h")

        # skip connection: pool node embeddings directly so the fine-mesh signal
        # reaches the classifier without going through the 32-patch bottleneck
        if self.node_readout:
            with bn.local_scope():
                bn.ndata["h"] = h
                hg = torch.cat([hg, dgl.mean_nodes(bn, "h"), dgl.max_nodes(bn, "h")], dim=1)
        if return_patch:
            return self.head(hg), ph
        if return_proj and return_embedding:
            return self.head(hg), F.normalize(self.proj_head(hg), dim=-1), hg
        if return_proj:
            return self.head(hg), F.normalize(self.proj_head(hg), dim=-1)
        if return_embedding:
            return self.head(hg), hg
        return self.head(hg)


def attn_pool_scatter(h, gpid, P, scorer):
    """Attention pooling of node embeddings h into P patches (softmax per patch)."""
    scores = scorer(h).squeeze(-1)
    pmax = torch.full((P,), float("-inf"), device=h.device, dtype=h.dtype)
    pmax = pmax.scatter_reduce(0, gpid, scores, reduce="amax", include_self=True)
    w = (scores - pmax[gpid]).exp()
    denom = torch.zeros(P, device=h.device, dtype=h.dtype).index_add_(0, gpid, w)
    w = w / denom[gpid].clamp(min=1e-8)
    return torch.zeros(P, h.shape[1], device=h.device, dtype=h.dtype).index_add_(
        0, gpid, h * w.unsqueeze(1))


class MultiScaleHierGNN(nn.Module):
    """HSNet/N2P with several patch resolutions on one shared node encoder.

    node encoder (shared) -> for each scale s: attention pooling into k_s patches
    -> patch GNN -> global max readout -> fuse the S graph vectors -> head.

    fusion:  "gate"   learned per-graph softmax weights over scales (weights are
                      returned by forward(..., return_gates=True) for analysis)
             "concat" concatenate the S vectors and project back to hidden_dim
             "mean"   plain average
    share_patch_gnn: one patch GNN for all scales, so the model has nearly the
                     parameter count of the single-scale model (param-matched arm).
    """

    def __init__(self, in_dim, hidden_dim, num_classes, n_scales, node_layers=3,
                 patch_layers=2, dropout=0.3, node_backbone="gvp", patch_backbone="gin",
                 pna_delta=6.0, xyz_slice=slice(5, 8), nor_slice=slice(2, 5),
                 invariant=False, inv_idx=None, gvp_normals=False,
                 fusion="gate", share_patch_gnn=False):
        super().__init__()
        self.n_scales, self.fusion, self.share = n_scales, fusion, share_patch_gnn
        self.invariant, self.inv_idx, self.xyz_slice = invariant, list(inv_idx or []), xyz_slice
        gvp_nor = nor_slice if gvp_normals else None
        self.node_convs, self.node_norms = nn.ModuleList(), nn.ModuleList()
        d = in_dim
        for _ in range(node_layers):
            self.node_convs.append(make_conv(node_backbone, d, hidden_dim, pna_delta,
                                             xyz_slice, nor_slice, gvp_nor))
            self.node_norms.append(nn.LayerNorm(hidden_dim))
            d = hidden_dim
        self.attn_pools = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_scales)])
        n_gnn = 1 if share_patch_gnn else n_scales
        self.patch_convs = nn.ModuleList([
            nn.ModuleList([make_conv(patch_backbone, hidden_dim, hidden_dim, pna_delta)
                           for _ in range(patch_layers)]) for _ in range(n_gnn)])
        self.patch_norms = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(patch_layers)])
            for _ in range(n_gnn)])
        if fusion == "gate":
            self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
                                      nn.Linear(hidden_dim // 2, 1))
        elif fusion == "concat":
            self.fuse = nn.Linear(n_scales * hidden_dim, hidden_dim)
        elif fusion != "mean":
            raise ValueError(f"unknown fusion {fusion!r}")
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes))

    node_input = HierarchicalGNN.node_input

    def forward(self, bn, bps, gpids, return_gates=False, return_patch=False, **_):
        h = self.node_input(bn.ndata["attr"])
        n = len(self.node_convs)
        for i, (conv, norm) in enumerate(zip(self.node_convs, self.node_norms)):
            h_in = h
            h = norm(conv(bn, h))
            if h_in.shape[-1] == h.shape[-1]:
                h = h + h_in
            if i < n - 1:
                h = self.dropout(F.elu(h))
        graph_vecs, patch_embs = [], []
        for s in range(self.n_scales):
            bp, gpid = bps[s], gpids[s]
            ph = attn_pool_scatter(h, gpid, bp.num_nodes(), self.attn_pools[s])
            convs = self.patch_convs[0 if self.share else s]
            norms = self.patch_norms[0 if self.share else s]
            for i, (conv, norm) in enumerate(zip(convs, norms)):
                ph_in = ph
                ph = norm(conv(bp, ph)) + ph_in
                if i < len(convs) - 1:
                    ph = self.dropout(F.elu(ph))
            patch_embs.append(ph)
            with bp.local_scope():
                bp.ndata["h"] = ph
                graph_vecs.append(dgl.max_nodes(bp, "h"))
        H = torch.stack(graph_vecs, dim=1)                       # [B, S, hidden]
        gates = None
        if self.fusion == "gate":
            gates = torch.softmax(self.gate(H).squeeze(-1), dim=1)  # [B, S]
            hg = (gates.unsqueeze(-1) * H).sum(dim=1)
        elif self.fusion == "concat":
            hg = self.fuse(H.flatten(1))
        else:
            hg = H.mean(dim=1)
        logits = self.head(hg)
        if return_patch:
            return logits, patch_embs, gates
        if return_gates:
            return logits, gates
        return logits


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def supcon_loss(proj, labels, temp=0.1):
    """Supervised contrastive loss (Khosla et al. 2020) on L2-normalised
    projections. Ported from train_ri_hiernet.py's tested implementation."""
    B = proj.shape[0]
    sim = (proj @ proj.T) / temp
    self_mask = ~torch.eye(B, dtype=torch.bool, device=proj.device)
    pos_mask = (labels.view(-1, 1) == labels.view(1, -1)) & self_mask
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
    exp = torch.exp(sim) * self_mask
    log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
    pos_count = pos_mask.sum(1)
    loss = -(pos_mask * log_prob).sum(1) / pos_count.clamp(min=1)
    loss = loss[pos_count > 0]
    return loss.mean() if loss.numel() > 0 else proj.new_zeros(())


def run_epoch(model, loader, criterion, device, optimizer=None, num_classes=None, desc="",
              supcon_weight=0.0, supcon_temp=0.1, collect_scores=False):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, preds, labs = 0.0, [], []
    scores = [] if collect_scores else None
    ctx = torch.enable_grad() if train else torch.no_grad()
    use_supcon = supcon_weight > 0
    with ctx:
        for batch in tqdm(loader, desc=desc, leave=False):
            if isinstance(batch[1], list):  # multi-scale: (bn, [bp_s], [gpid_s], y)
                bn, bps, gpids, y = batch
                bn = bn.to(device); y = y.to(device)
                bps = [b.to(device) for b in bps]; gpids = [g.to(device) for g in gpids]
                out = model(bn, bps, gpids)
            elif len(batch) == 6:  # three-level: (bn, bm, bp, mid_gpid, gpid, y)
                bn, bm, bp, mid_gpid, gpid, y = batch
                bn = bn.to(device); bm = bm.to(device); bp = bp.to(device)
                mid_gpid = mid_gpid.to(device); gpid = gpid.to(device); y = y.to(device)
                out = model(bn, bp, gpid, bm, mid_gpid, return_proj=use_supcon)
            else:
                bn, bp, gpid, y = batch
                bn = bn.to(device); bp = bp.to(device); gpid = gpid.to(device); y = y.to(device)
                out = model(bn, bp, gpid, return_proj=use_supcon)
            if use_supcon:
                logits, proj = out
                loss = criterion(logits, y) + supcon_weight * supcon_loss(proj, y, supcon_temp)
            else:
                logits = out
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            preds.extend(logits.argmax(1).cpu().numpy()); labs.extend(y.cpu().numpy())
            if collect_scores:
                scores.append(F.softmax(logits, dim=1).detach().cpu().numpy())
    m = compute_metrics(labs, preds, num_classes=num_classes)
    m["loss"] = total_loss / max(len(loader), 1)
    if collect_scores:
        return m, np.asarray(labs), np.concatenate(scores, axis=0)
    return m


def compute_map(labs, scores, num_classes):
    """One-vs-rest mean average precision from softmax scores, averaged over the
    classes present in `labs` (matches the appendix/ProCLR-style protocol)."""
    from sklearn.metrics import average_precision_score
    labs = np.asarray(labs)
    Y = np.zeros((len(labs), num_classes), dtype=float)
    Y[np.arange(len(labs)), labs] = 1.0
    present = Y.sum(0) > 0
    if present.sum() == 0:
        return float("nan")
    try:
        ap = average_precision_score(Y[:, present], scores[:, present], average=None)
        return float(np.mean(ap))
    except Exception:
        return float("nan")


def xyz_nor_slices(feature_mode):
    """xyz / normal slices of the node feature vector (as used in main())."""
    xyz = slice(0, 3) if feature_mode in ("coords", "coords_normals", "coords_potential",
                                          "coords_hks", "coords_normals_hks") else slice(5, 8)
    nor = slice(3, 6) if feature_mode in ("coords_normals", "coords_normals_hks") else slice(2, 5)
    return xyz, nor


def build_model(cfg, raw_in_dim, num_classes):
    """Build the model described by a run config (argparse namespace or the
    `config` dict of final_results.json). Used by main() and by the analysis
    scripts, so a checkpoint is always rebuilt exactly as it was trained.
    Keys missing from old configs fall back to the behaviour of the old code."""
    get = (lambda k, d=None: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    fmode = get("feature_mode", "all")
    xyz_sl, nor_sl = xyz_nor_slices(fmode)
    invariant = bool(get("invariant", False))
    inv_idx, gvp_normals, in_dim = None, False, raw_in_dim
    if invariant:
        if fmode not in FEATURE_LAYOUT:
            raise ValueError(f"--invariant needs a feature mode in {list(FEATURE_LAYOUT)}, got {fmode!r}")
        if get("node_backbone") != "gvp":
            raise ValueError("--invariant is defined for the GVP-inspired (HiPatchNet) node encoder only")
        _, layout_nor, inv_idx = FEATURE_LAYOUT[fmode]
        gvp_normals = get("gvp_normals")
        gvp_normals = (layout_nor is not None) if gvp_normals is None else bool(gvp_normals)
        if gvp_normals and layout_nor is None:
            raise ValueError(f"--gvp-normals: feature mode {fmode!r} has no normals")
        in_dim = len(inv_idx) + 1
    common = dict(node_layers=get("node_layers", 6), patch_layers=get("patch_layers", 2),
                  dropout=get("dropout", 0.3), node_backbone=get("node_backbone", "gin"),
                  patch_backbone=get("patch_backbone", "gin"), pna_delta=get("pna_delta") or 6.0,
                  xyz_slice=xyz_sl, nor_slice=nor_sl, invariant=invariant, inv_idx=inv_idx,
                  gvp_normals=gvp_normals)
    if get("multi_assign_dirs"):
        return MultiScaleHierGNN(in_dim, get("hidden_dim", 512), num_classes,
                                 n_scales=len(get("multi_assign_dirs")),
                                 fusion=get("fusion", "gate"),
                                 share_patch_gnn=bool(get("share_patch_gnn", False)), **common)
    return HierarchicalGNN(in_dim, get("hidden_dim", 512), num_classes,
                           pool=get("pool", "meanmax"), readout=get("readout", "max"),
                           node_readout=bool(get("node_readout", False)),
                           scalar_inject=bool(get("scalar_inject", False)),
                           patch_scalar_inject=bool(get("patch_scalar_inject", False)),
                           patch_inject_all=bool(get("patch_inject_all", False)),
                           three_level=get("assign_dir2") is not None,
                           mid_layers=get("mid_layers", 2), proj_dim=get("proj_dim", 128),
                           flat=bool(get("flat", False)), **common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="/home/ogharbi/datasets/SHREC25/ssl_graphs_1k")
    ap.add_argument("--assign-dir", default="/home/ogharbi/datasets/SHREC25/ssl_graphs_1k_patchassign_k32")
    ap.add_argument("--train-labels-csv", default=None,
                    help="default: <graph-dir>/train_split.csv if present")
    ap.add_argument("--val-labels-csv", default=None,
                    help="default: <graph-dir>/val_split.csv if present")
    ap.add_argument("--test-labels-csv", default=None)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--node-layers", type=int, default=6)
    ap.add_argument("--patch-layers", type=int, default=2)
    ap.add_argument("--pool", choices=["mean", "max", "meanmax", "attn"], default="meanmax")
    ap.add_argument("--readout", choices=["mean", "max", "attn"], default="max")
    ap.add_argument("--node-readout", action=argparse.BooleanOptionalAction, default=False,
                    help="HARMFUL — do not enable. A global node mean+max readout skip was tested "
                         "as a fix for the patch bottleneck; controlled run (hier_skip_clean) showed "
                         "it CUTS test macro-F1 from 0.22 to 0.045 (it shortcuts to majority classes "
                         "and drowns the patch path). Kept only for reproducibility of that negative result.")
    ap.add_argument("--flat", action="store_true", default=False,
                    help="No-hierarchy ablation: skip patch pooling and the patch GNN; the "
                         "--readout is applied over all node embeddings. Data, splits, loss "
                         "and head are unchanged, so the patch stage is the only difference.")
    ap.add_argument("--node-backbone",
                    choices=["gin", "sage", "gcn", "gat", "gatv2", "pna", "gt", "gps", "sgformer",
                             "ri_gin", "lrf_gin", "msg_gin", "gvp"],
                    default="gin", help="GNN backbone for the node-level encoder. "
                    "lrf_gin: uses precomputed 8D LRF edge features (requires ssl_graphs_1k_geom_enriched_ri). "
                    "msg_gin: multi-scale kNN-6/16/32 aggregation (built on-the-fly from xyz).")
    ap.add_argument("--patch-backbone",
                    choices=["gin", "sage", "gcn", "gat", "gatv2", "pna", "gt", "gps", "sgformer"],
                    default="gin", help="GNN backbone for the patch-level encoder")
    ap.add_argument("--pna-delta", type=float, default=None,
                    help="PNA avg in-degree (auto-computed from data if not set)")
    ap.add_argument("--feature-mode", choices=["all", "coords", "hks", "coords_hks", "coords_normals_hks", "coords_potential", "coords_normals", "all_no_normpot", "all_no_pot", "geom_14d", "geom_14d_raw"],
                    default="all",
                    help="all=8 features (default); coords=xyz only (3-dim, pos field); "
                         "hks=16D intrinsic heat-kernel signature (no xyz: not for the gvp backbone); coords_hks=xyz+HKS; "
                         "coords_normals_hks=xyz+normals+HKS (22-dim, supports --invariant); "
                         "coords_normals=xyz+normals (6-dim); "
                         "geom_14d=14D (8D base + dist_centroid/anchors + surf_variation, requires geom-enriched graph-dir)")
    ap.add_argument("--rwpe-dim", type=int, default=0,
                    help="Append k-step Random Walk PE to node features (0=disabled). "
                         "Adds global topology signal; k=16 is standard.")
    ap.add_argument("--scalar-inject", action="store_true", default=False,
                    help="Re-inject Potential+NormalPotential scalars between every node GNN layer "
                         "(negative result — kept for reproducibility).")
    ap.add_argument("--patch-scalar-inject", action="store_true", default=False,
                    help="Inject patch-mean of (Potential, NormalPotential) between patch GNN layers. "
                         "Coarser-scale electrostatic signal at the patch level.")
    ap.add_argument("--patch-inject-all", action="store_true", default=False,
                    help="Inject patch-mean of all signal scalars between patch GNN layers: "
                         "dims [0,1] (Potential, NormalPotential) + dims [8:] (geom features if any). "
                         "Skips structural dims [2:8] (normals + xyz). "
                         "For 8D: injects 2 features; for 14D: injects 8 features.")
    ap.add_argument("--ssl-emb-dir", default=None,
                    help="If set, load pre-extracted SSL node embeddings from this directory "
                         "(organized as {split}/{graph_id}.pt) instead of raw node features.")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--loss-type", choices=["ce", "focal"], default="ce")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="Label smoothing for CrossEntropyLoss (0=off, 0.1 recommended)")
    ap.add_argument("--focal-gamma", type=float, default=2.0,
                    help="Focal loss focusing parameter (only used when --loss-type focal)")
    ap.add_argument("--supcon-weight", type=float, default=0.0,
                    help="Weight of the supervised contrastive loss added to CE (0=off)")
    ap.add_argument("--supcon-temp", type=float, default=0.1,
                    help="Temperature for the supervised contrastive loss")
    ap.add_argument("--proj-dim", type=int, default=128,
                    help="Output dim of the SupCon projection head")
    ap.add_argument("--class-weight-method", default="inverse_sqrt",
                    choices=["none", "inverse", "inverse_sqrt", "effective_samples"])
    ap.add_argument("--augment", action="store_true",
                    help="Apply SO(3) rotation + random scale + jitter during training")
    ap.add_argument("--aug-scale-std", type=float, default=0.1,
                    help="Std of uniform scale perturbation (±aug_scale_std)")
    ap.add_argument("--aug-jitter-std", type=float, default=0.005,
                    help="Std of Gaussian jitter added to xyz coords")
    ap.add_argument("--balanced-sampler", action="store_true",
                    help="Use WeightedRandomSampler to oversample minority classes")
    ap.add_argument("--sampler-power", type=float, default=0.5,
                    help="Sampling weight = 1/count^power (0=uniform, 1=fully balanced)")
    ap.add_argument("--lr-schedule", choices=["none", "cosine"], default="none",
                    help="LR schedule: none (patience-based early stop) or cosine annealing")
    ap.add_argument("--warmup-epochs", type=int, default=5,
                    help="Linear LR warmup epochs (used with --lr-schedule cosine)")
    ap.add_argument("--assign-dir2", default=None,
                    help="Option 3 — three-level hierarchy: intermediate patch assignment dir "
                         "(e.g. ssl_graphs_1k_patchassign_k128). --assign-dir becomes the FINAL level.")
    ap.add_argument("--use-bidirectional-edges", action="store_true",
                    help="Use edge_index when available. Intended for the OFF-derived "
                         "TOSCA/SHREC11 graphs; disabled by default to preserve the "
                         "legacy SHREC19/SHREC25 protocol.")
    ap.add_argument("--mid-layers", type=int, default=2,
                    help="GIN layers in the intermediate (mid) pooling stage (three-level only).")
    ap.add_argument("--topk-classes", type=int, default=None)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--output-dir", default="runs")
    # ---- future-work options (all off by default: old runs are unchanged) ----
    ap.add_argument("--invariant", action="store_true", default=False,
                    help="Exactly rotation-invariant HiPatchNet: the node encoder gets only the "
                         "invariant channels (+ distance to centre); xyz/normals enter only as "
                         "relative geometry / GVP vectors. Requires --node-backbone gvp.")
    ap.add_argument("--gvp-normals", action=argparse.BooleanOptionalAction, default=None,
                    help="With --invariant: feed surface normals to the GVP messages as vectors "
                         "(default: on when the feature mode has normals).")
    ap.add_argument("--test-rotations", type=int, default=0,
                    help="After training, also evaluate the test set under N fixed random SO(3) "
                         "rotations (rotation test); results in final_results['rotation_test'].")
    ap.add_argument("--multi-assign-dirs", nargs="+", default=None,
                    help="Multi-scale model: one patch-assignment dir per resolution "
                         "(e.g. ..._k16 ..._k64 ..._k256). Replaces --assign-dir.")
    ap.add_argument("--fusion", choices=["gate", "concat", "mean"], default="gate",
                    help="Multi-scale fusion of the per-resolution graph vectors.")
    ap.add_argument("--share-patch-gnn", action="store_true", default=False,
                    help="Multi-scale: one patch GNN shared by all resolutions (param-matched).")
    args = ap.parse_args()
    multi = args.multi_assign_dirs is not None
    if multi:
        bad = [f for f, on in (("--assign-dir2", args.assign_dir2 is not None), ("--flat", args.flat),
                               ("--augment", args.augment), ("--supcon-weight", args.supcon_weight > 0),
                               ("--ssl-emb-dir", args.ssl_emb_dir is not None)) if on]
        if bad:
            ap.error(f"--multi-assign-dirs cannot be combined with {bad}")
        args.assign_dir = args.multi_assign_dirs[0]

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # default to dataset-local splits (matches the 81% full-graph baseline protocol)
    gdir = Path(args.graph_dir)
    if args.train_labels_csv is None and (gdir / "train_split.csv").exists():
        args.train_labels_csv = str(gdir / "train_split.csv")
    if args.val_labels_csv is None and (gdir / "val_split.csv").exists():
        args.val_labels_csv = str(gdir / "val_split.csv")

    ssl_tag = "_ssl" if args.ssl_emb_dir else ""
    bb_tag = f"_{args.node_backbone}" if args.node_backbone != "gin" else ""
    run_name = args.run_name or f"hier_h{args.hidden_dim}_n{args.node_layers}_p{args.patch_layers}{ssl_tag}{bb_tag}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_dir} | device {device}")
    print(f"node_layers={args.node_layers} patch_layers={args.patch_layers} "
          f"pool={args.pool} readout={args.readout} hidden={args.hidden_dim}")

    # labels
    train_labels, num_classes = load_labels(args.graph_dir, "train", args.train_labels_csv)
    test_labels, _ = load_labels(args.graph_dir, "test", args.test_labels_csv)
    train_labels, test_labels, _ = select_topk_populated_classes(train_labels, test_labels, args.topk_classes)
    if args.topk_classes:
        num_classes = args.topk_classes

    use_lrf = (args.node_backbone == "lrf_gin")
    three_level = (args.assign_dir2 is not None)
    _ds_kw = dict(ssl_emb_dir=args.ssl_emb_dir, feature_mode=args.feature_mode,
                  rwpe_dim=args.rwpe_dim, use_lrf_feat=use_lrf, assign_dir2=args.assign_dir2,
                  use_bidirectional_edges=args.use_bidirectional_edges)
    _col = collate_multi if multi else (collate_three if three_level else collate)
    if multi:
        def make_ds(split, labels):
            return MultiScalePatchDataset(args.graph_dir, args.multi_assign_dirs, split, labels, **_ds_kw)
    else:
        def make_ds(split, labels):
            return HierPatchDataset(args.graph_dir, args.assign_dir, split, labels, **_ds_kw)

    full_train = make_ds("train", train_labels)
    test_ds = make_ds("test", test_labels)

    if args.val_labels_csv:
        val_labels, _ = load_labels(args.graph_dir, "train", args.val_labels_csv)
        train_ds = full_train
        val_ds = make_ds("train", val_labels)
        num_classes = len(set(train_labels.values()) | set(val_labels.values()))
    else:
        tr_idx, va_idx = create_robust_train_val_split(full_train, train_labels,
                                                       val_split=args.val_split, seed=args.seed)
        train_ds, val_ds = Subset(full_train, tr_idx), Subset(full_train, va_idx)

    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} classes={num_classes}")

    # record the exact split ids so paired runs (e.g. hier vs --flat) can be
    # checked to have trained, early-stopped and been tested on identical data
    def _split_ids(ds):
        base = ds.dataset if isinstance(ds, Subset) else ds
        idx = ds.indices if isinstance(ds, Subset) else range(len(ds))
        return [str(base.items[i][0]) for i in idx]
    with open(run_dir / "split_ids.json", "w") as f:
        json.dump({"train": _split_ids(train_ds), "val": _split_ids(val_ds),
                   "test": _split_ids(test_ds)}, f)

    # DataLoader overhead-only tuning (numerically neutral — identical batches/order):
    # keep workers alive across epochs instead of re-spawning them every epoch, and
    # prefetch a few more batches (spare RAM) to keep the GPU fed. No effect on results.
    _loader_extra = {}
    if args.num_workers > 0:
        _loader_extra["persistent_workers"] = True
        _loader_extra["prefetch_factor"] = 4

    # oversampling: weight each sample by 1/count^power
    if args.balanced_sampler:
        labels_list = [train_labels[full_train.items[i][0]] for i in
                       (train_ds.indices if hasattr(train_ds, "indices") else range(len(train_ds)))]
        counts = np.bincount(labels_list, minlength=num_classes).astype(np.float64)
        sw = [1.0 / max(counts[l], 1) ** args.sampler_power for l in labels_list]
        sampler = WeightedRandomSampler(torch.as_tensor(sw, dtype=torch.double),
                                        num_samples=len(sw), replacement=True)
        print(f"  WeightedRandomSampler: power={args.sampler_power}")
        train_collate = (collate_aug(args.aug_scale_std, args.aug_jitter_std, args.feature_mode)
                         if args.augment and not three_level else _col)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, collate_fn=train_collate,
                                  **_loader_extra)
    else:
        train_collate = (collate_aug(args.aug_scale_std, args.aug_jitter_std, args.feature_mode)
                         if args.augment and not three_level else _col)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=train_collate,
                                  **_loader_extra)
    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=_col, **_loader_extra)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=_col, **_loader_extra)

    sample = full_train[0][0]
    in_dim = sample.ndata["attr"].shape[1]

    # auto-compute PNA delta (avg in-degree) from a sample of training graphs
    pna_delta = args.pna_delta
    if pna_delta is None and ("pna" in (args.node_backbone, args.patch_backbone)):
        degs = []
        for i in range(min(200, len(full_train))):
            g, _, _ = full_train[i]
            degs.append(g.in_degrees().float().mean().item())
        pna_delta = float(np.mean(degs))
        print(f"  PNA delta (avg in-degree): {pna_delta:.2f}")
    elif pna_delta is None:
        pna_delta = 6.0
    args.pna_delta = pna_delta  # persist the resolved value so it round-trips
                                 # through final_results.json / cfg_out for any
                                 # downstream reload (ensemble_predict.py,
                                 # reeval_all_90class.py) to reconstruct the
                                 # exact trained model instead of silently
                                 # falling back to the class default (6.0).

    # xyz / normal slices are resolved inside build_model (xyz_nor_slices); the
    # construction is shared with the analysis scripts so checkpoints round-trip.
    try:
        model = build_model(args, in_dim, num_classes).to(device)
    except ValueError as e:
        ap.error(str(e))
    if args.invariant:
        print(f"  invariant HiPatchNet: node input = channels {FEATURE_LAYOUT[args.feature_mode][2]} "
              f"+ distance to centre; GVP normals={model.node_convs[0].nor_slice is not None}")
    n_params = int(sum(p.numel() for p in model.parameters()))
    n_params_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"in_dim={in_dim} | node_backbone={args.node_backbone} | patch_backbone={args.patch_backbone}"
          f" | ssl={args.ssl_emb_dir is not None}"
          f" | params={n_params:,}")

    if args.class_weight_method == "none":
        weights = None
    else:
        weights = compute_class_weights(train_labels, num_classes, device, method=args.class_weight_method)
    if args.loss_type == "focal":
        criterion = FocalLoss(alpha=weights, gamma=args.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # LR schedule: cosine annealing with linear warmup, or none (patience-based)
    if args.lr_schedule == "cosine":
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=args.lr * 0.01)
        print(f"  CosineAnnealingLR: warmup={args.warmup_epochs} epochs, T_max={args.epochs - args.warmup_epochs}")
    else:
        cosine_sched = None

    best_val_f1, best_state, wait = -1.0, None, 0
    best_epoch = 0
    history = []
    train_t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        # linear warmup: scale LR from lr/10 → lr over warmup_epochs
        if cosine_sched and ep <= args.warmup_epochs:
            warmup_factor = ep / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * warmup_factor

        tr = run_epoch(model, train_loader, criterion, device, optimizer, num_classes, f"ep{ep} train",
                      supcon_weight=args.supcon_weight, supcon_temp=args.supcon_temp)
        # SupCon is a training-time representation regulariser only; val/test stay
        # pure-CE so metrics remain directly comparable across SupCon and non-SupCon runs.
        va = run_epoch(model, val_loader, criterion, device, None, num_classes, f"ep{ep} val")
        history.append({"epoch": ep, "train": tr, "val": va})

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {ep}/{args.epochs}  lr={cur_lr:.2e}  "
              f"Train loss {tr['loss']:.3f} acc {tr['accuracy']:.3f} f1 {tr['f1_macro']:.3f} | "
              f"Val loss {va['loss']:.3f} acc {va['accuracy']:.3f} f1 {va['f1_macro']:.3f} balacc {va['balanced_accuracy']:.3f}")

        if va["f1_macro"] > best_val_f1:
            best_val_f1 = va["f1_macro"]
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, run_dir / "best_model.pt")
            wait = 0
            print(f"  * new best val f1 {best_val_f1:.4f} (acc {va['accuracy']:.4f})")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  early stop at epoch {ep} (no val f1 improvement for {args.patience} epochs)")
                break

        if cosine_sched and ep > args.warmup_epochs:
            cosine_sched.step()

    train_time_sec = time.perf_counter() - train_t0
    epochs_run = len(history)

    if best_state:
        model.load_state_dict(best_state)
    test_m, test_labs, test_scores = run_epoch(model, test_loader, criterion, device, None,
                                               num_classes, "test", collect_scores=True)
    test_m["mAP"] = compute_map(test_labs, test_scores, num_classes)
    test_m.update(present_class_metrics(test_labs, test_scores.argmax(1)))
    print("\n=== FINAL TEST (best val model) ===")
    for k in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
              "tpr_macro", "tnr_macro", "mAP", "f1_macro_present", "balanced_accuracy_present"]:
        print(f"  {k:18s}: {test_m[k]:.4f}")

    # rotation test: same test set, every graph under a fixed random rotation
    rotation_test = None
    if args.test_rotations > 0:
        per_rot = []
        for r in range(args.test_rotations):
            rot_loader = DataLoader(RotatedView(test_ds, args.feature_mode, seed=1000 + r),
                                    batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, collate_fn=_col)
            rm, rl, rs = run_epoch(model, rot_loader, criterion, device, None, num_classes,
                                   f"rot{r}", collect_scores=True)
            rm["mAP"] = compute_map(rl, rs, num_classes)
            rm.update(present_class_metrics(rl, rs.argmax(1)))
            per_rot.append(rm)
            print(f"  rotation {r}: acc {rm['accuracy']:.4f}  f1w {rm['f1_weighted']:.4f}  "
                  f"f1m(present) {rm['f1_macro_present']:.4f}")
        keys = ["accuracy", "f1_weighted", "f1_macro_present", "balanced_accuracy_present", "mAP"]
        rotation_test = {"n_rotations": args.test_rotations, "per_rotation": per_rot,
                         "mean": {k: float(np.mean([m[k] for m in per_rot])) for k in keys},
                         "std": {k: float(np.std([m[k] for m in per_rot])) for k in keys}}
        print(f"  rotated test (mean of {args.test_rotations}): "
              f"acc {rotation_test['mean']['accuracy']:.4f}  f1w {rotation_test['mean']['f1_weighted']:.4f}")
    print(f"\n[timing] params={n_params:,} | trainable={n_params_trainable:,} | "
          f"epochs_run={epochs_run} | best_epoch={best_epoch} | "
          f"train_time_sec={train_time_sec:.1f} ({train_time_sec/60:.1f} min) | "
          f"sec_per_epoch={train_time_sec/max(epochs_run,1):.2f}")

    cfg_out = vars(args)
    cfg_out["in_dim"] = in_dim
    cfg_out["num_classes"] = num_classes
    json.dump({"final_test_metrics": test_m, "config": cfg_out,
               "rotation_test": rotation_test,
               "best_val_f1": best_val_f1,
               "n_params": n_params, "n_params_trainable": n_params_trainable,
               "epochs_run": epochs_run, "best_epoch": best_epoch,
               "train_time_sec": train_time_sec,
               "sec_per_epoch": train_time_sec / max(epochs_run, 1),
               "history": history},
              open(run_dir / "final_results.json", "w"), indent=2, default=float)
    print(f"\nsaved -> {run_dir/'final_results.json'}")


if __name__ == "__main__":
    main()
