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
import sys, argparse, os, json, math
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
                 use_lrf_feat: bool = False, assign_dir2=None):
        self.gdir = resolve_split_dir(graph_dir, split)
        self.adir = resolve_split_dir(assign_dir, split)
        self.use_lrf_feat = use_lrf_feat
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

        ei = d.get("edge_index_mesh", d.get("edge_index"))

        if self._ssl_index:
            ssl_path = self._ssl_index.get(gp.stem)
            if ssl_path is None:
                raise FileNotFoundError(f"No SSL embedding for {gp.stem!r}")
            sd = torch.load(ssl_path, map_location="cpu", weights_only=False)
            x = sd["ssl_emb"].float()
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


def _random_so3():
    """Sample a uniformly random 3×3 rotation matrix via QR decomposition."""
    Q, R = torch.linalg.qr(torch.randn(3, 3))
    Q = Q * torch.sign(torch.diag(R))  # ensure det = +1
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


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


def make_conv(backbone, in_dim, out_dim, pna_delta=6.0, xyz_slice=slice(5, 8)):
    if backbone == "gin":
        return GINConv(gin_mlp(in_dim, out_dim), "sum", learn_eps=True)
    elif backbone == "lrf_gin":
        return LRFConvLayer(in_dim, out_dim)
    elif backbone == "msg_gin":
        return MSGINLayer(in_dim, out_dim, xyz_slice=xyz_slice)
    elif backbone == "ri_gin":
        return RIConvLayer(in_dim, out_dim)
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
                 xyz_slice=slice(5, 8)):
        super().__init__()
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
            self.node_convs.append(make_conv(node_backbone, d, hidden_dim, pna_delta, xyz_slice))
            self.node_norms.append(nn.LayerNorm(hidden_dim))
            d = hidden_dim
        # when node_layers=0, project raw features to hidden_dim for pooling
        self.node_proj = nn.Linear(in_dim, hidden_dim) if node_layers == 0 else nn.Identity()

        # node-level scalar re-injection (failed experiment — left for reproducibility)
        self.scalar_projs = nn.ModuleList(
            [nn.Linear(hidden_dim + 2, hidden_dim) for _ in range(max(node_layers - 1, 0))]
        ) if scalar_inject else nn.ModuleList()

        # learned attention scorer for attention pooling
        self.attn_pool = nn.Linear(hidden_dim, 1) if pool == "attn" else None

        # projection after node->patch pooling (meanmax doubles width)
        pooled_dim = hidden_dim * (2 if pool == "meanmax" else 1)
        self.pool_proj = nn.Linear(pooled_dim, hidden_dim) if pooled_dim != hidden_dim else nn.Identity()

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

    def forward(self, bn, bp, gpid, bm=None, mid_gpid=None):
        """Forward pass.

        Standard mode: forward(bn, bp, gpid)
        Three-level:   forward(bn, bp, gpid, bm=mid_graph, mid_gpid=node_to_mid)
          where gpid = mid_to_final assignment.
        """
        h = bn.ndata["attr"]
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

        # Three-level path: pool nodes → mid-patches → final-patches
        if self.three_level and bm is not None:
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

        with bp.local_scope():
            bp.ndata["h"] = ph
            if self.readout == "attn":
                # attention-weighted sum over patches within each graph
                scores = self.readout_attn(ph).squeeze(-1)  # [total_patches]
                # softmax within each graph using batch node counts
                batch_nids = torch.zeros(bp.num_nodes(), dtype=torch.long, device=ph.device)
                offset = 0
                for i, n in enumerate(bp.batch_num_nodes()):
                    batch_nids[offset:offset + n] = i
                    offset += n
                B = bp.batch_size
                patch_max = torch.full((B,), float("-inf"), device=ph.device)
                patch_max = patch_max.scatter_reduce(0, batch_nids, scores, reduce="amax", include_self=True)
                scores = scores - patch_max[batch_nids]
                w = scores.exp()
                denom = torch.zeros(B, device=ph.device).index_add_(0, batch_nids, w)
                w = w / denom[batch_nids].clamp(min=1e-8)
                hg = torch.zeros(B, ph.shape[1], device=ph.device).index_add_(0, batch_nids, ph * w.unsqueeze(1))
            elif self.readout == "max":
                hg = dgl.max_nodes(bp, "h")
            else:
                hg = dgl.mean_nodes(bp, "h")

        # skip connection: pool node embeddings directly so the fine-mesh signal
        # reaches the classifier without going through the 32-patch bottleneck
        if self.node_readout:
            with bn.local_scope():
                bn.ndata["h"] = h
                hg = torch.cat([hg, dgl.mean_nodes(bn, "h"), dgl.max_nodes(bn, "h")], dim=1)
        return self.head(hg)


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion, device, optimizer=None, num_classes=None, desc=""):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, preds, labs = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=desc, leave=False):
            if len(batch) == 6:  # three-level: (bn, bm, bp, mid_gpid, gpid, y)
                bn, bm, bp, mid_gpid, gpid, y = batch
                bn = bn.to(device); bm = bm.to(device); bp = bp.to(device)
                mid_gpid = mid_gpid.to(device); gpid = gpid.to(device); y = y.to(device)
                logits = model(bn, bp, gpid, bm, mid_gpid)
            else:
                bn, bp, gpid, y = batch
                bn = bn.to(device); bp = bp.to(device); gpid = gpid.to(device); y = y.to(device)
                logits = model(bn, bp, gpid)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            preds.extend(logits.argmax(1).cpu().numpy()); labs.extend(y.cpu().numpy())
    m = compute_metrics(labs, preds, num_classes=num_classes)
    m["loss"] = total_loss / max(len(loader), 1)
    return m


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
    ap.add_argument("--node-backbone",
                    choices=["gin", "sage", "gcn", "gat", "gatv2", "pna", "gt", "gps", "sgformer",
                             "ri_gin", "lrf_gin", "msg_gin"],
                    default="gin", help="GNN backbone for the node-level encoder. "
                    "lrf_gin: uses precomputed 8D LRF edge features (requires ssl_graphs_1k_geom_enriched_ri). "
                    "msg_gin: multi-scale kNN-6/16/32 aggregation (built on-the-fly from xyz).")
    ap.add_argument("--patch-backbone",
                    choices=["gin", "sage", "gcn", "gat", "gatv2", "pna", "gt", "gps", "sgformer"],
                    default="gin", help="GNN backbone for the patch-level encoder")
    ap.add_argument("--pna-delta", type=float, default=None,
                    help="PNA avg in-degree (auto-computed from data if not set)")
    ap.add_argument("--feature-mode", choices=["all", "coords", "coords_potential", "coords_normals", "all_no_normpot", "all_no_pot", "geom_14d", "geom_14d_raw"],
                    default="all",
                    help="all=8 features (default); coords=xyz only (3-dim, pos field); "
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
    ap.add_argument("--mid-layers", type=int, default=2,
                    help="GIN layers in the intermediate (mid) pooling stage (three-level only).")
    ap.add_argument("--topk-classes", type=int, default=None)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--output-dir", default="runs")
    args = ap.parse_args()

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
                  rwpe_dim=args.rwpe_dim, use_lrf_feat=use_lrf, assign_dir2=args.assign_dir2)
    _col = collate_three if three_level else collate

    full_train = HierPatchDataset(args.graph_dir, args.assign_dir, "train", train_labels, **_ds_kw)
    test_ds = HierPatchDataset(args.graph_dir, args.assign_dir, "test", test_labels, **_ds_kw)

    if args.val_labels_csv:
        val_labels, _ = load_labels(args.graph_dir, "train", args.val_labels_csv)
        train_ds = full_train
        val_ds = HierPatchDataset(args.graph_dir, args.assign_dir, "train", val_labels, **_ds_kw)
        num_classes = len(set(train_labels.values()) | set(val_labels.values()))
    else:
        tr_idx, va_idx = create_robust_train_val_split(full_train, train_labels,
                                                       val_split=args.val_split, seed=args.seed)
        train_ds, val_ds = Subset(full_train, tr_idx), Subset(full_train, va_idx)

    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} classes={num_classes}")

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
                                  num_workers=args.num_workers, collate_fn=train_collate)
    else:
        train_collate = (collate_aug(args.aug_scale_std, args.aug_jitter_std, args.feature_mode)
                         if args.augment and not three_level else _col)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=train_collate)
    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=_col)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=_col)

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

    # xyz_slice: dims of xyz in the node feature vector (for msg_gin and ri_gin)
    _xyz_sl = slice(5, 8)  # default for all/geom_14d_raw; coords_normals uses 0:3
    if args.feature_mode in ("coords", "coords_normals", "coords_potential"):
        _xyz_sl = slice(0, 3)

    model = HierarchicalGNN(in_dim, args.hidden_dim, num_classes, args.node_layers,
                            args.patch_layers, args.pool, args.readout, args.dropout,
                            node_readout=args.node_readout,
                            node_backbone=args.node_backbone,
                            patch_backbone=args.patch_backbone,
                            pna_delta=pna_delta,
                            scalar_inject=args.scalar_inject,
                            patch_scalar_inject=args.patch_scalar_inject,
                            patch_inject_all=args.patch_inject_all,
                            three_level=three_level,
                            mid_layers=args.mid_layers,
                            xyz_slice=_xyz_sl).to(device)
    print(f"in_dim={in_dim} | node_backbone={args.node_backbone} | patch_backbone={args.patch_backbone}"
          f" | ssl={args.ssl_emb_dir is not None}"
          f" | params={sum(p.numel() for p in model.parameters()):,}")

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
    history = []
    for ep in range(1, args.epochs + 1):
        # linear warmup: scale LR from lr/10 → lr over warmup_epochs
        if cosine_sched and ep <= args.warmup_epochs:
            warmup_factor = ep / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * warmup_factor

        tr = run_epoch(model, train_loader, criterion, device, optimizer, num_classes, f"ep{ep} train")
        va = run_epoch(model, val_loader, criterion, device, None, num_classes, f"ep{ep} val")
        history.append({"epoch": ep, "train": tr, "val": va})

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {ep}/{args.epochs}  lr={cur_lr:.2e}  "
              f"Train loss {tr['loss']:.3f} acc {tr['accuracy']:.3f} f1 {tr['f1_macro']:.3f} | "
              f"Val loss {va['loss']:.3f} acc {va['accuracy']:.3f} f1 {va['f1_macro']:.3f} balacc {va['balanced_accuracy']:.3f}")

        if va["f1_macro"] > best_val_f1:
            best_val_f1 = va["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, run_dir / "best_model.pt")
            wait = 0
            print(f"  * new best val f1 {best_val_f1:.4f} (acc {va['accuracy']:.4f})")
        else:
            wait += 1
            if cosine_sched is None and wait >= args.patience:
                print(f"  early stop at epoch {ep}")
                break

        if cosine_sched and ep > args.warmup_epochs:
            cosine_sched.step()

    if best_state:
        model.load_state_dict(best_state)
    test_m = run_epoch(model, test_loader, criterion, device, None, num_classes, "test")
    print("\n=== FINAL TEST (best val model) ===")
    for k in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]:
        print(f"  {k:18s}: {test_m[k]:.4f}")

    cfg_out = vars(args)
    cfg_out["in_dim"] = in_dim
    cfg_out["num_classes"] = num_classes
    json.dump({"final_test_metrics": test_m, "config": cfg_out,
               "best_val_f1": best_val_f1, "history": history},
              open(run_dir / "final_results.json", "w"), indent=2, default=float)
    print(f"\nsaved -> {run_dir/'final_results.json'}")


if __name__ == "__main__":
    main()
