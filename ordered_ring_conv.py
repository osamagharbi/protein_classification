#!/usr/bin/env python3
"""
Ordered-ring rotation-invariant convolution — a faithful port of RIConv2's
`order_index` + `RI_features` (Zhang et al.), to be used as a node backbone
inside the proven HierGIN pipeline (classify_hierarchical_gnn.py).

Why this and not my earlier PPF: RIConv2 canonically *orders* each point's
neighbours (project onto the tangent plane, sort by signed angle around a
reference) and computes RI features *between consecutive* neighbours — capturing
local surface shape as an ordered ring, which unordered max-pooled PPF discards.

All features are SO(3)+translation invariant by construction (chirality terms are
sign-flipped under reflection, which is desirable for chiral proteins). Run
`python ordered_ring_conv.py` for the numerical invariance gate.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-7


# ── RIConv2 helpers (ported verbatim) ────────────────────────────────────────

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    B = points.shape[0]
    view = list(idx.shape); view[1:] = [1] * (len(view) - 1)
    rep = list(idx.shape); rep[0] = 1
    bidx = torch.arange(B, dtype=torch.long, device=points.device).view(view).repeat(rep)
    return points[bidx, idx, :]


def knn_self(k, xyz):
    """k nearest neighbours INCLUDING self (RIConv2 groups include the centre)."""
    return square_distance(xyz, xyz).topk(k, dim=-1, largest=False)[1]   # [B,N,k]


def order_index(xyz, new_xyz, new_norm, idx):
    """RIConv2 canonical ordering: project neighbours to the tangent plane, pick
    the longest as reference, sort the rest by signed angle around the normal."""
    B, S, C = new_xyz.shape
    nsample = idx.shape[2]
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_local = grouped_xyz - new_xyz.view(B, S, 1, C)

    dist_plane = torch.matmul(grouped_xyz_local, new_norm)
    proj_xyz = grouped_xyz_local - dist_plane * new_norm.view(B, S, 1, C)
    proj_len = torch.norm(proj_xyz, dim=-1, keepdim=True)
    proj_unit = proj_xyz / proj_len
    proj_unit[proj_unit != proj_unit] = 0

    length_max_idx = torch.argmax(proj_len, dim=2)
    vec_ref = proj_unit.gather(2, length_max_idx.unsqueeze(-1).repeat(1, 1, 1, 3))
    dots = torch.matmul(proj_unit, vec_ref.view(B, S, C, 1))
    sign = torch.cross(proj_unit, vec_ref.view(B, S, 1, C).repeat(1, 1, nsample, 1), dim=-1)
    sign = torch.matmul(sign, new_norm)
    sign = torch.sign(sign)
    sign[:, :, 0, 0] = 1.0                       # centre is its own first neighbour
    dots = sign * dots - (1 - sign)
    dots_sorted, indices = torch.sort(dots, dim=2, descending=True)
    idx_ordered = idx.gather(2, indices.squeeze(-1))
    return dots_sorted, idx_ordered


def ri_features(xyz, norm, new_xyz, new_norm, idx):
    """Full 8-D RIConv2 RI feature set per ordered (centre, neighbour) pair."""
    B, S, C = new_xyz.shape
    new_norm = new_norm.unsqueeze(-1)                                   # [B,S,3,1]
    dots_sorted, idx_ord = order_index(xyz, new_xyz, new_norm, idx)

    grouped_xyz = index_points(xyz, idx_ord)
    grouped_local = grouped_xyz - new_xyz.view(B, S, 1, C)
    length = torch.norm(grouped_local, dim=-1, keepdim=True)
    unit = grouped_local / length
    unit[unit != unit] = 0
    g_norm = index_points(norm, idx_ord)

    angle_0 = torch.matmul(unit, new_norm)                             # ∠(d, n_c)
    angle_1 = (unit * g_norm).sum(-1, keepdim=True)                    # ∠(d, n_j)
    angle_n = torch.matmul(g_norm, new_norm)                          # ∠(n_c, n_j)
    angle_n = torch.acos(torch.clamp(angle_n, -1 + EPS, 1 - EPS))
    D0 = (angle_0 < angle_1); D0 = torch.where(D0, 1.0, -1.0)
    angle_n = D0 * angle_n

    inner_vec = grouped_local - torch.roll(grouped_local, 1, 2)        # consecutive (ring)
    inner_unit = inner_vec / torch.norm(inner_vec, dim=-1, keepdim=True)
    inner_unit[inner_unit != inner_unit] = 0
    in_a0 = (inner_unit * g_norm).sum(-1, keepdim=True)
    in_a1 = (inner_unit * torch.roll(g_norm, 1, 2)).sum(-1, keepdim=True)
    in_a2 = (g_norm * torch.roll(g_norm, 1, 2)).sum(-1, keepdim=True)
    in_a2 = torch.acos(torch.clamp(in_a2, -1 + EPS, 1 - EPS))
    D1 = (in_a0 < in_a1); D1 = torch.where(D1, 1.0, -1.0)
    in_a2 = D1 * in_a2

    proj_inner = dots_sorted - torch.roll(dots_sorted, 1, 2)
    proj_inner[:, :, 0, 0] = (-3) - dots_sorted[:, :, -1, 0]

    ri = torch.cat([length, proj_inner, angle_0, angle_1, angle_n,
                    in_a0, in_a1, in_a2], dim=-1)                       # [B,S,k,8]
    return ri, idx_ord


# ── Conv layer ───────────────────────────────────────────────────────────────

class OrderedRingConvLayer(nn.Module):
    """Per-node ordered-ring RI conv: kNN group → order → 8-D RI features →
    shared MLP over [RI ‖ carried neighbour feature] → max-pool over neighbours.
    Operates on dense [B, N, ·] tensors (all SHREC graphs are N=1000)."""

    def __init__(self, in_dim: int, out_dim: int, k: int = 16):
        super().__init__()
        self.k = k
        c = 8 + in_dim
        self.mlp = nn.Sequential(
            nn.Conv2d(c, out_dim, 1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
        )

    def forward(self, pos, normal, feat):
        # pos,normal [B,N,3]; feat [B,N,in_dim] or None
        idx = knn_self(self.k, pos)
        ri, idx_ord = ri_features(pos, normal, pos, normal, idx)        # [B,N,k,8]
        grouped = ri if feat is None else torch.cat([ri, index_points(feat, idx_ord)], dim=-1)
        x = self.mlp(grouped.permute(0, 3, 1, 2))                       # [B,out,N,k]
        return x.max(dim=-1)[0].transpose(1, 2)                         # [B,N,out]


# ── numerical RI gate ────────────────────────────────────────────────────────

def _rand_rotation(device, dtype):
    a = torch.randn(3, 3, device=device, dtype=dtype)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


if __name__ == "__main__":
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for dt in (torch.float32, torch.float64):
        torch.manual_seed(0)
        B, N = 2, 1000
        pos = (torch.randn(B, N, 3, device=dev) * 0.4).to(dt)
        normal = F.normalize(torch.randn(B, N, 3, device=dev), dim=-1).to(dt)
        R = _rand_rotation(dev, dt); t = (torch.randn(1, 1, 3, device=dev) * 2).to(dt)
        idx = knn_self(16, pos)
        ri1, _ = ri_features(pos, normal, pos, normal, idx)
        # SAME neighbour selection, rotated coords -> features must match
        ri2, _ = ri_features(pos @ R.T + t, normal @ R.T, pos @ R.T + t, normal @ R.T, idx)
        d = (ri1 - ri2).abs().max().item()
        print(f"{str(dt):14s} ordered-ring RI features  max|Δ| = {d:.2e}")
    # full layer shape check
    layer = OrderedRingConvLayer(0, 64, k=16).to(dev).eval()
    out = layer(pos.float(), normal.float(), None)
    print(f"layer out: {tuple(out.shape)}  (expect (2,1000,64))")
