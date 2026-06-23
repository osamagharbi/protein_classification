"""GeoChem-MSRINet: pure-PyTorch multi-scale RI protein surface model."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import (
    farthest_point_sample,
    index_points,
    knn_indices,
    ppf_features,
    radius_topk_query,
)


class PairMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MSRIBlock(nn.Module):
    """Nested FPS + multi-radius point-pair invariant aggregation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_out: int,
        radii: tuple[float, float],
        max_neighbors: tuple[int, int],
        chem_dim: int = 2,
        geom_dim: int = 0,
        dropout: float = 0.10,
        min_neighbors: int = 8,
    ):
        super().__init__()
        self.n_out = int(n_out)
        self.radii = tuple(float(r) for r in radii)
        self.max_neighbors = tuple(int(k) for k in max_neighbors)
        self.min_neighbors = int(min_neighbors)
        self.chem_dim = int(chem_dim)
        self.geom_dim = int(geom_dim)
        pair_dim = in_dim + 8 + (chem_dim * 4) + (geom_dim * 2)
        hidden = max(out_dim, 64)
        self.branches = nn.ModuleList([PairMLP(pair_dim, hidden, out_dim, dropout) for _ in self.radii])
        self.branch_proj = nn.ModuleList([nn.Linear(out_dim * 2, out_dim) for _ in self.radii])
        self.gate = nn.Linear(out_dim * len(self.radii), len(self.radii))
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        pos: torch.Tensor,
        normal: torch.Tensor,
        h: torch.Tensor,
        chem: torch.Tensor,
        geom: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        idx = farthest_point_sample(pos, self.n_out)
        new_pos = index_points(pos, idx)
        new_normal = F.normalize(index_points(normal, idx), dim=-1)
        center_h = index_points(h, idx)
        center_chem = index_points(chem, idx)
        center_geom = index_points(geom, idx) if geom is not None and geom.shape[-1] else None

        branch_outs = []
        for radius, k, branch, proj in zip(self.radii, self.max_neighbors, self.branches, self.branch_proj):
            nidx = radius_topk_query(pos, new_pos, radius, k, self.min_neighbors)
            neigh_pos = index_points(pos, nidx)
            neigh_normal = F.normalize(index_points(normal, nidx), dim=-1)
            neigh_h = index_points(h, nidx)
            neigh_chem = index_points(chem, nidx)
            ppf = ppf_features(new_pos, new_normal, neigh_pos, neigh_normal, radius)

            parts = [
                neigh_h,
                ppf,
                center_chem.unsqueeze(2).expand_as(neigh_chem),
                neigh_chem,
                neigh_chem - center_chem.unsqueeze(2),
                (neigh_chem - center_chem.unsqueeze(2)).abs(),
            ]
            if self.geom_dim and center_geom is not None and geom is not None:
                neigh_geom = index_points(geom, nidx)
                parts.extend([
                    center_geom.unsqueeze(2).expand_as(neigh_geom),
                    neigh_geom - center_geom.unsqueeze(2),
                ])
            pair = torch.cat(parts, dim=-1)
            emb = branch(pair)
            pooled = torch.cat([emb.max(dim=2).values, emb.mean(dim=2)], dim=-1)
            branch_outs.append(proj(pooled))

        stacked = torch.stack(branch_outs, dim=2)  # [B,S,R,C]
        gates = torch.softmax(self.gate(torch.cat(branch_outs, dim=-1)), dim=-1).unsqueeze(-1)
        out = (stacked * gates).sum(dim=2)
        out = self.norm(out + self.residual(center_h))
        out = self.dropout(F.silu(out))
        return new_pos, new_normal, out, center_chem, center_geom


def patch_edge_features(pos: torch.Tensor, normal: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    neigh_pos = index_points(pos, idx)
    neigh_normal = F.normalize(index_points(normal, idx), dim=-1)
    center_pos = pos.unsqueeze(2)
    center_normal = F.normalize(normal, dim=-1).unsqueeze(2)
    d = neigh_pos - center_pos
    dist = d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    e = d / dist
    ni = center_normal.expand_as(neigh_normal)
    nj = neigh_normal
    return torch.cat(
        [
            dist,
            dist / 0.75,
            (ni * e).sum(dim=-1, keepdim=True),
            (nj * e).sum(dim=-1, keepdim=True),
            (ni * nj).sum(dim=-1, keepdim=True),
            torch.linalg.cross(nj, ni, dim=-1).mul(e).sum(dim=-1, keepdim=True),
        ],
        dim=-1,
    )


class PatchGINBlock(nn.Module):
    def __init__(self, dim: int = 512, edge_dim: int = 6, dropout: float = 0.15, drop_path: float = 0.0):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(dim + edge_dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop_path)

    def forward(self, h: torch.Tensor, idx: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        neigh_h = index_points(h, idx)
        msg = self.msg(torch.cat([neigh_h, edge_feat], dim=-1)).sum(dim=2)
        h = self.norm1(h + self.drop(msg))
        h = self.norm2(h + self.drop(self.ffn(h)))
        return h


class EdgeBiasedTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        edge_dim: int = 6,
        heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.15,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.d_head = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.edge_bias = nn.Linear(edge_dim, heads)
        self.out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.attn_drop = nn.Dropout(dropout)
        self.drop = nn.Dropout(drop_path)

    def forward(self, h: torch.Tensor, idx: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        b, s, c = h.shape
        k_in = idx.shape[2]
        neigh_h = index_points(h, idx)
        q = self.q(h).view(b, s, self.heads, self.d_head)
        k = self.k(neigh_h).view(b, s, k_in, self.heads, self.d_head)
        v = self.v(neigh_h).view(b, s, k_in, self.heads, self.d_head)
        scores = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(self.d_head)
        scores = scores + self.edge_bias(edge_feat)
        attn = self.attn_drop(torch.softmax(scores, dim=2))
        out = (attn.unsqueeze(-1) * v).sum(dim=2).reshape(b, s, c)
        h = self.norm1(h + self.drop(self.out(out)))
        h = self.norm2(h + self.drop(self.ffn(h)))
        return h


class ClassTokenAttentionMaxReadout(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.cls_query = nn.Parameter(torch.randn(dim) * 0.02)
        self.attn = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        cls_scores = (h * self.cls_query.view(1, 1, -1)).sum(dim=-1) / math.sqrt(h.shape[-1])
        r_cls = (torch.softmax(cls_scores, dim=1).unsqueeze(-1) * h).sum(dim=1)
        r_attn = (torch.softmax(self.attn(h), dim=1) * h).sum(dim=1)
        r_max = h.max(dim=1).values
        return torch.cat([r_cls, r_attn, r_max], dim=-1)


class CosineClassifier(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, num_classes: int, scale: float = 30.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)
        self.weight = nn.Parameter(torch.randn(num_classes, embed_dim))
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = F.normalize(self.proj(x), dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        return self.scale * emb @ weight.t(), emb


class GeoChemMSRINet(nn.Module):
    """Geometry-Chemistry Multi-Scale Rotation-Invariant Network."""

    def __init__(
        self,
        num_classes: int = 99,
        points: int = 1000,
        chem_dim: int = 2,
        geom_dim: int = 0,
        stage_dims: tuple[int, ...] | None = None,
        n_out: tuple[int, ...] | None = None,
        radii: tuple[tuple[float, float], ...] | None = None,
        max_neighbors: tuple[tuple[int, int], ...] | None = None,
        patch_knn: int = 8,
        patch_gin_layers: int = 2,
        patch_transformer_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.30,
        stage_dropout: float = 0.10,
        patch_dropout: float = 0.15,
        logit_blend: float = 0.65,
    ):
        super().__init__()
        if stage_dims is None:
            stage_dims = (96, 160, 256, 384, 512) if points > 1024 else (96, 160, 256, 512)
        if n_out is None:
            n_out = (1024, 512, 256, 128, 64) if points > 1024 else (512, 256, 128, 64)
        if radii is None:
            radii = (
                ((0.045, 0.075), (0.075, 0.140), (0.140, 0.280), (0.280, 0.520), (0.520, 0.900))
                if points > 1024
                else ((0.060, 0.100), (0.100, 0.200), (0.200, 0.400), (0.400, 0.800))
            )
        if max_neighbors is None:
            max_neighbors = (
                ((24, 32), (32, 40), (32, 48), (48, 64), (64, 96))
                if points > 1024
                else ((24, 32), (32, 40), (40, 64), (64, 96))
            )
        if not (len(stage_dims) == len(n_out) == len(radii) == len(max_neighbors)):
            raise ValueError("stage_dims, n_out, radii, and max_neighbors must have equal length")

        self.num_classes = num_classes
        self.chem_dim = chem_dim
        self.geom_dim = geom_dim
        self.logit_blend = float(logit_blend)
        self.patch_knn = int(patch_knn)

        blocks = []
        in_dim = chem_dim
        for out_dim, n_stage, r_stage, k_stage in zip(stage_dims, n_out, radii, max_neighbors):
            blocks.append(
                MSRIBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    n_out=n_stage,
                    radii=r_stage,
                    max_neighbors=k_stage,
                    chem_dim=chem_dim,
                    geom_dim=geom_dim,
                    dropout=stage_dropout,
                )
            )
            in_dim = out_dim
        self.blocks = nn.ModuleList(blocks)
        dim = stage_dims[-1]

        self.patch_gin = nn.ModuleList(
            [PatchGINBlock(dim, dropout=patch_dropout, drop_path=0.10) for _ in range(patch_gin_layers)]
        )
        self.patch_attn = nn.ModuleList(
            [
                EdgeBiasedTransformerBlock(
                    dim=dim,
                    heads=heads,
                    ffn_dim=dim * 4,
                    dropout=patch_dropout,
                    drop_path=0.10,
                )
                for _ in range(patch_transformer_layers)
            ]
        )
        self.readout = ClassTokenAttentionMaxReadout(dim)
        graph_dim = dim * 3
        self.mlp_head = nn.Sequential(
            nn.Linear(graph_dim, max(dim, 768 if dim >= 512 else dim * 2)),
            nn.LayerNorm(max(dim, 768 if dim >= 512 else dim * 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(dim, 768 if dim >= 512 else dim * 2), num_classes),
        )
        self.cos_head = CosineClassifier(graph_dim, min(512, dim), num_classes)
        self.proj_head = nn.Sequential(nn.Linear(graph_dim, 512), nn.GELU(), nn.Linear(512, 128))

    def forward(
        self,
        pos: torch.Tensor,
        normal: torch.Tensor,
        chem: torch.Tensor,
        geom: torch.Tensor | None = None,
        use_chem: bool = True,
    ) -> dict[str, torch.Tensor]:
        normal = F.normalize(normal, dim=-1)
        if not use_chem:
            chem = torch.zeros_like(chem)
        if geom is None or not self.geom_dim:
            geom = None
        h = chem
        for block in self.blocks:
            pos, normal, h, chem, geom = block(pos, normal, h, chem, geom)

        k = min(self.patch_knn, pos.shape[1])
        idx = knn_indices(pos, k, include_self=True)
        edge_feat = patch_edge_features(pos, normal, idx)
        for block in self.patch_gin:
            h = block(h, idx, edge_feat)
        for block in self.patch_attn:
            h = block(h, idx, edge_feat)

        graph = self.readout(h)
        logits_mlp = self.mlp_head(graph)
        logits_cos, embed = self.cos_head(graph)
        logits = self.logit_blend * logits_mlp + (1.0 - self.logit_blend) * logits_cos
        proj = F.normalize(self.proj_head(graph), dim=-1)
        return {
            "logits": logits,
            "mlp_logits": logits_mlp,
            "cos_logits": logits_cos,
            "embed": embed,
            "proj": proj,
            "graph": graph,
        }
