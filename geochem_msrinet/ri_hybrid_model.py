"""RIConv2-fronted GeoChem model.

This module implements the "option 3" integration: replace GeoChem's pure
PyTorch local MSRI encoder with RIConv2 set-abstraction blocks, then keep the
in-repository patch graph/readout/classifier head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import (
    ClassTokenAttentionMaxReadout,
    CosineClassifier,
    EdgeBiasedTransformerBlock,
    PatchGINBlock,
    patch_edge_features,
)
from .ops import knn_indices
from .riconv2_utils_features import RIConv2SetAbstraction


class RIConv2FeatureEncoder(nn.Module):
    """RIConv2 set abstraction stack over graph-node point clouds.

    The input graph edges are intentionally ignored here. The graph file is used
    as the canonical storage for sampled surface nodes: xyz, normals, and
    optional chemistry channels. Chemistry channels are carried along as
    point-wise features in the same style as Barisin's feature variant.
    """

    def __init__(
        self,
        chem_dim: int = 2,
        width_mult: int = 1,
        token_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        width_mult = max(1, int(width_mult))
        self.chem_dim = int(chem_dim)
        self.width_mult = width_mult
        self.out_dim = int(token_dim)

        self.sa0 = RIConv2SetAbstraction(
            npoint=512 * width_mult,
            radius=0.12,
            nsample=8,
            in_channel=self.chem_dim + 64,
            mlp=[128],
            group_all=False,
        )
        self.sa1 = RIConv2SetAbstraction(
            npoint=256 * width_mult,
            radius=0.16,
            nsample=16,
            in_channel=self.chem_dim + 128 + 64,
            mlp=[256],
            group_all=False,
        )
        self.sa2 = RIConv2SetAbstraction(
            npoint=128 * width_mult,
            radius=0.24,
            nsample=32,
            in_channel=self.chem_dim + 256 + 64,
            mlp=[512],
            group_all=False,
        )
        self.sa3 = RIConv2SetAbstraction(
            npoint=64 * width_mult,
            radius=0.48,
            nsample=64,
            in_channel=self.chem_dim + 512 + 64,
            mlp=[1024],
            group_all=False,
        )
        self.sa4 = RIConv2SetAbstraction(
            npoint=32 * width_mult,
            radius=0.64,
            nsample=64 * width_mult,
            in_channel=self.chem_dim + 1024 + 64,
            mlp=[2048],
            group_all=False,
        )
        self.token_proj = nn.Linear(2048 + self.chem_dim, self.out_dim)
        self.drop = nn.Dropout(dropout)

    def _append_point_channels(self, points: torch.Tensor, sampled_norm: torch.Tensor) -> torch.Tensor:
        if self.chem_dim <= 0:
            return points
        return torch.cat((points, sampled_norm[:, :, 3 : 3 + self.chem_dim].permute(0, 2, 1)), dim=1)

    def forward(
        self,
        pos: torch.Tensor,
        normal: torch.Tensor,
        chem: torch.Tensor,
        use_chem: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not use_chem:
            chem = torch.zeros_like(chem)
        normal = F.normalize(normal, dim=-1)
        norm_feat = torch.cat([normal, chem], dim=-1)

        l0_xyz, l0_norm, l0_points = self.sa0(pos, norm_feat, norm_feat[:, :, 3:].permute(0, 2, 1))
        l0_points = self._append_point_channels(l0_points, l0_norm)

        l1_xyz, l1_norm, l1_points = self.sa1(l0_xyz, l0_norm, l0_points)
        l1_points = self._append_point_channels(l1_points, l1_norm)

        l2_xyz, l2_norm, l2_points = self.sa2(l1_xyz, l1_norm, l1_points)
        l2_points = self._append_point_channels(l2_points, l2_norm)

        l3_xyz, l3_norm, l3_points = self.sa3(l2_xyz, l2_norm, l2_points)
        l3_points = self._append_point_channels(l3_points, l3_norm)

        l4_xyz, l4_norm, l4_points = self.sa4(l3_xyz, l3_norm, l3_points)
        l4_points = self._append_point_channels(l4_points, l4_norm)

        h = self.drop(self.token_proj(l4_points.permute(0, 2, 1).contiguous()))
        final_normal = F.normalize(l4_norm[:, :, :3], dim=-1)
        return l4_xyz, final_normal, h


class RIConv2GeoChemHybrid(nn.Module):
    """RIConv2 local encoder + GeoChem patch graph/readout head."""

    def __init__(
        self,
        num_classes: int = 99,
        chem_dim: int = 2,
        ri_width_mult: int = 1,
        ri_token_dim: int = 512,
        patch_knn: int = 8,
        patch_gin_layers: int = 1,
        patch_transformer_layers: int = 1,
        heads: int = 8,
        dropout: float = 0.30,
        patch_dropout: float = 0.15,
        logit_blend: float = 1.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.chem_dim = int(chem_dim)
        self.patch_knn = int(patch_knn)
        self.logit_blend = float(logit_blend)

        self.encoder = RIConv2FeatureEncoder(
            chem_dim=chem_dim,
            width_mult=ri_width_mult,
            token_dim=ri_token_dim,
            dropout=patch_dropout,
        )
        dim = self.encoder.out_dim

        self.patch_gin = nn.ModuleList(
            [PatchGINBlock(dim, dropout=patch_dropout, drop_path=0.10) for _ in range(patch_gin_layers)]
        )
        self.patch_attn = nn.ModuleList(
            [
                EdgeBiasedTransformerBlock(
                    dim=dim,
                    heads=heads,
                    ffn_dim=dim * 2,
                    dropout=patch_dropout,
                    drop_path=0.10,
                )
                for _ in range(patch_transformer_layers)
            ]
        )
        self.readout = ClassTokenAttentionMaxReadout(dim)
        graph_dim = dim * 3
        hidden = max(1024, dim // 2)
        self.mlp_head = nn.Sequential(
            nn.Linear(graph_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
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
        del geom
        pos, normal, h = self.encoder(pos, normal, chem, use_chem=use_chem)

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
