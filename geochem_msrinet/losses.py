"""Losses and class weighting for GeoChem-MSRINet."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sqrt_class_weights(labels: list[int], num_classes: int, device=None) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
    for y in labels:
        if 0 <= int(y) < num_classes:
            counts[int(y)] += 1
    weights = torch.zeros_like(counts)
    mask = counts > 0
    weights[mask] = 1.0 / counts[mask].sqrt()
    if mask.any():
        weights[mask] = weights[mask] / weights[mask].mean()
    # Keep absent classes neutral so validation/test losses do not become NaN
    # in small smoke runs or unusual split subsets.
    weights[~mask] = 1.0
    return weights


def supervised_contrastive_loss(proj: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Supervised contrastive loss on L2-normalized projections."""
    if proj.shape[0] < 2:
        return proj.new_zeros(())
    proj = F.normalize(proj, dim=-1)
    logits = proj @ proj.t() / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(proj.shape[0], dtype=torch.bool, device=proj.device)
    pos = labels[:, None].eq(labels[None, :]) & ~eye
    denom_mask = ~eye
    exp_logits = torch.exp(logits) * denom_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    pos_count = pos.sum(dim=1)
    loss = -(log_prob * pos).sum(dim=1) / pos_count.clamp_min(1)
    loss = loss[pos_count > 0]
    return loss.mean() if loss.numel() else proj.new_zeros(())


class PrototypeMemory:
    """EMA prototype memory for optional rare-class metric supervision."""

    def __init__(self, num_classes: int, dim: int, momentum: float = 0.95, device=None):
        self.num_classes = num_classes
        self.dim = dim
        self.momentum = momentum
        self.prototypes = torch.zeros(num_classes, dim, device=device)
        self.initialized = torch.zeros(num_classes, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, emb: torch.Tensor, labels: torch.Tensor) -> None:
        emb = F.normalize(emb.detach(), dim=-1)
        for c in labels.unique():
            c_int = int(c.item())
            mask = labels == c
            mean = emb[mask].mean(dim=0)
            if self.initialized[c_int]:
                self.prototypes[c_int].mul_(self.momentum).add_(mean, alpha=1.0 - self.momentum)
            else:
                self.prototypes[c_int].copy_(mean)
                self.initialized[c_int] = True
        self.prototypes[:] = F.normalize(self.prototypes, dim=-1)

    def logits(self, emb: torch.Tensor, scale: float = 20.0) -> torch.Tensor:
        emb = F.normalize(emb, dim=-1)
        logits = scale * (emb @ self.prototypes.t())
        missing = ~self.initialized
        if missing.any():
            logits[:, missing] = -1e4
        return logits


class GeoChemLoss(nn.Module):
    """Combined CE, cosine-margin CE, SupCon, and prototype loss."""

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.1,
        cosine_weight: float = 0.30,
        supcon_weight: float = 0.0,
        proto_weight: float = 0.0,
        cosine_margin: float = 0.15,
        supcon_temperature: float = 0.07,
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.cosine_weight = cosine_weight
        self.supcon_weight = supcon_weight
        self.proto_weight = proto_weight
        self.cosine_margin = cosine_margin
        self.supcon_temperature = supcon_temperature

    def forward(self, outputs: dict[str, torch.Tensor], labels: torch.Tensor, proto: PrototypeMemory | None = None):
        loss_ce = self.ce(outputs["logits"], labels)
        loss_cos = outputs["logits"].new_zeros(())
        if self.cosine_weight > 0 and "cos_logits" in outputs:
            cos_logits = outputs["cos_logits"].clone()
            cos_logits[torch.arange(labels.numel(), device=labels.device), labels] -= self.cosine_margin
            loss_cos = self.ce(cos_logits, labels)

        loss_sup = outputs["logits"].new_zeros(())
        if self.supcon_weight > 0:
            loss_sup = supervised_contrastive_loss(outputs["proj"], labels, self.supcon_temperature)

        loss_proto = outputs["logits"].new_zeros(())
        if self.proto_weight > 0 and proto is not None:
            loss_proto = self.ce(proto.logits(outputs["embed"]), labels)

        total = (
            loss_ce
            + self.cosine_weight * loss_cos
            + self.supcon_weight * loss_sup
            + self.proto_weight * loss_proto
        )
        return total, {
            "ce": float(loss_ce.detach().cpu()),
            "cosine": float(loss_cos.detach().cpu()),
            "supcon": float(loss_sup.detach().cpu()),
            "proto": float(loss_proto.detach().cpu()),
        }
