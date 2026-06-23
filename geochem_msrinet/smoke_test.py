#!/usr/bin/env python3
"""Small correctness checks for GeoChem-MSRINet."""

from __future__ import annotations

import argparse

import torch

from geochem_msrinet.model import GeoChemMSRINet
from geochem_msrinet.ops import random_rotation_matrix, rotate_points_and_normals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    # Use a tiny custom model by overriding point schedule for fast smoke tests.
    model = GeoChemMSRINet(
        points=args.points,
        stage_dims=(32, 64, 128),
        n_out=(128, 64, 32),
        radii=((0.12, 0.24), (0.24, 0.48), (0.48, 0.96)),
        max_neighbors=((12, 16), (16, 24), (24, 32)),
        patch_gin_layers=1,
        patch_transformer_layers=1,
        heads=4,
    ).to(device)
    model.eval()

    pos = torch.randn(args.batch_size, args.points, 3, device=device)
    pos = pos - pos.mean(dim=1, keepdim=True)
    pos = pos / pos.norm(dim=-1).amax(dim=1, keepdim=True).clamp_min(1e-6).unsqueeze(-1)
    normal = torch.nn.functional.normalize(torch.randn_like(pos), dim=-1)
    chem = torch.randn(args.batch_size, args.points, 2, device=device).clamp(-5, 5)

    with torch.no_grad():
        out = model(pos, normal, chem)
        rot = random_rotation_matrix(args.batch_size, device, pos.dtype)
        pos_r, normal_r = rotate_points_and_normals(pos, normal, rot)
        out_r = model(pos_r, normal_r, chem)

    max_delta = (out["logits"] - out_r["logits"]).abs().max().item()
    print(f"logits shape: {tuple(out['logits'].shape)}")
    print(f"projection shape: {tuple(out['proj'].shape)}")
    print(f"rotation max |delta logits|: {max_delta:.6g}")
    if not torch.isfinite(out["logits"]).all():
        raise SystemExit("non-finite logits")
    if max_delta > 1e-3:
        raise SystemExit("rotation invariance smoke check failed")


if __name__ == "__main__":
    main()
