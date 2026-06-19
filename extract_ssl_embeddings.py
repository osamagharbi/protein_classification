#!/usr/bin/env python3
"""
Extract per-node SSL embeddings from the best FAG-MAE checkpoint
(graphsage + coords_potential, macro-F1 0.7704) for all SHREC25 1k graphs.

Output: one .pt file per graph at --out/{split}/{graph_id}.pt
Each file contains {"ssl_emb": Tensor[N, 256]} alongside the original graph data.
"""
import argparse, json, sys, os
from pathlib import Path
import torch
from tqdm import tqdm

GRAPHMAE = "/home/ogharbi/projects/GraphMAE_fast"
sys.path.insert(0, GRAPHMAE)
from graphmae.models import build_model
from graphmae.datasets.shrec25_ssl_loader import load_shrec25_ssl_split

BEST_CKPT = (
    f"{GRAPHMAE}/experiments/"
    "shrec25_ssl1k_coords_potential_graphsage_graphsage_mixed_norm_mask75_bidirected_archstudy_20260519_215548"
    "/checkpoint_best.pt"
)

def build_ssl_model(ckpt_path):
    cfg = json.load(open(Path(ckpt_path).parent / "config.json"))
    import argparse as ap
    args = ap.Namespace(**cfg)
    for k, v in dict(num_out_heads=1, concat_hidden=False, in_drop=0.0,
                     attn_drop=0.0, negative_slope=0.2, alpha_l=2,
                     residual=False, norm=None).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    model = build_model(args)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
    assert all(k.startswith("criterion.") for k in unexpected), f"unexpected: {unexpected}"
    model.eval()
    print(f"  loaded epoch={ck.get('epoch')}  encoder={cfg['encoder']}  "
          f"features={cfg['feature_spec']}  hidden={cfg['num_hidden']}")
    return model, cfg


@torch.no_grad()
def extract_split(model, cfg, split, out_dir, device):
    ds, _ = load_shrec25_ssl_split(
        root_dir=cfg["ssl_root"],
        label_root=cfg["ssl_label_root"],
        split=split, preload=False, use_mesh_edges=False,
        feature_spec=cfg["feature_spec"],
    )
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(len(ds)), desc=split, ncols=80):
        g, label = ds[idx]
        gid = getattr(g, "graph_id", idx)
        out_path = split_dir / f"{gid}.pt"
        if out_path.exists():
            continue
        x = g.ndata["attr"].float().to(device)
        g = g.to(device)
        emb = model.embed(g, x).cpu()         # [N, 256]
        torch.save({"ssl_emb": emb, "label": label, "graph_id": gid}, out_path)

    saved = len(list(split_dir.glob("*.pt")))
    print(f"  {split}: {saved} files in {split_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=BEST_CKPT)
    ap.add_argument("--out", default="/home/ogharbi/datasets/SHREC25/ssl_embeddings_graphsage_coordspotential")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    print(f"Output: {out_dir}  |  device: {device}")

    model, cfg = build_ssl_model(args.ckpt)
    model = model.to(device)

    for split in ("train", "test"):
        extract_split(model, cfg, split, out_dir, device)

    print("Done.")


if __name__ == "__main__":
    main()
