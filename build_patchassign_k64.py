#!/usr/bin/env python3
"""
build_patchassign_k64.py

Build ssl_graphs_1k_patchassign_k64 from an existing 1k graph dataset.
For each protein, runs FPS to select k=64 seed nodes, assigns every node
to its nearest seed (geodesic Voronoi approximated by Euclidean), builds
a kNN-8 patch graph, and saves {patch_ids, patch_edge_index, num_patches}.

Usage:
    python build_patchassign_k64.py \
        --graph-dir /path/to/ssl_graphs_1k_geom_enriched_ri \
        --out-dir   /path/to/ssl_graphs_1k_patchassign_k64 \
        [--k 64] [--patch-knn 8] [--workers 8]
"""

import argparse
import multiprocessing as mp
from pathlib import Path

import torch
from tqdm import tqdm


# ── FPS ────────────────────────────────────────────────────────────────────────

def fps(pos: torch.Tensor, k: int, seed: int = 0) -> torch.Tensor:
    """Farthest Point Sampling — returns k seed indices."""
    N = pos.shape[0]
    k = min(k, N)
    sel = torch.zeros(k, dtype=torch.long)
    dist = torch.full((N,), float("inf"))
    torch.manual_seed(seed)
    cur = torch.randint(0, N, (1,)).item()
    for i in range(k):
        sel[i] = cur
        d = ((pos - pos[cur]) ** 2).sum(dim=1)
        dist = torch.minimum(dist, d)
        cur = int(dist.argmax())
    return sel


# ── Per-file worker ────────────────────────────────────────────────────────────

def _process(args_tuple):
    gp, ap, k, patch_knn = args_tuple
    try:
        d = torch.load(gp, map_location="cpu", weights_only=False)
        pos = d.get("pos", d.get("xyz_raw"))
        if pos is None:
            return gp.name, "no pos"
        pos = pos.float()
        N = pos.shape[0]

        seeds = fps(pos, k)                                    # [k]
        centers = pos[seeds]                                   # [k, 3]

        # assign each node to nearest seed
        D = torch.cdist(pos, centers)                          # [N, k]
        patch_ids = D.argmin(dim=1).long()                     # [N]

        # kNN graph among patch centers
        Dc = torch.cdist(centers, centers)
        Dc.fill_diagonal_(float("inf"))
        kc = min(patch_knn, k - 1)
        nbrs = Dc.topk(kc, dim=1, largest=False).indices       # [k, kc]
        src = torch.arange(k).repeat_interleave(kc)
        dst = nbrs.reshape(-1)
        # add self-loops
        src = torch.cat([src, torch.arange(k)])
        dst = torch.cat([dst, torch.arange(k)])
        patch_ei = torch.stack([src, dst])

        out = {"patch_ids": patch_ids, "patch_edge_index": patch_ei, "num_patches": k}
        ap.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, ap)
        return gp.name, None
    except Exception as e:
        return gp.name, str(e)


def _init():
    torch.set_num_threads(1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", required=True)
    ap.add_argument("--out-dir",   required=True)
    ap.add_argument("--k",         type=int, default=64)
    ap.add_argument("--patch-knn", type=int, default=8)
    ap.add_argument("--workers",   type=int, default=min(8, mp.cpu_count()))
    args = ap.parse_args()

    gdir = Path(args.graph_dir)
    odir = Path(args.out_dir)

    tasks = []
    for subset in ["train_set", "test_set"]:
        src = gdir / subset
        if not src.exists():
            print(f"  skip {src}")
            continue
        pts = sorted(src.glob("*.pt"))
        print(f"  {subset}: {len(pts)} graphs")
        for p in pts:
            tasks.append((p, odir / subset / p.name, args.k, args.patch_knn))

    print(f"\nTotal: {len(tasks)} files  k={args.k}  patch_knn={args.patch_knn}  workers={args.workers}")

    errors = []
    with mp.Pool(args.workers, initializer=_init) as pool:
        for name, err in tqdm(pool.imap_unordered(_process, tasks),
                              total=len(tasks), unit="graph"):
            if err:
                errors.append((name, err))

    print(f"\nDone: {len(tasks)-len(errors)}/{len(tasks)}  errors={len(errors)}")
    for n, e in errors[:10]:
        print(f"  {n}: {e}")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
