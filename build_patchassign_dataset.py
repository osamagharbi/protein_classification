#!/usr/bin/env python3
"""
Precompute geodesic patch assignments for ssl_graphs_1k.

For each input graph we run the SAME geodesic decomposition used by
utils/geodesic_patch_graph_builder.py and store a tiny sidecar holding:
  - patch_ids        LongTensor [N]      node -> patch id (compacted to 0..P-1)
  - patch_edge_index LongTensor [2, 2E]  patch adjacency (boundary-crossing), bidirectional
  - num_patches      int                 P (non-empty patches kept)

These sidecars let the hierarchical GNN pool the full mesh into patches at
train time without recomputing geodesic Voronoi every epoch.

Usage:
  python build_patchassign_dataset.py \
    --input /home/ogharbi/datasets/SHREC25/ssl_graphs_1k \
    --out   /home/ogharbi/datasets/SHREC25/ssl_graphs_1k_patchassign_k32 \
    --num-patches 32
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

sys.path.insert(0, "/home/ogharbi/projects/utils")
from geodesic_patch_graph_builder import decompose


def build_one(args):
    path, out_path, k, seed_mode, edge_key, weight_mode, radius, fallback = args
    try:
        _, pos, edge_index, patch_ids, geo, seeds = decompose(
            Path(path), k, seed_mode, edge_key, weight_mode, radius, fallback,
            None, "zscore", "auto", np.random.default_rng(42),
        )
        present = sorted(int(p) for p in np.unique(patch_ids[patch_ids >= 0]).tolist())
        pid_to_row = {pid: r for r, pid in enumerate(present)}
        # compact node->patch
        compact = np.full(patch_ids.shape[0], -1, dtype=np.int64)
        for pid, row in pid_to_row.items():
            compact[patch_ids == pid] = row
        # patch adjacency (boundary-crossing), bidirectional
        pu = patch_ids[edge_index[0]]; pv = patch_ids[edge_index[1]]
        valid = (pu >= 0) & (pv >= 0) & (pu != pv)
        pairs = set()
        for a, b in zip(pu[valid].tolist(), pv[valid].tolist()):
            lo, hi = (a, b) if a < b else (b, a)
            pairs.add((lo, hi))
        if pairs:
            s, d = [], []
            for lo, hi in pairs:
                rs, rd = pid_to_row[lo], pid_to_row[hi]
                s += [rs, rd]; d += [rd, rs]
            pei = np.asarray([s, d], dtype=np.int64)
        else:
            pei = np.zeros((2, 0), dtype=np.int64)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "patch_ids": torch.from_numpy(compact),
            "patch_edge_index": torch.from_numpy(pei),
            "num_patches": len(present),
        }, out_path)
        return str(path), None
    except Exception as e:  # noqa: BLE001
        return str(path), str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-patches", type=int, default=32)
    ap.add_argument("--seed-mode", default="fps")
    ap.add_argument("--edge-key", default="auto")
    ap.add_argument("--weight-mode", default="euclidean")
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--fallback-mode", default="nearest")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    inp = Path(args.input); out = Path(args.out)
    items = []
    for split in ("train_set", "test_set"):
        sdir = inp / split
        if not sdir.exists():
            continue
        for f in sorted(sdir.glob("*.pt")):
            op = out / split / f.name
            items.append((str(f), op, args.num_patches, args.seed_mode, args.edge_key,
                          args.weight_mode, args.radius, args.fallback_mode))
    print(f"[INFO] {len(items)} graphs to process -> {out}", flush=True)

    nw = args.workers if args.workers > 0 else min(len(items), os.cpu_count() or 4)
    done = err = 0
    with ProcessPoolExecutor(max_workers=nw) as pool:
        futs = [pool.submit(build_one, it) for it in items]
        for i, fut in enumerate(as_completed(futs), 1):
            _, e = fut.result()
            if e:
                err += 1
                if err <= 5:
                    print(f"[ERROR] {e}", flush=True)
            else:
                done += 1
            if i % 1000 == 0:
                print(f"  {i}/{len(items)} done", flush=True)
    print(f"[DONE] ok={done} err={err}", flush=True)


if __name__ == "__main__":
    main()
