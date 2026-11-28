#!/usr/bin/env python3
"""Build HSNet/N2P patch-assignment sidecars at any resolution k (self-contained).

Re-implements, without the external utils/ package, exactly what
build_patchassign_dataset.py + utils/geodesic_patch_decomposer.py do:

  1. seeds: farthest-point sampling in Euclidean space; the first seed is drawn
     with a fresh numpy default_rng(42) per graph (so seeds are NESTED across k:
     the first k seeds of a k'>k run are the k seeds);
  2. assignment: geodesic Voronoi = multi-source Dijkstra over the mesh edges
     (edge_index_mesh if present, else edge_index), edge weight = Euclidean length;
     unreachable nodes fall back to the Euclidean-nearest seed;
  3. patch graph: two patches are adjacent iff a mesh edge crosses their boundary;
     empty patches are dropped and ids compacted to 0..P-1.

Output per graph (same format the training code reads):
  {"patch_ids": LongTensor[N], "patch_edge_index": LongTensor[2, 2E], "num_patches": P}

Usage:
  python tools/build_patchassign.py --input $WORK/ssl_graphs_1k_geom_enriched_ri \
      --out $WORK/ssl_graphs_1k_patchassign_k16 --num-patches 16 --workers 10
  python tools/build_patchassign.py --verify-against $WORK/ssl_graphs_1k_patchassign_k64 \
      --input $WORK/ssl_graphs_1k_geom_enriched_ri --num-patches 64 --max-graphs 50
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def load_graph(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    ei = d["edge_index_mesh"] if "edge_index_mesh" in d else d["edge_index"]
    pos = d.get("pos", d.get("xyz_raw"))
    if pos is None:
        pos = d["x"][:, :3]
    return torch.as_tensor(pos).float().numpy(), torch.as_tensor(ei).long().numpy()


def fps_seeds(pos, k):
    n = pos.shape[0]
    k = min(k, n)
    rng = np.random.default_rng(42)
    seeds = np.empty(k, dtype=np.int64)
    seeds[0] = int(rng.integers(0, n))
    min_dist = np.linalg.norm(pos - pos[seeds[0]], axis=1)
    for i in range(1, k):
        seeds[i] = int(np.argmax(min_dist))
        min_dist = np.minimum(min_dist, np.linalg.norm(pos - pos[seeds[i]], axis=1))
    return seeds


def decompose(pos, ei, k):
    n = pos.shape[0]
    src, dst = ei[0], ei[1]
    ok = (src >= 0) & (dst >= 0) & (src < n) & (dst < n) & (src != dst)
    src, dst = src[ok], dst[ok]
    w = np.linalg.norm(pos[src] - pos[dst], axis=1).astype(np.float64)
    w = np.where((w > 0) & np.isfinite(w), w, 1e-6)
    # both directions; duplicate (u, v) entries are summed by csr_matrix, as in the original
    rows = np.concatenate([src, dst]).astype(np.int32)
    cols = np.concatenate([dst, src]).astype(np.int32)
    graph = csr_matrix((np.concatenate([w, w]), (rows, cols)), shape=(n, n))
    seeds = fps_seeds(pos, k)
    dist = shortest_path(graph, method="D", directed=False, indices=seeds.tolist(),
                         return_predecessors=False)
    owner = np.argmin(dist, axis=0).astype(np.int64)
    owner = np.where(np.isfinite(dist[owner, np.arange(n)]), owner, -1)
    for v in np.where(owner < 0)[0]:                         # unreachable -> nearest seed
        owner[v] = int(np.argmin(np.linalg.norm(pos[seeds] - pos[v], axis=1)))
    return owner, ei


def sidecar(pos, ei, k):
    owner, ei = decompose(pos, ei, k)
    present = sorted(int(p) for p in np.unique(owner[owner >= 0]))
    row = {p: r for r, p in enumerate(present)}
    compact = np.full(owner.shape[0], -1, dtype=np.int64)
    for p, r in row.items():
        compact[owner == p] = r
    pu, pv = owner[ei[0]], owner[ei[1]]
    ok = (pu >= 0) & (pv >= 0) & (pu != pv)
    pairs = {(min(a, b), max(a, b)) for a, b in zip(pu[ok].tolist(), pv[ok].tolist())}
    s, d = [], []
    for lo, hi in pairs:
        s += [row[lo], row[hi]]
        d += [row[hi], row[lo]]
    pei = np.asarray([s, d], dtype=np.int64) if s else np.zeros((2, 0), dtype=np.int64)
    return {"patch_ids": torch.from_numpy(compact), "patch_edge_index": torch.from_numpy(pei),
            "num_patches": len(present)}


def build_one(job):
    src, out, k = job
    try:
        pos, ei = load_graph(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sidecar(pos, ei, k), out)
        return None
    except Exception as e:  # noqa: BLE001
        return f"{src}: {e}"


def edge_set(pei):
    return set(zip(pei[0].tolist(), pei[1].tolist()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="graph dir containing train_set/ and test_set/")
    ap.add_argument("--out", help="output dir (mirrors train_set/ and test_set/)")
    ap.add_argument("--num-patches", type=int, required=True)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--verify-against", default=None,
                    help="existing sidecar dir built by the original pipeline: compare, don't write")
    ap.add_argument("--max-graphs", type=int, default=0)
    args = ap.parse_args()

    inp = Path(args.input)
    # train_set and test_set may be the same folder (OFF meshes: both link to all/): build once
    splits, seen_dirs, aliases = [], {}, {}
    for split in ("train_set", "test_set"):
        if (inp / split).exists():
            real = (inp / split).resolve()
            if real in seen_dirs:
                aliases[split] = seen_dirs[real]
            else:
                seen_dirs[real] = split
                splits.append(split)
    files = [f for split in splits for f in sorted((inp / split).glob("*.pt"))]
    if args.max_graphs:
        files = files[:args.max_graphs] + [f for f in files if f.parent.name == "test_set"][:args.max_graphs]

    if args.verify_against:
        ref = Path(args.verify_against)
        same = diff = missing = 0
        for f in files:
            rf = ref / f.parent.name / f.name
            if not rf.exists():
                missing += 1
                continue
            a = torch.load(rf, map_location="cpu", weights_only=False)
            pos, ei = load_graph(f)
            b = sidecar(pos, ei, args.num_patches)
            ok = (torch.equal(a["patch_ids"].long(), b["patch_ids"]) and
                  int(a["num_patches"]) == b["num_patches"] and
                  edge_set(a["patch_edge_index"]) == edge_set(b["patch_edge_index"]))
            same += ok
            diff += not ok
        print(f"verify k={args.num_patches}: identical={same} different={diff} missing_ref={missing}")
        raise SystemExit(0 if diff == 0 and same > 0 else 1)

    if not args.out:
        ap.error("--out is required unless --verify-against is given")
    out = Path(args.out)
    jobs = [(f, out / f.parent.name / f.name, args.num_patches) for f in files
            if not (out / f.parent.name / f.name).exists()]
    print(f"[build_patchassign] k={args.num_patches}: {len(jobs)} graphs to build -> {out}", flush=True)
    nw = args.workers or min(len(jobs), os.cpu_count() or 4) or 1
    errors = 0
    with ProcessPoolExecutor(max_workers=nw) as pool:
        for i, fut in enumerate(as_completed([pool.submit(build_one, j) for j in jobs]), 1):
            e = fut.result()
            if e:
                errors += 1
                if errors <= 5:
                    print("[ERROR]", e, flush=True)
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    for alias, target in aliases.items():                    # mirror the input's split links
        link = out / alias
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
    print(f"[build_patchassign] done: ok={len(jobs) - errors} errors={errors}", flush=True)
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
