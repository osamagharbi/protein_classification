#!/usr/bin/env python3
"""
Sample N nodes from ssl_graphs protein mesh graphs using Farthest Point Sampling (FPS).

Produces graphs with fewer nodes while preserving:
  - All 8 original features: [Potential_norm, NormalPotential_norm, nx, ny, nz, x_ssl, y_ssl, z_ssl]
  - All per-vertex tensors:  x, pos, xyz_raw, normals, scalars_raw, scalars_norm
  - Mesh-like connectivity:  original mesh edges (both endpoints sampled) + kNN augmentation
  - Same .pt file format as ssl_graphs (drop-in replacement)

Connectivity strategy (two-pass):
  Pass 1 – keep original mesh edges whose both endpoints were sampled.
           Preserves local mesh topology where FPS picks nearby vertices.
  Pass 2 – augment with kNN-k edges on sampled 3D positions.
           Ensures the graph stays connected and has ~k neighbours per node
           regardless of how spread-out the sampled points are.

Usage:
    # Fixed node count (same N for every graph)
    python sample_ssl_graphs.py --n-points 1000
    python sample_ssl_graphs.py --n-points 2500 --knn-k 8

    # Relative to original size (different N per graph)
    python sample_ssl_graphs.py --ratio 0.1

    # Custom I/O
    python sample_ssl_graphs.py --n-points 1000 \\
        --input-dir datasets/SHREC25/ssl_graphs \\
        --output-dir datasets/SHREC25/ssl_graphs_1k

    # Re-process already-existing files
    python sample_ssl_graphs.py --n-points 1000 --overwrite

Output directory is auto-named  <input_dir>_<N>pts  or  <input_dir>_r<ratio>
when --output-dir is not given.
"""

import argparse
import multiprocessing as mp
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Farthest Point Sampling
# ─────────────────────────────────────────────────────────────────────────────

def fps(pos: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Farthest Point Sampling on 3-D positions.

    Starts from the vertex nearest the centroid (deterministic, no random seed)
    and greedily picks the farthest unselected vertex at each step.

    Uses the dot-product identity ‖a-b‖² = ‖a‖²-2(a·b)+‖b‖² to replace
    an [N,3] subtraction tensor per iteration with a single BLAS matmul.
    Operates on numpy for minimal dispatch overhead on CPU.

    Args:
        pos:       [N, 3] float32 tensor of 3-D positions
        n_samples: number of points to select (≤ N)

    Returns:
        [n_samples] long tensor of selected node indices (unsorted)
    """
    N = pos.shape[0]
    if n_samples >= N:
        return torch.arange(N, dtype=torch.long)

    p = pos.numpy().astype(np.float32, copy=False)   # [N, 3], no copy if already float32
    pos_norm2 = (p * p).sum(axis=1)                  # [N], precomputed once

    selected = np.empty(n_samples, dtype=np.int64)
    min_dist2 = np.full(N, np.inf, dtype=np.float32)

    # Deterministic start: vertex closest to the centroid
    centroid = p.mean(axis=0)
    diff = p - centroid
    current = int((diff * diff).sum(axis=1).argmin())

    for i in range(n_samples):
        selected[i] = current
        # ‖a - b‖² = ‖a‖² - 2(a·b) + ‖b‖²  — only a matmul needed per step
        d2 = pos_norm2 - 2.0 * (p @ p[current]) + pos_norm2[current]
        np.minimum(min_dist2, d2, out=min_dist2)
        current = int(min_dist2.argmax())

    return torch.from_numpy(selected)


# ─────────────────────────────────────────────────────────────────────────────
# Edge construction
# ─────────────────────────────────────────────────────────────────────────────

def build_subgraph_edges(
    pos_sampled: torch.Tensor,
    sampled_idx: torch.Tensor,
    edge_index_mesh_orig: torch.Tensor,
    knn_k: int = 6,
) -> tuple:
    """
    Build edges for the sampled subgraph.

    Args:
        pos_sampled:         [n, 3] positions of sampled nodes
        sampled_idx:         [n]    original indices of sampled nodes
        edge_index_mesh_orig:[2, E] undirected mesh edges of the full graph
        knn_k:               k for kNN augmentation

    Returns:
        edge_index_mesh: [2, E'] undirected edges (u ≤ v), no self-loops, no duplicates
        edge_index:      [2, 2E'] bidirectional edges
    """
    n = pos_sampled.shape[0]

    # ── Pass 1: filter original mesh edges ───────────────────────────────────
    max_orig = int(sampled_idx.max().item()) + 1
    # id_map[original_id] → new_id  (-1 = not sampled)
    id_map = torch.full((max_orig,), -1, dtype=torch.long)
    id_map[sampled_idx] = torch.arange(n, dtype=torch.long)

    src_o, dst_o = edge_index_mesh_orig[0], edge_index_mesh_orig[1]
    in_range = (src_o < max_orig) & (dst_o < max_orig)
    new_src = id_map[src_o[in_range]]
    new_dst = id_map[dst_o[in_range]]
    both_ok = (new_src >= 0) & (new_dst >= 0) & (new_src != new_dst)
    mesh_src = new_src[both_ok].numpy()
    mesh_dst = new_dst[both_ok].numpy()

    # ── Pass 2: kNN on sampled positions ─────────────────────────────────────
    k = min(knn_k, n - 1)
    tree = cKDTree(pos_sampled.numpy())
    _, neigh = tree.query(pos_sampled.numpy(), k=k + 1)   # k+1 includes self
    neigh = neigh[:, 1:]                                   # drop self (col 0)

    knn_src = np.repeat(np.arange(n), k)
    knn_dst = neigh.flatten()

    # ── Union & deduplication ─────────────────────────────────────────────────
    all_src = np.concatenate([mesh_src, knn_src])
    all_dst = np.concatenate([mesh_dst, knn_dst])

    # Canonicalize to (u, v) with u ≤ v to deduplicate undirected edges
    u = np.minimum(all_src, all_dst)
    v = np.maximum(all_src, all_dst)
    edges = np.unique(np.stack([u, v], axis=1), axis=0)   # [E', 2]

    src_t = torch.from_numpy(edges[:, 0]).long()
    dst_t = torch.from_numpy(edges[:, 1]).long()

    edge_index_mesh = torch.stack([src_t, dst_t], dim=0)          # [2, E']
    edge_index = torch.cat([                                       # [2, 2E']
        edge_index_mesh,
        torch.stack([dst_t, src_t], dim=0),
    ], dim=1)

    return edge_index_mesh, edge_index


# ─────────────────────────────────────────────────────────────────────────────
# Per-file worker
# ─────────────────────────────────────────────────────────────────────────────

# Keys that are per-vertex tensors and must be indexed by sampled_idx
_PER_VERTEX_KEYS = ('x', 'pos', 'xyz_raw', 'normals', 'scalars_raw', 'scalars_norm')


def process_one(task: tuple) -> tuple:
    """
    Worker: load one .pt file, FPS-sample it, save the result.

    Args:
        task: (input_path, output_path, n_points, ratio, knn_k)

    Returns:
        (ok, input_path, n_orig, n_sampled, error_msg)
    """
    input_path, output_path, n_points, ratio, knn_k = task
    try:
        data = torch.load(input_path, map_location='cpu', weights_only=False)

        pos    = data['pos']   # [N, 3]  SSL-normalised positions (used for FPS)
        N_orig = pos.shape[0]

        # ── Determine target size ─────────────────────────────────────────────
        if n_points is not None:
            n_target = min(n_points, N_orig)
        else:
            n_target = max(10, int(round(N_orig * ratio)))

        # ── FPS ───────────────────────────────────────────────────────────────
        sampled_idx = fps(pos, n_target)
        n_actual = len(sampled_idx)

        # ── Build output dict ─────────────────────────────────────────────────
        out = {}

        # Per-vertex tensors: direct index (no interpolation)
        for key in _PER_VERTEX_KEYS:
            v = data.get(key)
            if torch.is_tensor(v):
                out[key] = v[sampled_idx]

        # ── Edges ─────────────────────────────────────────────────────────────
        edge_mesh_orig = data.get('edge_index_mesh', data['edge_index'])
        edge_index_mesh, edge_index = build_subgraph_edges(
            out['pos'], sampled_idx, edge_mesh_orig, knn_k=knn_k,
        )
        out['edge_index']      = edge_index        # [2, 2E']
        out['edge_index_mesh'] = edge_index_mesh   # [2, E']

        # ── Meta ──────────────────────────────────────────────────────────────
        orig_meta = dict(data.get('meta', {}))
        out['meta'] = {
            **orig_meta,
            'num_nodes':                 n_actual,
            'num_mesh_edges_undirected': edge_index_mesh.shape[1],
            'num_edges_bidir':           edge_index.shape[1],
            'sampling': {
                'method':     'fps',
                'n_original': N_orig,
                'n_sampled':  n_actual,
                'ratio':      n_actual / N_orig,
                'knn_k':      knn_k,
            },
        }

        # ── Save ──────────────────────────────────────────────────────────────
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, output_path)

        return True, input_path, N_orig, n_actual, ''

    except Exception:
        return False, input_path, 0, 0, traceback.format_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='FPS-sample ssl_graphs protein meshes, preserving features & connectivity.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--input-dir', type=str,
        default='datasets/SHREC25/ssl_graphs',
        help='Root directory with train_set/ and test_set/ subdirs',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output root (default: <input_dir>_<N>pts  or  <input_dir>_r<ratio>)',
    )

    # Exactly one of these is required
    size_group = parser.add_mutually_exclusive_group(required=True)
    size_group.add_argument(
        '--n-points', type=int,
        help='Fixed number of nodes per output graph (e.g. 1000)',
    )
    size_group.add_argument(
        '--ratio', type=float,
        help='Fraction of original nodes to keep, e.g. 0.1 for 10%%',
    )

    parser.add_argument(
        '--knn-k', type=int, default=6,
        help='k for kNN augmentation (matches original mesh avg degree of 6)',
    )
    parser.add_argument(
        '--splits', nargs='+', default=['train_set', 'test_set'],
        help='Subdirectories to process',
    )
    parser.add_argument(
        '--num-workers', type=int, default=None,
        help='Parallel workers (default: all logical CPUs)',
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Re-process and overwrite already-existing output files',
    )

    args = parser.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────────
    if args.ratio is not None and not (0.0 < args.ratio <= 1.0):
        parser.error('--ratio must be in (0, 1]')
    if args.n_points is not None and args.n_points < 1:
        parser.error('--n-points must be ≥ 1')

    # ── Paths ─────────────────────────────────────────────────────────────────
    input_root = Path(args.input_dir)
    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        suffix = f'{args.n_points}pts' if args.n_points else f'r{args.ratio}'
        out_root = input_root.parent / f'{input_root.name}_{suffix}'

    n_workers = args.num_workers or mp.cpu_count()

    print(f'Input dir:  {input_root}')
    print(f'Output dir: {out_root}')
    if args.n_points:
        print(f'Sampling:   FPS → {args.n_points} nodes per graph')
    else:
        print(f'Sampling:   FPS → {args.ratio:.1%} of original nodes per graph')
    print(f'kNN-k:      {args.knn_k}  (connectivity augmentation)')
    print(f'Workers:    {n_workers}')
    print()

    total_ok = total_skip = total_fail = 0

    for split in args.splits:
        split_dir = input_root / split
        if not split_dir.exists():
            print(f'[SKIP] {split_dir} not found')
            continue

        files = sorted(split_dir.glob('*.pt'))
        if not files:
            print(f'[SKIP] No .pt files in {split_dir}')
            continue

        out_split = out_root / split
        out_split.mkdir(parents=True, exist_ok=True)

        # Build task list (skip already-done unless --overwrite)
        tasks = []
        for f in files:
            out_f = out_split / f.name
            if not args.overwrite and out_f.exists():
                total_skip += 1
                continue
            tasks.append((str(f), str(out_f), args.n_points, args.ratio, args.knn_k))

        n_already = len(files) - len(tasks)
        if not tasks:
            print(f'{split}: all {len(files)} files already done (--overwrite to redo)')
            continue

        print(f'{split}: {len(tasks)} to process'
              + (f'  ({n_already} already exist, skipped)' if n_already else ''))

        # ── Parallel processing ───────────────────────────────────────────────
        import time
        n_orig_list, n_sampled_list = [], []
        t0 = time.time()

        ctx = mp.get_context('spawn')
        with ctx.Pool(n_workers) as pool:
            for ok, path, n_orig, n_sampled, err in tqdm(
                pool.imap_unordered(process_one, tasks),
                total=len(tasks),
                desc=split,
                unit='graph',
            ):
                if ok:
                    total_ok += 1
                    n_orig_list.append(n_orig)
                    n_sampled_list.append(n_sampled)
                else:
                    total_fail += 1
                    print(f'\n[FAIL] {path}\n{err}')

        elapsed = time.time() - t0
        if n_orig_list:
            orig_arr    = np.array(n_orig_list)
            sampled_arr = np.array(n_sampled_list)
            print(
                f'  Done {elapsed:.1f}s | '
                f'orig: {orig_arr.mean():.0f}±{orig_arr.std():.0f} nodes | '
                f'sampled: {sampled_arr.mean():.0f}±{sampled_arr.std():.0f} nodes | '
                f'ratio: {(sampled_arr / orig_arr).mean():.1%}'
            )
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'Summary:  OK={total_ok}  skipped={total_skip}  failed={total_fail}')
    if total_fail:
        print(f'WARNING: {total_fail} files failed — see errors above')
    if total_ok:
        print(f'Output: {out_root}')


if __name__ == '__main__':
    main()
