#!/usr/bin/env python3
"""
Offline feature enrichment for SHREC25 SSL graph .pt files.

This script loads saved graph dicts from a dataset root such as:

  datasets/SHREC25/ssl_graphs_1k
  datasets/SHREC25/ssl_graphs_500

and appends graph-only geometric features to each node feature vector `x`.
The original dataset is left untouched; enriched graphs are written to a new
output root with the same split structure.

New features are computed from the saved graph fields only, without going back
to VTK meshes. By default, the script uses normalized positions (`pos`), unit
normals (`normals`), and mesh topology edges (`edge_index_mesh`).

Default added features per node:
  - dist_to_centroid
  - dist_to_anchor_0 .. dist_to_anchor_{K-1}
  - mean_neighbor_dist
  - std_neighbor_dist
  - inv_mean_neighbor_dist
  - laplacian_curvature_mag
  - laplacian_curvature_signed
  - normal_variation
  - planarity
  - surface_variation

The raw geometric features are stored in `geom_features_raw`, and a normalized
copy is stored in `geom_features_norm`. The graph `x` tensor is updated to:

  concat([original_x, geom_features_norm], dim=1)

The feature order is tracked in meta['feature_order_x'].
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append graph-only geometric features to SHREC25 SSL graph .pt files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Dataset root containing split subdirectories with .pt graph files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output dataset root (default: <input-dir>_geom_enriched)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train_set", "test_set"],
        help="Split subdirectories to process",
    )
    parser.add_argument(
        "--position-key",
        type=str,
        default="pos",
        choices=["pos", "xyz_raw"],
        help="Position tensor used to compute geometry features",
    )
    parser.add_argument(
        "--num-anchors",
        type=int,
        default=4,
        help="Number of deterministic FPS-like anchors used for anchor-distance features",
    )
    parser.add_argument(
        "--normalize-mode",
        type=str,
        default="graph_zscore",
        choices=["none", "graph_zscore"],
        help="How to normalize appended geometric features before concatenating into x",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel worker count (default: cpu_count)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional per-split limit for quick tests (0 = no limit)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist",
    )
    return parser.parse_args()


def _resolve_output_root(input_root: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    return input_root.parent / f"{input_root.name}_geom_enriched"


def _feature_names(num_anchors: int) -> List[str]:
    return [
        "dist_to_centroid",
        *[f"dist_to_anchor_{index}" for index in range(num_anchors)],
        "mean_neighbor_dist",
        "std_neighbor_dist",
        "inv_mean_neighbor_dist",
        "laplacian_curvature_mag",
        "laplacian_curvature_signed",
        "normal_variation",
        "planarity",
        "surface_variation",
    ]


def _ensure_numpy_2d(value, name: str, dtype=np.float32) -> np.ndarray:
    arr = torch.as_tensor(value).detach().cpu().numpy().astype(dtype, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    return arr


def _get_undirected_edges(data: Dict[str, object]) -> np.ndarray:
    edge_index = data.get("edge_index_mesh")
    if edge_index is None:
        edge_index = data.get("edge_index")
    if edge_index is None:
        raise KeyError("Graph is missing both edge_index_mesh and edge_index")

    edges = torch.as_tensor(edge_index).detach().cpu().numpy().astype(np.int64, copy=False)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {edges.shape}")

    src, dst = edges[0], edges[1]
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    mask = lo != hi
    if not np.any(mask):
        return np.zeros((0, 2), dtype=np.int64)

    undirected = np.stack([lo[mask], hi[mask]], axis=1)
    undirected = np.unique(undirected, axis=0)
    return undirected.astype(np.int64, copy=False)


def _pick_anchor_indices(pos: np.ndarray, num_anchors: int) -> np.ndarray:
    if num_anchors <= 0:
        return np.zeros((0,), dtype=np.int64)

    n_nodes = pos.shape[0]
    if n_nodes == 0:
        return np.zeros((0,), dtype=np.int64)

    num_anchors = min(num_anchors, n_nodes)
    centroid = pos.mean(axis=0, keepdims=True)
    dists_to_center = np.linalg.norm(pos - centroid, axis=1)
    first = int(np.argmax(dists_to_center))

    chosen = [first]
    min_dists = np.linalg.norm(pos - pos[first], axis=1)

    for _ in range(1, num_anchors):
        next_idx = int(np.argmax(min_dists))
        chosen.append(next_idx)
        min_dists = np.minimum(min_dists, np.linalg.norm(pos - pos[next_idx], axis=1))

    return np.asarray(chosen, dtype=np.int64)


def _graph_zscore(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    safe_std = np.where(std < EPS, 1.0, std)
    normalized = (features - mean) / safe_std
    normalized[:, (std[0] < EPS)] = 0.0
    return normalized.astype(np.float32, copy=False)


def compute_geometry_features(
    data: Dict[str, object],
    position_key: str,
    num_anchors: int,
    normalize_mode: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    pos = _ensure_numpy_2d(data[position_key], position_key)
    n_nodes = pos.shape[0]

    normals_value = data.get("normals")
    if normals_value is None:
        normals = np.zeros_like(pos, dtype=np.float32)
    else:
        normals = _ensure_numpy_2d(normals_value, "normals")
        if normals.shape != pos.shape:
            raise ValueError(f"normals shape {normals.shape} does not match positions {pos.shape}")

    edges = _get_undirected_edges(data)
    feature_names = _feature_names(num_anchors)

    centered = pos - pos.mean(axis=0, keepdims=True)
    dist_to_centroid = np.linalg.norm(centered, axis=1, keepdims=True)

    anchor_indices = _pick_anchor_indices(pos, num_anchors)
    anchor_features: List[np.ndarray] = []
    for idx in anchor_indices.tolist():
        anchor_features.append(np.linalg.norm(pos - pos[idx], axis=1, keepdims=True))
    while len(anchor_features) < num_anchors:
        anchor_features.append(np.zeros((n_nodes, 1), dtype=np.float32))

    if edges.shape[0] == 0:
        zeros = np.zeros((n_nodes, 1), dtype=np.float32)
        raw = np.concatenate(
            [
                dist_to_centroid.astype(np.float32, copy=False),
                *anchor_features,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
            ],
            axis=1,
        )
    else:
        src = edges[:, 0]
        dst = edges[:, 1]
        diff = pos[dst] - pos[src]
        dist = np.linalg.norm(diff, axis=1)

        degree = np.bincount(src, minlength=n_nodes) + np.bincount(dst, minlength=n_nodes)
        degree_f = np.maximum(degree.astype(np.float32), 1.0)

        sum_dist = np.zeros(n_nodes, dtype=np.float32)
        sum_dist2 = np.zeros(n_nodes, dtype=np.float32)
        np.add.at(sum_dist, src, dist)
        np.add.at(sum_dist, dst, dist)
        np.add.at(sum_dist2, src, dist * dist)
        np.add.at(sum_dist2, dst, dist * dist)

        mean_neighbor_dist = (sum_dist / degree_f)[:, None]
        var_neighbor_dist = np.maximum(sum_dist2 / degree_f - mean_neighbor_dist[:, 0] ** 2, 0.0)
        std_neighbor_dist = np.sqrt(var_neighbor_dist)[:, None]
        inv_mean_neighbor_dist = (1.0 / np.maximum(mean_neighbor_dist, EPS)).astype(np.float32, copy=False)

        sum_neighbor_pos = np.zeros((n_nodes, 3), dtype=np.float32)
        np.add.at(sum_neighbor_pos, src, pos[dst])
        np.add.at(sum_neighbor_pos, dst, pos[src])
        mean_neighbor_pos = sum_neighbor_pos / degree_f[:, None]
        lap_vec = mean_neighbor_pos - pos
        lap_mag = np.linalg.norm(lap_vec, axis=1, keepdims=True)
        lap_signed = np.sum(lap_vec * normals, axis=1, keepdims=True)

        normal_dot = np.sum(normals[src] * normals[dst], axis=1)
        normal_dot = np.clip(normal_dot, -1.0, 1.0)
        normal_delta = 1.0 - normal_dot
        sum_normal_delta = np.zeros(n_nodes, dtype=np.float32)
        np.add.at(sum_normal_delta, src, normal_delta)
        np.add.at(sum_normal_delta, dst, normal_delta)
        normal_variation = (sum_normal_delta / degree_f)[:, None]

        outer = diff[:, :, None] * diff[:, None, :]
        sum_outer = np.zeros((n_nodes, 3, 3), dtype=np.float32)
        np.add.at(sum_outer, src, outer)
        np.add.at(sum_outer, dst, outer)
        second_moment = sum_outer / degree_f[:, None, None]
        covariance = second_moment - lap_vec[:, :, None] * lap_vec[:, None, :]
        covariance = 0.5 * (covariance + np.swapaxes(covariance, 1, 2))
        eigvals = np.linalg.eigvalsh(covariance)
        eigvals = np.clip(eigvals, 0.0, None)
        lambda3 = eigvals[:, 0]
        lambda2 = eigvals[:, 1]
        lambda1 = eigvals[:, 2]
        planarity = ((lambda2 - lambda3) / np.maximum(lambda1, EPS))[:, None]
        surface_variation = (lambda3 / np.maximum(lambda1 + lambda2 + lambda3, EPS))[:, None]

        raw = np.concatenate(
            [
                dist_to_centroid.astype(np.float32, copy=False),
                *anchor_features,
                mean_neighbor_dist.astype(np.float32, copy=False),
                std_neighbor_dist.astype(np.float32, copy=False),
                inv_mean_neighbor_dist.astype(np.float32, copy=False),
                lap_mag.astype(np.float32, copy=False),
                lap_signed.astype(np.float32, copy=False),
                normal_variation.astype(np.float32, copy=False),
                planarity.astype(np.float32, copy=False),
                surface_variation.astype(np.float32, copy=False),
            ],
            axis=1,
        )

    if normalize_mode == "graph_zscore":
        normed = _graph_zscore(raw)
    else:
        normed = raw.astype(np.float32, copy=False)

    return raw.astype(np.float32, copy=False), normed, feature_names


def process_one(task: Tuple[str, str, str, int, str]) -> Tuple[bool, str, int, int, str]:
    input_path, output_path, position_key, num_anchors, normalize_mode = task
    try:
        data = torch.load(input_path, map_location="cpu", weights_only=False)
        original_x = torch.as_tensor(data["x"]).detach().cpu().float()

        geom_raw, geom_norm, feature_names = compute_geometry_features(
            data,
            position_key=position_key,
            num_anchors=num_anchors,
            normalize_mode=normalize_mode,
        )

        out = dict(data)
        out["geom_features_raw"] = torch.from_numpy(geom_raw)
        out["geom_features_norm"] = torch.from_numpy(geom_norm)
        out["x"] = torch.cat([original_x, torch.from_numpy(geom_norm)], dim=1)

        meta = dict(data.get("meta", {}))
        base_feature_order = meta.get("feature_order_x")
        if not isinstance(base_feature_order, list) or len(base_feature_order) != original_x.shape[1]:
            base_feature_order = [f"feat_{idx}" for idx in range(original_x.shape[1])]

        meta["feature_order_x"] = [*base_feature_order, *feature_names]
        meta["geom_feature_order_raw"] = feature_names
        meta["geom_feature_order_norm"] = feature_names
        meta["feature_enrichment"] = {
            "script": "enrich_ssl_graph_features.py",
            "position_key": position_key,
            "normalize_mode": normalize_mode,
            "num_anchors": num_anchors,
            "num_added_features": len(feature_names),
            "output_feature_dim": int(out["x"].shape[1]),
        }
        out["meta"] = meta

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, output_path)
        return True, input_path, int(original_x.shape[0]), int(out["x"].shape[1]), ""
    except Exception:
        return False, input_path, 0, 0, traceback.format_exc()


def _iter_split_files(split_dir: Path, limit: int) -> List[Path]:
    files = sorted(split_dir.glob("*.pt"), key=lambda path: (path.stem if not path.stem.isdigit() else int(path.stem)))
    if limit > 0:
        return files[:limit]
    return files


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_dir).expanduser().resolve()
    output_root = _resolve_output_root(input_root, args.output_dir).expanduser().resolve()
    num_workers = args.num_workers or mp.cpu_count()

    print(f"Input dir:        {input_root}")
    print(f"Output dir:       {output_root}")
    print(f"Splits:           {args.splits}")
    print(f"Position key:     {args.position_key}")
    print(f"Normalize mode:   {args.normalize_mode}")
    print(f"Anchor count:     {args.num_anchors}")
    print(f"Workers:          {num_workers}")
    if args.limit:
        print(f"Per-split limit:  {args.limit}")
    print()

    total_ok = 0
    total_skip = 0
    total_fail = 0

    for split in args.splits:
        split_dir = input_root / split
        if not split_dir.exists():
            print(f"[SKIP] split dir not found: {split_dir}")
            continue

        files = _iter_split_files(split_dir, args.limit)
        if not files:
            print(f"[SKIP] no .pt files in {split_dir}")
            continue

        output_split = output_root / split
        output_split.mkdir(parents=True, exist_ok=True)

        tasks = []
        for input_file in files:
            output_file = output_split / input_file.name
            if output_file.exists() and not args.overwrite:
                total_skip += 1
                continue
            tasks.append((str(input_file), str(output_file), args.position_key, args.num_anchors, args.normalize_mode))

        if not tasks:
            print(f"{split}: all {len(files)} files already exist in output")
            continue

        print(f"{split}: processing {len(tasks)} graphs")

        ctx = mp.get_context("spawn")
        with ctx.Pool(num_workers) as pool:
            for ok, path, n_nodes, out_dim, err in tqdm(
                pool.imap_unordered(process_one, tasks),
                total=len(tasks),
                desc=split,
                unit="graph",
            ):
                if ok:
                    total_ok += 1
                else:
                    total_fail += 1
                    print(f"\n[FAIL] {path}\n{err}")

        print()

    print(f"Summary: OK={total_ok} skipped={total_skip} failed={total_fail}")
    if total_ok:
        print(f"Enriched dataset root: {output_root}")
    if total_fail:
        print("Some files failed; see traceback above.")


if __name__ == "__main__":
    main()