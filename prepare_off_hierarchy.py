#!/usr/bin/env python3
"""Prepare OFF mesh collections for the coordinate-only HiPatch pipeline.

This mirrors the SHREC19 representation used by the classifier:

* vertices are FPS-sampled to 1024 points;
* points are centered and scaled to the unit ball;
* a 16-nearest-neighbour graph is built (unique undirected edges plus a
  bidirectional ``edge_index``);
* face-derived vertex normals are retained for invariant geometric encoders;
* a 16-channel heat-kernel signature (HKS) on the sampled graph (intrinsic: unchanged by
  rotations and by near-isometric pose changes).

All graphs go to ``<output>/all/``; ``train_set`` and ``test_set`` are symlinks to it, so any
split (the committed files in ``data_labels/shrec11`` and ``data_labels/tosca``, written by
``tools/offmesh/make_splits.py``) can be used without copying. Patches are built afterwards with
``tools/build_patchassign.py`` (same geodesic Voronoi construction as SHREC2025).  The small serializer
below writes standard PyTorch zip checkpoints without importing torch, which
keeps preparation usable on machines where importing the very large torch
library is slow.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import pickle
import random
import sys
import types
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Minimal standard-PyTorch zip writer
# ---------------------------------------------------------------------------

class _StorageRef:
    def __init__(self, dtype: str, key: str, size: int, raw: bytes):
        self.dtype = dtype
        self.key = key
        self.size = int(size)
        self.raw = raw


class _TensorRef:
    def __init__(self, array: np.ndarray, dtype: str, storage: _StorageRef):
        self.array = array
        self.dtype = dtype
        self.storage = storage

    def __reduce__(self):
        # The real torch loader resolves this global and rebuilds a Tensor.
        return (_torch_rebuild_tensor_v2, (
            self.storage,
            0,
            tuple(int(x) for x in self.array.shape),
            tuple(int(self.array.strides[i] // self.array.itemsize)
                  for i in range(self.array.ndim)),
            False,
            None,
        ))


def _torch_rebuild_tensor_v2(storage, storage_offset, size, stride,
                             requires_grad, backward_hooks):
    # Never called while writing; this function only supplies the pickle global
    # name expected by torch.load.
    raise RuntimeError("placeholder used only for serialization")


_torch_rebuild_tensor_v2.__module__ = "torch._utils"
_torch_rebuild_tensor_v2.__name__ = "_rebuild_tensor_v2"
_torch_rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"

# Make pickle emit the canonical torch global names without importing torch.
_torch_stub = types.ModuleType("torch")
_torch_utils_stub = types.ModuleType("torch._utils")
_torch_utils_stub._rebuild_tensor_v2 = _torch_rebuild_tensor_v2
_torch_stub._utils = _torch_utils_stub
sys.modules.setdefault("torch", _torch_stub)
sys.modules.setdefault("torch._utils", _torch_utils_stub)

FloatStorage = type("FloatStorage", (), {"__module__": "torch"})
LongStorage = type("LongStorage", (), {"__module__": "torch"})
setattr(_torch_stub, "FloatStorage", FloatStorage)
setattr(_torch_stub, "LongStorage", LongStorage)


class _TorchPickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, _StorageRef):
            storage_type = FloatStorage if obj.dtype == "float32" else LongStorage
            return ("storage", storage_type, obj.key, "cpu", obj.size)
        return None


def _tensor_ref(array: np.ndarray, key: str):
    array = np.ascontiguousarray(array)
    if array.dtype == np.float32:
        dtype = "float32"
    elif array.dtype == np.int64:
        dtype = "int64"
    else:
        raise TypeError(f"Unsupported tensor dtype: {array.dtype}")
    storage = _StorageRef(dtype, key, int(array.size), array.tobytes(order="C"))
    return _TensorRef(array, dtype, storage), storage


def save_torch_dict(path: Path, obj: dict):
    tensors = []
    converted = {}

    def convert(value):
        if isinstance(value, np.ndarray):
            ref, storage = _tensor_ref(value, str(len(tensors)))
            tensors.append(storage)
            return ref
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        return value

    converted = convert(obj)
    buffer = io.BytesIO()
    _TorchPickler(buffer, protocol=2).dump(converted)
    payload = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("archive/data.pkl", payload)
        zf.writestr("archive/byteorder", "little")
        zf.writestr("archive/version", "3\n")
        zf.writestr("archive/.data/serialization_id", "00000000000000000000000000000000")
        for storage in tensors:
            zf.writestr(f"archive/data/{storage.key}", storage.raw)


# ---------------------------------------------------------------------------
# OFF, sampling, graph and patch preparation
# ---------------------------------------------------------------------------

def read_off(path: Path):
    tokens = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens.extend(line.split())
    if not tokens or tokens[0] != "OFF":
        raise ValueError(f"{path} is not an OFF file")
    nverts, nfaces = int(tokens[1]), int(tokens[2])
    p = 4  # OFF, vertex count, face count, edge count
    verts = np.asarray([[float(tokens[p + 3*i + j]) for j in range(3)]
                        for i in range(nverts)], dtype=np.float32)
    p += 3 * nverts
    faces = []
    for _ in range(nfaces):
        n = int(tokens[p]); p += 1
        ids = [int(x) for x in tokens[p:p+n]]; p += n
        if n >= 3:
            for j in range(1, n - 1):
                faces.append((ids[0], ids[j], ids[j+1]))
    return verts, np.asarray(faces, dtype=np.int64)


def fps(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    n = min(int(n), len(points))
    rng = np.random.default_rng(seed)
    selected = np.empty(n, dtype=np.int64)
    dmin = np.full(len(points), np.inf, dtype=np.float32)
    current = int(rng.integers(0, len(points)))
    for i in range(n):
        selected[i] = current
        d = ((points - points[current]) ** 2).sum(axis=1)
        dmin = np.minimum(dmin, d)
        current = int(np.argmax(dmin))
    return selected


def knn_edges(pos: np.ndarray, k: int = 16):
    k_eff = min(int(k), len(pos) - 1)
    _, nbr = cKDTree(pos).query(pos, k=k_eff + 1, workers=-1)
    pairs = set()
    for u in range(len(pos)):
        for v in np.asarray(nbr[u]).reshape(-1):
            v = int(v)
            if v != u:
                pairs.add((u, v) if u < v else (v, u))
    und = np.asarray(sorted(pairs), dtype=np.int64)
    edge_mesh = und.T.copy()
    edge_bidir = np.concatenate([und, und[:, ::-1]], axis=0).T.copy()
    return edge_mesh, edge_bidir


def patch_assignment(pos, edge_mesh, num_patches=64, seed=42):
    seeds = fps(pos, num_patches, seed=seed)
    dmin = np.full(len(pos), np.inf, dtype=np.float32)
    seed_ids = np.empty(len(pos), dtype=np.int64)
    # The SHREC19 patch builder uses Euclidean FPS seeds and geodesic Voronoi
    # assignment over the graph edges weighted by Euclidean edge length.
    rows = np.concatenate([edge_mesh[0], edge_mesh[1]])
    cols = np.concatenate([edge_mesh[1], edge_mesh[0]])
    weights = np.linalg.norm(pos[rows] - pos[cols], axis=1).astype(np.float64)
    csr = csr_matrix((weights, (rows, cols)), shape=(len(pos), len(pos)))
    dist = shortest_path(csr, directed=False, indices=seeds, method="D")
    owner = np.argmin(dist, axis=0).astype(np.int64)
    best = dist[owner, np.arange(len(pos))]
    owner[~np.isfinite(best)] = -1

    present = sorted(int(x) for x in np.unique(owner[owner >= 0]))
    remap = {old: i for i, old in enumerate(present)}
    compact = np.asarray([remap.get(int(x), -1) for x in owner], dtype=np.int64)
    cross = set()
    for u, v in edge_mesh.T.tolist():
        a, b = int(compact[u]), int(compact[v])
        if a >= 0 and b >= 0 and a != b:
            cross.add((a, b) if a < b else (b, a))
    if cross:
        pairs = np.asarray(sorted(cross), dtype=np.int64)
        patch_edges = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T
    else:
        patch_edges = np.empty((2, 0), dtype=np.int64)
    return compact, patch_edges, len(present)


def normalize(points):
    center = points.mean(axis=0)
    centered = points - center
    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale <= 1e-12:
        scale = 1.0
    return (centered / scale).astype(np.float32), center.astype(np.float32), scale


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute area-weighted per-vertex normals from triangulated OFF faces."""
    normals = np.zeros_like(vertices, dtype=np.float32)
    if faces.size:
        tri = vertices[faces]
        face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        for corner in range(3):
            np.add.at(normals, faces[:, corner], face_normals)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(length, 1e-12)
    return normals.astype(np.float32)


def graph_hks(pos: np.ndarray, edge_mesh: np.ndarray, n_times: int = 16) -> np.ndarray:
    """Compute a compact heat-kernel signature on the sampled surface graph.

    This is deliberately used only by the OFF-derived datasets.  The HKS is
    built from a symmetric weighted KNN Laplacian and is invariant to rigid
    rotations and translations, unlike raw XYZ coordinates.
    """
    n = len(pos)
    rows, cols = edge_mesh
    lengths = np.linalg.norm(pos[rows] - pos[cols], axis=1).astype(np.float64)
    scale = max(float(np.median(lengths)), 1e-8)
    weights = np.exp(-((lengths / scale) ** 2))
    a = coo_matrix((weights, (rows, cols)), shape=(n, n))
    a = a + a.T
    lap = diags(np.asarray(a.sum(axis=1)).ravel()) - a
    k = min(n - 2, n_times + 1)
    try:
        evals, evecs = eigsh(lap, k=k, which="SM", tol=1e-3)
    except Exception:
        # The fallback keeps preparation robust for an unusual disconnected
        # or numerically degenerate OFF graph.
        evals, evecs = eigsh(lap, k=k, which="SM", tol=1e-2, maxiter=5000)
    order = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]
    positive = np.maximum(evals[1:], 1e-8)
    if len(positive) == 0:
        return np.zeros((n, n_times), dtype=np.float32)
    t_min = 4.0 * math.log(10.0) / positive[-1]
    t_max = 4.0 * math.log(10.0) / positive[0]
    times = np.exp(np.linspace(np.log(t_min), np.log(max(t_max, t_min)), n_times))
    hks = (evecs[:, 1:] ** 2) @ np.exp(-positive[:, None] * times[None, :])
    hks = (hks - hks.mean(axis=0, keepdims=True)) / (
        hks.std(axis=0, keepdims=True) + 1e-8
    )
    return hks.astype(np.float32)


def graph_for_off(path: Path, sample_seed: int, patch_seed: int):
    verts, faces = read_off(path)
    if len(verts) < 4:
        raise ValueError(f"Too few vertices in {path}")
    normals_full = vertex_normals(verts, faces)
    idx = fps(verts, 1024, seed=sample_seed) if len(verts) > 1024 else np.arange(len(verts))
    raw = verts[idx].astype(np.float32, copy=False)
    normals = normals_full[idx].copy()
    pos, center, scale = normalize(raw)
    if len(pos) < 1024:
        # This is not expected for TOSCA/SHREC11, but preserves SHREC19's
        # fixed-node contract if a future OFF file is unusually small.
        rng = np.random.default_rng(sample_seed)
        pad = rng.choice(len(pos), 1024 - len(pos), replace=True)
        raw = np.concatenate([raw, raw[pad]], axis=0)
        normals = np.concatenate([normals, normals[pad]], axis=0)
        pos, center, scale = normalize(raw)
    edge_mesh, edge_index = knn_edges(pos, 16)
    patch_ids, patch_edge_index, n_patch = patch_assignment(pos, edge_mesh, 64, patch_seed)
    hks = graph_hks(pos, edge_mesh, n_times=16)
    meta = {
        "source_file": str(path),
        "feature_order_x": ["x_ssl", "y_ssl", "z_ssl"],
        "feature_order_normals": ["nx", "ny", "nz"],
        "dataset": path.parent.name,
        "num_nodes": int(len(pos)),
        "num_mesh_edges_undirected": int(edge_mesh.shape[1]),
        "num_edges_bidir": int(edge_index.shape[1]),
        "topology": "knn",
        "knn_k": 16,
        "xyz_ssl_norm": {
            "type": "centered_unit_sphere",
            "center": [float(x) for x in center],
            "scale": float(scale),
        },
    }
    graph = {
        "x": pos,
        "edge_index": edge_index,
        "edge_index_mesh": edge_mesh,
        "pos": pos.copy(),
        "normals": normals,
        "hks": hks,
        "xyz_raw": raw,
        "meta": meta,
    }
    assign = {
        "patch_ids": patch_ids,
        "patch_edge_index": patch_edge_index,
        "num_patches": int(n_patch),
    }
    return graph, assign


def stratified_split(items, class_of, seed=42, test_ratio=0.2):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for item in items:
        by_class[class_of[item]].append(item)
    train, test = [], []
    for cls in sorted(by_class, key=str):
        group = sorted(by_class[cls])
        rng.shuffle(group)
        ntest = min(max(1, int(round(len(group) * test_ratio))), max(0, len(group)-1))
        test.extend(group[:ntest]); train.extend(group[ntest:])
    return sorted(train), sorted(test)


def write_labels(path, ids, class_of, class_name):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["protein_id", "class_id", "class_name", "surface_id"])
        for gid in sorted(ids):
            cls = class_of[gid]
            w.writerow([gid, cls, class_name[cls], gid])


def _process_one(task):
    """Worker for one mesh; kept at module scope for multiprocessing."""
    path, out_dir, gid, sample_seed = task
    out = Path(out_dir) / f"{gid}.pt"
    if out.exists():
        return gid
    graph, _ = graph_for_off(Path(path), sample_seed=sample_seed, patch_seed=42)
    tmp = out.with_suffix(".part")
    save_torch_dict(tmp, graph)
    tmp.rename(out)
    return gid


def prepare(input_dir: Path, output_dir: Path, dataset_name: str, class_parser,
            seed=42, workers=1):
    files = sorted(input_dir.glob("*.off"))
    if not files:
        raise RuntimeError(f"No OFF files found in {input_dir}")
    class_name, class_of = {}, {}
    for f in files:
        cls, name = class_parser(f)
        class_of[f.stem] = int(cls)
        class_name[int(cls)] = name
    all_dir = output_dir / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train_set", "test_set"):          # every split reads the same folder
        link = output_dir / split
        if not link.exists():
            link.symlink_to("all", target_is_directory=True)
    # the FPS start of each mesh depends only on its position in the sorted file list
    tasks = [(str(f), str(all_dir), f.stem, seed + i) for i, f in enumerate(files)]
    workers = max(1, int(workers))
    print(f"Processing {len(tasks)} meshes with {workers} worker(s)...", flush=True)
    if workers == 1:
        for i, task in enumerate(tasks, 1):
            _process_one(task)
            if i % 25 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_one, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), 1):
                future.result()
                if i % 25 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)}", flush=True)
    write_labels(output_dir / "all_labels.csv", list(class_of), class_of, class_name)
    summary = {
        "dataset": dataset_name,
        "num_files": len(files),
        "num_classes": len(class_name),
        "nodes": 1024, "knn_k": 16,
        "sampling": "farthest-point sampling over OFF vertices (start = rng(seed + file index))",
        "normalization": "centered and scaled to unit sphere",
        "features": ["xyz", "vertex_normals", "hks_16"],
        "layout": "all/ with train_set and test_set symlinks; splits in data_labels/",
    }
    (output_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def tosca_class(path):
    import re
    m = re.match(r"([A-Za-z]+)", path.stem)
    name = m.group(1).lower() if m else "unknown"
    names = ["cat", "centaur", "david", "dog", "gorilla", "horse", "michael", "victoria", "wolf"]
    return names.index(name), name


def load_shrec11_categories(path: Path):
    """Read the official SHREC11 ``categories.txt`` mapping.

    SHREC11 filenames are not grouped by class: the 20 members of a class
    are scattered throughout T0--T599.  The categories file is therefore
    required; assigning ``T<n>`` to ``n // 20`` silently corrupts every label.
    """
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    mapping = {}
    names = {}
    i = 2  # skip the PSB header and the ``30 600`` line
    class_id = 0
    while i < len(lines) and class_id < 30:
        if not lines[i]:
            i += 1
            continue
        header = lines[i].split()
        if len(header) != 3 or header[1] != "0":
            raise ValueError(f"Unexpected SHREC11 categories line: {lines[i]!r}")
        name, _, count_text = header
        count = int(count_text)
        names[class_id] = name
        i += 1
        members = []
        while i < len(lines) and len(members) < count:
            if lines[i]:
                members.append(lines[i])
            i += 1
        if len(members) != count:
            raise ValueError(f"Category {name!r} has {len(members)} members, expected {count}")
        for member in members:
            mapping[f"T{int(member)}"] = (class_id, name)
        class_id += 1
    if len(mapping) != 600 or class_id != 30:
        raise ValueError(f"Expected 600 SHREC11 mappings, got {len(mapping)}")
    return mapping


def shrec11_class(path, category_map):
    try:
        return category_map[path.stem]
    except KeyError as exc:
        raise KeyError(f"{path.stem} is missing from the SHREC11 categories file") from exc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["tosca", "shrec11"], required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument(
        "--categories",
        type=Path,
        default=None,
        help="Official SHREC11 class file (test.cla from For_evaluation.zip, or DiffusionNet's "
             "categories.txt); required for shrec11.",
    )
    args = ap.parse_args()
    if args.dataset == "tosca":
        parser = tosca_class
    else:
        if args.categories is None:
            ap.error("--categories is required for shrec11")
        category_map = load_shrec11_categories(args.categories)
        parser = lambda path: shrec11_class(path, category_map)
    prepare(args.input, args.output, args.dataset, parser, args.seed, args.workers)


if __name__ == "__main__":
    main()
