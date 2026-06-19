#!/usr/bin/env python3
"""
build_ri_feat_dataset.py

Precompute SO(3)-invariant RIConv2 edge features for every graph in a dataset
and write an enriched copy to a new directory.

For each directed edge i→j, five invariant scalars are appended as
d["ri_feat"]  (shape [E_mesh, 5], float32):

    [0]  ‖p_j − p_i‖                    edge length
    [1]  ⟨n_i , ê_ij⟩                   src-normal / edge-direction angle
    [2]  ⟨n_j , ê_ij⟩                   dst-normal / edge-direction angle
    [3]  ⟨n_i , n_j⟩                    normal-normal angle
    [4]  ⟨(n_j × n_i) , ê_ij⟩          cross-product / edge-direction scalar

Features are computed from d["pos"] (SSL xyz) and d["normals"] using the
mesh edge index d["edge_index_mesh"].  Self-loops get ri_feat = 0.

All other keys in the dict are preserved unchanged.  CSV split files found
in the source directory are copied to the destination.

Usage
-----
    python3 build_ri_feat_dataset.py \\
        --src-dir  /path/to/ssl_graphs_1k \\
        --dst-dir  /path/to/ssl_graphs_1k_ri \\
        [--test-csv /path/to/graphs_old/test_labels.csv] \\
        [--workers 16]

The destination will mirror the source layout:
    <dst-dir>/
        train_set/   *.pt  (enriched)
        test_set/    *.pt  (enriched)
        train_split.csv    (copied)
        val_split.csv      (copied)
        test_labels.csv    (copied from --test-csv if provided)
        <any other *.csv or *.json files from src>

Integration with the training script
--------------------------------------
After generating the dataset, pass it as --graph-dir.  The training script's
HierPatchDataset.__getitem__ already builds a DGL graph by calling:

    g = dgl.graph(edge_index_mesh).remove_self_loop().add_self_loop()

After this call the edge ordering is:
    edges 0 .. E_mesh-1   : mesh edges  (precomputed ri_feat available)
    edges E_mesh .. E_mesh+N-1 : self-loops (ri_feat = 0)

To use precomputed features, modify RIConvLayer.forward to check:
    if "ri_feat" in g.edata:
        ri = g.edata["ri_feat"]
    else:
        ri = self._compute_ri(g)    # fallback: on-the-fly
"""

import argparse
import shutil
from pathlib import Path
from functools import partial
import multiprocessing as mp

import torch
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Core computation (pure torch, no DGL required in workers)
# ──────────────────────────────────────────────────────────────────────────────

def compute_ri_features(pos: torch.Tensor, normals: torch.Tensor,
                         edge_index: torch.Tensor) -> torch.Tensor:
    """
    Compute 5 SO(3)-invariant edge scalars.

    Args:
        pos        : [N, 3] node positions (SSL normalised xyz)
        normals    : [N, 3] unit surface normals
        edge_index : [2, E] directed edge indices (src, dst)

    Returns:
        ri_feat    : [E, 5] float32 tensor; self-loop rows are zero
    """
    src, dst = edge_index[0], edge_index[1]

    pi = pos[src]       # [E, 3]
    pj = pos[dst]
    ni = normals[src]   # [E, 3]
    nj = normals[dst]

    d    = pj - pi                                  # [E, 3]  raw edge vector
    dist = d.norm(dim=1, keepdim=True)              # [E, 1]  edge length

    is_self = (dist.squeeze(1) < 1e-8)             # self-loops: distance ≈ 0
    dn = d / dist.clamp(min=1e-8)                  # [E, 3]  unit edge direction

    ri = torch.stack([
        dist.squeeze(1),                                      # ‖p_j − p_i‖
        (ni * dn).sum(dim=1),                                 # ⟨n_i, ê_ij⟩
        (nj * dn).sum(dim=1),                                 # ⟨n_j, ê_ij⟩
        (ni * nj).sum(dim=1),                                 # ⟨n_i, n_j⟩
        torch.linalg.cross(nj, ni).mul(dn).sum(dim=1),       # ⟨n_j×n_i, ê_ij⟩
    ], dim=1).float()                                         # [E, 5]

    ri[is_self] = 0.0
    return ri


# ──────────────────────────────────────────────────────────────────────────────
# Worker (called in subprocess)
# ──────────────────────────────────────────────────────────────────────────────

def _process_file(args):
    """Load one .pt dict, add ri_feat, save to destination."""
    src_path, dst_path = args
    try:
        d = torch.load(str(src_path), map_location="cpu", weights_only=False)

        ei = d.get("edge_index_mesh", d.get("edge_index"))
        if ei is None:
            return src_path.name, "no edge_index found"

        pos     = d.get("pos")
        normals = d.get("normals")
        if pos is None or normals is None:
            return src_path.name, "missing pos or normals"

        d["ri_feat"] = compute_ri_features(pos.float(), normals.float(), ei)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(d, str(dst_path))
        return src_path.name, None

    except Exception as exc:
        return src_path.name, str(exc)


def _worker_init():
    """Suppress torch inter-op threads to avoid oversubscription."""
    torch.set_num_threads(1)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Precompute SO(3)-invariant RI edge features for an SSL graph dataset."
    )
    ap.add_argument("--src-dir",  required=True,
                    help="Source dataset directory (contains train_set/ and test_set/)")
    ap.add_argument("--dst-dir",  required=True,
                    help="Destination directory for enriched dataset")
    ap.add_argument("--test-csv", default=None,
                    help="Path to test_labels.csv (copied from a different directory, "
                         "e.g. graphs_old/test_labels.csv)")
    ap.add_argument("--workers",  type=int, default=min(16, mp.cpu_count()),
                    help="Number of parallel worker processes (default: min(16, nproc))")
    ap.add_argument("--subsets",  nargs="+", default=["train_set", "test_set"],
                    help="Subdirectory names to process (default: train_set test_set)")
    args = ap.parse_args()

    src = Path(args.src_dir).resolve()
    dst = Path(args.dst_dir).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    # ── Collect .pt files ──────────────────────────────────────────────────────
    work_items = []
    for subset in args.subsets:
        src_sub = src / subset
        dst_sub = dst / subset
        if not src_sub.exists():
            print(f"  [skip] {src_sub} does not exist")
            continue
        pt_files = sorted(src_sub.glob("*.pt"))
        for f in pt_files:
            work_items.append((f, dst_sub / f.name))
        print(f"  {subset}: {len(pt_files)} graphs")

    if not work_items:
        print("No .pt files found — check --src-dir and --subsets.")
        return

    print(f"\nTotal: {len(work_items)} graphs → {dst}  (workers={args.workers})")

    # ── Copy metadata files ────────────────────────────────────────────────────
    # CSV split files
    for fname in ["train_split.csv", "val_split.csv",
                  "train_labels.csv", "val_labels.csv"]:
        src_f = src / fname
        if src_f.exists():
            shutil.copy2(src_f, dst / fname)
            print(f"  Copied {fname}")

    # JSON stats files (normalisation constants used by training script)
    for json_f in src.glob("*.json"):
        shutil.copy2(json_f, dst / json_f.name)
        print(f"  Copied {json_f.name}")

    # External test labels CSV
    if args.test_csv:
        src_test = Path(args.test_csv)
        if src_test.exists():
            shutil.copy2(src_test, dst / src_test.name)
            print(f"  Copied {src_test.name} (from --test-csv)")
        else:
            print(f"  [warn] --test-csv path not found: {src_test}")

    # ── Process graphs in parallel ─────────────────────────────────────────────
    print()
    errors = []
    with mp.Pool(processes=args.workers, initializer=_worker_init) as pool:
        for name, err in tqdm(
            pool.imap_unordered(_process_file, work_items),
            total=len(work_items),
            desc="Computing RI features",
            unit="graph",
        ):
            if err:
                errors.append((name, err))

    # ── Summary ────────────────────────────────────────────────────────────────
    n_ok = len(work_items) - len(errors)
    print(f"\n{'─'*60}")
    print(f"Done:  {n_ok}/{len(work_items)} graphs written to {dst}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for name, msg in errors[:20]:
            print(f"  {name}: {msg}")
        if len(errors) > 20:
            print(f"  ... and {len(errors)-20} more")
    else:
        print("No errors.")
    print(f"{'─'*60}")
    print()
    print("ri_feat shape per graph: [E_mesh, 5]  (float32)")
    print("  dim 0: ‖p_j − p_i‖")
    print("  dim 1: ⟨n_i, ê_ij⟩")
    print("  dim 2: ⟨n_j, ê_ij⟩")
    print("  dim 3: ⟨n_i, n_j⟩")
    print("  dim 4: ⟨(n_j × n_i), ê_ij⟩")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
