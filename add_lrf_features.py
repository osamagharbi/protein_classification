#!/usr/bin/env python3
"""
add_lrf_features.py

Enrich an existing RI dataset (produced by build_ri_feat_dataset.py) with
Local Reference Frame (LRF) edge features and adds them as a new key
`ri_lrf_feat`  [E_mesh, 8]  (float32) to every .pt file in-place.

LRF at node i
─────────────
Built from the eigenvectors of the covariance matrix of relative neighbor
positions in the mesh graph:

    C_i  =  (1/|N_i|) Σ_{j∈N_i}  (p_j − p_i)(p_j − p_i)ᵀ

Eigenvectors [e1, e2, e3] sorted by descending eigenvalue:
  e1  ─  largest eigenvalue  →  dominant surface tangent (principal curvature direction)
  e2  ─  middle eigenvalue   →  secondary tangent
  e3  ─  smallest eigenvalue →  surface normal direction
  (e3 is sign-disambiguated to align with the stored vertex normal)

8 LRF edge features per directed edge i→j
──────────────────────────────────────────
  [0]  ‖p_j − p_i‖            edge length
  [1]  ⟨ê_ij, e1_i⟩           edge toward principal curvature direction
  [2]  ⟨ê_ij, e2_i⟩           edge toward secondary curvature direction
  [3]  ⟨ê_ij, e3_i⟩           edge toward normal  (≈ ⟨n_i, ê⟩ from ri_feat)
  [4]  ⟨n_j, e1_i⟩            dst normal in principal direction
  [5]  ⟨n_j, e2_i⟩            dst normal in secondary direction
  [6]  ⟨n_j, e3_i⟩            dst normal in normal direction  (≈ ⟨n_i,n_j⟩)
  [7]  ⟨(n_j × n_i), ê_ij⟩   cross-product term  (same as ri_feat dim 4)

Dimensions [1,2] capture WHICH DIRECTION along the surface the edge travels —
information invisible to the existing 5D ri_feat that only knows the angle with
the normal axis.

Self-loop rows are set to zero.

Usage
─────
    python3 add_lrf_features.py \\
        --dataset-dir /path/to/ssl_graphs_1k_geom_enriched_ri \\
        [--workers 16]

The script modifies .pt files in-place (reads → adds key → overwrites).
It is idempotent: running it twice just overwrites ri_lrf_feat with the same
values.
"""

import argparse
from pathlib import Path
import multiprocessing as mp

import torch
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_lrf(pos: torch.Tensor, normals: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
    """
    Compute a Local Reference Frame (LRF) per node via PCA of mesh neighbours.

    Args:
        pos        : [N, 3]
        normals    : [N, 3]  unit vertex normals (used for sign disambiguation)
        edge_index : [2, E]  mesh edge index (may include self-loops)

    Returns:
        lrf        : [N, 3, 3]  columns = [e1, e2, e3] (descending eigenvalue order)
                     e3 sign-disambiguated to align with vertex normal.
    """
    N = pos.shape[0]
    src, dst = edge_index[0], edge_index[1]

    # Remove self-loops for covariance computation
    mask = (src != dst)
    s, d_ = src[mask], dst[mask]

    rel = pos[d_] - pos[s]                      # [E_nsl, 3] relative vectors
    outer = rel.unsqueeze(2) * rel.unsqueeze(1)  # [E_nsl, 3, 3] outer products

    # Accumulate per source node
    C = torch.zeros(N, 3, 3, dtype=pos.dtype)
    C.index_add_(0, s, outer)

    counts = torch.bincount(s, minlength=N).float().clamp(min=1.0)
    C = C / counts.view(N, 1, 1)                # normalised covariance [N, 3, 3]

    # Eigendecomposition: eigh returns eigenvalues ascending, eigenvecs as columns
    _, e_vecs = torch.linalg.eigh(C)            # e_vecs[n, :, k] = k-th eigenvector
    # Reorder: descending eigenvalue → [e1(max), e2(mid), e3(min)]
    e_vecs = e_vecs.flip(dims=[2])              # [N, 3, 3]

    # Sign disambiguation for e3 (≈ normal direction): align with vertex normal
    e3 = e_vecs[:, :, 2]                        # [N, 3]
    flip = (e3 * normals).sum(dim=1) < 0        # [N] bool
    e_vecs[flip, :, 2] *= -1

    # Make e1 consistent: align with mean of outgoing edge directions at each node
    # Compute mean outgoing direction per node
    mean_d = torch.zeros(N, 3, dtype=pos.dtype)
    dn = rel / rel.norm(dim=1, keepdim=True).clamp(min=1e-8)
    mean_d.index_add_(0, s, dn)
    flip_e1 = (mean_d * e_vecs[:, :, 0]).sum(dim=1) < 0
    e_vecs[flip_e1, :, 0] *= -1
    # e2 = e3 × e1 for right-hand consistency
    e_vecs[:, :, 1] = torch.linalg.cross(e_vecs[:, :, 2], e_vecs[:, :, 0])

    return e_vecs  # [N, 3, 3]


def compute_ri_lrf_features(pos: torch.Tensor, normals: torch.Tensor,
                             edge_index: torch.Tensor,
                             lrf: torch.Tensor) -> torch.Tensor:
    """
    Compute 8 LRF-based RI edge features per directed edge.

    Args:
        pos        : [N, 3]
        normals    : [N, 3]
        edge_index : [2, E]
        lrf        : [N, 3, 3]  columns = [e1, e2, e3]

    Returns:
        ri_lrf     : [E, 8]  float32; self-loop rows are zero
    """
    src, dst = edge_index[0], edge_index[1]

    pi, pj   = pos[src], pos[dst]
    ni, nj   = normals[src], normals[dst]
    lrf_i    = lrf[src]                          # [E, 3, 3]

    d    = pj - pi
    dist = d.norm(dim=1, keepdim=True)
    is_self = (dist.squeeze(1) < 1e-8)
    dn   = d / dist.clamp(min=1e-8)             # [E, 3] unit edge direction

    e1_i = lrf_i[:, :, 0]                       # [E, 3]  principal tangent at src
    e2_i = lrf_i[:, :, 1]                       # [E, 3]  secondary tangent at src
    e3_i = lrf_i[:, :, 2]                       # [E, 3]  normal direction at src

    ri_lrf = torch.stack([
        dist.squeeze(1),                              # [0] edge length
        (dn * e1_i).sum(dim=1),                       # [1] edge → principal tangent
        (dn * e2_i).sum(dim=1),                       # [2] edge → secondary tangent
        (dn * e3_i).sum(dim=1),                       # [3] edge → normal  (≈ ⟨ni,ê⟩)
        (nj * e1_i).sum(dim=1),                       # [4] dst normal in e1
        (nj * e2_i).sum(dim=1),                       # [5] dst normal in e2
        (nj * e3_i).sum(dim=1),                       # [6] dst normal in e3 (≈⟨ni,nj⟩)
        torch.linalg.cross(nj, ni).mul(dn).sum(dim=1),# [7] cross-product term
    ], dim=1).float()                                 # [E, 8]

    ri_lrf[is_self] = 0.0
    return ri_lrf


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

def _process_file(pt_path: Path):
    """Load one .pt dict, add ri_lrf_feat, save in-place."""
    try:
        d = torch.load(str(pt_path), map_location="cpu", weights_only=False)

        ei = d.get("edge_index_mesh", d.get("edge_index"))
        if ei is None:
            return pt_path.name, "no edge_index found"

        pos     = d.get("pos")
        normals = d.get("normals")
        if pos is None or normals is None:
            return pt_path.name, "missing pos or normals"

        pos     = pos.float()
        normals = normals.float()

        lrf             = compute_lrf(pos, normals, ei)
        d["ri_lrf_feat"] = compute_ri_lrf_features(pos, normals, ei, lrf)

        torch.save(d, str(pt_path))
        return pt_path.name, None

    except Exception as exc:
        return pt_path.name, str(exc)


def _worker_init():
    torch.set_num_threads(1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Add LRF edge features (ri_lrf_feat [E,8]) to an RI dataset in-place."
    )
    ap.add_argument(
        "--dataset-dir",
        default="/home/ogharbi/datasets/SHREC25/ssl_graphs_1k_geom_enriched_ri",
        help="Dataset directory containing train_set/ and test_set/ subdirs",
    )
    ap.add_argument(
        "--subsets", nargs="+", default=["train_set", "test_set"],
        help="Subdirectories to process",
    )
    ap.add_argument(
        "--workers", type=int, default=min(16, mp.cpu_count()),
        help="Parallel worker processes",
    )
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()

    # Collect files
    files = []
    for subset in args.subsets:
        sub = dataset_dir / subset
        if not sub.exists():
            print(f"  [skip] {sub} does not exist")
            continue
        pts = sorted(sub.glob("*.pt"))
        files.extend(pts)
        print(f"  {subset}: {len(pts)} graphs")

    if not files:
        print("No .pt files found.")
        return

    print(f"\nTotal: {len(files)} files  (workers={args.workers})")
    print("Adding key: ri_lrf_feat  [E_mesh, 8]  in-place\n")

    errors = []
    with mp.Pool(processes=args.workers, initializer=_worker_init) as pool:
        for name, err in tqdm(
            pool.imap_unordered(_process_file, files),
            total=len(files),
            desc="LRF features",
            unit="graph",
        ):
            if err:
                errors.append((name, err))

    n_ok = len(files) - len(errors)
    print(f"\n{'─'*60}")
    print(f"Done:  {n_ok}/{len(files)} files updated")
    if errors:
        print(f"Errors ({len(errors)}):")
        for name, msg in errors[:20]:
            print(f"  {name}: {msg}")
    else:
        print("No errors.")
    print(f"{'─'*60}")
    print()
    print("ri_lrf_feat shape per graph: [E_mesh, 8]  (float32)")
    print("  [0] ‖p_j − p_i‖          edge length")
    print("  [1] ⟨ê_ij, e1_i⟩         edge → principal curvature direction")
    print("  [2] ⟨ê_ij, e2_i⟩         edge → secondary curvature direction")
    print("  [3] ⟨ê_ij, e3_i⟩         edge → normal  (≈ ⟨n_i, ê⟩)")
    print("  [4] ⟨n_j, e1_i⟩          dst normal in principal direction")
    print("  [5] ⟨n_j, e2_i⟩          dst normal in secondary direction")
    print("  [6] ⟨n_j, e3_i⟩          dst normal in normal  (≈ ⟨n_i, n_j⟩)")
    print("  [7] ⟨(n_j × n_i), ê_ij⟩  cross-product term")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
