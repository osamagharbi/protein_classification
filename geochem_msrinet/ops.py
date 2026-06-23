"""Pure PyTorch point-cloud operations for GeoChem-MSRINet.

The project environment used by Codex does not currently expose DGL,
torch-cluster, torch-scatter, or PyTorch3D, so this module intentionally uses
plain torch tensors. It is slower than specialized CUDA kernels but keeps the
new architecture runnable and easy to validate.
"""

from __future__ import annotations

import math
import torch


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather batched point tensors.

    Args:
        points: Tensor shaped ``[B, N, C]``.
        idx: Tensor shaped ``[B, S]`` or ``[B, S, K]``.

    Returns:
        Gathered tensor shaped ``idx.shape + [C]``.
    """
    if points.dim() != 3:
        raise ValueError(f"points must be [B,N,C], got {tuple(points.shape)}")
    if idx.dim() not in (2, 3):
        raise ValueError(f"idx must be [B,S] or [B,S,K], got {tuple(idx.shape)}")
    b = points.shape[0]
    view_shape = [b] + [1] * (idx.dim() - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_idx = torch.arange(b, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_idx, idx]


def pairwise_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Squared pairwise distances from ``src`` to ``dst``.

    Args:
        src: ``[B, N, 3]``.
        dst: ``[B, M, 3]``.
    """
    return torch.cdist(src, dst).pow(2)


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Deterministic batched farthest-point sampling.

    The starting point is fixed to index 0 rather than random. That makes smoke
    tests and rotation-invariance checks deterministic.
    """
    b, n, _ = xyz.shape
    npoint = min(int(npoint), n)
    device = xyz.device
    centroids = torch.zeros(b, npoint, dtype=torch.long, device=device)
    distance = torch.full((b, n), 1e10, device=device)
    farthest = torch.zeros(b, dtype=torch.long, device=device)
    batch_idx = torch.arange(b, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest].view(b, 1, 3)
        dist = (xyz - centroid).pow(2).sum(-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.max(dim=1).indices
    return centroids


def radius_topk_query(
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
    radius: float,
    max_neighbors: int,
    min_neighbors: int = 8,
) -> torch.Tensor:
    """Query neighbors by radius with nearest-neighbor fallback.

    This returns exactly ``max_neighbors`` indices per centroid. Points outside
    the radius are masked out when possible, but if a ball is too sparse the
    nearest points are used to satisfy ``min_neighbors`` and then padded.
    """
    b, n, _ = xyz.shape
    s = new_xyz.shape[1]
    k = min(int(max_neighbors), n)
    min_k = min(int(min_neighbors), k)
    dist = pairwise_distance(new_xyz, xyz)  # [B,S,N]
    nearest = dist.topk(k, dim=-1, largest=False).indices
    nearest_dist = dist.gather(-1, nearest)
    in_radius = nearest_dist <= (radius * radius)

    # Keep radius members. Ensure at least min_k nearest points survive.
    force_keep = torch.arange(k, device=xyz.device).view(1, 1, k) < min_k
    keep = in_radius | force_keep

    # Replace rejected slots with the nearest point so gathers are valid; the
    # duplicate padding is harmless under max/mean pooling and avoids dead balls.
    first = nearest[:, :, :1].expand(b, s, k)
    return torch.where(keep, nearest, first)


def knn_indices(xyz: torch.Tensor, k: int, include_self: bool = True) -> torch.Tensor:
    """Batched kNN indices over ``xyz`` shaped ``[B,N,3]``."""
    n = xyz.shape[1]
    k_eff = min(int(k), n)
    dist = pairwise_distance(xyz, xyz)
    if not include_self:
        eye = torch.eye(n, dtype=torch.bool, device=xyz.device).unsqueeze(0)
        dist = dist.masked_fill(eye, float("inf"))
    return dist.topk(k_eff, dim=-1, largest=False).indices


def random_rotation_matrix(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate random SO(3) rotation matrices with QR decomposition."""
    a = torch.randn(batch, 3, 3, device=device, dtype=dtype)
    q, r = torch.linalg.qr(a)
    # Make Q uniformly oriented and right-handed.
    sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    q = q * sign.unsqueeze(-2)
    det = torch.linalg.det(q)
    q[det < 0, :, -1] *= -1
    return q


def rotate_points_and_normals(
    pos: torch.Tensor, normal: torch.Tensor, rot: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply batched rotation matrices to positions and normals."""
    return pos @ rot, normal @ rot


def ppf_features(
    center_pos: torch.Tensor,
    center_normal: torch.Tensor,
    neigh_pos: torch.Tensor,
    neigh_normal: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """Eight point-pair invariant features.

    Args:
        center_pos: ``[B,S,3]``.
        center_normal: ``[B,S,3]``.
        neigh_pos: ``[B,S,K,3]``.
        neigh_normal: ``[B,S,K,3]``.

    Returns:
        ``[B,S,K,8]`` containing distance, normalized distance, signed normal
        projections, normal-normal dot, chirality, and absolute projections.
    """
    n_i = torch.nn.functional.normalize(center_normal, dim=-1).unsqueeze(2)
    n_j = torch.nn.functional.normalize(neigh_normal, dim=-1)
    d = neigh_pos - center_pos.unsqueeze(2)
    dist = d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    e = d / dist
    n_i_exp = n_i.expand_as(n_j)

    ni_e = (n_i_exp * e).sum(dim=-1, keepdim=True)
    nj_e = (n_j * e).sum(dim=-1, keepdim=True)
    ni_nj = (n_i_exp * n_j).sum(dim=-1, keepdim=True)
    chirality = torch.linalg.cross(n_j, n_i_exp, dim=-1).mul(e).sum(dim=-1, keepdim=True)

    return torch.cat(
        [
            dist,
            dist / max(float(radius), 1e-8),
            ni_e,
            nj_e,
            ni_nj,
            chirality,
            ni_e.abs(),
            nj_e.abs(),
        ],
        dim=-1,
    )


def sinusoidal_radius_encoding(radius: float, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Small deterministic encoding for a radius branch."""
    half = dim // 2
    freq = torch.exp(
        torch.arange(half, device=device, dtype=dtype) * (-math.log(10000.0) / max(half - 1, 1))
    )
    r = torch.tensor(float(radius), device=device, dtype=dtype)
    enc = torch.cat([torch.sin(r * freq), torch.cos(r * freq)], dim=0)
    if dim % 2:
        enc = torch.cat([enc, r.view(1)], dim=0)
    return enc
