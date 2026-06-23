#!/usr/bin/env python3
"""
RI-HierNet — Rotation-Invariant Hierarchical Network for SHREC25 protein
surface classification.

Design rationale and evidence base: see RI_HIERNET_DESIGN.md.

Core idea (the one thing every HierGIN variant lacked): extract multi-scale,
fixed-radius, rotation-invariant local surface descriptors via PPF features
computed on-the-fly (no PCA/LRF eigendecomposition — that instability sank
LRF-GIN), then keep HierGIN's proven coarse patch-level GNN for spatial
reasoning.

Architecture
------------
    INPUT  : 1000 surface points  ·  pos (unit-ball)  ·  normals  ·  chem(2)
    STAGE A: 4× RI Set-Abstraction  (FPS → ball-query → PPF → shared MLP → max)
             1000 → 512 → 256 → 128 → 64    radii 0.10 / 0.20 / 0.40 / 0.80
             (FPS centroids are nested → correct multi-level pooling for free)
    STAGE B: patch GNN  (kNN-8 over the 64 SA centroids, 2× dense GIN)
    STAGE C: readout [max ‖ attention]  →  classifier(99)  +  projection(128)

Everything downstream of the raw (pos, normal) inputs is SO(3)+translation
invariant by construction; `python ri_hiernet.py` runs a numerical invariance
check as a correctness gate.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Pure-torch point-cloud ops (batched, dense [B, N, ·]; PointNet++ style)
# ─────────────────────────────────────────────────────────────────────────────

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Pairwise squared L2 distance. src [B,N,3], dst [B,M,3] -> [B,N,M]."""
    return torch.cdist(src, dst) ** 2


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index. points [B,N,C], idx [B,S] or [B,S,K] -> [...,C]."""
    B = points.shape[0]
    view = [B] + [1] * (idx.dim() - 1)
    rep = list(idx.shape); rep[0] = 1
    bidx = torch.arange(B, device=points.device).view(view).repeat(rep)
    return points[bidx, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """FPS. xyz [B,N,3] -> idx [B,npoint] (nested-compatible subset)."""
    B, N, _ = xyz.shape
    dev = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=dev)
    distance = torch.full((B, N), 1e10, device=dev)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=dev)
    bidx = torch.arange(B, dtype=torch.long, device=dev)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[bidx, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def ball_query(radius: float, nsample: int, xyz: torch.Tensor,
               new_xyz: torch.Tensor) -> torch.Tensor:
    """Fixed-radius neighbourhood. Returns idx [B,S,nsample]; if fewer than
    nsample points fall in the ball, pad by repeating the nearest member."""
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    dev = xyz.device
    group_idx = torch.arange(N, device=dev).view(1, 1, N).repeat(B, S, 1)
    sqrdist = square_distance(new_xyz, xyz)                       # [B,S,N]
    group_idx = group_idx.clone()
    group_idx[sqrdist > radius ** 2] = N                         # mark out-of-ball
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]        # in-ball first
    group_first = group_idx[:, :, 0:1].repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]                          # pad w/ nearest
    return group_idx


def knn_index(nsample: int, xyz: torch.Tensor) -> torch.Tensor:
    """k-nearest-neighbour indices (incl. self). xyz [B,S,3] -> [B,S,nsample]."""
    dist = square_distance(xyz, xyz)
    return dist.topk(nsample, dim=-1, largest=False)[1]


def _angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Robust angle in [0,π] between vectors along last dim -> [...,1]."""
    cross = torch.linalg.cross(a, b, dim=-1).norm(dim=-1, keepdim=True)
    dot = (a * b).sum(dim=-1, keepdim=True)
    return torch.atan2(cross, dot)


def ppf_features(new_xyz, new_normal, grouped_xyz, grouped_normal,
                 radius: float) -> torch.Tensor:
    """On-the-fly 6-D Point-Pair-Feature descriptor per (centroid, neighbour).

    All six are invariant to global SO(3) rotation + translation: they use only
    relative displacement and angles between the surface normals and that
    displacement — no covariance eigendecomposition (the LRF-GIN failure mode).
    Chirality is a signed triple product: invariant under proper rotations,
    sign-flipped under reflection — desirable, since proteins are chiral.

    (An earlier 7th feature, the angle to the group centre-of-mass, was dropped:
    the centroid sits at the centre of its own ball, so that vector is ≈0 and its
    normalised direction is float-noise — not rotation-invariant.)

    Returns [B,S,K,6] = (dist, ∠(n_c,d), ∠(n_j,d), ∠(n_c,n_j), dist/r, chirality).
    """
    nc = new_normal.unsqueeze(2)                                 # [B,S,1,3]
    d = grouped_xyz - new_xyz.unsqueeze(2)                       # [B,S,K,3]
    dist = d.norm(dim=-1, keepdim=True)                          # [B,S,K,1]
    d_hat = d / (dist + 1e-8)
    nj = grouped_normal                                          # [B,S,K,3]
    nc_b = nc.expand_as(nj)

    phi_ncd = _angle(nc_b, d_hat)        # ∠(centroid normal, displacement)
    phi_njd = _angle(nj, d_hat)          # ∠(neighbour normal, displacement)
    phi_ncnj = _angle(nc_b, nj)          # ∠(normal, normal)
    rad = dist / radius                  # scale-normalised radial position
    chir = (nc_b * torch.linalg.cross(nj, d_hat, dim=-1)).sum(-1, keepdim=True)
    return torch.cat([dist, phi_ncd, phi_njd, phi_ncnj, rad, chir], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Stage A — RI Set Abstraction
# ─────────────────────────────────────────────────────────────────────────────

class RISetAbstraction(nn.Module):
    """FPS-downsample → fixed-radius ball-query → PPF + carried features →
    shared 1×1 MLP → max-pool over the neighbourhood."""

    def __init__(self, npoint: int, radius: float, nsample: int,
                 in_ch: int, mlp: list[int]):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        layers, c = [], 6 + in_ch                                # 6 PPF dims + carried feat
        for out in mlp:
            layers += [nn.Conv2d(c, out, 1), nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
            c = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, normal, feat):
        idx = farthest_point_sample(xyz, self.npoint)            # [B,S]
        new_xyz = index_points(xyz, idx)
        new_normal = index_points(normal, idx)
        new_normal = F.normalize(new_normal, dim=-1)

        gidx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, gidx)                    # [B,S,K,3]
        grouped_normal = F.normalize(index_points(normal, gidx), dim=-1)

        ppf = ppf_features(new_xyz, new_normal, grouped_xyz, grouped_normal,
                           self.radius)                          # [B,S,K,7]
        grouped = ppf if feat is None else torch.cat(
            [ppf, index_points(feat, gidx)], dim=-1)             # [B,S,K,7+C]

        x = grouped.permute(0, 3, 1, 2)                          # [B,7+C,S,K]
        x = self.mlp(x)                                          # [B,Cout,S,K]
        new_feat = x.max(dim=-1)[0].transpose(1, 2)              # [B,S,Cout]
        return new_xyz, new_normal, new_feat


# ─────────────────────────────────────────────────────────────────────────────
# Stage B — dense GIN on the patch graph
# ─────────────────────────────────────────────────────────────────────────────

class DenseGIN(nn.Module):
    """GIN layer over a fixed kNN graph of patch centroids (dense tensors)."""

    def __init__(self, ch: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(ch, ch), nn.BatchNorm1d(ch), nn.ReLU(inplace=True),
            nn.Linear(ch, ch),
        )
        self.norm = nn.LayerNorm(ch)

    def forward(self, h, nbr_idx):
        # h [B,S,C], nbr_idx [B,S,K] (kNN incl. self)
        agg = index_points(h, nbr_idx).sum(dim=2)                # Σ_{j∈N(i)∪i} h_j
        out = self.eps * h + agg                                 # = (1+eps)h_i + Σ_{j≠i}h_j
        B, S, C = out.shape
        out = self.mlp(out.reshape(B * S, C)).reshape(B, S, C)
        return F.relu(self.norm(out + h))                        # residual


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class RIHierNet(nn.Module):
    def __init__(self, num_classes: int = 99, chem_dim: int = 2,
                 patch_knn: int = 8, patch_layers: int = 2,
                 proj_dim: int = 128, dropout: float = 0.3,
                 radii=(0.10, 0.20, 0.40, 0.80),
                 npoints=(512, 256, 128, 64), nsample: int = 32):
        super().__init__()
        self.chem_dim = chem_dim
        r1, r2, r3, r4 = radii
        n1, n2, n3, n4 = npoints
        self.sa1 = RISetAbstraction(n1, r1, nsample, chem_dim, [32, 32, 64])
        self.sa2 = RISetAbstraction(n2, r2, nsample, 64,       [64, 64, 128])
        self.sa3 = RISetAbstraction(n3, r3, nsample, 128,      [128, 128, 256])
        self.sa4 = RISetAbstraction(n4, r4, nsample, 256,      [256, 256, 512])

        self.patch_knn = patch_knn
        self.patch_gins = nn.ModuleList([DenseGIN(512) for _ in range(patch_layers)])

        self.att = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self.proj = nn.Linear(1024, proj_dim)

    def forward(self, xyz, normal, chem=None, use_chem: bool = True):
        """xyz [B,N,3], normal [B,N,3], chem [B,N,chem_dim] (or None).
        Returns (logits [B,C], proj [B,proj_dim] L2-normalised).

        Two-phase curriculum: phase-1 (geometry-only) sets use_chem=False, which
        zeros the chemical *values* but keeps the channels, so the same weights
        and architecture serve both phases."""
        if self.chem_dim > 0:
            feat = chem if (use_chem and chem is not None) else torch.zeros(
                xyz.shape[0], xyz.shape[1], self.chem_dim,
                device=xyz.device, dtype=xyz.dtype)
        else:
            feat = None

        xyz, normal, feat = self.sa1(xyz, normal, feat)
        xyz, normal, feat = self.sa2(xyz, normal, feat)
        xyz, normal, feat = self.sa3(xyz, normal, feat)
        xyz, normal, feat = self.sa4(xyz, normal, feat)          # [B,64,512]

        nbr = knn_index(min(self.patch_knn, xyz.shape[1]), xyz)
        h = feat
        for gin in self.patch_gins:
            h = gin(h, nbr)

        h_max = h.max(dim=1)[0]                                   # [B,512]
        a = torch.softmax(self.att(h), dim=1)                    # [B,64,1]
        h_att = (a * h).sum(dim=1)                                # [B,512]
        g = torch.cat([h_max, h_att], dim=-1)                    # [B,1024]

        return self.classifier(g), F.normalize(self.proj(g), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test + numerical SO(3)-invariance gate
# ─────────────────────────────────────────────────────────────────────────────

def _random_rotation(device):
    a = torch.randn(3, 3, device=device)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))                        # proper rotation
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    B, N = 4, 1000
    model = RIHierNet(num_classes=99).to(dev).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={dev}  params={n_params:,}")

    xyz = torch.randn(B, N, 3, device=dev) * 0.4
    normal = F.normalize(torch.randn(B, N, 3, device=dev), dim=-1)
    chem = torch.randn(B, N, 2, device=dev)

    # NOTE: FPS draws its seed point from the global RNG, so it is stochastic
    # by design (acts as train-time augmentation). For the invariance gate we
    # reseed identically before each pass so FPS selects the SAME physical
    # points; any residual delta is then a true RI bug, not FPS jitter.
    with torch.no_grad():
        torch.manual_seed(123)
        logits, proj = model(xyz, normal, chem)
    print(f"logits {tuple(logits.shape)}  proj {tuple(proj.shape)}")
    assert logits.shape == (B, 99) and proj.shape == (B, 128)

    # SO(3) + translation invariance: rotate xyz & normals, translate xyz
    R = _random_rotation(dev)
    t = torch.randn(1, 1, 3, device=dev) * 2.0
    xyz_r = xyz @ R.T + t
    normal_r = normal @ R.T
    with torch.no_grad():
        torch.manual_seed(123)
        logits_r, _ = model(xyz_r, normal_r, chem)
    max_dev = (logits - logits_r).abs().max().item()
    rel = max_dev / (logits.abs().max().item() + 1e-9)
    print(f"SO(3)+T invariance: max|Δlogit|={max_dev:.2e}  rel={rel:.2e}")
    print("PASS ✓" if max_dev < 1e-3 else "FAIL ✗ — RI construction broken")
