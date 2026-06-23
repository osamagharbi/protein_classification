"""Dataset and batching utilities for GeoChem-MSRINet."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .ops import random_rotation_matrix, rotate_points_and_normals


DEFAULT_GRAPH_DIR = Path("/home/ogharbi/datasets/SHREC25/ssl_graphs_1k_geom_enriched_ri")
DEFAULT_TEST_LABELS = Path("/home/ogharbi/datasets/SHREC25/graphs/test_labels.csv")

GEOM14_RAW_STATS = {
    "dist_centroid": (0.5304, 0.1879),
    "dist_anchor_0": (1.0685, 0.4176),
    "dist_anchor_1": (0.9485, 0.4168),
    "dist_anchor_2": (0.7922, 0.3202),
    "dist_anchor_3": (0.7632, 0.3046),
    "surf_variation": (0.0648, 0.0491),
}


def canonical_id(value: str) -> str:
    """Normalize graph ids from CSV rows and file stems."""
    v = str(value).strip()
    if v.endswith(".pt"):
        v = v[:-3]
    return v


def read_label_csv(path: str | Path) -> dict[str, int]:
    path = Path(path)
    labels: dict[str, int] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("protein_id") or row.get("id") or row.get("graph_id")
            cls = row.get("class_id") or row.get("label") or row.get("y")
            if pid is None or cls is None:
                raise ValueError(f"{path} must contain protein_id/class_id columns")
            labels[canonical_id(pid)] = int(cls)
    return labels


def resolve_split_dir(graph_dir: str | Path, split: str) -> Path:
    root = Path(graph_dir)
    candidates = [root / split, root / f"{split}_set"]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not resolve split {split!r} under {root}")


def default_label_csv(graph_dir: str | Path, split: str) -> Path:
    root = Path(graph_dir)
    fallback_ssl_1k = root.parent / "ssl_graphs_1k"
    if split == "train":
        for p in (
            root / "train_split.csv",
            root / "train_labels.csv",
            fallback_ssl_1k / "train_split.csv",
            fallback_ssl_1k / "train_labels.csv",
        ):
            if p.exists():
                return p
    if split == "val":
        for p in (root / "val_split.csv", fallback_ssl_1k / "val_split.csv"):
            if p.exists():
                return p
    if split == "test":
        for p in (root / "test_labels.csv", DEFAULT_TEST_LABELS):
            if p.exists():
                return p
    raise FileNotFoundError(f"No default label CSV found for split {split!r}")


def available_items(graph_dir: str | Path, split: str, labels: dict[str, int]) -> list[tuple[str, Path, int]]:
    split_dir = resolve_split_dir(graph_dir, split if split != "val" else "train")
    files = {canonical_id(p.stem): p for p in split_dir.glob("*.pt")}
    items: list[tuple[str, Path, int]] = []
    missing = 0
    for gid, label in labels.items():
        p = files.get(canonical_id(gid))
        if p is None:
            missing += 1
            continue
        items.append((gid, p, int(label)))
    if missing:
        print(f"[{split}] skipped {missing} labels with no .pt file in {split_dir}")
    return items


def class_counts(labels: Iterable[int], num_classes: int = 99) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for y in labels:
        if 0 <= int(y) < num_classes:
            counts[int(y)] += 1
    return counts


@dataclass
class BatchAugment:
    rotate: bool = True
    jitter: float = 0.0
    point_dropout: float = 0.0
    normal_noise: float = 0.0
    scalar_noise: float = 0.0


class ProteinPointDataset(Dataset):
    """Loads SHREC25 graph ``.pt`` files as dense point clouds.

    Existing graph files already contain 1,000 FPS-sampled points. The loader
    uses ``pos``, ``normals``, and ``scalars_norm`` by default, and can optionally
    append the six globally normalized raw geometric features from the 14D run.
    """

    def __init__(
        self,
        graph_dir: str | Path = DEFAULT_GRAPH_DIR,
        split: str = "train",
        labels_csv: str | Path | None = None,
        npoint: int = 1000,
        use_geom_raw: bool = False,
        clamp: float = 5.0,
    ):
        self.graph_dir = Path(graph_dir)
        self.split = split
        self.npoint = int(npoint)
        self.use_geom_raw = bool(use_geom_raw)
        self.clamp = float(clamp)
        labels_path = Path(labels_csv) if labels_csv else default_label_csv(self.graph_dir, split)
        self.labels = read_label_csv(labels_path)
        self.items = available_items(self.graph_dir, split, self.labels)
        if not self.items:
            raise RuntimeError(f"No items found for split={split} graph_dir={graph_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def ordered_labels(self) -> list[int]:
        return [label for _, _, label in self.items]

    def _select_or_pad(self, *xs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        n = xs[0].shape[0]
        if self.npoint <= 0 or n == self.npoint:
            return xs
        if n > self.npoint:
            # Random subset supports FPS-TTA if a future dataset has >1k points.
            idx = torch.randperm(n)[: self.npoint]
        else:
            extra = torch.randint(0, n, (self.npoint - n,))
            idx = torch.cat([torch.arange(n), extra])
        return tuple(x[idx] for x in xs)

    def _geom14(self, d: dict) -> torch.Tensor:
        raw = d.get("geom_features_raw")
        if raw is None:
            return torch.empty(d["pos"].shape[0], 0)
        names = d.get("meta", {}).get("geom_feature_order_raw", [])
        columns = []
        for key, (mean, std) in GEOM14_RAW_STATS.items():
            if key not in names:
                continue
            idx = names.index(key)
            columns.append((raw[:, idx].float() - mean) / max(std, 1e-8))
        if not columns:
            return torch.empty(d["pos"].shape[0], 0)
        return torch.stack(columns, dim=1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        gid, path, label = self.items[idx]
        d = torch.load(path, map_location="cpu", weights_only=False)
        pos = d["pos"].float()
        normal = F.normalize(d["normals"].float(), dim=-1)
        chem = d["scalars_norm"].float().clamp(-self.clamp, self.clamp)
        geom = self._geom14(d) if self.use_geom_raw else torch.empty(pos.shape[0], 0)
        pos, normal, chem, geom = self._select_or_pad(pos, normal, chem, geom)
        return {
            "id": gid,
            "pos": pos,
            "normal": normal,
            "chem": chem,
            "geom": geom,
            "label": torch.tensor(label, dtype=torch.long),
        }


def make_collate(augment: BatchAugment | None = None):
    augment = augment or BatchAugment(rotate=False)

    def collate(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
        ids = [str(item["id"]) for item in batch]
        pos = torch.stack([item["pos"] for item in batch]).float()
        normal = torch.stack([item["normal"] for item in batch]).float()
        chem = torch.stack([item["chem"] for item in batch]).float()
        geom = torch.stack([item["geom"] for item in batch]).float()
        label = torch.stack([item["label"] for item in batch]).long()

        if augment.point_dropout > 0:
            b, n, _ = pos.shape
            keep = torch.rand(b, n) > augment.point_dropout
            keep[:, 0] = True
            for bi in range(b):
                dropped = (~keep[bi]).nonzero(as_tuple=False).flatten()
                kept = keep[bi].nonzero(as_tuple=False).flatten()
                repl = kept[torch.randint(0, kept.numel(), (dropped.numel(),))]
                pos[bi, dropped] = pos[bi, repl]
                normal[bi, dropped] = normal[bi, repl]
                chem[bi, dropped] = chem[bi, repl]
                if geom.shape[-1]:
                    geom[bi, dropped] = geom[bi, repl]

        if augment.jitter > 0:
            pos = pos + torch.randn_like(pos) * augment.jitter
        if augment.normal_noise > 0:
            normal = F.normalize(normal + torch.randn_like(normal) * augment.normal_noise, dim=-1)
        if augment.scalar_noise > 0:
            chem = chem + torch.randn_like(chem) * augment.scalar_noise
        if augment.rotate:
            rot = random_rotation_matrix(pos.shape[0], pos.device, pos.dtype)
            pos, normal = rotate_points_and_normals(pos, normal, rot)

        return {"id": ids, "pos": pos, "normal": normal, "chem": chem, "geom": geom, "label": label}

    return collate
