#!/usr/bin/env python3
"""
Patch-level GNN classification for SHREC25 with graph-level voting.

This script trains a classifier on decomposed patch graphs, predicts a class for
each patch, then aggregates patch predictions per source graph to obtain the
final graph class.

Usage:
    python classify_patch_gnn.py --device 0
    python classify_patch_gnn.py --device 0 --patch-subdir patches_k16
    python classify_patch_gnn.py --device 0 --patch-subdir patches_k32 --model gin --hidden-dim 512
    python classify_patch_gnn.py --device 0 --vote-method majority --rotation-aug

Expected patch layout:
    <data-dir>/train_set/<patch-subdir>/*.pt
    <data-dir>/test_set/<patch-subdir>/*.pt

Each patch file must contain the source graph id and graph label. The patch
decomposer used in this workspace writes both.
"""

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm import tqdm

import dgl

from classify_gnn import (
    canonical_graph_id,
    FocalLoss,
    GNNClassifier,
    compute_metrics,
    load_graph_file,
    rotate_graph_z,
    select_topk_populated_classes,
)
from fix_training import compute_class_weights


PATCH_NAME_RE = re.compile(r"^(?P<gid>.+)_patch_(?P<pid>\d+)_graph_label_(?P<label>-?\d+)\.pt$")


@dataclass(frozen=True)
class PatchRecord:
    path: Path
    graph_id: str
    label: int
    patch_id: int


def resolve_patch_split_dir(root_dir: str, split: str, patch_subdir: str) -> Path:
    """Resolve train/test patch directory across naming variants."""
    root = Path(root_dir)
    candidates = [
        root / split / patch_subdir,
        root / f"{split}_set" / patch_subdir,
        root / split,
        root / f"{split}_set",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir() and any(candidate.glob("*.pt")):
            return candidate
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return candidates[0]


def _parse_patch_record(path: Path) -> PatchRecord:
    match = PATCH_NAME_RE.match(path.name)
    if match:
        return PatchRecord(
            path=path,
            graph_id=str(match.group("gid")),
            label=int(match.group("label")),
            patch_id=int(match.group("pid")),
        )

    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"Unsupported patch content in {path}")

    meta = data.get("meta", {}) if isinstance(data.get("meta", {}), dict) else {}
    graph_id = data.get("source_graph_id", meta.get("source_graph_id"))
    graph_label = data.get("graph_label", meta.get("source_graph_label"))
    patch_id = data.get("patch_id", meta.get("patch_id"))

    if graph_id is None or graph_label is None or patch_id is None:
        raise KeyError(
            f"Could not infer patch metadata from {path}. Expected source_graph_id, graph_label, and patch_id."
        )

    return PatchRecord(
        path=path,
        graph_id=str(graph_id),
        label=int(graph_label),
        patch_id=int(patch_id),
    )


def collect_patch_records(split_dir: Path) -> List[PatchRecord]:
    records = [_parse_patch_record(path) for path in sorted(split_dir.glob("*.pt"))]
    if not records:
        raise FileNotFoundError(f"No patch .pt files found in {split_dir}")
    return sorted(records, key=lambda record: (record.graph_id, record.patch_id, record.path.name))


def safe_compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
        }
    return compute_metrics(y_true, y_pred, num_classes=num_classes)


def build_graph_label_dict(records: Sequence[PatchRecord]) -> Dict[str, int]:
    graph_labels: Dict[str, int] = {}
    for record in records:
        graph_id = canonical_graph_id(record.graph_id)
        previous = graph_labels.get(graph_id)
        if previous is not None and previous != record.label:
            raise ValueError(
                f"Inconsistent labels for graph {graph_id!r}: saw {previous} and {record.label}"
            )
        graph_labels[graph_id] = record.label
    return graph_labels


def remap_record_labels(records: Sequence[PatchRecord], graph_label_dict: Dict[str, int]) -> List[PatchRecord]:
    remapped: List[PatchRecord] = []
    for record in records:
        graph_id = canonical_graph_id(record.graph_id)
        if graph_id not in graph_label_dict:
            continue
        remapped.append(replace(record, label=int(graph_label_dict[graph_id])))
    return remapped


def remap_records_with_class_map(records: Sequence[PatchRecord], class_map: Optional[Dict[int, int]]) -> List[PatchRecord]:
    if class_map is None:
        return list(records)

    remapped: List[PatchRecord] = []
    for record in records:
        if record.label not in class_map:
            continue
        remapped.append(replace(record, label=int(class_map[record.label])))
    return remapped


def split_records_by_graph(
    records: Sequence[PatchRecord],
    graph_ids: Iterable[str],
) -> List[PatchRecord]:
    allowed = {canonical_graph_id(graph_id) for graph_id in graph_ids}
    return [record for record in records if canonical_graph_id(record.graph_id) in allowed]


def load_graph_ids_from_csv(csv_path: str) -> List[str]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Graph split CSV not found: {path}")

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "protein_id" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a protein_id column")
        graph_ids = [canonical_graph_id(str(row["protein_id"])) for row in reader if row.get("protein_id")]

    if not graph_ids:
        raise ValueError(f"No graph ids found in {path}")

    unique_graph_ids = sorted(set(graph_ids))
    print(f"Loaded {len(unique_graph_ids)} graph ids from: {path}")
    return unique_graph_ids


def validate_requested_graph_ids(
    requested_graph_ids: Sequence[str],
    available_graph_ids: Sequence[str],
    split_name: str,
    csv_path: str,
) -> None:
    available = set(available_graph_ids)
    missing = sorted(set(requested_graph_ids) - available)
    if not missing:
        return

    preview = ", ".join(missing[:5])
    raise ValueError(
        f"{len(missing)} graph ids from {csv_path} were not found in the available {split_name} patch set. "
        f"Examples: {preview}"
    )


def create_robust_graph_train_val_split(
    graph_label_dict: Dict[str, int],
    val_split: float = 0.2,
    min_graphs_for_stratify: int = 2,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Split train graphs into train/val without leaking patches across graphs."""
    if not 0.0 <= val_split < 1.0:
        raise ValueError(f"--val-split must be in range [0.0, 1.0), got {val_split}")
    if val_split == 0.0:
        return sorted(graph_label_dict.keys()), []

    class_counts = Counter(graph_label_dict.values())
    stratifiable = [gid for gid, label in graph_label_dict.items() if class_counts[label] >= min_graphs_for_stratify]
    non_stratifiable = [gid for gid, label in graph_label_dict.items() if class_counts[label] < min_graphs_for_stratify]

    print("\nTrain/Val Graph Split Strategy:")
    print(f"  Stratifiable graphs:     {len(stratifiable)}")
    print(f"  Train-only rare graphs:  {len(non_stratifiable)}")
    print(f"  Rare classes (< {min_graphs_for_stratify} graphs) stay in training only")

    if stratifiable:
        strat_labels = [graph_label_dict[gid] for gid in stratifiable]
        try:
            train_graphs, val_graphs = train_test_split(
                stratifiable,
                test_size=val_split,
                stratify=strat_labels,
                random_state=seed,
            )
        except ValueError as exc:
            print(f"  Warning: stratified split failed ({exc}). Falling back to random graph split.")
            train_graphs, val_graphs = train_test_split(
                stratifiable,
                test_size=val_split,
                random_state=seed,
                shuffle=True,
            )
    else:
        train_graphs, val_graphs = [], []

    train_graphs = sorted(train_graphs + non_stratifiable)
    val_graphs = sorted(val_graphs)

    train_classes = {graph_label_dict[gid] for gid in train_graphs}
    val_classes = {graph_label_dict[gid] for gid in val_graphs}
    print("\nFinal graph split:")
    print(f"  Train graphs: {len(train_graphs)}")
    print(f"  Val graphs:   {len(val_graphs)}")
    print(f"  Train classes: {len(train_classes)}")
    print(f"  Val classes:   {len(val_classes)}")
    print(f"  Classes only in train: {len(train_classes - val_classes)}")

    return train_graphs, val_graphs


def summarize_records(name: str, records: Sequence[PatchRecord]) -> None:
    if not records:
        print(f"{name} patches: 0")
        print(f"{name} graphs:  0")
        return
    graph_label_dict = build_graph_label_dict(records)
    patch_counts = Counter(record.graph_id for record in records)
    patch_per_graph = np.array(list(patch_counts.values()), dtype=np.float32)
    print(f"{name} patches: {len(records)}")
    print(f"{name} graphs:  {len(graph_label_dict)}")
    print(
        f"{name} patches/graph: mean={patch_per_graph.mean():.2f}, min={patch_per_graph.min():.0f}, max={patch_per_graph.max():.0f}"
    )


def prepare_patch_graph(dgl_graph):
    dgl_graph = dgl_graph.remove_self_loop().add_self_loop()

    if "_ID" in dgl_graph.ndata:
        dgl_graph.ndata.pop("_ID")
    if "_ID" in dgl_graph.edata:
        dgl_graph.edata.pop("_ID")

    return dgl_graph


def group_records_by_graph(records: Sequence[PatchRecord]) -> List[Tuple[str, int, List[PatchRecord]]]:
    grouped: Dict[str, List[PatchRecord]] = defaultdict(list)
    labels: Dict[str, int] = {}

    for record in records:
        graph_id = canonical_graph_id(record.graph_id)
        grouped[graph_id].append(record)
        previous = labels.get(graph_id)
        if previous is not None and previous != record.label:
            raise ValueError(
                f"Inconsistent labels for grouped graph {graph_id!r}: saw {previous} and {record.label}"
            )
        labels[graph_id] = int(record.label)

    grouped_records: List[Tuple[str, int, List[PatchRecord]]] = []
    for graph_id in sorted(grouped.keys()):
        graph_records = sorted(grouped[graph_id], key=lambda record: record.patch_id)
        grouped_records.append((graph_id, labels[graph_id], graph_records))
    return grouped_records


class PatchDataset(Dataset):
    """Lazy patch dataset backed by decomposed patch .pt files."""

    def __init__(self, records: Sequence[PatchRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        dgl_graph = prepare_patch_graph(load_graph_file(record.path))

        return dgl_graph, torch.tensor(record.label, dtype=torch.long), record.graph_id


class GraphPatchBagDataset(Dataset):
    """Dataset of graphs represented as bags of patch graphs."""

    def __init__(self, records: Sequence[PatchRecord]):
        self.bags = group_records_by_graph(records)

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int):
        graph_id, label, records = self.bags[idx]
        graphs = [prepare_patch_graph(load_graph_file(record.path)) for record in records]
        return graphs, torch.tensor(label, dtype=torch.long), graph_id


def collate_patch_fn(batch):
    graphs, labels, graph_ids = zip(*batch)
    batch_g = dgl.batch(list(graphs))
    batch_labels = torch.stack(list(labels), dim=0)
    return batch_g, batch_labels, list(graph_ids)


def collate_graph_patch_bag_fn(batch):
    flat_graphs = []
    graph_labels = []
    graph_ids = []
    bag_sizes: List[int] = []
    patch_labels = []
    patch_graph_ids: List[str] = []

    for graphs, label, graph_id in batch:
        flat_graphs.extend(graphs)
        graph_labels.append(label)
        graph_ids.append(graph_id)
        bag_sizes.append(len(graphs))
        patch_labels.extend([label.clone() for _ in graphs])
        patch_graph_ids.extend([graph_id] * len(graphs))

    batch_g = dgl.batch(flat_graphs)
    batch_graph_labels = torch.stack(graph_labels, dim=0)
    batch_patch_labels = torch.stack(patch_labels, dim=0)
    return batch_g, batch_graph_labels, graph_ids, bag_sizes, batch_patch_labels, patch_graph_ids


def aggregate_patch_logits_to_graph_logits(
    patch_logits: torch.Tensor,
    bag_sizes: Sequence[int],
    vote_method: str = "logit_mean",
) -> torch.Tensor:
    splits = torch.split(patch_logits, [int(size) for size in bag_sizes], dim=0)
    train_vote_method = "logit_mean" if vote_method == "majority" else vote_method
    return torch.stack([aggregate_patch_logits(split, train_vote_method) for split in splits], dim=0)


def aggregate_patch_logits(
    patch_logits: torch.Tensor,
    vote_method: str,
) -> torch.Tensor:
    probs = torch.softmax(patch_logits, dim=1)

    if vote_method == "logit_mean":
        return patch_logits.mean(dim=0)

    if vote_method == "confidence_logit_mean":
        weights = probs.max(dim=1).values
        weights = weights / weights.sum().clamp(min=1e-12)
        return (patch_logits * weights.unsqueeze(1)).sum(dim=0)

    if vote_method == "prob_mean":
        return probs.mean(dim=0).clamp(min=1e-12).log()

    if vote_method == "confidence_prob_mean":
        weights = probs.max(dim=1).values
        weights = weights / weights.sum().clamp(min=1e-12)
        return (probs * weights.unsqueeze(1)).sum(dim=0).clamp(min=1e-12).log()

    raise ValueError(f"Unknown vote method: {vote_method}")


def aggregate_graph_predictions(
    graph_logits: Dict[str, Dict[str, object]],
    vote_method: str,
    num_classes: int,
) -> Dict[str, float]:
    if not graph_logits:
        metrics = safe_compute_metrics([], [], num_classes=num_classes)
        metrics["num_graphs"] = 0
        return metrics

    y_true: List[int] = []
    y_pred: List[int] = []

    for graph_id in sorted(graph_logits.keys()):
        entry = graph_logits[graph_id]
        true_label = int(entry["label"])
        logits = torch.stack(entry["logits"])
        probs = torch.softmax(logits, dim=1)
        patch_preds = torch.argmax(probs, dim=1).cpu().numpy()

        if vote_method == "majority":
            counts = np.bincount(patch_preds, minlength=num_classes)
            winners = np.flatnonzero(counts == counts.max())
            if len(winners) == 1:
                pred_label = int(winners[0])
            else:
                winner_scores = probs[:, winners].sum(dim=0)
                pred_label = int(winners[int(torch.argmax(winner_scores).item())])
        else:
            pred_label = int(torch.argmax(aggregate_patch_logits(logits, vote_method)).item())

        y_true.append(true_label)
        y_pred.append(pred_label)

    metrics = safe_compute_metrics(y_true, y_pred, num_classes=num_classes)
    metrics["num_graphs"] = len(y_true)
    return metrics


def merge_patch_and_graph_metrics(
    patch_true: Sequence[int],
    patch_pred: Sequence[int],
    graph_logits: Dict[str, Dict[str, object]],
    num_classes: int,
    vote_method: str,
    loss: float,
) -> Dict[str, float]:
    patch_metrics = safe_compute_metrics(patch_true, patch_pred, num_classes=num_classes)
    graph_metrics = aggregate_graph_predictions(graph_logits, vote_method, num_classes)

    merged = {"loss": float(loss)}
    for key, value in patch_metrics.items():
        merged[f"patch_{key}"] = float(value)
    for key, value in graph_metrics.items():
        merged[f"graph_{key}"] = float(value)
    return merged


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    num_classes,
    vote_method,
    use_rotation_aug=False,
    rotation_weight=0.1,
):
    model.train()
    total_loss = 0.0
    total_class_loss = 0.0
    total_rot_loss = 0.0
    patch_true: List[int] = []
    patch_pred: List[int] = []
    graph_logits: Dict[str, Dict[str, object]] = {}

    correct = 0
    total = 0
    rotation_criterion = nn.CrossEntropyLoss() if model.use_rotation_head else None

    pbar = tqdm(loader, desc="Training")
    for batch_g, labels, graph_ids in pbar:
        batch_g = batch_g.to(device)
        labels = labels.view(-1).to(device)
        feat = batch_g.ndata["attr"]

        if use_rotation_aug and not model.use_rotation_head:
            rotated_graphs = []
            for graph in dgl.unbatch(batch_g):
                angle = int(np.random.choice([0, 90, 180, 270]))
                rotated_graphs.append(rotate_graph_z(graph, angle))
            batch_g = dgl.batch(rotated_graphs).to(device)
            feat = batch_g.ndata["attr"]

        optimizer.zero_grad()

        if model.use_rotation_head:
            expanded_graphs = []
            rotation_labels = []
            expanded_labels = []
            for label, graph in zip(labels, dgl.unbatch(batch_g)):
                for rotation_index, angle in enumerate([0, 90, 180, 270]):
                    expanded_graphs.append(rotate_graph_z(graph, angle))
                    rotation_labels.append(rotation_index)
                    expanded_labels.append(int(label.item()))

            batch_g = dgl.batch(expanded_graphs).to(device)
            feat = batch_g.ndata["attr"]
            rotation_labels = torch.tensor(rotation_labels, dtype=torch.long, device=device)
            expanded_labels = torch.tensor(expanded_labels, dtype=torch.long, device=device)

            class_logits, rotation_logits = model(batch_g, feat)
            class_loss = criterion(class_logits, expanded_labels)
            rot_loss = rotation_criterion(rotation_logits, rotation_labels)
            loss = class_loss + rotation_weight * rot_loss

            total_class_loss += float(class_loss.item())
            total_rot_loss += float(rot_loss.item())
            logits = class_logits.reshape(len(labels), 4, -1).mean(dim=1)
        else:
            logits = model(batch_g, feat)
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        preds = torch.argmax(logits, dim=1)
        probs_cpu = torch.softmax(logits.detach(), dim=1).cpu()
        logits_cpu = logits.detach().cpu()
        labels_cpu = labels.detach().cpu()

        patch_true.extend(labels_cpu.tolist())
        patch_pred.extend(preds.detach().cpu().tolist())

        for idx, graph_id in enumerate(graph_ids):
            entry = graph_logits.setdefault(graph_id, {"label": int(labels_cpu[idx].item()), "logits": []})
            entry["logits"].append(logits_cpu[idx])

        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))
        acc = correct / max(total, 1)

        if model.use_rotation_head:
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                cls=f"{class_loss.item():.4f}",
                rot=f"{rot_loss.item():.4f}",
                acc=f"{acc:.4f}",
            )
        else:
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

    metrics = merge_patch_and_graph_metrics(
        patch_true,
        patch_pred,
        graph_logits,
        num_classes=num_classes,
        vote_method=vote_method,
        loss=total_loss / max(len(loader), 1),
    )
    if model.use_rotation_head:
        metrics["rotation_class_loss"] = total_class_loss / max(len(loader), 1)
        metrics["rotation_aux_loss"] = total_rot_loss / max(len(loader), 1)
    return metrics


def evaluate(
    model,
    loader,
    criterion,
    device,
    desc,
    num_classes,
    vote_method,
    test_time_aug=False,
):
    model.eval()
    total_loss = 0.0
    patch_true: List[int] = []
    patch_pred: List[int] = []
    graph_logits: Dict[str, Dict[str, object]] = {}

    with torch.no_grad():
        for batch_g, labels, graph_ids in tqdm(loader, desc=desc):
            batch_g = batch_g.to(device)
            labels = labels.view(-1).to(device)
            feat = batch_g.ndata["attr"]

            if test_time_aug:
                batch_logits = []
                for graph in dgl.unbatch(batch_g):
                    rotated_logits = []
                    for angle in [0, 90, 180, 270]:
                        rotated_graph = rotate_graph_z(graph, angle).to(device)
                        rotated_feat = rotated_graph.ndata["attr"]
                        output = model(rotated_graph, rotated_feat)
                        class_logits = output[0] if isinstance(output, tuple) else output
                        rotated_logits.append(class_logits.squeeze(0))
                    batch_logits.append(torch.stack(rotated_logits).mean(dim=0))
                logits = torch.stack(batch_logits)
            else:
                output = model(batch_g, feat)
                logits = output[0] if isinstance(output, tuple) else output

            loss = criterion(logits, labels)
            total_loss += float(loss.item())

            preds = torch.argmax(logits, dim=1)
            logits_cpu = logits.detach().cpu()
            labels_cpu = labels.detach().cpu()

            patch_true.extend(labels_cpu.tolist())
            patch_pred.extend(preds.detach().cpu().tolist())

            for idx, graph_id in enumerate(graph_ids):
                entry = graph_logits.setdefault(graph_id, {"label": int(labels_cpu[idx].item()), "logits": []})
                entry["logits"].append(logits_cpu[idx])

    return merge_patch_and_graph_metrics(
        patch_true,
        patch_pred,
        graph_logits,
        num_classes=num_classes,
        vote_method=vote_method,
        loss=total_loss / max(len(loader), 1),
    )


def train_epoch_graph_objective(
    model,
    loader,
    optimizer,
    criterion,
    device,
    num_classes,
    vote_method,
    use_rotation_aug=False,
    graph_loss_weight=1.0,
    patch_loss_weight=0.0,
):
    model.train()
    total_loss = 0.0
    total_graph_loss = 0.0
    total_patch_loss = 0.0
    patch_true: List[int] = []
    patch_pred: List[int] = []
    graph_logits: Dict[str, Dict[str, object]] = {}
    graph_correct = 0
    graph_total = 0

    pbar = tqdm(loader, desc="Training")
    for batch_g, graph_labels, graph_ids, bag_sizes, patch_labels, patch_graph_ids in pbar:
        batch_g = batch_g.to(device)
        graph_labels = graph_labels.view(-1).to(device)
        patch_labels = patch_labels.view(-1).to(device)
        feat = batch_g.ndata["attr"]

        if use_rotation_aug:
            rotated_graphs = []
            for graph in dgl.unbatch(batch_g):
                angle = int(np.random.choice([0, 90, 180, 270]))
                rotated_graphs.append(rotate_graph_z(graph, angle))
            batch_g = dgl.batch(rotated_graphs).to(device)
            feat = batch_g.ndata["attr"]

        optimizer.zero_grad()
        patch_level_logits = model(batch_g, feat)
        if isinstance(patch_level_logits, tuple):
            patch_level_logits = patch_level_logits[0]
        graph_level_logits = aggregate_patch_logits_to_graph_logits(
            patch_level_logits,
            bag_sizes,
            vote_method=vote_method,
        )
        graph_loss = criterion(graph_level_logits, graph_labels)
        patch_loss = criterion(patch_level_logits, patch_labels)
        loss = graph_loss_weight * graph_loss + patch_loss_weight * patch_loss

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_graph_loss += float(graph_loss.item())
        total_patch_loss += float(patch_loss.item())
        graph_preds = torch.argmax(graph_level_logits, dim=1)
        graph_correct += int((graph_preds == graph_labels).sum().item())
        graph_total += int(graph_labels.size(0))

        patch_preds = torch.argmax(patch_level_logits, dim=1)
        patch_level_logits_cpu = patch_level_logits.detach().cpu()
        patch_labels_cpu = patch_labels.detach().cpu()
        patch_true.extend(patch_labels_cpu.tolist())
        patch_pred.extend(patch_preds.detach().cpu().tolist())

        for idx, graph_id in enumerate(patch_graph_ids):
            entry = graph_logits.setdefault(graph_id, {"label": int(patch_labels_cpu[idx].item()), "logits": []})
            entry["logits"].append(patch_level_logits_cpu[idx])

        graph_acc = graph_correct / max(graph_total, 1)
        postfix = {
            "loss": f"{loss.item():.4f}",
            "graph": f"{graph_loss.item():.4f}",
            "graph_acc": f"{graph_acc:.4f}",
        }
        if patch_loss_weight > 0.0:
            postfix["patch"] = f"{patch_loss.item():.4f}"
        pbar.set_postfix(**postfix)

    metrics = merge_patch_and_graph_metrics(
        patch_true,
        patch_pred,
        graph_logits,
        num_classes=num_classes,
        vote_method=vote_method,
        loss=total_loss / max(len(loader), 1),
    )
    metrics["graph_loss_component"] = total_graph_loss / max(len(loader), 1)
    metrics["patch_loss_component"] = total_patch_loss / max(len(loader), 1)
    return metrics


def evaluate_graph_objective(
    model,
    loader,
    criterion,
    device,
    desc,
    num_classes,
    vote_method,
    test_time_aug=False,
    graph_loss_weight=1.0,
    patch_loss_weight=0.0,
):
    model.eval()
    total_loss = 0.0
    total_graph_loss = 0.0
    total_patch_loss = 0.0
    patch_true: List[int] = []
    patch_pred: List[int] = []
    graph_logits: Dict[str, Dict[str, object]] = {}

    with torch.no_grad():
        for batch_g, graph_labels, graph_ids, bag_sizes, patch_labels, patch_graph_ids in tqdm(loader, desc=desc):
            batch_g = batch_g.to(device)
            graph_labels = graph_labels.view(-1).to(device)
            patch_labels = patch_labels.view(-1).to(device)
            feat = batch_g.ndata["attr"]

            if test_time_aug:
                patch_level_outputs = []
                for graph in dgl.unbatch(batch_g):
                    rotated_logits = []
                    for angle in [0, 90, 180, 270]:
                        rotated_graph = rotate_graph_z(graph, angle).to(device)
                        rotated_feat = rotated_graph.ndata["attr"]
                        output = model(rotated_graph, rotated_feat)
                        class_logits = output[0] if isinstance(output, tuple) else output
                        rotated_logits.append(class_logits.squeeze(0))
                    patch_level_outputs.append(torch.stack(rotated_logits).mean(dim=0))
                patch_level_logits = torch.stack(patch_level_outputs)
            else:
                output = model(batch_g, feat)
                patch_level_logits = output[0] if isinstance(output, tuple) else output

            graph_level_logits = aggregate_patch_logits_to_graph_logits(
                patch_level_logits,
                bag_sizes,
                vote_method=vote_method,
            )
            graph_loss = criterion(graph_level_logits, graph_labels)
            patch_loss = criterion(patch_level_logits, patch_labels)
            loss = graph_loss_weight * graph_loss + patch_loss_weight * patch_loss
            total_loss += float(loss.item())
            total_graph_loss += float(graph_loss.item())
            total_patch_loss += float(patch_loss.item())

            patch_preds = torch.argmax(patch_level_logits, dim=1)
            patch_level_logits_cpu = patch_level_logits.detach().cpu()
            patch_labels_cpu = patch_labels.detach().cpu()
            patch_true.extend(patch_labels_cpu.tolist())
            patch_pred.extend(patch_preds.detach().cpu().tolist())

            for idx, graph_id in enumerate(patch_graph_ids):
                entry = graph_logits.setdefault(graph_id, {"label": int(patch_labels_cpu[idx].item()), "logits": []})
                entry["logits"].append(patch_level_logits_cpu[idx])

    metrics = merge_patch_and_graph_metrics(
        patch_true,
        patch_pred,
        graph_logits,
        num_classes=num_classes,
        vote_method=vote_method,
        loss=total_loss / max(len(loader), 1),
    )
    metrics["graph_loss_component"] = total_graph_loss / max(len(loader), 1)
    metrics["patch_loss_component"] = total_patch_loss / max(len(loader), 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch-level GNN classification with graph voting")

    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.getenv("SHREC25_PATCH_DATA_DIR", "datasets/SHREC25/ssl_graph_patchs_1k"),
        help="Patch dataset root containing train_set/test_set folders",
    )
    parser.add_argument(
        "--patch-subdir",
        type=str,
        default="patches_k8",
        help="Patch folder inside each split, e.g. patches_k8, patches_k16, patches_k32",
    )
    parser.add_argument(
        "--train-graphs-csv",
        type=str,
        default=None,
        help="Optional CSV listing train graph ids via a protein_id column",
    )
    parser.add_argument(
        "--val-graphs-csv",
        type=str,
        default=None,
        help="Optional CSV listing validation graph ids via a protein_id column",
    )
    parser.add_argument(
        "--test-graphs-csv",
        type=str,
        default=None,
        help="Optional CSV listing test graph ids via a protein_id column",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument(
        "--train-objective",
        type=str,
        default="patch",
        choices=["patch", "graph", "mixed"],
        help="Optimize patch labels, graph labels from aggregated patch logits, or a weighted combination of both",
    )
    parser.add_argument(
        "--graph-loss-weight",
        type=float,
        default=1.0,
        help="Graph-loss contribution when --train-objective mixed",
    )
    parser.add_argument(
        "--patch-loss-weight",
        type=float,
        default=1.0,
        help="Patch-loss contribution when --train-objective mixed",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gat",
        choices=["gat", "gcn", "resgcn", "gin", "sage", "egnn", "gvp"],
        help="Backbone architecture",
    )
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads (GAT)")
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "max", "sum"],
        help="Pooling method",
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument(
        "--drop-features",
        type=int,
        nargs="*",
        default=None,
        help="Feature indices to drop",
    )
    parser.add_argument(
        "--keep-features",
        type=int,
        nargs="*",
        default=None,
        help="Feature indices to keep",
    )
    parser.add_argument("--skip-xyz", action="store_true", help="Shortcut for dropping features 0,1,2")

    parser.add_argument("--rotation-aug", action="store_true", help="Apply random Z-rotation to patches during training")
    parser.add_argument("--rotation-ssl", action="store_true", help="Predict patch class and rotation jointly")
    parser.add_argument("--rotation-weight", type=float, default=0.1, help="Rotation loss weight")
    parser.add_argument("--test-time-aug", action="store_true", help="Average predictions over 4 Z-rotations")

    parser.add_argument("--epochs", type=int, default=None, help="Max epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping")
    parser.add_argument("--device", type=int, default=0, help="GPU device")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio on training graphs")
    parser.add_argument(
        "--loss-type",
        type=str,
        default="ce",
        choices=["ce", "focal"],
        help="Classification loss type",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--balanced-sampler", action="store_true", help="Use class-balanced patch sampling")
    parser.add_argument("--sampler-power", type=float, default=1.0, help="Sampler weight exponent")
    parser.add_argument("--topk-classes", type=int, default=None, help="Train/evaluate only top-k train classes")
    parser.add_argument(
        "--vote-method",
        type=str,
        default="majority",
        choices=["majority", "prob_mean", "logit_mean", "confidence_logit_mean", "confidence_prob_mean"],
        help="How to turn patch predictions into graph predictions",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-interval", type=int, default=10, help="Save progress every N epochs")
    parser.add_argument("--run-name", type=str, default=None, help="Run name")
    parser.add_argument("--output-dir", type=str, default="runs", help="Output directory")

    args = parser.parse_args()

    if args.epochs is None:
        args.epochs = 500 if not args.no_early_stopping else 200

    if args.drop_features is not None and args.keep_features is not None:
        raise ValueError("Cannot use both --drop-features and --keep-features")
    if not 0.0 <= args.val_split < 1.0:
        raise ValueError(f"--val-split must be in range [0.0, 1.0), got {args.val_split}")
    if args.sampler_power < 0.0:
        raise ValueError(f"--sampler-power must be >= 0, got {args.sampler_power}")
    if args.graph_loss_weight < 0.0:
        raise ValueError(f"--graph-loss-weight must be >= 0, got {args.graph_loss_weight}")
    if args.patch_loss_weight < 0.0:
        raise ValueError(f"--patch-loss-weight must be >= 0, got {args.patch_loss_weight}")
    if args.train_objective == "mixed" and args.graph_loss_weight == 0.0 and args.patch_loss_weight == 0.0:
        raise ValueError("--train-objective mixed requires at least one positive loss weight")
    if args.train_objective in {"graph", "mixed"} and args.rotation_ssl:
        raise ValueError("--rotation-ssl is not supported with --train-objective graph or mixed")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"patch_{args.patch_subdir}_{args.model}_h{args.hidden_dim}_l{args.num_layers}_{timestamp}"
    else:
        run_name = args.run_name

    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    print(f"Run directory: {run_dir}")
    print(f"Using device: {device}")
    print(f"\n{'=' * 70}")
    print("SHREC25 PATCH GNN CLASSIFICATION")
    print(f"{'=' * 70}")
    print(f"Run name:      {run_name}")
    print(f"Data dir:      {args.data_dir}")
    print(f"Patch folder:  {args.patch_subdir}")
    print(f"Model:         {args.model.upper()}")
    print(f"Train obj:     {args.train_objective}")
    if args.train_objective == "mixed":
        print(f"Loss weights:  graph={args.graph_loss_weight}, patch={args.patch_loss_weight}")
    if args.train_objective in {"graph", "mixed"} and args.vote_method == "majority":
        print("Warning: --vote-method majority is non-differentiable; graph loss will use logit_mean aggregation during training.")
    print(f"Vote method:   {args.vote_method}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Hidden dim:    {args.hidden_dim}")
    print(f"Num layers:    {args.num_layers}")
    if args.model == "gat":
        print(f"Num heads:     {args.num_heads}")
    print(f"Pooling:       {args.pooling}")
    print(f"Dropout:       {args.dropout}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs:        {args.epochs}")
    print(f"Val split:     {args.val_split:.1%}")
    print(f"Loss type:     {args.loss_type}")
    print(f"Sampler:       {'balanced' if args.balanced_sampler else 'standard shuffle'}")
    print(f"{'=' * 70}\n")

    if args.skip_xyz:
        print("Feature mode:  Skipping XYZ coordinates (features 0-2)")
    elif args.drop_features is not None:
        print(f"Feature mode:  Dropping features {args.drop_features}")
    elif args.keep_features is not None:
        print(f"Feature mode:  Keeping only features {args.keep_features}")
    else:
        print("Feature mode:  Using all features")
    print()

    train_split_dir = resolve_patch_split_dir(args.data_dir, "train", args.patch_subdir)
    test_split_dir = resolve_patch_split_dir(args.data_dir, "test", args.patch_subdir)
    print(f"Resolved train patches: {train_split_dir}")
    print(f"Resolved test patches:  {test_split_dir}")

    print("\nScanning patch files...")
    all_train_records = collect_patch_records(train_split_dir)
    test_records = collect_patch_records(test_split_dir)

    if args.test_graphs_csv is not None:
        test_graph_ids = load_graph_ids_from_csv(args.test_graphs_csv)
        validate_requested_graph_ids(
            test_graph_ids,
            build_graph_label_dict(test_records).keys(),
            split_name="test",
            csv_path=args.test_graphs_csv,
        )
        test_records = split_records_by_graph(test_records, test_graph_ids)

    use_external_graph_splits = args.train_graphs_csv is not None or args.val_graphs_csv is not None
    if use_external_graph_splits:
        all_train_graph_ids = sorted(build_graph_label_dict(all_train_records).keys())
        val_graph_ids = load_graph_ids_from_csv(args.val_graphs_csv) if args.val_graphs_csv is not None else []

        if args.train_graphs_csv is not None:
            train_graph_ids = load_graph_ids_from_csv(args.train_graphs_csv)
        else:
            val_graph_id_set = set(val_graph_ids)
            train_graph_ids = [graph_id for graph_id in all_train_graph_ids if graph_id not in val_graph_id_set]

        if args.train_graphs_csv is not None:
            validate_requested_graph_ids(
                train_graph_ids,
                all_train_graph_ids,
                split_name="train",
                csv_path=args.train_graphs_csv,
            )
        if args.val_graphs_csv is not None:
            validate_requested_graph_ids(
                val_graph_ids,
                all_train_graph_ids,
                split_name="validation",
                csv_path=args.val_graphs_csv,
            )

        overlap = set(train_graph_ids) & set(val_graph_ids)
        if overlap:
            raise ValueError(
                f"External train/val graph splits overlap on {len(overlap)} graph ids; they must be disjoint"
            )

        train_records = split_records_by_graph(all_train_records, train_graph_ids)
        val_records = split_records_by_graph(all_train_records, val_graph_ids)
        print("\nUsing external graph split CSVs")
    else:
        all_train_graph_labels = build_graph_label_dict(all_train_records)
        train_graph_ids, val_graph_ids = create_robust_graph_train_val_split(
            all_train_graph_labels,
            val_split=args.val_split,
            min_graphs_for_stratify=2,
            seed=args.seed,
        )
        train_records = split_records_by_graph(all_train_records, train_graph_ids)
        val_records = split_records_by_graph(all_train_records, val_graph_ids)
        if args.val_split == 0.0:
            val_records = []

    train_graph_labels = build_graph_label_dict(train_records)
    val_graph_labels = build_graph_label_dict(val_records)
    test_graph_labels = build_graph_label_dict(test_records)

    train_graph_labels, test_graph_labels, class_map = select_topk_populated_classes(
        train_graph_labels,
        test_graph_labels,
        args.topk_classes,
    )
    if class_map is not None:
        train_records = remap_records_with_class_map(train_records, class_map)
        val_records = remap_records_with_class_map(val_records, class_map)
        test_records = remap_records_with_class_map(test_records, class_map)
        train_graph_labels = build_graph_label_dict(train_records)
        val_graph_labels = build_graph_label_dict(val_records)
        test_graph_labels = build_graph_label_dict(test_records)

    if not train_graph_labels:
        raise ValueError("No training graphs left after filtering. Check split CSVs, --patch-subdir, and --topk-classes.")
    if not test_records:
        raise ValueError("No test patches left after filtering. Check --patch-subdir, --test-graphs-csv, and --topk-classes.")

    if args.topk_classes is not None:
        num_classes = args.topk_classes
    else:
        all_labels = (
            list(train_graph_labels.values())
            + list(val_graph_labels.values())
            + list(test_graph_labels.values())
        )
        num_classes = max(all_labels) + 1

    print("\nDataset summary:")
    summarize_records("Train", train_records)
    if val_records:
        summarize_records("Val", val_records)
    else:
        print("Val patches:   0 (no validation split)")
    summarize_records("Test", test_records)
    print(f"Num classes:   {num_classes}")

    if args.train_objective in {"graph", "mixed"}:
        train_dataset = GraphPatchBagDataset(train_records)
        val_dataset = GraphPatchBagDataset(val_records) if val_records else None
        test_dataset = GraphPatchBagDataset(test_records)
    else:
        train_dataset = PatchDataset(train_records)
        val_dataset = PatchDataset(val_records) if val_records else None
        test_dataset = PatchDataset(test_records)

    sample_graph = prepare_patch_graph(load_graph_file(train_records[0].path))
    in_dim = int(sample_graph.ndata["attr"].shape[1])
    feature_indices: Optional[List[int]] = None
    if args.skip_xyz:
        feature_indices = list(range(3, in_dim))
        print(f"Input dim:     {in_dim} -> {len(feature_indices)} (skipping XYZ)")
    elif args.drop_features is not None:
        feature_indices = [idx for idx in range(in_dim) if idx not in args.drop_features]
        print(f"Input dim:     {in_dim} -> {len(feature_indices)} (dropped {args.drop_features})")
    elif args.keep_features is not None:
        feature_indices = list(args.keep_features)
        print(f"Input dim:     {in_dim} -> {len(feature_indices)} (keeping {args.keep_features})")
    else:
        print(f"Input dim:     {in_dim} (using all features)")

    if feature_indices is not None:
        if not all(0 <= idx < in_dim for idx in feature_indices):
            raise ValueError(f"Feature indices must be in range [0, {in_dim - 1}]")
        if len(feature_indices) == 0:
            raise ValueError("Must keep at least one feature")

    sampler = None
    shuffle_train = True
    if args.train_objective in {"graph", "mixed"}:
        train_sample_labels = list(build_graph_label_dict(train_records).values())
    else:
        train_sample_labels = [record.label for record in train_records]

    if args.balanced_sampler:
        class_counts = np.bincount(train_sample_labels, minlength=num_classes).astype(np.float64)
        sample_weights = [1.0 / (max(class_counts[label], 1.0) ** args.sampler_power) for label in train_sample_labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle_train = False
        print(f"Using balanced sampler (power={args.sampler_power}, num_samples={len(sample_weights)})")

    use_cuda = torch.cuda.is_available()
    collate_fn = collate_graph_patch_bag_fn if args.train_objective in {"graph", "mixed"} else collate_patch_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_train,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=use_cuda,
            persistent_workers=args.num_workers > 0,
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
    )

    print("\nInitializing model...")
    model = GNNClassifier(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.num_layers,
        model_type=args.model,
        num_heads=args.num_heads,
        pooling=args.pooling,
        dropout=args.dropout,
        feature_indices=feature_indices,
        use_rotation_head=args.rotation_ssl,
    ).to(device)
    print(f"Model parameters: {sum(param.numel() for param in model.parameters()):,}")

    if args.rotation_aug or args.rotation_ssl or args.test_time_aug:
        print(f"\n{'=' * 70}")
        print("ROTATION AUGMENTATION CONFIG")
        print(f"{'=' * 70}")
        if args.rotation_aug:
            print("✓ Random rotation augmentation enabled (training)")
        if args.rotation_ssl:
            print(f"✓ Multi-task learning: rotation prediction (weight={args.rotation_weight})")
        if args.test_time_aug:
            print("✓ Test-time augmentation: 4-rotation ensemble")
        print(f"{'=' * 70}\n")

    print("\nComputing class weights...")
    if args.train_objective in {"graph", "mixed"}:
        train_label_dict = build_graph_label_dict(train_records)
    else:
        train_label_dict = {f"{record.graph_id}::patch{record.patch_id}": record.label for record in train_records}
    class_weights = compute_class_weights(
        train_label_dict,
        num_classes,
        device,
        method="inverse_sqrt",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )
    if args.loss_type == "ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted CrossEntropyLoss with class weights")
    else:
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
        print(f"Using FocalLoss (gamma={args.focal_gamma}) with class weights")

    print(f"\n{'=' * 70}")
    print("TRAINING")
    print(f"{'=' * 70}\n")

    best_val_graph_f1 = 0.0
    best_train_graph_f1 = -1.0
    epochs_without_improvement = 0
    history = {"train": [], "val": [] if val_loader is not None else None}

    if args.no_early_stopping:
        print(f"Early stopping: DISABLED (will run all {args.epochs} epochs)")
    else:
        print(f"Early stopping: ENABLED (patience={args.patience} epochs)")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 70)

        if args.train_objective in {"graph", "mixed"}:
            train_metrics = train_epoch_graph_objective(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                num_classes=num_classes,
                vote_method=args.vote_method,
                use_rotation_aug=args.rotation_aug,
                graph_loss_weight=1.0 if args.train_objective == "graph" else args.graph_loss_weight,
                patch_loss_weight=0.0 if args.train_objective == "graph" else args.patch_loss_weight,
            )
        else:
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                num_classes=num_classes,
                vote_method=args.vote_method,
                use_rotation_aug=args.rotation_aug,
                rotation_weight=args.rotation_weight,
            )
        scheduler.step()
        history["train"].append(train_metrics)

        if val_loader is not None:
            if args.train_objective in {"graph", "mixed"}:
                val_metrics = evaluate_graph_objective(
                    model,
                    val_loader,
                    criterion,
                    device,
                    desc="Validating",
                    num_classes=num_classes,
                    vote_method=args.vote_method,
                    test_time_aug=args.test_time_aug,
                    graph_loss_weight=1.0 if args.train_objective == "graph" else args.graph_loss_weight,
                    patch_loss_weight=0.0 if args.train_objective == "graph" else args.patch_loss_weight,
                )
            else:
                val_metrics = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device,
                    desc="Validating",
                    num_classes=num_classes,
                    vote_method=args.vote_method,
                    test_time_aug=args.test_time_aug,
                )
            history["val"].append(val_metrics)
        else:
            val_metrics = None

        print("\nEpoch Summary:")
        if args.train_objective == "graph":
            print(
                f"  Train graph - Loss: {train_metrics['graph_loss_component']:.4f} | Acc: {train_metrics['graph_accuracy']:.4f} | "
                f"Bal Acc: {train_metrics['graph_balanced_accuracy']:.4f} | F1: {train_metrics['graph_f1_macro']:.4f}"
            )
        elif args.train_objective == "patch":
            print(
                f"  Train patch - Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['patch_accuracy']:.4f} | "
                f"Bal Acc: {train_metrics['patch_balanced_accuracy']:.4f} | F1: {train_metrics['patch_f1_macro']:.4f}"
            )
            print(
                f"  Train graph - Acc: {train_metrics['graph_accuracy']:.4f} | "
                f"Bal Acc: {train_metrics['graph_balanced_accuracy']:.4f} | F1: {train_metrics['graph_f1_macro']:.4f}"
            )
        else:
            print(f"  Train total - Loss: {train_metrics['loss']:.4f}")
            print(
                f"  Train graph - Acc: {train_metrics['graph_accuracy']:.4f} | "
                f"Bal Acc: {train_metrics['graph_balanced_accuracy']:.4f} | F1: {train_metrics['graph_f1_macro']:.4f}"
            )
            if args.patch_loss_weight > 0.0:
                print(
                    f"  Train patch - Acc: {train_metrics['patch_accuracy']:.4f} | "
                    f"Bal Acc: {train_metrics['patch_balanced_accuracy']:.4f} | F1: {train_metrics['patch_f1_macro']:.4f}"
                )
            print(
                f"  Train loss components - graph: {train_metrics['graph_loss_component']:.4f} x {args.graph_loss_weight:.3f} | "
                f"patch: {train_metrics['patch_loss_component']:.4f} x {args.patch_loss_weight:.3f}"
            )
        if val_metrics is not None:
            if args.train_objective == "graph":
                print(
                    f"  Val graph   - Loss: {val_metrics['graph_loss_component']:.4f} | Acc: {val_metrics['graph_accuracy']:.4f} | "
                    f"Bal Acc: {val_metrics['graph_balanced_accuracy']:.4f} | F1: {val_metrics['graph_f1_macro']:.4f}"
                )
            elif args.train_objective == "patch":
                print(
                    f"  Val patch   - Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['patch_accuracy']:.4f} | "
                    f"Bal Acc: {val_metrics['patch_balanced_accuracy']:.4f} | F1: {val_metrics['patch_f1_macro']:.4f}"
                )
                print(
                    f"  Val graph   - Acc: {val_metrics['graph_accuracy']:.4f} | "
                    f"Bal Acc: {val_metrics['graph_balanced_accuracy']:.4f} | F1: {val_metrics['graph_f1_macro']:.4f}"
                )
            else:
                print(f"  Val total   - Loss: {val_metrics['loss']:.4f}")
                print(
                    f"  Val graph   - Acc: {val_metrics['graph_accuracy']:.4f} | "
                    f"Bal Acc: {val_metrics['graph_balanced_accuracy']:.4f} | F1: {val_metrics['graph_f1_macro']:.4f}"
                )
                if args.patch_loss_weight > 0.0:
                    print(
                        f"  Val patch   - Acc: {val_metrics['patch_accuracy']:.4f} | "
                        f"Bal Acc: {val_metrics['patch_balanced_accuracy']:.4f} | F1: {val_metrics['patch_f1_macro']:.4f}"
                    )
                print(
                    f"  Val loss components   - graph: {val_metrics['graph_loss_component']:.4f} x {args.graph_loss_weight:.3f} | "
                    f"patch: {val_metrics['patch_loss_component']:.4f} x {args.patch_loss_weight:.3f}"
                )
            if not args.no_early_stopping and epochs_without_improvement > 0:
                print(f"  No graph-F1 improvement for {epochs_without_improvement}/{args.patience} epochs")

        if val_metrics is not None:
            current_graph_f1 = val_metrics["graph_f1_macro"]
            if current_graph_f1 > best_val_graph_f1:
                best_val_graph_f1 = current_graph_f1
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "train_metrics": train_metrics,
                    },
                    run_dir / "best_model.pth",
                )
                print(f"  ✓ New best val graph F1: {best_val_graph_f1:.4f}")
            else:
                epochs_without_improvement += 1
                if not args.no_early_stopping and epochs_without_improvement >= args.patience:
                    print(f"\n  Early stopping triggered at epoch {epoch + 1}")
                    print(f"  Best validation graph F1: {best_val_graph_f1:.4f}")
                    break
        else:
            current_graph_f1 = train_metrics["graph_f1_macro"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                },
                run_dir / "last_model.pth",
            )
            if current_graph_f1 > best_train_graph_f1:
                best_train_graph_f1 = current_graph_f1
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_metrics": train_metrics,
                    },
                    run_dir / "best_model.pth",
                )
                print(f"  ✓ New best train graph F1: {best_train_graph_f1:.4f}")
            else:
                print(
                    f"  ✓ Epoch {epoch + 1} saved to last_model.pth "
                    f"(graph F1: {current_graph_f1:.4f}, best: {best_train_graph_f1:.4f})"
                )

        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = run_dir / f"checkpoint_epoch{epoch + 1}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                checkpoint_path,
            )
            with open(run_dir / "training_history.json", "w") as handle:
                json.dump(history, handle, indent=2)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")

    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")

    best_checkpoint = torch.load(run_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    if args.train_objective in {"graph", "mixed"}:
        final_test_metrics = evaluate_graph_objective(
            model,
            test_loader,
            criterion,
            device,
            desc="Final Test",
            num_classes=num_classes,
            vote_method=args.vote_method,
            test_time_aug=args.test_time_aug,
            graph_loss_weight=1.0 if args.train_objective == "graph" else args.graph_loss_weight,
            patch_loss_weight=0.0 if args.train_objective == "graph" else args.patch_loss_weight,
        )
    else:
        final_test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            desc="Final Test",
            num_classes=num_classes,
            vote_method=args.vote_method,
            test_time_aug=args.test_time_aug,
        )

    if "val_metrics" in best_checkpoint:
        print(f"\nBest Validation Graph Metrics (epoch {best_checkpoint['epoch'] + 1}):")
        print(f"  Accuracy:          {best_checkpoint['val_metrics']['graph_accuracy']:.4f}")
        print(f"  Balanced Accuracy: {best_checkpoint['val_metrics']['graph_balanced_accuracy']:.4f}")
        print(f"  F1 (macro):        {best_checkpoint['val_metrics']['graph_f1_macro']:.4f}")
        print(f"  F1 (weighted):     {best_checkpoint['val_metrics']['graph_f1_weighted']:.4f}")
        print(f"  Precision:         {best_checkpoint['val_metrics']['graph_precision_macro']:.4f}")
        print(f"  Recall:            {best_checkpoint['val_metrics']['graph_recall_macro']:.4f}")
        print("\nFinal Test Graph Metrics (from best val model):")
    else:
        print(f"\nBest Training Graph Metrics (epoch {best_checkpoint['epoch'] + 1}):")
        print(f"  Accuracy:          {best_checkpoint['train_metrics']['graph_accuracy']:.4f}")
        print(f"  Balanced Accuracy: {best_checkpoint['train_metrics']['graph_balanced_accuracy']:.4f}")
        print(f"  F1 (macro):        {best_checkpoint['train_metrics']['graph_f1_macro']:.4f}")
        print(f"  F1 (weighted):     {best_checkpoint['train_metrics']['graph_f1_weighted']:.4f}")
        print(f"  Precision:         {best_checkpoint['train_metrics']['graph_precision_macro']:.4f}")
        print(f"  Recall:            {best_checkpoint['train_metrics']['graph_recall_macro']:.4f}")
        print("\nFinal Test Graph Metrics (from best train graph-F1 model):")

    print(f"  Accuracy:          {final_test_metrics['graph_accuracy']:.4f}")
    print(f"  Balanced Accuracy: {final_test_metrics['graph_balanced_accuracy']:.4f}")
    print(f"  F1 (macro):        {final_test_metrics['graph_f1_macro']:.4f}")
    print(f"  F1 (weighted):     {final_test_metrics['graph_f1_weighted']:.4f}")
    print(f"  Precision:         {final_test_metrics['graph_precision_macro']:.4f}")
    print(f"  Recall:            {final_test_metrics['graph_recall_macro']:.4f}")

    if args.train_objective != "graph" and args.patch_loss_weight > 0.0:
        print("\nFinal Test Patch Metrics:")
        print(f"  Accuracy:          {final_test_metrics['patch_accuracy']:.4f}")
        print(f"  Balanced Accuracy: {final_test_metrics['patch_balanced_accuracy']:.4f}")
        print(f"  F1 (macro):        {final_test_metrics['patch_f1_macro']:.4f}")
        print(f"  F1 (weighted):     {final_test_metrics['patch_f1_weighted']:.4f}")
        print(f"  Precision:         {final_test_metrics['patch_precision_macro']:.4f}")
        print(f"  Recall:            {final_test_metrics['patch_recall_macro']:.4f}")
    else:
        print("\nPatch-level metrics omitted because patch loss weight is 0 in graph-objective training.")

    with open(run_dir / "training_history.json", "w") as handle:
        json.dump(history, handle, indent=2)

    with open(run_dir / "final_results.json", "w") as handle:
        results = {
            "final_test_metrics": final_test_metrics,
            "config": vars(args),
        }
        if "val_metrics" in best_checkpoint:
            results["best_val_metrics"] = best_checkpoint["val_metrics"]
        else:
            results["best_train_metrics"] = best_checkpoint["train_metrics"]
        json.dump(results, handle, indent=2)

    print(f"\n✓ All outputs saved to: {run_dir}")
    print("  - training_history.json")
    print("  - final_results.json")
    print("  - best_model.pth", end="")
    if val_loader is None:
        print(" (best by train graph F1)")
        print("  - last_model.pth (last epoch)")
    else:
        print(" (best by val graph F1)")
    print("  - checkpoint_epoch*.pth")


if __name__ == "__main__":
    main()