#!/usr/bin/env python3
"""Train the upstream WangYueFt DGCNN on SHREC25 protein point clouds.

This script intentionally imports the original DGCNN implementation from
dgcnn/pytorch/model.py and does not modify the model architecture. Since the
upstream classifier is a 3-channel point-cloud model, the default input signal
is the normalized xyz coordinate tensor stored as ``pos`` in our graph files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from dataset_split_guard import resolve_dataset_local_split_csvs


ROOT = Path(__file__).resolve().parent
DGCNN_PYTORCH = ROOT / "dgcnn" / "pytorch"
if str(DGCNN_PYTORCH) not in sys.path:
    sys.path.insert(0, str(DGCNN_PYTORCH))

from model import DGCNN  # noqa: E402  (upstream WangYueFt/dgcnn model)
from util import cal_loss  # noqa: E402


DEFAULT_DATA_DIR = "/home/ogharbi/datasets/SHREC25/ssl_graphs_1k_geom_enriched_ri"


def canonical_id(value) -> str:
    return str(value).replace("_graph", "").replace("_pc", "")


def resolve_split_dir(data_dir: Path, split: str) -> Path:
    candidates = [data_dir / f"{split}_set", data_dir / split]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {split} split under {data_dir}")


def load_label_map(data_dir: Path, split: str, labels_csv: str | None = None) -> dict[str, int]:
    split_dir = resolve_split_dir(data_dir, split)
    stems = {canonical_id(path.stem) for path in split_dir.glob("*.pt")}

    if labels_csv:
        candidates = [Path(labels_csv)]
    else:
        candidates = [
            data_dir / f"{split}_labels.csv",
            data_dir.parent / "graphs" / f"{split}_labels.csv",
            data_dir.parent / "graphs_old" / f"{split}_labels.csv",
            data_dir.parent / "pc" / f"{split}_labels.csv",
            data_dir.parent / "pc_8f" / f"{split}_labels.csv",
        ]

    best_path = None
    best_df = None
    best_overlap = -1
    for candidate in candidates:
        if not candidate.exists():
            continue
        df = pd.read_csv(candidate)
        if "protein_id" not in df.columns or "class_id" not in df.columns:
            continue
        ids = {canonical_id(value) for value in df["protein_id"]}
        overlap = len(ids & stems)
        if overlap > best_overlap:
            best_path = candidate
            best_df = df
            best_overlap = overlap

    if best_df is None or best_overlap == 0:
        raise FileNotFoundError(f"No label CSV for split={split} matched files in {split_dir}")

    print(f"Labels [{split}]: {best_path} overlap={best_overlap}/{len(stems)}")
    return {
        canonical_id(row.protein_id): int(row.class_id)
        for row in best_df.itertuples(index=False)
    }


class SHREC25DGCNNDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        label_map: dict[str, int],
        *,
        num_points: int,
        input_signal: str,
        augment: bool,
        limit: int | None = None,
    ) -> None:
        self.num_points = num_points
        self.input_signal = input_signal
        self.augment = augment
        self.samples: list[tuple[Path, int]] = []

        for stem, label in label_map.items():
            for name in (stem, f"{stem}_graph"):
                path = split_dir / f"{name}.pt"
                if path.exists():
                    self.samples.append((path, label))
                    break

        self.samples.sort(key=lambda item: str(item[0]))
        if limit is not None:
            self.samples = self.samples[:limit]

        if not self.samples:
            raise RuntimeError(f"No samples matched labels in {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _select_signal(self, data: dict) -> torch.Tensor:
        if self.input_signal == "pos":
            x = data["pos"]
        elif self.input_signal == "normals":
            x = data["normals"]
        elif self.input_signal == "xyz_raw":
            x = data["xyz_raw"]
            x = x - x.mean(dim=0, keepdim=True)
            scale = torch.linalg.norm(x, dim=-1).max().clamp_min(1e-8)
            x = x / scale
        elif self.input_signal == "x_ssl":
            feat = data["x"]
            x = feat[:, 5:8]
        else:
            raise ValueError(f"Unknown input signal: {self.input_signal}")
        return torch.as_tensor(x, dtype=torch.float32)

    def _sample_points(self, x: torch.Tensor) -> torch.Tensor:
        n_points = x.size(0)
        if n_points >= self.num_points:
            index = torch.randperm(n_points)[: self.num_points]
            return x[index]

        repeats = math.ceil(self.num_points / n_points)
        return x.repeat(repeats, 1)[: self.num_points]

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.empty(1).uniform_(0.8, 1.25)
        shift = torch.empty(1, 3).uniform_(-0.1, 0.1)
        x = x * scale + shift

        dropout = torch.empty(1).uniform_(0.0, 0.875).item()
        mask = torch.rand(x.size(0)) <= dropout
        if mask.any():
            x[mask] = x[0].clone()
        return x

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        data = torch.load(path, map_location="cpu", weights_only=False)
        x = self._sample_points(self._select_signal(data)).contiguous()
        if self.augment:
            x = self._augment(x)
        return x, torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    x = torch.stack([item[0] for item in batch], dim=0)
    y = torch.stack([item[1] for item in batch], dim=0)
    return x, y


def make_loaders(args):
    data_dir = Path(args.data_dir)
    train_csv, val_csv, messages = resolve_dataset_local_split_csvs(
        data_dir,
        args.train_labels_csv,
        args.val_labels_csv,
        allow_mismatch=args.allow_split_mismatch,
        train_flag="--train-labels-csv",
        val_flag="--val-labels-csv",
    )
    for message in messages:
        print(message)

    train_dir = resolve_split_dir(data_dir, "train")
    test_dir = resolve_split_dir(data_dir, "test")
    train_labels = load_label_map(data_dir, "train", train_csv)
    if val_csv is None:
        raise ValueError(
            "This launcher expects a fixed validation CSV. The active SHREC25 dataset "
            "provides train_split.csv/val_split.csv automatically."
        )
    val_labels = load_label_map(data_dir, "train", val_csv)
    test_labels = load_label_map(data_dir, "test", args.test_labels_csv)

    num_classes = max(
        max(train_labels.values()),
        max(val_labels.values()),
        max(test_labels.values()),
    ) + 1

    train_ds = SHREC25DGCNNDataset(
        train_dir,
        train_labels,
        num_points=args.num_points,
        input_signal=args.input_signal,
        augment=not args.no_augment,
        limit=args.limit_train,
    )
    val_ds = SHREC25DGCNNDataset(
        train_dir,
        val_labels,
        num_points=args.num_points,
        input_signal=args.input_signal,
        augment=False,
        limit=args.limit_val,
    )
    test_ds = SHREC25DGCNNDataset(
        test_dir,
        test_labels,
        num_points=args.num_points,
        input_signal=args.input_signal,
        augment=False,
        limit=args.limit_test,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader, num_classes


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, *, use_amp: bool):
    model.eval()
    total_loss = 0.0
    total = 0
    preds = []
    labels = []
    for x, y in loader:
        x = x.to(device, non_blocking=True).permute(0, 2, 1).contiguous()
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += float(loss.item()) * y.numel()
        total += y.numel()
        preds.append(logits.argmax(dim=1).detach().cpu())
        labels.append(y.detach().cpu())

    y_true = torch.cat(labels).numpy()
    y_pred = torch.cat(preds).numpy()
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(total, 1)
    return metrics


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The upstream DGCNN get_graph_feature implementation constructs CUDA indices. "
            "Run this adapter with a CUDA device to keep model.py unchanged."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    train_loader, val_loader, test_loader, num_classes = make_loaders(args)

    model_args = SimpleNamespace(k=args.k, emb_dims=args.emb_dims, dropout=args.dropout)
    model = DGCNN(model_args, output_channels=num_classes).to(device)
    n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)

    criterion = cal_loss if args.label_smoothing else nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    run_name = args.run_name or f"upstream_dgcnn_shrec25_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update({"num_classes": num_classes, "params": n_params, "upstream_model": str(DGCNN_PYTORCH / "model.py")})
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"run={run_name} device={device} params={n_params:,}")
    print(
        f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)} "
        f"classes={num_classes} batch={args.batch_size} input={args.input_signal}"
    )
    print(f"savedir={run_dir}")

    best_metric = -1.0
    best_epoch = 0
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        total = 0
        train_preds = []
        train_labels = []
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True).permute(0, 2, 1).contiguous()
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                logits = model(x)
                loss = criterion(logits, y) / args.grad_accum
            scaler.scale(loss).backward()

            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_loss = float(loss.item()) * args.grad_accum
            total_loss += batch_loss * y.numel()
            total += y.numel()
            train_preds.append(logits.argmax(dim=1).detach().cpu())
            train_labels.append(y.detach().cpu())

        scheduler.step()

        train_true = torch.cat(train_labels).numpy()
        train_pred = torch.cat(train_preds).numpy()
        train_metrics = compute_metrics(train_true, train_pred)
        train_metrics["loss"] = total_loss / max(total, 1)
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp=args.amp)

        current = val_metrics[args.monitor]
        improved = current > best_metric
        if improved:
            best_metric = current
            best_epoch = epoch
            no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, run_dir / "best_model.pt")
        else:
            no_improve += 1

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - t0,
            "train": train_metrics,
            "val": val_metrics,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
        }
        history.append(row)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        print(
            f"ep {epoch:03d} train acc={train_metrics['accuracy']:.4f} "
            f"wf1={train_metrics['f1_weighted']:.4f} mf1={train_metrics['f1_macro']:.4f} "
            f"val acc={val_metrics['accuracy']:.4f} wf1={val_metrics['f1_weighted']:.4f} "
            f"mf1={val_metrics['f1_macro']:.4f} bal={val_metrics['balanced_accuracy']:.4f} "
            f"loss={val_metrics['loss']:.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if args.patience > 0 and no_improve >= args.patience:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch} best_{args.monitor}={best_metric:.4f}")
            break

    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, criterion, device, use_amp=args.amp)
    final = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        f"best_val_{args.monitor}": best_metric,
        "val_at_best": history[best_epoch - 1]["val"],
        "test": test_metrics,
    }
    (run_dir / "final_results.json").write_text(json.dumps(final, indent=2))
    torch.save({"model": model.state_dict(), "epoch": best_epoch, "config": config, "test": test_metrics}, run_dir / "final_model.pt")
    print(json.dumps(final, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Train upstream WangYueFt DGCNN on SHREC25")
    parser.add_argument("--data-dir", default=os.getenv("SHREC25_DGCNN_DATA_DIR", DEFAULT_DATA_DIR))
    parser.add_argument("--train-labels-csv", default=None)
    parser.add_argument("--val-labels-csv", default=None)
    parser.add_argument("--test-labels-csv", default=None)
    parser.add_argument("--allow-split-mismatch", action="store_true")
    parser.add_argument("--input-signal", choices=["pos", "normals", "xyz_raw", "x_ssl"], default="pos")
    parser.add_argument("--num-points", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--emb-dims", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["step", "cosine"], default="step")
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--eta-min", type=float, default=1e-5)
    parser.add_argument("--monitor", choices=["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"], default="f1_weighted")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--label-smoothing", action="store_true", help="Use upstream cal_loss with smoothing.")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--output-dir", default="runs_pc")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
