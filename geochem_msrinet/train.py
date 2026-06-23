#!/usr/bin/env python3
"""Train GeoChem-MSRINet on SHREC25 point tensors.

Example:
    python -m geochem_msrinet.train --epochs 2 --batch-size 4 --limit-train 128

The implementation is intentionally pure PyTorch. It starts from the existing
1k FPS `.pt` graph tensors, so the first practical target is the 1024/1000
point version of the architecture.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from geochem_msrinet.data import BatchAugment, ProteinPointDataset, class_counts, make_collate
from geochem_msrinet.losses import GeoChemLoss, PrototypeMemory, inverse_sqrt_class_weights
from geochem_msrinet.metrics import classification_metrics
from geochem_msrinet.model import GeoChemMSRINet
from geochem_msrinet.ri_hybrid_model import RIConv2GeoChemHybrid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-dir", default="/home/ogharbi/datasets/SHREC25/ssl_graphs_1k_geom_enriched_ri")
    p.add_argument("--train-labels-csv", default=None)
    p.add_argument("--val-labels-csv", default=None)
    p.add_argument("--test-labels-csv", default=None)
    p.add_argument("--output-dir", default="/home/ogharbi/projects/protein_classification/runs")
    p.add_argument("--run-name", default=None)

    p.add_argument("--points", type=int, default=1000)
    p.add_argument("--model-kind", choices=["geochem", "ri_hybrid"], default="geochem")
    p.add_argument("--ri-width-mult", type=int, default=1, help="RIConv2 set-abstraction width multiplier.")
    p.add_argument("--ri-token-dim", type=int, default=512, help="Projected RIConv2 token dimension for patch graph head.")
    p.add_argument("--use-geom-raw", action="store_true")
    p.add_argument("--patch-knn", type=int, default=8)
    p.add_argument("--patch-gin-layers", type=int, default=2)
    p.add_argument("--patch-transformer-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--logit-blend", type=float, default=0.65)

    p.add_argument("--phase", choices=["geom", "chem", "curriculum"], default="curriculum")
    p.add_argument("--geom-epochs", type=int, default=40)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.10)
    p.add_argument("--class-weight-method", choices=["inverse_sqrt", "none"], default="inverse_sqrt")
    p.add_argument("--cosine-weight", type=float, default=0.30)
    p.add_argument("--supcon-weight", type=float, default=0.0)
    p.add_argument("--proto-weight", type=float, default=0.0)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--balanced-sampler", action="store_true", help="Sample training examples with inverse class frequency.")
    p.add_argument("--sampler-power", type=float, default=1.0, help="Class-frequency exponent for --balanced-sampler.")
    p.add_argument(
        "--samples-per-epoch",
        type=int,
        default=0,
        help="Number of sampled training examples per epoch when using --balanced-sampler; default is train size.",
    )

    p.add_argument("--rotate", action="store_true", default=True)
    p.add_argument("--no-rotate", dest="rotate", action="store_false")
    p.add_argument("--jitter", type=float, default=0.0)
    p.add_argument("--point-dropout", type=float, default=0.0)
    p.add_argument("--normal-noise", type=float, default=0.0)
    p.add_argument("--scalar-noise", type=float, default=0.0)

    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--limit-val", type=int, default=0)
    p.add_argument("--limit-test", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limit_dataset(ds, limit: int):
    if limit and limit > 0:
        return Subset(ds, list(range(min(limit, len(ds)))))
    return ds


def dataset_labels(ds) -> list[int]:
    if isinstance(ds, Subset):
        base = ds.dataset.ordered_labels()
        return [base[i] for i in ds.indices]
    return ds.ordered_labels()


def make_balanced_sampler(labels: list[int], samples_per_epoch: int = 0, power: float = 1.0) -> WeightedRandomSampler:
    counts = class_counts(labels, 99).float()
    power = max(0.0, float(power))
    weights = torch.tensor([1.0 / counts[int(y)].clamp_min(1.0).pow(power) for y in labels], dtype=torch.float32)
    n_samples = int(samples_per_epoch) if samples_per_epoch and samples_per_epoch > 0 else len(labels)
    return WeightedRandomSampler(weights, num_samples=n_samples, replacement=True)


def chem_enabled(args: argparse.Namespace, epoch: int) -> bool:
    if args.phase == "geom":
        return False
    if args.phase == "chem":
        return True
    return epoch >= args.geom_epochs


def make_model(args: argparse.Namespace):
    if args.model_kind == "ri_hybrid":
        return RIConv2GeoChemHybrid(
            num_classes=99,
            chem_dim=2,
            ri_width_mult=args.ri_width_mult,
            ri_token_dim=args.ri_token_dim,
            patch_knn=args.patch_knn,
            patch_gin_layers=args.patch_gin_layers,
            patch_transformer_layers=args.patch_transformer_layers,
            dropout=args.dropout,
            logit_blend=args.logit_blend,
        )
    return GeoChemMSRINet(
        num_classes=99,
        points=args.points,
        chem_dim=2,
        geom_dim=6 if args.use_geom_raw else 0,
        patch_knn=args.patch_knn,
        patch_gin_layers=args.patch_gin_layers,
        patch_transformer_layers=args.patch_transformer_layers,
        dropout=args.dropout,
        logit_blend=args.logit_blend,
    )


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    use_chem: bool = True,
    proto: PrototypeMemory | None = None,
    grad_accum: int = 1,
    clip_grad: float = 1.0,
) -> dict:
    train = optimizer is not None
    model.train(train)
    losses = []
    parts: dict[str, list[float]] = {"ce": [], "cosine": [], "supcon": [], "proto": []}
    preds = []
    labels_all = []
    optimizer_steps = 0

    if train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        pos = batch["pos"].to(device, non_blocking=True)
        normal = batch["normal"].to(device, non_blocking=True)
        chem = batch["chem"].to(device, non_blocking=True)
        geom = batch["geom"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            use_amp = scaler is not None and device.type == "cuda"
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(pos, normal, chem, geom=geom if geom.shape[-1] else None, use_chem=use_chem)
                loss, loss_parts = criterion(outputs, labels, proto=proto)
                loss_for_backward = loss / max(1, grad_accum)

            if train:
                if scaler is not None and device.type == "cuda":
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

                should_step = (step + 1) % grad_accum == 0 or (step + 1) == len(loader)
                if should_step:
                    if scaler is not None and device.type == "cuda":
                        scaler.unscale_(optimizer)
                    if clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    if scaler is not None and device.type == "cuda":
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                if proto is not None:
                    proto.update(outputs["embed"], labels)

        losses.append(float(loss.detach().cpu()))
        for k, v in loss_parts.items():
            parts[k].append(v)
        preds.extend(outputs["logits"].argmax(dim=1).detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())

    metrics = classification_metrics(labels_all, preds)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    for k, values in parts.items():
        metrics[f"loss_{k}"] = float(np.mean(values)) if values else 0.0
    metrics["optimizer_steps"] = optimizer_steps
    return metrics


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    run_name = args.run_name or f"geochem_msrinet_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ProteinPointDataset(args.graph_dir, "train", args.train_labels_csv, args.points, args.use_geom_raw)
    val_ds = ProteinPointDataset(args.graph_dir, "val", args.val_labels_csv, args.points, args.use_geom_raw)
    test_ds = ProteinPointDataset(args.graph_dir, "test", args.test_labels_csv, args.points, args.use_geom_raw)
    train_ds = limit_dataset(train_ds, args.limit_train)
    val_ds = limit_dataset(val_ds, args.limit_val)
    test_ds = limit_dataset(test_ds, args.limit_test)

    train_aug = BatchAugment(
        rotate=args.rotate,
        jitter=args.jitter,
        point_dropout=args.point_dropout,
        normal_noise=args.normal_noise,
        scalar_noise=args.scalar_noise,
    )
    train_label_list = dataset_labels(train_ds)
    train_sampler = (
        make_balanced_sampler(train_label_list, args.samples_per_epoch, args.sampler_power)
        if args.balanced_sampler
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=make_collate(train_aug),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate(BatchAugment(rotate=False)),
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate(BatchAugment(rotate=False)),
        pin_memory=device.type == "cuda",
    )

    model = make_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    weights = (
        inverse_sqrt_class_weights(train_label_list, 99, device=device)
        if args.class_weight_method == "inverse_sqrt"
        else None
    )
    criterion = GeoChemLoss(
        class_weights=weights,
        label_smoothing=args.label_smoothing,
        cosine_weight=args.cosine_weight,
        supcon_weight=args.supcon_weight,
        proto_weight=args.proto_weight,
    )
    proto = PrototypeMemory(99, min(512, 512), device=device) if args.proto_weight > 0 else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch: int) -> float:
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(1, args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    config = vars(args).copy()
    config.update(
        {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "device_resolved": str(device),
            "n_params": n_params,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "test_size": len(test_ds),
            "class_counts_train": class_counts(train_label_list, 99).tolist(),
            "balanced_sampler_active": bool(args.balanced_sampler),
            "sampler_power": args.sampler_power,
            "effective_train_samples_per_epoch": len(train_sampler) if train_sampler is not None else len(train_ds),
        }
    )
    save_json(run_dir / "config.json", config)
    print(f"run={run_name} device={device} params={n_params:,}")
    sampler_msg = (
        f" balanced_sampler power={args.sampler_power:g} samples_per_epoch={len(train_sampler)}"
        if train_sampler is not None
        else ""
    )
    print(
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"batch={args.batch_size} class_weights={args.class_weight_method}{sampler_msg}"
    )

    best_f1 = -1.0
    best_epoch = -1
    best_use_chem = args.phase != "geom"
    history = []
    bad_epochs = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        use_chem = chem_enabled(args, epoch)
        tr = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_chem=use_chem,
            proto=proto,
            grad_accum=args.grad_accum,
            clip_grad=args.clip_grad,
        )
        va = run_epoch(model, val_loader, criterion, device, use_chem=use_chem, proto=proto)
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "use_chem": use_chem,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - t0,
            "train": tr,
            "val": va,
        }
        history.append(row)
        save_json(run_dir / "history.json", {"history": history})

        print(
            f"ep {epoch+1:03d} chem={int(use_chem)} "
            f"train f1={tr['f1_macro']:.4f} val f1={va['f1_macro']:.4f} "
            f"val acc={va['accuracy']:.4f} loss={va['loss']:.4f} lr={row['lr']:.2e}"
        )

        if va["f1_macro"] > best_f1:
            best_f1 = va["f1_macro"]
            best_epoch = epoch + 1
            best_use_chem = use_chem
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": best_epoch,
                    "best_val_f1": best_f1,
                    "use_chem": best_use_chem,
                },
                run_dir / "best_model.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping at epoch {epoch+1}")
                break

    ckpt = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    final_use_chem = bool(ckpt.get("use_chem", best_use_chem))
    final_val = run_epoch(model, val_loader, criterion, device, use_chem=final_use_chem, proto=proto)
    final_test = run_epoch(model, test_loader, criterion, device, use_chem=final_use_chem, proto=proto)
    results = {
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "use_chem": final_use_chem,
        "final_val": final_val,
        "final_test": final_test,
        "config": config,
    }
    save_json(run_dir / "final_results.json", results)
    print(f"best_epoch={best_epoch} best_val_f1={best_f1:.4f}")
    print(f"test macro_f1={final_test['f1_macro']:.4f} acc={final_test['accuracy']:.4f}")


if __name__ == "__main__":
    main()
