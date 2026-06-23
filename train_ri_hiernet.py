#!/usr/bin/env python3
"""
Training driver for RI-HierNet (ri_hiernet.py) on SHREC25.

Reads pos / normals / scalars_norm directly from the existing 1k graph .pt
files into dense [B,1000,*] point-cloud batches — no new precomputed dataset.

Supports the two-phase curriculum from the design doc:
  phase=geom        geometry only (chem zeroed) the whole run
  phase=chem        chemistry enabled the whole run (a.k.a. joint)
  phase=curriculum  geometry-only for --geom-epochs, then enable chemistry

Eval uses the official 88-class basis (macro-F1 over the classes present in the
test set), directly comparable to the report table. See RI_HIERNET_DESIGN.md.

Example:
  conda run --no-capture-output -n graphmae python3 -u train_ri_hiernet.py \
      --graph-dir /home/ogharbi/datasets/SHREC25/ssl_graphs_1k_geom_enriched \
      --phase curriculum --geom-epochs 40 --epochs 120 --batch-size 64 \
      --run-name ri_hiernet_curriculum
"""
from __future__ import annotations
import sys, argparse, json, math, time, re
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, accuracy_score

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from classify_gnn import load_labels, canonical_graph_id, resolve_split_dir
from fix_training import compute_class_weights, create_robust_train_val_split
from ri_hiernet import RIHierNet


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PointCloudDataset(Dataset):
    """Returns (pos[N,3], normal[N,3], chem[N,2], label) per protein."""

    def __init__(self, graph_dir, split, label_dict, npoint=1000, clamp=5.0):
        self.gdir = resolve_split_dir(graph_dir, split)
        self.npoint, self.clamp = npoint, clamp
        avail = {canonical_graph_id(p.stem): p for p in self.gdir.glob("*.pt")}
        self.items, missing = [], 0
        for gid, lab in label_dict.items():
            p = avail.get(gid)
            if p is not None:
                self.items.append((gid, p, int(lab)))
            else:
                missing += 1
        print(f"  [{split}] {len(self.items)} clouds ({missing} missing)")

    def ordered_labels(self) -> dict:
        """label_dict in dataset order (so split indices align with the dataset)."""
        return {gid: lab for gid, _, lab in self.items}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        _, p, lab = self.items[idx]
        d = torch.load(p, map_location="cpu", weights_only=False)
        pos = d["pos"].float()
        normal = F.normalize(d["normals"].float(), dim=-1)
        chem = d["scalars_norm"].float().clamp(-self.clamp, self.clamp)
        n = pos.shape[0]
        if n >= self.npoint:
            pos, normal, chem = pos[:self.npoint], normal[:self.npoint], chem[:self.npoint]
        elif n < self.npoint:                              # pad small proteins
            r = torch.randint(0, n, (self.npoint - n,))
            pos = torch.cat([pos, pos[r]]); normal = torch.cat([normal, normal[r]])
            chem = torch.cat([chem, chem[r]])
        return pos, normal, chem, lab


def load_labels_csv(csv_path: str) -> dict:
    """protein_id -> class_id from one CSV, stripping the '_pc' suffix so ids
    match the graph stems. Using ONE source for train+test guarantees a single
    consistent class numbering (the `graphs/` CSV uses a different numbering — the
    bug that gave RI-HierNet random test scores)."""
    df = pd.read_csv(csv_path)
    out = {}
    for r in df.itertuples():
        pid = str(r.protein_id)
        if pid.endswith("_pc"):
            pid = pid[:-3]
        out[canonical_graph_id(pid)] = int(r.class_id)
    return out


def base_pdb_id(gid: str) -> str:
    """Strip chain/model variants -> PDB base id (e.g. 105m_1_A_A_model1 -> 105m)."""
    return re.split(r'[_:]', str(gid))[0]


def grouped_train_val_split(dataset, val_split=0.2, seed=42, min_groups_for_stratify=4):
    """Leakage-free split: whole PDB groups go entirely to train OR val, so no
    near-duplicate protein (same PDB, different chain/model) straddles the split.
    Rare classes (< min_groups_for_stratify groups) are kept train-only."""
    import numpy as np
    groups = defaultdict(list)
    for idx, (gid, _, lab) in enumerate(dataset.items):
        groups[base_pdb_id(gid)].append((idx, lab))
    gkeys = list(groups.keys())
    glabel = {g: Counter(l for _, l in groups[g]).most_common(1)[0][0] for g in gkeys}
    by_class = defaultdict(list)
    for g in gkeys:
        by_class[glabel[g]].append(g)

    rng = np.random.RandomState(seed)
    train_groups, val_groups = [], []
    for lab, gs in by_class.items():
        gs = list(gs); rng.shuffle(gs)
        if len(gs) < min_groups_for_stratify:
            train_groups += gs                                    # rare -> all train
        else:
            n_val = max(1, int(round(len(gs) * val_split)))
            val_groups += gs[:n_val]; train_groups += gs[n_val:]
    tr_idx = [i for g in train_groups for i, _ in groups[g]]
    va_idx = [i for g in val_groups for i, _ in groups[g]]
    # report leakage check
    trb = {base_pdb_id(dataset.items[i][0]) for i in tr_idx}
    vab = {base_pdb_id(dataset.items[i][0]) for i in va_idx}
    print(f"  grouped split: {len(tr_idx)} train / {len(va_idx)} val  "
          f"({len(trb)}/{len(vab)} base-IDs, overlap={len(trb & vab)})")
    return tr_idx, va_idx


def make_collate(jitter: float = 0.0, dropout: float = 0.0,
                 scale: float = 0.0, shift: float = 0.0, training: bool = True):
    """Collate with optional RIConv2-style augmentations (applied only during training).

    dropout: probability each point is zeroed (set pos/normal to 0)
    scale:   max relative scale perturbation, i.e. scale ~ Uniform(1-s, 1+s)
    shift:   max absolute xyz shift, i.e. t ~ Uniform(-s, s) per axis
    """
    def collate(batch):
        pos = torch.stack([b[0] for b in batch])      # [B,N,3]
        normal = torch.stack([b[1] for b in batch])
        chem = torch.stack([b[2] for b in batch])
        y = torch.tensor([b[3] for b in batch], dtype=torch.long)
        if jitter > 0:
            pos = pos + torch.randn_like(pos) * jitter
        if training:
            if dropout > 0:
                mask = torch.rand(pos.shape[0], pos.shape[1], 1) > dropout
                pos = pos * mask.float()
                normal = normal * mask.float()
            if scale > 0:
                s = 1.0 + (torch.rand(pos.shape[0], 1, 1) * 2 - 1) * scale
                pos = pos * s
            if shift > 0:
                t = (torch.rand(pos.shape[0], 1, 3) * 2 - 1) * shift
                pos = pos + t
        return pos, normal, chem, y
    return collate


# ─────────────────────────────────────────────────────────────────────────────
# Losses & metrics
# ─────────────────────────────────────────────────────────────────────────────

def supcon_loss(proj, labels, temp=0.1):
    """Supervised contrastive loss (Khosla et al.) on L2-normalised projections."""
    B = proj.shape[0]
    sim = (proj @ proj.T) / temp
    self_mask = ~torch.eye(B, dtype=torch.bool, device=proj.device)
    pos_mask = (labels.view(-1, 1) == labels.view(1, -1)) & self_mask
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
    exp = torch.exp(sim) * self_mask
    log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
    pos_count = pos_mask.sum(1)
    loss = -(pos_mask * log_prob).sum(1) / pos_count.clamp(min=1)
    loss = loss[pos_count > 0]
    return loss.mean() if loss.numel() > 0 else proj.new_zeros(())


def metrics(y_true, y_pred):
    """88-class basis (over classes present in y_true) + raw-99 macro-F1."""
    valid = sorted(set(int(c) for c in y_true))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": recall_score(y_true, y_pred, average="macro", zero_division=0, labels=valid),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0, labels=valid),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=valid),
        "f1_macro_99": f1_score(y_true, y_pred, average="macro", zero_division=0, labels=list(range(99))),
        "n_classes": len(valid),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, device, optimizer=None, use_chem=True,
              supcon_w=0.0, desc=""):
    train = optimizer is not None
    model.train() if train else model.eval()
    total, ce_tot, sc_tot, preds, labs = 0.0, 0.0, 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for pos, normal, chem, y in tqdm(loader, desc=desc, leave=False):
            pos, normal, chem, y = pos.to(device), normal.to(device), chem.to(device), y.to(device)
            logits, proj = model(pos, normal, chem, use_chem=use_chem)
            ce = criterion(logits, y)
            sc = supcon_loss(proj, y) if (supcon_w > 0) else proj.new_zeros(())
            loss = ce + supcon_w * sc
            if train:
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item(); ce_tot += ce.item(); sc_tot += float(sc)
            preds.extend(logits.argmax(1).cpu().numpy()); labs.extend(y.cpu().numpy())
    m = metrics(np.array(labs), np.array(preds))
    m["loss"] = total / max(1, len(loader)); m["ce"] = ce_tot / max(1, len(loader))
    m["supcon"] = sc_tot / max(1, len(loader))
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/ogharbi/datasets/SHREC25",
                    help="SHREC25 dataset root. Derives --graph-dir (<root>/ssl_graphs_1k_geom_enriched), "
                         "--train-labels-csv (<root>/pc/train_labels.csv), and "
                         "--test-labels-csv (<root>/pc/test_labels.csv) unless those are passed explicitly.")
    ap.add_argument("--graph-subdir", default="ssl_graphs_1k_geom_enriched",
                    help="graph folder name under --data-dir (used only when --graph-dir is not given)")
    ap.add_argument("--graph-dir", default=None,
                    help="explicit graph dir (overrides --data-dir/--graph-subdir)")
    ap.add_argument("--train-labels-csv", default=None,
                    help="explicit train labels CSV (overrides --data-dir)")
    ap.add_argument("--test-labels-csv", default=None,
                    help="explicit test labels CSV (overrides --data-dir)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--output-dir", default="/home/ogharbi/projects/protein_classification/runs")
    # curriculum
    ap.add_argument("--phase", choices=["geom", "chem", "curriculum"], default="curriculum")
    ap.add_argument("--geom-epochs", type=int, default=40, help="geometry-only epochs before enabling chem (curriculum)")
    # model
    ap.add_argument("--npoint", type=int, default=1000)
    ap.add_argument("--nsample", type=int, default=32)
    ap.add_argument("--patch-knn", type=int, default=8)
    ap.add_argument("--patch-layers", type=int, default=2)
    ap.add_argument("--patch-backbone", default="gin",
                    choices=["gin", "gat", "sage", "transformer", "graphormer"],
                    help="patch-stage GNN: gin (default) | gat (kNN attention) | "
                         "sage (GraphSAGE mean) | transformer (full self-attention) | "
                         "graphormer (full attention + invariant distance bias)")
    ap.add_argument("--readout", default="max+attn",
                    help="'+'-joined graph pooling over 64 patches; atoms: "
                         "max|mean|min|sum|attn. e.g. 'max', 'mean+max', 'max+attn' (default)")
    ap.add_argument("--radii", default="0.10,0.20,0.40,0.80")
    ap.add_argument("--dropout", type=float, default=0.3)
    # optim
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--class-weight-method", default="none",
                    help="'none' (best for accuracy/weighted-F1) or e.g. inverse_sqrt")
    ap.add_argument("--select-metric", default="accuracy",
                    choices=["accuracy", "f1_weighted", "f1_macro", "balanced_accuracy"],
                    help="val metric used for checkpoint selection / early stopping")
    ap.add_argument("--supcon-weight", type=float, default=0.0)
    ap.add_argument("--aug-jitter",   type=float, default=0.0)
    ap.add_argument("--aug-dropout", type=float, default=0.0,
                    help="point dropout probability (RIConv2 style)")
    ap.add_argument("--aug-scale",   type=float, default=0.0,
                    help="max relative scale perturbation, e.g. 0.2 → scale in [0.8,1.2]")
    ap.add_argument("--aug-shift",   type=float, default=0.0,
                    help="max absolute xyz shift per axis, e.g. 0.1")
    ap.add_argument("--width-scale", type=int, default=1,
                    help="SA encoder width multiplier: 1=current(512D out), 2=wider(1024D out)")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--grouped-val", action="store_true", default=True,
                    help="leakage-free train/val split by PDB id (default on)")
    ap.add_argument("--random-val", dest="grouped_val", action="store_false",
                    help="old leaky random split")
    ap.add_argument("--monitor-test", action="store_true", default=True,
                    help="log clean test metrics each epoch (diagnosis only)")
    args = ap.parse_args()

    # Derive dataset paths from --data-dir unless overridden explicitly.
    root = Path(args.data_dir)
    if args.graph_dir is None:
        args.graph_dir = str(root / args.graph_subdir)
    if args.train_labels_csv is None:
        args.train_labels_csv = str(root / "pc" / "train_labels.csv")
    if args.test_labels_csv is None:
        args.test_labels_csv = str(root / "pc" / "test_labels.csv")
    print(f"data-dir={args.data_dir}\n  graph-dir={args.graph_dir}\n"
          f"  train-csv={args.train_labels_csv}\n  test-csv={args.test_labels_csv}")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or f"ri_hiernet_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run={run_name}  device={device}  phase={args.phase}")

    # ── data ── (single consistent label source for train AND test)
    train_labels = load_labels_csv(args.train_labels_csv)
    test_labels = load_labels_csv(args.test_labels_csv)
    num_classes = 99
    print(f"labels: train {len(train_labels)} ({len(set(train_labels.values()))} cls) from {args.train_labels_csv}")
    print(f"        test  {len(test_labels)} ({len(set(test_labels.values()))} cls) from {args.test_labels_csv}")

    full_train = PointCloudDataset(args.graph_dir, "train", train_labels, npoint=args.npoint)
    test_ds = PointCloudDataset(args.graph_dir, "test", test_labels, npoint=args.npoint)
    if args.grouped_val:
        tr_idx, va_idx = grouped_train_val_split(full_train, val_split=args.val_split, seed=args.seed)
    else:
        tr_idx, va_idx = create_robust_train_val_split(full_train, full_train.ordered_labels(),
                                                       val_split=args.val_split, seed=args.seed)
    train_ds, val_ds = Subset(full_train, tr_idx), Subset(full_train, va_idx)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} classes={num_classes}")

    train_collate = make_collate(args.aug_jitter, args.aug_dropout, args.aug_scale,
                                 args.aug_shift, training=True)
    eval_collate  = make_collate(training=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=train_collate,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=eval_collate, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=eval_collate, pin_memory=True)

    # ── model / loss / optim ──
    radii = tuple(float(x) for x in args.radii.split(","))
    model = RIHierNet(num_classes=num_classes, chem_dim=2, patch_knn=args.patch_knn,
                      patch_layers=args.patch_layers, dropout=args.dropout,
                      radii=radii, nsample=args.nsample,
                      width_scale=args.width_scale,
                      patch_backbone=args.patch_backbone,
                      readout=args.readout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}  radii={radii}  width_scale={args.width_scale}  "
          f"patch_backbone={args.patch_backbone}  readout={args.readout}")

    # No class weighting by default: we optimise accuracy / weighted-F1, which are
    # support-weighted — up-weighting rare classes trades those away.
    weights = (None if args.class_weight_method in (None, "none")
               else compute_class_weights(full_train.ordered_labels(), num_classes, device,
                                           method=args.class_weight_method))
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    print(f"class weighting: {args.class_weight_method}  | select checkpoint by val '{args.select_metric}'")
    print(f"augmentation: jitter={args.aug_jitter} dropout={args.aug_dropout} "
          f"scale={args.aug_scale} shift={args.aug_shift}  label_smoothing={args.label_smoothing}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup, total = args.warmup_epochs, args.epochs
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / max(1, warmup)
        prog = (ep - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def chem_on(ep):
        if args.phase == "geom":  return False
        if args.phase == "chem":  return True
        return ep >= args.geom_epochs                      # curriculum

    # ── train loop ──
    best_sel, best_state, since_best, history = -1.0, None, 0, []
    for ep in range(args.epochs):
        uc = chem_on(ep)
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device, optimizer, use_chem=uc,
                       supcon_w=args.supcon_weight, desc=f"ep{ep} train")
        va = run_epoch(model, val_loader, criterion, device, None, use_chem=uc,
                       desc=f"ep{ep} val")
        te = (run_epoch(model, test_loader, criterion, device, None, use_chem=uc, desc=f"ep{ep} test")
              if args.monitor_test else None)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        te_str = (f" | TEST acc {te['accuracy']:.3f} wf1 {te['f1_weighted']:.3f} mf1 {te['f1_macro']:.3f}"
                  if te else "")
        print(f"Epoch {ep+1}/{args.epochs} lr={lr_now:.2e} chem={int(uc)} "
              f"({time.time()-t0:.0f}s) | Train acc {tr['accuracy']:.3f} | "
              f"Val acc {va['accuracy']:.3f} wf1 {va['f1_weighted']:.3f} mf1 {va['f1_macro']:.3f}{te_str}")
        mkeys = ("loss", "accuracy", "f1_weighted", "f1_macro", "balanced_accuracy")
        history.append({"epoch": ep, "lr": lr_now, "chem": int(uc),
                        "train": {k: tr[k] for k in ("loss", "accuracy", "f1_weighted")},
                        "val": {k: va[k] for k in mkeys},
                        "test": ({k: te[k] for k in mkeys[1:]} if te else None)})
        # select checkpoint by the metric we care about (default: accuracy)
        if va[args.select_metric] > best_sel:
            best_sel = va[args.select_metric]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, run_dir / "best_model.pt")
            since_best = 0
            print(f"  * new best val {args.select_metric} {best_sel:.4f}"
                  + (f"  (test acc {te['accuracy']:.4f} wf1 {te['f1_weighted']:.4f})" if te else ""))
        else:
            since_best += 1
            if since_best >= args.patience:
                print(f"  early stop (no val improvement for {args.patience} epochs)")
                break

    # ── test ──
    if best_state is not None:
        model.load_state_dict(best_state)
    test_m = run_epoch(model, test_loader, criterion, device, None, use_chem=True, desc="test")
    print(f"\nTEST ({test_m['n_classes']}-class): acc {test_m['accuracy']:.4f}  "
          f"wtd-F1 {test_m['f1_weighted']:.4f}  macro-F1 {test_m['f1_macro']:.4f}  "
          f"balacc {test_m['balanced_accuracy']:.4f}")

    (run_dir / "final_results.json").write_text(json.dumps({
        "final_test_metrics": test_m, "best_val_select": best_sel,
        "config": {**vars(args), "num_classes": num_classes, "params": n_params, "radii": list(radii)},
        "history": history,
    }, indent=2, default=float))
    print(f"saved -> {run_dir}/final_results.json")


if __name__ == "__main__":
    main()
