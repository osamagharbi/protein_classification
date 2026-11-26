#!/usr/bin/env python3
"""Write SHREC2025 label CSVs in the id format of ssl_graphs_1k_geom_enriched_ri.

Produces (committed under data_labels/, so Jean-Zay needs no upload):
  shrec25_final/{train_split,val_split,test_labels}.csv
      FINAL release (organisers, Sept 2026): 98 classes (0..97; former class 73
      removed, former 74..98 shifted by -1), 9,219 train / 2,305 test. The meshes of
      that release are randomly rotated; our graphs are the unrotated originals of
      the same surfaces, which is equivalent for an exactly rotation-invariant model
      (use --test-rotations to measure non-invariant models under rotation).
      The train/val membership of our existing split is kept.
  shrec25_update_281025/test_labels.csv
      Labels of the organisers' 28 Oct 2025 update (2,308 test surfaces), for
      re-scoring runs that were evaluated on the original 2,311-surface labels.

Usage (paths of the local copies used to build the committed files):
  python tools/make_shrec25_label_sets.py \
      --ours /home/ogharbi/datasets/SHREC25/ssl_graphs_1k_geom_enriched_ri \
      --final-labels /mnt/d/SHREC25_final_rot/labels \
      --update-test /home/ogharbi/datasets/SHREC25/pc/test_labels.csv \
      --out data_labels
"""
import argparse
import csv
from collections import Counter
from pathlib import Path


def read(path, key="protein_id"):
    return {r[key]: r for r in csv.DictReader(open(path))}


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protein_id", "class_id"])
        w.writerows(rows)
    print(f"  {path}: {len(rows)} rows, {len({c for _, c in rows})} classes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--final-labels", required=True, help="dir with train_labels.csv / test_labels.csv")
    ap.add_argument("--update-test", required=True)
    ap.add_argument("--out", default="data_labels")
    a = ap.parse_args()
    ours, out = Path(a.ours), Path(a.out)
    norm = lambda s: s.replace(":", "_").replace("_centered", "")   # FINAL train ids -> our ids

    fin_tr = {norm(k): int(r["class_id"]) for k, r in read(Path(a.final_labels) / "train_labels.csv").items()}
    fin_te = {k: int(r["class_id"]) for k, r in read(Path(a.final_labels) / "test_labels.csv").items()}
    for split in ("train_split", "val_split"):
        rows = [(k, fin_tr[k]) for k in read(ours / f"{split}.csv") if k in fin_tr]
        write(out / "shrec25_final" / f"{split}.csv", rows)
    test_rows = [(k, fin_te[k]) for k in read(ours / "test_labels.csv") if k in fin_te]
    write(out / "shrec25_final" / "test_labels.csv", test_rows)
    n_classes = len({c for c in fin_tr.values()})
    missing = sorted(set(range(max(fin_tr.values()) + 1)) - set(fin_tr.values()))
    print(f"  FINAL train classes: {n_classes} (missing ids: {missing or 'none'}); "
          f"test classes present: {len(Counter(c for _, c in test_rows))}")

    upd = [(r["protein_id"].replace("_pc", ""), int(r["class_id"])) for r in csv.DictReader(open(a.update_test))]
    write(out / "shrec25_update_281025" / "test_labels.csv", upd)


if __name__ == "__main__":
    main()
