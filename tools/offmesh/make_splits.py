#!/usr/bin/env python3
"""Write the train / val / test label files for the OFF-mesh datasets (SHREC2011, TOSCA).

Both datasets have no official classification split, so the splits are fixed here, committed
under data_labels/, and every run reads them (graphs live in one folder, see prepare_off_hierarchy.py).

  SHREC2011 non-rigid (600 meshes, 30 classes x 20): the usual 'split10' protocol of the mesh
  classification literature (MeshCNN, DiffusionNet): 10 train / 10 test meshes per class, drawn
  at random. Three draws (seeds 1-3). Of the 10 training meshes per class, 2 are held out for
  validation (early stopping), so the model trains on 8.
  TOSCA (80 meshes, 9 classes of 3 to 20): stratified 4-fold cross-validation (fold = seed 1-4);
  from the 3 training folds, 1 mesh per class is held out for validation.

  python tools/offmesh/make_splits.py --shrec11-cla <test.cla or categories.txt> --tosca-meshes <dir of *.off>
"""
import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOSCA_NAMES = ["cat", "centaur", "david", "dog", "gorilla", "horse", "michael", "victoria", "wolf"]


def shrec11_classes(cla):
    lines = [l.strip() for l in Path(cla).read_text(encoding="utf-8", errors="replace").splitlines()]
    out, names, i, c = {}, [], 2, 0
    while i < len(lines) and c < 30:
        if not lines[i]:
            i += 1; continue
        name, _, n = lines[i].split(); n = int(n); i += 1; names.append(name); got = 0
        while got < n:
            if lines[i]:
                out[f"T{int(lines[i])}"] = c; got += 1
            i += 1
        c += 1
    assert len(out) == 600 and c == 30, (len(out), c)
    return out, names


def write(path, rows, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["protein_id", "class_id", "class_name"])
        for gid, c in sorted(rows, key=lambda r: (r[1], int(re.sub(r"\D", "", r[0]) or 0), r[0])):
            w.writerow([gid, c, names[c]])


def by_class(cls_of):
    g = defaultdict(list)
    for gid, c in cls_of.items():
        g[c].append(gid)
    return {c: sorted(v) for c, v in sorted(g.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shrec11-cla", required=True)
    ap.add_argument("--tosca-meshes", required=True)
    a = ap.parse_args()

    cls, names = shrec11_classes(a.shrec11_cla)
    for seed in (1, 2, 3):
        rng = random.Random(seed); tr, va, te = [], [], []
        for c, ids in by_class(cls).items():
            ids = ids[:]; rng.shuffle(ids)
            te += [(g, c) for g in ids[10:]]; va += [(g, c) for g in ids[8:10]]; tr += [(g, c) for g in ids[:8]]
        d = REPO / "data_labels" / "shrec11" / f"split10_seed{seed}"
        write(d / "train.csv", tr, names); write(d / "val.csv", va, names); write(d / "test.csv", te, names)
        print(f"shrec11 split10 seed {seed}: train {len(tr)} val {len(va)} test {len(te)}")

    tos = {}
    for p in sorted(Path(a.tosca_meshes).glob("*.off")):
        tos[p.stem] = TOSCA_NAMES.index(re.match(r"[A-Za-z]+", p.stem).group(0).lower())
    assert len(tos) == 80, len(tos)
    rng = random.Random(0); fold_of = {}
    for c, ids in by_class(tos).items():
        ids = ids[:]; rng.shuffle(ids)
        for i, g in enumerate(ids):
            fold_of[g] = i % 4
    for f in range(4):
        te = [(g, c) for g, c in tos.items() if fold_of[g] == f]
        rest = by_class({g: c for g, c in tos.items() if fold_of[g] != f})
        tr, va = [], []
        rng_v = random.Random(100 + f)
        for c, ids in rest.items():
            ids = ids[:]; rng_v.shuffle(ids)
            va += [(ids[0], c)]; tr += [(g, c) for g in ids[1:]]
        assert len({c for _, c in tr}) == 9, f"fold {f + 1}: a class has no training mesh"
        d = REPO / "data_labels" / "tosca" / f"cv4_fold{f + 1}"
        write(d / "train.csv", tr, TOSCA_NAMES); write(d / "val.csv", va, TOSCA_NAMES); write(d / "test.csv", te, TOSCA_NAMES)
        print(f"tosca fold {f + 1}: train {len(tr)} val {len(va)} test {len(te)} "
              f"(test classes {len({c for _, c in te})})")


if __name__ == "__main__":
    main()
