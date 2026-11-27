#!/usr/bin/env python
"""Fingerprint a sample of SHREC'25 graphs (and patch files) so a rebuild elsewhere can be checked.

  make    on the machine that holds the reference graphs (writes a small JSON, committed to git)
  verify  on the rebuilt graphs: exits 1 if any sampled graph differs

Per sampled graph:
  exact     : node count, sha256 of edge_index_mesh (which vertices FPS kept and how they connect),
              sha256 of the k-patch assignment (patch_ids, patch_edge_index) when --patches is given
  tolerance : per-column mean / min / max of every float tensor (x, pos, normals, geom_features_*,
              ri_feat, ri_lrf_feat, ...), compared with |a - b| <= atol + rtol * |b|

  python tools/shrec25_rebuild/fingerprint.py make   --graphs .../ssl_graphs_1k_geom_enriched_ri \
         --patches .../ssl_graphs_1k_patchassign_k64 --out tools/shrec25_rebuild/fingerprints_paper.json
  python tools/shrec25_rebuild/fingerprint.py verify --graphs $DATA_ROOT/ssl_graphs_1k_geom_enriched_ri \
         [--patches $DATA_ROOT/ssl_graphs_1k_patchassign_k64] --ref tools/shrec25_rebuild/fingerprints_paper.json

--rotated: graphs built from the FINAL release's rotated meshes. These are rotated copies of the paper
graphs, but float32 near-ties in FPS swap ~1% of the node order, so exact hashes cannot match. Checked
instead: node count, tensor shapes (edge count), and per-column mean/min/max of every rotation- and
order-invariant feature (potentials, the 13 geometric descriptors, the invariant columns of x); xyz,
normals and the per-edge RI features (direction-dependent) are shape-checked only; patches: count only.
Sampled graphs missing from the build (surfaces dropped by the FINAL release) are skipped.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _sha(t):
    return hashlib.sha256(np.ascontiguousarray(t.cpu().numpy().astype(np.int64)).tobytes()).hexdigest()


def sample_ids(graph_dir, n_train, n_test):
    out = {}
    for split, n in (("train_set", n_train), ("test_set", n_test)):
        stems = sorted(p.stem for p in (Path(graph_dir) / split).glob("*.pt"))
        step = max(1, len(stems) // n)
        out[split] = stems[::step][:n]
    return out


def graph_fp(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    fp = {"keys": sorted(k for k in d if k != "meta"), "num_nodes": int(d["pos"].shape[0]),
          "edge_index_mesh": _sha(d["edge_index_mesh"]), "edge_index": _sha(d["edge_index"]), "float": {}}
    for k, v in d.items():
        if torch.is_tensor(v) and v.is_floating_point():
            a = v.double().reshape(v.shape[0], -1).numpy()
            fp["float"][k] = {"shape": list(v.shape), "mean": a.mean(0).tolist(),
                              "min": a.min(0).tolist(), "max": a.max(0).tolist()}
    return fp


def patch_fp(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return {"num_patches": int(d["num_patches"]), "patch_ids": _sha(d["patch_ids"]),
            "patch_edge_index": _sha(d["patch_edge_index"])}


def make(a):
    ids = sample_ids(a.graphs, a.n_train, a.n_test)
    ref = {"graphs": {}, "patches": {}, "patch_dir": Path(a.patches).name if a.patches else None}
    for split, stems in ids.items():
        for s in stems:
            ref["graphs"][f"{split}/{s}"] = graph_fp(Path(a.graphs) / split / f"{s}.pt")
            if a.patches:
                ref["patches"][f"{split}/{s}"] = patch_fp(Path(a.patches) / split / f"{s}.pt")
    Path(a.out).write_text(json.dumps(ref, separators=(",", ":")))
    print(f"wrote {len(ref['graphs'])} graph fingerprints -> {a.out}")


def close(a, b, atol, rtol):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(np.all(np.abs(a - b) <= atol + rtol * np.abs(b)))


ROTATING = {"pos", "xyz_raw", "normals", "ri_feat", "ri_lrf_feat"}   # --rotated: shape check only
X_INVARIANT = [0, 1] + list(range(8, 21))   # x: potentials + 13 geometric descriptors (2-7 = normals, xyz)


def verify(a):
    ref = json.loads(Path(a.ref).read_text())
    bad, worst, skipped = [], 0.0, 0
    for key, r in ref["graphs"].items():
        p = Path(a.graphs) / f"{key}.pt"
        if not p.is_file():
            if a.present_only or a.rotated:
                skipped += 1; continue
            bad.append(f"{key}: missing"); continue
        g = graph_fp(p)
        if g["keys"] != r["keys"]:
            bad.append(f"{key}: keys {g['keys']} != {r['keys']}"); continue
        if g["num_nodes"] != r["num_nodes"] or (not a.rotated and g["edge_index_mesh"] != r["edge_index_mesh"]):
            bad.append(f"{key}: different sampled vertices / mesh edges"); continue
        for k, rv in r["float"].items():
            gv = g["float"].get(k)
            if gv is None or gv["shape"] != rv["shape"]:
                bad.append(f"{key}: {k} shape {gv and gv['shape']} != {rv['shape']}"); continue
            if a.rotated and k in ROTATING:
                continue
            for stat in ("mean", "min", "max"):
                gs, rs = gv[stat], rv[stat]
                if a.rotated and k == "x":
                    gs, rs = [gs[i] for i in X_INVARIANT], [rs[i] for i in X_INVARIANT]
                d = float(np.max(np.abs(np.asarray(gs) - np.asarray(rs)))) if rs else 0.0
                worst = max(worst, d)
                if not close(gs, rs, a.atol, a.rtol):
                    bad.append(f"{key}: {k}.{stat} differs (max abs diff {d:.3g})"); break
        if a.patches and key in ref["patches"]:
            pp = Path(a.patches) / f"{key}.pt"
            if not pp.is_file():
                bad.append(f"{key}: patch file missing"); continue
            pf, rf = patch_fp(pp), ref["patches"][key]
            if (pf["num_patches"] != rf["num_patches"]) if a.rotated else (pf != rf):
                bad.append(f"{key}: patch assignment differs")
    n = len(ref["graphs"]) - skipped
    if n == 0:
        sys.exit("no sampled graph is present")
    print(f"checked {n} graphs{' + patch files' if a.patches else ''}: {n - len({b.split(':')[0] for b in bad})} match, "
          f"{len({b.split(':')[0] for b in bad})} differ; largest float difference {worst:.3g}")
    for b in bad[:20]:
        print("  DIFF", b)
    if bad:
        sys.exit(1)
    if skipped:
        print(f"  ({skipped} sampled graphs not in this build, skipped)")
    print("VERIFIED: " + ("rotated copies of the reference graphs (invariant features identical)"
                          if a.rotated else "identical to the reference graphs"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["make", "verify"])
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--patches", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--present-only", action="store_true", help="skip sampled graphs that are not built (smoke tests)")
    ap.add_argument("--rotated", action="store_true", help="graphs from the FINAL release's rotated meshes")
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-4)
    a = ap.parse_args()
    make(a) if a.mode == "make" else verify(a)


if __name__ == "__main__":
    main()
