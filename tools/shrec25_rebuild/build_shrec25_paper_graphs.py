#!/usr/bin/env python
"""Rebuild the paper's SHREC'25 graphs (ssl_graphs_1k_geom_enriched_ri) from the organisers' ORIGINAL
(unrotated) archives train_set.tar.xz / test_set.tar.xz, with the original functions unchanged
(vendored copies of projects/utils in this folder):

  1. vtk_mesh_to_ssl_graph_parallel.make_graph_from_vtk(scalar_stats=train_scalar_stats.json)
       full mesh graph; xyz centred + unit ball; Potential / NormalPotential z-scored with the
       paper's training-set statistics (stored, not recomputed)
  2. sample_ssl_graphs.process_one(n_points=1000, ratio=None, knn_k=6)
       FPS from the vertex nearest the centroid (deterministic); mesh edges between kept vertices
  3. enrich_ssl_graph_features.process_one(position_key="pos", num_anchors=4, normalize_mode="graph_zscore")
       13 geometric descriptors -> x is 21-D
  4. build_ri_feat_dataset.compute_ri_features            -> ri_feat      [E_mesh, 5]
     add_lrf_features.compute_lrf + compute_ri_lrf_features -> ri_lrf_feat [E_mesh, 8]

Output (same layout as the paper folder; the split CSVs come from data_labels/shrec25_paper):
  OUT/ssl_graphs_1k_geom_enriched_ri/{train_set,test_set}/<id>.pt   train: name with ':' -> '_'; test: anon id
  OUT/ssl_graphs_1k_geom_enriched_ri/{train_split,val_split,test_labels}.csv, foundation_*.json, build_manifest.json

--release rot: the FINAL release instead (organisers, Sept 2026): ROTATED meshes train_set_rot.tar.xz /
test_set_rot.tar.xz (members X_model1_rot.vtk / <anon>_rot.vtk), FINAL labels (data_labels/shrec25_final:
98 classes, 9,219 train / 2,305 test, our train/val membership), same pipeline and the same (paper) potential
z-score statistics -> OUT/ssl_graphs_1k_geom_enriched_ri_rot. Every step is rotation-invariant except
xyz / normals, so `fingerprint.py verify --rotated` checks these graphs against the paper fingerprints.

Meshes are streamed out of the .tar.xz one at a time into a RAM/temp folder; nothing is extracted to disk.
Resumable (finished graphs are skipped). Check the result with fingerprint.py verify.

  python tools/shrec25_rebuild/build_shrec25_paper_graphs.py --train-archive A --test-archive B --out $DATA_ROOT -j 16
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

RELEASES = {   # release -> (label folder in data_labels/, output folder)
    "paper": ("shrec25_paper", "ssl_graphs_1k_geom_enriched_ri"),
    "rot":   ("shrec25_final", "ssl_graphs_1k_geom_enriched_ri_rot"),
}
LABELS = REPO / "data_labels" / "shrec25_paper"
SPLIT_FILES = ["train_split.csv", "val_split.csv", "test_labels.csv"]
STAT_FILES = ["foundation_scalar_stats_train.json", "foundation_xyz_global_scale_train_p99.json"]
OUT_NAME = RELEASES["paper"][1]
N_POINTS, KNN_K = 1000, 6
ENRICH = ("pos", 4, "graph_zscore")

_STATS = None
_TMP = None


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def stem_of(member_base):
    """Archive member -> our graph id: drop .vtk and the FINAL release's _rot suffix, ':' -> '_'."""
    stem = member_base[:-4]
    if stem.endswith("_rot"):
        stem = stem[:-4]
    return stem.replace("_centered", "").replace(":", "_")


def ids(csv_name):
    with open(LABELS / csv_name, newline="") as f:
        return {r["protein_id"] for r in csv.DictReader(f)}


def _init_worker(stats, tmp):
    global _STATS, _TMP
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    _STATS, _TMP = stats, tmp


def convert(job):
    """VTK bytes -> final paper graph, written atomically to out_path."""
    import vtk_mesh_to_ssl_graph_parallel as V
    import sample_ssl_graphs as S
    import enrich_ssl_graph_features as E
    from build_ri_feat_dataset import compute_ri_features
    from add_lrf_features import compute_lrf, compute_ri_lrf_features
    member, payload, out_path, source = job
    tag = os.path.join(_TMP, f"s25paper_{os.getpid()}")
    vtk_tmp, full_tmp, k1_tmp, en_tmp = f"{tag}.vtk", f"{tag}_full.pt", f"{tag}_1k.pt", f"{tag}_en.pt"
    try:
        with open(vtk_tmp, "wb") as f:
            f.write(payload)
        g = V.make_graph_from_vtk(vtk_tmp, scalar_stats=_STATS)
        g["meta"]["source_file"] = source
        torch.save(g, full_tmp)
        ok, _, _, _, err = S.process_one((full_tmp, k1_tmp, N_POINTS, None, KNN_K))
        if not ok:
            raise RuntimeError(f"sample_ssl_graphs: {err}")
        ok, _, _, _, err = E.process_one((k1_tmp, en_tmp, *ENRICH))
        if not ok:
            raise RuntimeError(f"enrich_ssl_graph_features: {err}")
        d = torch.load(en_tmp, map_location="cpu", weights_only=False)
        ei = d["edge_index_mesh"]
        pos, nor = d["pos"].float(), d["normals"].float()
        d["ri_feat"] = compute_ri_features(pos, nor, ei)
        d["ri_lrf_feat"] = compute_ri_lrf_features(pos, nor, ei, compute_lrf(pos, nor, ei))
        torch.save(d, out_path + ".tmp")
        os.replace(out_path + ".tmp", out_path)
        return member, True, ""
    except Exception as e:  # noqa: BLE001
        return member, False, f"{type(e).__name__}: {e}"
    finally:
        for p in (vtk_tmp, full_tmp, k1_tmp, en_tmp):
            try:
                os.remove(p)
            except OSError:
                pass


def open_stream(archive):
    if shutil.which("xz"):
        xz = subprocess.Popen(["xz", "-dc", "-T0", str(archive)], stdout=subprocess.PIPE, bufsize=1 << 20)
        return xz, tarfile.open(fileobj=xz.stdout, mode="r|")
    return None, tarfile.open(str(archive), mode="r|xz")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-archive", required=True)
    ap.add_argument("--test-archive", required=True)
    ap.add_argument("--out", required=True, help="data root; the graphs go to OUT/" + "/".join(o for _, o in RELEASES.values()) + " (by --release)")
    ap.add_argument("-j", "--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N needed meshes per split")
    ap.add_argument("--only", default="", help="smoke test: comma-separated graph ids (train ids with '_'), "
                    "e.g. ids sampled in fingerprints_paper.json")
    ap.add_argument("--tmp", default=None, help="scratch for one mesh per worker (default /dev/shm)")
    ap.add_argument("--release", choices=list(RELEASES), default="paper",
                    help="paper: original unrotated archives + paper labels; rot: FINAL rotated archives + labels")
    a = ap.parse_args()
    global LABELS, OUT_NAME
    LABELS = REPO / "data_labels" / RELEASES[a.release][0]
    OUT_NAME = RELEASES[a.release][1]

    stats = json.loads((HERE / "train_scalar_stats.json").read_text())
    tmp = a.tmp or ("/dev/shm" if os.access("/dev/shm", os.W_OK) else tempfile.gettempdir())
    gdir = Path(a.out) / OUT_NAME
    for s in ("train_set", "test_set"):
        (gdir / s).mkdir(parents=True, exist_ok=True)
    for f in SPLIT_FILES:
        shutil.copy2(LABELS / f, gdir / f)
    for f in STAT_FILES:   # train statistics of the paper graphs, kept for every release
        shutil.copy2(REPO / "data_labels" / "shrec25_paper" / f, gdir / f)
    want = {"train_set": ids("train_split.csv") | ids("val_split.csv"), "test_set": ids("test_labels.csv")}
    archives = {"train_set": Path(a.train_archive), "test_set": Path(a.test_archive)}
    log(f"target {gdir}: train {len(want['train_set'])}, test {len(want['test_set'])} graphs; "
        f"{a.workers} workers; temp {tmp}")

    failures = []
    for split in ("train_set", "test_set"):
        have = {p.stem for p in (gdir / split).glob("*.pt")}
        todo = want[split] - have
        if not todo:
            log(f"{split}: all {len(want[split])} graphs present -> skip")
            continue
        if a.only:
            todo &= {x.strip() for x in a.only.split(",") if x.strip()}
            if not todo:
                continue
        if a.limit:
            todo = set(sorted(todo)[:a.limit])
        log(f"{split}: streaming {archives[split].name} for {len(todo)} graphs")
        proc, tar = open_stream(archives[split])
        seen, built, t0 = set(), 0, time.time()
        with ProcessPoolExecutor(a.workers, initializer=_init_worker, initargs=(stats, tmp)) as pool, tar:
            inflight = set()

            def drain(block):
                nonlocal built
                done = wait(inflight, return_when=FIRST_COMPLETED)[0] if block else {f for f in inflight if f.done()}
                for fu in done:
                    inflight.discard(fu)
                    member, ok, err = fu.result()
                    if not ok:
                        failures.append((member, err)); log(f"  FAILED {member}: {err}"); continue
                    built += 1
                    if built % 250 == 0:
                        log(f"  {split}: {built}/{len(todo)} ({built / (time.time() - t0):.1f}/s)")

            for ti in tar:
                base = ti.name.split("/")[-1]
                if not ti.isfile() or not base.endswith(".vtk"):
                    continue
                stem = stem_of(base)
                if stem not in todo:
                    continue
                seen.add(stem)
                payload = tar.extractfile(ti).read()
                while len(inflight) >= 2 * a.workers:
                    drain(True)
                inflight.add(pool.submit(convert, (base, payload, str(gdir / split / f"{stem}.pt"),
                                                   f"{archives[split].name}:{ti.name}")))
                if (a.limit or a.only) and len(seen) >= len(todo):
                    break
            while inflight:
                drain(True)
        if proc is not None:
            proc.stdout.close(); proc.kill(); proc.wait()
        missing = sorted(todo - seen)
        if missing:
            raise SystemExit(f"{split}: {len(missing)} surfaces not found in {archives[split].name}, "
                             f"e.g. {missing[:5]} (wrong archive?)")
        log(f"{split}: built {built} graphs in {(time.time() - t0) / 60:.1f} min")
    if failures:
        raise SystemExit(f"{len(failures)} conversions failed, e.g. {failures[:3]}")

    counts = {s: len(list((gdir / s).glob("*.pt"))) for s in ("train_set", "test_set")}
    (gdir / "build_manifest.json").write_text(json.dumps({
        "created": time.strftime("%F %T"),
        "release": ("SHREC'25 original (unrotated) meshes: train_set.tar.xz / test_set.tar.xz; paper splits"
                    if a.release == "paper" else
                    "SHREC'25 FINAL release: rotated meshes train_set_rot.tar.xz / test_set_rot.tar.xz; final labels "
                    "(98 classes) on our train/val membership; potentials z-scored with the paper train statistics"),
        "archives": {s: str(p) for s, p in archives.items()},
        "pipeline": ["vtk_mesh_to_ssl_graph_parallel.make_graph_from_vtk(scalar_stats=train_scalar_stats.json)",
                     f"sample_ssl_graphs.process_one(n_points={N_POINTS}, knn_k={KNN_K})",
                     f"enrich_ssl_graph_features.process_one{ENRICH}",
                     "build_ri_feat_dataset.compute_ri_features + add_lrf_features (ri_feat, ri_lrf_feat)"],
        "counts": counts, "limit": a.limit, "only": a.only}, indent=1))
    log(f"done: {counts} -> {gdir}")


if __name__ == "__main__":
    main()
