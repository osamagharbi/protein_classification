#!/usr/bin/env python
"""Check the FINAL-release (rotated) SHREC'25 graphs against the paper graphs, graph by graph.

The rotated meshes are the paper meshes turned by a random rotation, so the same pipeline should give the
same graphs up to that rotation. Float32 near-ties in farthest-point sampling can still pick a different
vertex here and there (and reorder nodes), so exact hashes / summary statistics are the wrong test.
Instead, per graph id present in both builds:

  1. match vertices by rotation-invariant keys: the two potentials (read from the mesh, identical for the
     same vertex) and |pos| (distance to the centre of the unit-ball mesh); mutual nearest neighbours
     within --key-tol
  2. overlap  = matched vertices / vertices                  (same surface: ~1.0; another surface: ~0)
  3. Kabsch   : one proper rotation R with pos_rot ~ pos_ref R^T on the matched vertices; residual RMS
               (same surface: ~1e-7; anything else: >1e-2). Normals checked with the same R.
  4. edges    : Jaccard of the mesh edges among matched vertices
  5. patches  : (with --ref-patches/--rot-patches) number of patches + adjusted Rand index of the patch
               labels on matched vertices. build_patchassign picks its first seed by node INDEX, so a
               reordered graph can get different (equally valid) patches: ARI is reported, not required to be 1

Exit 1 if any graph has overlap < --min-overlap or a rotation residual > --max-resid (or, with patches,
a different number of patches or median ARI < --min-median-ari).

  python tools/shrec25_rebuild/compare_rotated.py --ref $DATA_ROOT/ssl_graphs_1k_geom_enriched_ri \
      --rot $DATA_ROOT/ssl_graphs_1k_geom_enriched_ri_rot [--sample tools/shrec25_rebuild/fingerprints_paper.json] \
      [--ref-patches $DATA_ROOT/ssl_graphs_1k_patchassign_k64 --rot-patches $DATA_ROOT/ssl_graphs_1k_patchassign_rot_k64]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree


def _load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def ari(a, b):
    """Adjusted Rand index of two integer labelings (numpy only)."""
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    c = np.zeros((ai.max() + 1, bi.max() + 1), np.int64)
    np.add.at(c, (ai, bi), 1)
    comb = lambda x: (x * (x - 1) / 2.0).sum()
    s, sa, sb, n2 = comb(c), comb(c.sum(1)), comb(c.sum(0)), len(a) * (len(a) - 1) / 2.0
    exp = sa * sb / n2
    den = 0.5 * (sa + sb) - exp
    return 1.0 if den == 0 else float((s - exp) / den)


def match(A, B, tol):
    """Mutual-nearest-neighbour vertex matching on (potential_1, potential_2, |pos|)."""
    ka = np.c_[A["scalars_raw"].double().numpy(), A["pos"].double().norm(dim=1).numpy()]
    kb = np.c_[B["scalars_raw"].double().numpy(), B["pos"].double().norm(dim=1).numpy()]
    scale = np.maximum(ka.std(0), 1e-12)
    ta, tb = cKDTree(kb / scale), cKDTree(ka / scale)
    _, ab = ta.query(ka / scale)          # A vertex -> nearest B vertex
    _, ba = tb.query(kb / scale)          # B vertex -> nearest A vertex
    ia = np.arange(len(ka))
    ok = (ba[ab] == ia) & np.all(np.abs(ka - kb[ab]) <= tol * np.maximum(1.0, np.abs(ka)), axis=1)
    return ia[ok], ab[ok]


def kabsch(P, Q):
    """Proper rotation R minimising |Q - P R^T|; returns R, residual RMS."""
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, float(np.sqrt(((Q - P @ R.T) ** 2).sum(1).mean()))


def undirected(ei):
    e = np.sort(ei, axis=0)
    return set(map(tuple, e.T.tolist()))


def compare_one(job):
    key, ref_p, rot_p, refpat_p, rotpat_p, tol = job
    A, B = _load(ref_p), _load(rot_p)
    ia, ib = match(A, B, tol)
    n = int(A["pos"].shape[0])
    out = {"key": key, "n_ref": n, "n_rot": int(B["pos"].shape[0]), "matched": int(len(ia))}
    out["overlap"] = len(ia) / max(n, out["n_rot"])
    if len(ia) >= 4:
        P, Q = A["pos"].double().numpy()[ia], B["pos"].double().numpy()[ib]
        R, out["resid"] = kabsch(P, Q)
        NA, NB = A["normals"].double().numpy()[ia], B["normals"].double().numpy()[ib]
        out["normal_resid"] = float(np.sqrt(((NB - NA @ R.T) ** 2).sum(1).mean()))
        a2b = -np.ones(n, np.int64); a2b[ia] = ib
        ea = A["edge_index_mesh"].numpy()
        ea = a2b[ea]; ea = ea[:, (ea >= 0).all(0)]
        inb = np.zeros(out["n_rot"], bool); inb[ib] = True
        eb = B["edge_index_mesh"].numpy(); eb = eb[:, inb[eb].all(0)]
        sa, sb = undirected(ea), undirected(eb)
        out["edge_jaccard"] = len(sa & sb) / max(1, len(sa | sb))
        ga = A["geom_features_raw"].double().numpy()[ia]; gb = B["geom_features_raw"].double().numpy()[ib]
        out["geom_same_frac"] = float(np.mean(np.all(np.abs(ga - gb) <= 1e-3 * np.maximum(1, np.abs(ga)), 1)))
    else:
        out.update(resid=float("inf"), normal_resid=float("inf"), edge_jaccard=0.0, geom_same_frac=0.0)
    if refpat_p is not None:
        if not Path(rotpat_p).is_file():
            out["patch_missing"] = True
        else:
            pa, pb = _load(refpat_p), _load(rotpat_p)
            out["np_ref"], out["np_rot"] = int(pa["num_patches"]), int(pb["num_patches"])
            out["ari"] = ari(pa["patch_ids"].numpy()[ia], pb["patch_ids"].numpy()[ib]) if len(ia) > 1 else 0.0
    return out


def q(v, p):
    with np.errstate(invalid="ignore"):
        return float(np.percentile(v, p)) if len(v) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="paper graphs (ssl_graphs_1k_geom_enriched_ri)")
    ap.add_argument("--rot", required=True, help="FINAL-release graphs (ssl_graphs_1k_geom_enriched_ri_rot)")
    ap.add_argument("--ref-patches"); ap.add_argument("--rot-patches")
    ap.add_argument("--sample", help="fingerprint JSON: compare only its sampled ids (default: every common id)")
    ap.add_argument("--ids", help="comma-separated split/id keys, e.g. train_set/3av2_3_G_G_model1")
    ap.add_argument("--key-tol", type=float, default=1e-4)
    ap.add_argument("--min-overlap", type=float, default=0.9)
    ap.add_argument("--max-resid", type=float, default=1e-4)
    ap.add_argument("--min-median-ari", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", help="write per-graph results (JSON)")
    a = ap.parse_args()
    if bool(a.ref_patches) != bool(a.rot_patches):
        sys.exit("--ref-patches and --rot-patches go together")

    ref, rot = Path(a.ref), Path(a.rot)
    if a.ids:
        keys = [k.strip() for k in a.ids.split(",") if k.strip()]
    elif a.sample:
        keys = list(json.loads(Path(a.sample).read_text())["graphs"])
    else:
        keys = []
        for split in ("train_set", "test_set"):
            r = {p.stem for p in (ref / split).glob("*.pt")}
            o = {p.stem for p in (rot / split).glob("*.pt")}
            print(f"{split}: {len(r)} paper, {len(o)} FINAL; {len(r & o)} in both, "
                  f"{len(r - o)} only paper (dropped by FINAL), {len(o - r)} only FINAL (cannot be checked)")
            keys += [f"{split}/{s}" for s in sorted(r & o)]
    keys = [k for k in keys if (ref / f"{k}.pt").is_file() and (rot / f"{k}.pt").is_file()]
    if not keys:
        sys.exit("no graph id present in both builds")
    jobs = [(k, ref / f"{k}.pt", rot / f"{k}.pt",
             Path(a.ref_patches) / f"{k}.pt" if a.ref_patches else None,
             Path(a.rot_patches) / f"{k}.pt" if a.rot_patches else None, a.key_tol) for k in keys]
    with ProcessPoolExecutor(a.workers) as pool:
        res = list(pool.map(compare_one, jobs, chunksize=16))

    ov = np.array([r["overlap"] for r in res]); rs = np.array([r["resid"] for r in res])
    full = int(np.sum([r["matched"] == r["n_ref"] == r["n_rot"] for r in res]))
    print(f"compared {len(res)} graphs: {full} share every vertex ({100 * full / len(res):.1f}%), "
          f"{len(res) - full} differ in some vertices")
    print(f"  vertex overlap   min {ov.min():.3f}  p1 {q(ov, 1):.3f}  median {q(ov, 50):.4f}")
    print(f"  rotation resid   max {rs.max():.2e}  median {q(rs, 50):.2e}   (pos in the unit ball)")
    print(f"  normal resid     max {max(r['normal_resid'] for r in res):.2e}")
    print(f"  mesh-edge Jaccard (matched vertices)  min {min(r['edge_jaccard'] for r in res):.4f}")
    print(f"  geom descriptors equal (1e-3)         min {min(r['geom_same_frac'] for r in res):.3f} of matched vertices")
    worst = sorted(res, key=lambda r: r["overlap"])[:5]
    for r in worst:
        if r["overlap"] < 1:
            print(f"    {r['key']}: {r['matched']}/{r['n_ref']} vertices shared, resid {r['resid']:.1e}, "
                  f"edge Jaccard {r['edge_jaccard']:.4f}" + (f", patch ARI {r['ari']:.3f}" if "ari" in r else ""))
    bad = [r for r in res if r["overlap"] < a.min_overlap or r["resid"] > a.max_resid]
    if a.ref_patches:
        miss = [r for r in res if r.get("patch_missing")]
        npd = [r for r in res if not r.get("patch_missing") and r["np_ref"] != r["np_rot"]]
        ar = np.array([r["ari"] for r in res if "ari" in r])
        same = [r for r in res if r["matched"] == r["n_ref"] == r["n_rot"] and "ari" in r]
        print(f"  patches: {len(miss)} missing, {len(npd)} with a different patch count; ARI median {q(ar, 50):.3f}, "
              f"p5 {q(ar, 5):.3f}, min {ar.min() if len(ar) else float('nan'):.3f}; "
              f"ARI = 1 in {np.mean([r['ari'] > 0.9999 for r in res if 'ari' in r]) * 100:.1f}% of graphs "
              f"({np.mean([r['ari'] > 0.9999 for r in same]) * 100 if same else float('nan'):.1f}% of those sharing every vertex)")
        bad += miss + npd
        if len(ar) and q(ar, 50) < a.min_median_ari:
            bad.append({"key": f"median patch ARI {q(ar, 50):.3f} < {a.min_median_ari}", "overlap": 0, "resid": 0})
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=0))
        print(f"  per-graph results -> {a.out}")
    if bad:
        for r in bad[:10]:
            print(f"  FAIL {r['key']}: overlap {r['overlap']:.3f}, resid {r['resid']:.2e}")
        sys.exit(1)
    print("VERIFIED: FINAL-release graphs are the paper graphs rotated "
          f"(min vertex overlap {ov.min():.3f}, max rotation residual {rs.max():.1e})")


if __name__ == "__main__":
    main()
