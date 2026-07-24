#!/usr/bin/env python3
"""
Compute standard shape-retrieval metrics (NN, first-tier, second-tier, mAP)
from the query-to-database distance matrices already extracted by
shrec21_retrieval_eval.py.

CAVEAT: query ground truth is NOT confirmed from an official source -- there is
no query-to-class mapping in any of the official SHREC21 download files. This
script uses the working hypothesis that query_N corresponds to the Nth class in
the .cla file's declared order (SH2, SH3, PDZ, SH3_2, m50, PDZ_6, Bromodomain,
PHD, ZF, Stat). A geometric sanity check (rotation-invariant shape fingerprints)
only weakly supports this (1/10 exact nearest-class matches) but the fingerprint
itself is known to be too coarse to discriminate these classes reliably even for
genuine same-class database members, so this is inconclusive either way.
Treat all numbers below as provisional pending real confirmation.
"""
import numpy as np

CLA_CLASS_ORDER = ["SH2", "SH3", "PDZ", "SH3_2", "m50", "PDZ_6", "Bromodomain", "PHD", "ZF", "Stat"]

# class_name -> class_id, from the single-label derived view (Stat has no
# class_id since it never wins as primary label -- handled as -1, meaning "no
# database surface will ever match this query under the primary-label scheme").
NAME_TO_ID = {"SH2": 0, "SH3": 1, "PDZ": 2, "SH3_2": 3, "m50": 4,
              "PDZ_6": 5, "Bromodomain": 6, "PHD": 7, "ZF": 8, "Stat": -1}


def retrieval_metrics(dist, query_class_ids, db_labels):
    """dist: [Q, N] distance matrix. query_class_ids: [Q] hypothesized class id
    per query (-1 = no possible match). db_labels: [N] single-label class id
    per database surface (the size of the true class = relevant-set size)."""
    Q, N = dist.shape
    results = []
    for qi in range(Q):
        cid = query_class_ids[qi]
        order = np.argsort(dist[qi])  # nearest first
        ranked_labels = db_labels[order]
        relevant = (ranked_labels == cid)
        n_rel = int(relevant.sum())
        if cid < 0 or n_rel == 0:
            results.append(dict(nn=None, ft=None, st=None, ap=None, n_rel=n_rel))
            continue

        nn_hit = bool(relevant[0])
        ft = relevant[:n_rel].sum() / n_rel
        st = relevant[:2 * n_rel].sum() / n_rel

        hits = np.cumsum(relevant)
        precision_at_k = hits / (np.arange(N) + 1)
        ap = (precision_at_k * relevant).sum() / n_rel

        results.append(dict(nn=float(nn_hit), ft=float(ft), st=float(st), ap=float(ap), n_rel=n_rel))
    return results


def main():
    for modality in ["shape", "electro"]:
        d = np.load(f"runs_shrec21/retrieval/{modality}_distance_matrix.npz", allow_pickle=True)
        dist = d["dist"]
        db_labels = d["db_labels"]
        query_ids = list(d["query_ids"])

        query_class_ids = np.array([NAME_TO_ID[CLA_CLASS_ORDER[i]] for i in range(10)])

        res = retrieval_metrics(dist, query_class_ids, db_labels)

        print(f"\n=== {modality} ===  (query -> hypothesized class -> #relevant in DB)")
        print(f"{'query':>10s} {'class':14s} {'#rel':>5s} {'NN':>6s} {'1st-tier':>9s} {'2nd-tier':>9s} {'mAP':>7s}")
        valid = [r for r in res if r["nn"] is not None]
        for qi, r in enumerate(res):
            cname = CLA_CLASS_ORDER[qi]
            if r["nn"] is None:
                print(f"{query_ids[qi]:>10s} {cname:14s} {r['n_rel']:>5d}   (no possible match -- class has no id in single-label scheme)")
            else:
                print(f"{query_ids[qi]:>10s} {cname:14s} {r['n_rel']:>5d} {r['nn']:6.3f} {r['ft']:9.3f} {r['st']:9.3f} {r['ap']:7.3f}")
        if valid:
            print(f"{'MEAN (valid queries)':>10s} {'':14s} {'':>5s} "
                  f"{np.mean([r['nn'] for r in valid]):6.3f} "
                  f"{np.mean([r['ft'] for r in valid]):9.3f} "
                  f"{np.mean([r['st'] for r in valid]):9.3f} "
                  f"{np.mean([r['ap'] for r in valid]):7.3f}")


if __name__ == "__main__":
    main()
