#!/usr/bin/env python
"""Is the multi-scale HiPatch model (MultiScaleHierGNN) implemented as designed?

Design (future-work point 4): one shared node encoder feeds S patch resolutions
(k = 16, 64, 256); each resolution has its own attention pooling and patch GNN
(or one patch GNN shared by all, --share-patch-gnn) and a max readout; a learned
gate (softmax over the S graph vectors of each protein) weighs the resolutions
before the classifier head.

Part 1, structure (real graphs, untrained models, CPU, seconds):
  A  data      scale s receives exactly the inputs (node graph, node->patch ids, patch graph)
               that a single-scale model reading that k receives; patch ids never cross proteins
  B  S = 1     with one resolution the multi-scale model is the single-scale HiPatch model:
               identical logits after copying weights (plain and invariant encoder)
  C  branches  with S = 3, branch s computes what a single-scale model with branch-s weights
               computes; the fused logits equal head(sum_s gate_s * maxreadout_s) recomputed by hand
  D  gate      one weight vector per protein, >= 0, sums to 1; a protein's output does not depend
               on the other proteins in the batch
  E  training  every parameter receives a gradient (gate, each attention pooling, each patch GNN);
               parameter counts explained
  F  invariant with --invariant, a random rotation of the input leaves the multi-scale logits unchanged
  G  edges     --use-bidirectional-edges gives every node incoming messages (the default mesh
               edge list is one-directional)

Part 2, trained runs (--runs-root; Jean-Zay, inference on the test set):
  per run: test accuracy / F1w of the full model, of each resolution alone (gate forced to
  that resolution), and without each resolution (gate renormalised over the others); gate
  statistics (mean weight per resolution, entropy, how often one resolution takes > 90%).
  If the gate collapses onto one resolution, the model is a single-scale model in practice.

  python tools/verify_multiscale.py --graph-dir $DATA_ROOT/ssl_graphs_1k_geom_enriched_ri \
      --assign-dirs $DATA_ROOT/ssl_graphs_1k_patchassign_k{16,64,256}
  python tools/verify_multiscale.py --runs-root runs/fw_multiscale25/runs --data-root $DATA_ROOT --skip-structure
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import classify_hierarchical_gnn as C  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


def cfg_for(args, invariant=False, dirs=None, fusion="gate", share=False):
    c = dict(feature_mode=args.feature_mode, node_backbone="gvp", hidden_dim=256, node_layers=3,
             patch_layers=2, pool="attn", readout="max", dropout=0.3, patch_backbone="gin",
             invariant=invariant)
    if dirs:
        c.update(multi_assign_dirs=list(dirs), fusion=fusion, share_patch_gnn=share)
    return c


def multi_key(k, s, share):
    """single-scale parameter name -> multi-scale name for branch s (None: no counterpart)."""
    if k.startswith(("node_convs.", "node_norms.", "head.")):
        return k
    if k.startswith("attn_pool."):
        return k.replace("attn_pool.", f"attn_pools.{s}.", 1)
    for p in ("patch_convs.", "patch_norms."):
        if k.startswith(p):
            return p + f"{0 if share else s}." + k[len(p):]
    return None


def copy_single_to_multi(single, multi, s, share):
    msd, used = multi.state_dict(), []
    for k, v in single.state_dict().items():
        mk = multi_key(k, s, share)
        if mk is not None and mk in msd:
            msd[mk] = v.clone(); used.append(k)
    multi.load_state_dict(msd)
    return [k for k in single.state_dict() if k not in used]


def copy_multi_to_single(multi, single, s, share):
    ssd, msd = single.state_dict(), multi.state_dict()
    for k in ssd:
        mk = multi_key(k, s, share)
        if mk is not None:
            ssd[k] = msd[mk].clone()
    single.load_state_dict(ssd)


def maxdiff(a, b):
    return float((a - b).abs().max())


def rotation(seed):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g))
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


# --------------------------------------------------------------------------- #
# part 1: structure
# --------------------------------------------------------------------------- #
def structure(args):
    torch.manual_seed(0)
    gdir, dirs = args.graph_dir, args.assign_dirs
    csv = args.labels_csv or str(Path(gdir) / "train_split.csv")
    labels, ncls = C.load_labels(gdir, "train", csv)
    kw = dict(feature_mode=args.feature_mode)
    probe = C.MultiScalePatchDataset(gdir, dirs, "train", labels, **kw)
    sub = {it[0]: labels[it[0]] for it in probe.items[:args.n]}
    ms = C.MultiScalePatchDataset(gdir, dirs, "train", sub, **kw)
    singles = [C.HierPatchDataset(gdir, d, "train", sub, **kw) for d in dirs]
    n, S = len(ms), len(dirs)
    in_dim = ms[0][0].ndata["attr"].shape[1]
    ks = [int(np.max([torch.load(Path(d) / "train_set" / it[1].name, weights_only=False)["num_patches"]
                      for it in ms.items])) for d in dirs]
    print(f"\n{n} proteins, {S} resolutions (max patches per protein: {ks}), in_dim {in_dim}, {ncls} classes")

    print("\nA. data: each resolution sees exactly the single-scale inputs")
    items = [ms[i] for i in range(n)]
    bn, bps, gpids, y = C.collate_multi(items)
    ok_ids = all([it[0] for it in sd.items] == [it[0] for it in ms.items] for sd in singles)
    check("same proteins in the same order at every resolution", ok_ids)
    for s in range(S):
        sbn, sbp, sgpid, sy = C.collate([singles[s][i] for i in range(n)])
        same = (torch.equal(torch.stack(bn.edges()), torch.stack(sbn.edges()))
                and torch.equal(bn.ndata["attr"], sbn.ndata["attr"]) and torch.equal(gpids[s], sgpid)
                and torch.equal(torch.stack(bps[s].edges()), torch.stack(sbp.edges())) and torch.equal(y, sy))
        node_graph = torch.repeat_interleave(torch.arange(n), bn.batch_num_nodes())
        patch_graph = torch.repeat_interleave(torch.arange(n), bps[s].batch_num_nodes())
        cross = int((node_graph != patch_graph[gpids[s]]).sum())
        empty = int((torch.bincount(gpids[s], minlength=bps[s].num_nodes()) == 0).sum())
        check(f"k={ks[s]}: node graph, patch ids, patch graph, labels identical to single-scale k={ks[s]}", same)
        check(f"k={ks[s]}: no node assigned to another protein's patch; no empty patch", cross == 0 and empty == 0,
              f"cross={cross}, empty={empty}")

    print("\nB. one resolution = the single-scale HiPatch model")
    for inv in (False, True):
        for s in range(S):
            single = C.build_model(cfg_for(args, inv), in_dim, ncls).eval()
            multi = C.build_model(cfg_for(args, inv, [dirs[s]]), in_dim, ncls).eval()
            unused = copy_single_to_multi(single, multi, 0, False)
            sbn, sbp, sgpid, _ = C.collate([singles[s][i] for i in range(n)])
            with torch.no_grad():
                d = maxdiff(single(sbn, sbp, sgpid), multi(sbn, [sbp], [sgpid]))
            check(f"{'invariant' if inv else 'plain'} encoder, k={ks[s]}: identical logits", d < 1e-5,
                  f"max |diff| {d:.1e}; single-scale params without counterpart: "
                  f"{sorted({u.split('.')[0] for u in unused}) or 'none'}")

    print("\nC. three resolutions: each branch = a single-scale model; fusion arithmetic")
    for share in (False, True):
        multi = C.build_model(cfg_for(args, False, dirs, "gate", share), in_dim, ncls).eval()
        with torch.no_grad():
            logits, pembs, gates = multi(bn, bps, gpids, return_patch=True)
            for s in range(S):
                single = C.build_model(cfg_for(args), in_dim, ncls).eval()
                copy_multi_to_single(multi, single, s, share)
                _, ph = single(bn, bps[s], gpids[s], return_patch=True)
                check(f"{'shared' if share else 'separate'} patch GNN, k={ks[s]}: branch patch embeddings "
                      f"= single-scale model", maxdiff(ph, pembs[s]) < 1e-5, f"max |diff| {maxdiff(ph, pembs[s]):.1e}")
            H = []
            for s in range(S):
                with bps[s].local_scope():
                    bps[s].ndata["h"] = pembs[s]
                    H.append(dgl.max_nodes(bps[s], "h"))
            H = torch.stack(H, 1)
            g2 = torch.softmax(multi.gate(H).squeeze(-1), 1)
            manual = multi.head((g2.unsqueeze(-1) * H).sum(1))
            check(f"{'shared' if share else 'separate'}: logits = head(sum_s gate_s * maxreadout_s)",
                  maxdiff(manual, logits) < 1e-5 and maxdiff(g2, gates) < 1e-6, f"max |diff| {maxdiff(manual, logits):.1e}")

    print("\nD. gate")
    multi = C.build_model(cfg_for(args, False, dirs), in_dim, ncls).eval()
    with torch.no_grad():
        logits, gates = multi(bn, bps, gpids, return_gates=True)
        b1 = C.collate_multi([ms[0]])
        l1 = multi(b1[0], b1[1], b1[2])
        perm = list(range(n))[::-1]
        bP = C.collate_multi([ms[i] for i in perm])
        lP = multi(bP[0], bP[1], bP[2])
    check("one weight per protein and resolution, >= 0, rows sum to 1",
          tuple(gates.shape) == (n, S) and bool((gates >= 0).all()) and maxdiff(gates.sum(1), torch.ones(n)) < 1e-6,
          f"shape {tuple(gates.shape)}")
    check("a protein's output does not depend on the rest of the batch",
          maxdiff(l1[0], logits[0]) < 1e-5 and maxdiff(lP.flip(0), logits) < 1e-5,
          f"alone vs batch {maxdiff(l1[0], logits[0]):.1e}, reversed batch {maxdiff(lP.flip(0), logits):.1e}")
    check("untrained gate is not degenerate", float(gates.std()) > 0, f"gate mean per resolution {gates.mean(0).numpy().round(3).tolist()}")

    print("\nE. training: gradients and parameter counts")
    counts = {}
    for name, c in (("single-scale k=64", cfg_for(args)),
                    ("multi-scale, separate patch GNNs", cfg_for(args, False, dirs)),
                    ("multi-scale, shared patch GNN", cfg_for(args, False, dirs, "gate", True))):
        m = C.build_model(c, in_dim, ncls).train()
        out = m(bn, bps, gpids) if "multi_assign_dirs" in c else m(bn, bps[min(1, S - 1)], gpids[min(1, S - 1)])
        F.cross_entropy(out, y).backward()
        # a bias that feeds a softmax directly has zero gradient by construction (softmax ignores a
        # constant shift): the attention-pooling scorers (softmax within a patch; single-scale model
        # too) and the gate's last layer (softmax over resolutions). Their weights must learn.
        def no_grad(k, p):
            return p.grad is None or float(p.grad.abs().max()) < 1e-6
        pre_softmax = [k for k, _ in m.named_parameters() if "attn_pool" in k or k.startswith("gate.2.")]
        shift = [k for k in pre_softmax if k.endswith(".bias")]
        dead = sorted({k.rsplit(".", 1)[0] for k, p in m.named_parameters() if no_grad(k, p) and k not in shift})
        weights_ok = all(not no_grad(k, p) for k, p in m.named_parameters() if k in pre_softmax and k not in shift)
        n_par = sum(p.numel() for p in m.parameters())
        n_dead = sum(p.numel() for k, p in m.named_parameters() if p.grad is None)
        counts[name] = (n_par, n_dead)
        expect_dead = all(d.startswith("proj_head") for d in dead) and weights_ok
        check(f"{name}: every parameter gets a gradient" + (" (except the unused SupCon head)" if dead else ""),
              expect_dead, f"{n_par:,} params, {n_par - n_dead:,} trained" + (f"; no gradient: {dead}" if dead else "")
              + "; biases right before a softmax: zero gradient by construction")
    for name, (n_par, n_dead) in counts.items():
        print(f"     {name:34s} {n_par:>9,} params  ({n_par - n_dead:,} used in the forward pass)")

    print("\nF. rotation invariance of the multi-scale model")
    R = rotation(1)
    for inv in (True, False):
        m = C.build_model(cfg_for(args, inv, dirs), in_dim, ncls).eval()
        bR = C.collate_multi([ms[i] for i in range(n)])
        C.rotate_attr(bR[0], args.feature_mode, R)
        with torch.no_grad():
            d = maxdiff(m(bn, bps, gpids), m(bR[0], bR[1], bR[2]))
        if inv:
            check("invariant encoder: logits unchanged under a random rotation", d < 1e-4, f"max |diff| {d:.1e}")
        else:
            check("plain encoder (control): logits change under rotation", d > 1e-4, f"max |diff| {d:.1e}")

    print("\nG. message direction")
    for bidir in (False, True):
        sd = C.HierPatchDataset(gdir, dirs[0], "train", sub, feature_mode=args.feature_mode,
                                use_bidirectional_edges=bidir)
        none = total = asym = 0
        for i in range(n):
            g = sd[i][0]
            src, dst = g.edges()
            keep = src != dst
            indeg = torch.bincount(dst[keep], minlength=g.num_nodes())
            none += int((indeg == 0).sum()); total += g.num_nodes()
            fwd = set(zip(src[keep].tolist(), dst[keep].tolist()))
            asym += sum((b, a) not in fwd for a, b in fwd)
        frac = none / total
        if bidir:
            check("bidirectional: every edge in both directions, every node receives messages",
                  asym == 0 and none == 0, f"{100 * frac:.1f}% nodes without incoming message")
        else:
            print(f"     default edge list: {100 * frac:.1f}% of nodes receive no neighbour message "
                  f"({asym} of the edges have no reverse edge)")
    for it in ms.items[:n]:
        d = torch.load(it[1], weights_only=False)
        mesh = set(map(tuple, d["edge_index_mesh"].t().tolist()))
        full = set(map(tuple, d["edge_index"].t().tolist())) - {(a, a) for a in range(d["pos"].shape[0])}
        if full != mesh | {(b, a) for a, b in mesh}:
            check("edge_index = edge_index_mesh in both directions (graph files)", False, it[1].name); break
    else:
        check("edge_index = edge_index_mesh in both directions (graph files)", True)


# --------------------------------------------------------------------------- #
# part 2: trained runs
# --------------------------------------------------------------------------- #
@torch.no_grad()
def trained(args):
    import ensemble_attribution as EA
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score, f1_score
    device = torch.device("cpu" if args.device == "cpu" or not torch.cuda.is_available() else f"cuda:{args.device}")
    runs = [r for r in EA.find_runs([args.runs_root], args.include, None)
            if json.loads((r / "final_results.json").read_text())["config"].get("multi_assign_dirs")]
    if not runs:
        sys.exit(f"no multi-scale runs with best_model.pt under {args.runs_root}")
    report = ["# Multi-scale runs: what each resolution contributes (test set)\n"]
    for run in runs:
        cfg = json.loads((run / "final_results.json").read_text())["config"]
        if cfg.get("fusion", "gate") != "gate":
            print(f"skip {run.name}: fusion {cfg.get('fusion')}"); continue
        ds, col = EA.test_dataset(cfg, args.data_root)
        model = EA.load_model(run, cfg, device)
        ks = [int(m.group(1)) if (m := re.search(r"k(\d+)$", Path(a).name)) else i
              for i, a in enumerate(cfg["multi_assign_dirs"])]
        S = len(ks)
        Hs, G, Y, L = [], [], [], []
        for batch in DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                collate_fn=col):
            bn, bps, gpids, y = EA.to_dev(batch, device)
            logits, pembs, gates = model(bn, bps, gpids, return_patch=True)
            H = []
            for s in range(S):
                with bps[s].local_scope():
                    bps[s].ndata["h"] = pembs[s]
                    H.append(dgl.max_nodes(bps[s], "h"))
            Hs.append(torch.stack(H, 1).cpu()); G.append(gates.cpu()); Y.append(y.cpu()); L.append(logits.cpu())
        H, G, Y, L = torch.cat(Hs), torch.cat(G), torch.cat(Y).numpy(), torch.cat(L)
        head = model.head.cpu()

        def score(w):
            pred = head((w.unsqueeze(-1) * H).sum(1)).argmax(1).numpy()
            return accuracy_score(Y, pred), f1_score(Y, pred, average="weighted", zero_division=0)

        full = score(G)
        check(f"{run.name}: recomputed fusion reproduces the model's test predictions",
              (head((G.unsqueeze(-1) * H).sum(1)).argmax(1) == L.argmax(1)).all())
        rows = [("all resolutions (learned gate)", *full)]
        for s in range(S):
            one = torch.zeros_like(G); one[:, s] = 1
            rows.append((f"k={ks[s]} only", *score(one)))
        for s in range(S):
            w = G.clone(); w[:, s] = 0; w = w / w.sum(1, keepdim=True).clamp(min=1e-12)
            rows.append((f"without k={ks[s]}", *score(w)))
        rows.append(("equal weights (no gate)", *score(torch.full_like(G, 1.0 / S))))
        ent = float((-(G.clamp(min=1e-12)).log().mul(G).sum(1) / math.log(S)).mean())
        dom = (G.max(1).values > 0.9).float().mean().item()
        report += [f"\n## {run.name}\n",
                   f"gate mean per resolution: " + ", ".join(f"k={k} {v:.3f}" for k, v in zip(ks, G.mean(0).tolist())),
                   f"  \nnormalised gate entropy (1 = uniform, 0 = one resolution): {ent:.3f}; "
                   f"proteins with one resolution > 90%: {100 * dom:.1f}%\n",
                   "| test-time weights | acc | F1w |", "|---|---|---|"]
        report += [f"| {n} | {a:.4f} | {f:.4f} |" for n, a, f in rows]
        cls = {}
        for c in np.unique(Y):
            cls[int(c)] = G[torch.from_numpy(Y == c)].mean(0)
        top = {k: sum(int(v.argmax()) == i for v in cls.values()) for i, k in enumerate(ks)}
        report.append(f"\nclasses whose mean gate is highest at: " + ", ".join(f"k={k}: {n}" for k, n in top.items()))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report) + "\n")
    print("\n".join(report)); print(f"\n-> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph-dir")
    ap.add_argument("--assign-dirs", nargs="+", help="one patch-assignment dir per resolution, as in the runs")
    ap.add_argument("--labels-csv", default=None, help="default: <graph-dir>/train_split.csv")
    ap.add_argument("--feature-mode", default="geom_14d_raw")
    ap.add_argument("--n", type=int, default=16, help="proteins used by the structural checks")
    ap.add_argument("--skip-structure", action="store_true")
    ap.add_argument("--runs-root", default=None, help="also check trained multi-scale runs (part 2)")
    ap.add_argument("--include", default=None, help="regex on run names (part 2)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="runs/fw_analysis/multiscale_check/report.md")
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    a = ap.parse_args()
    if not a.skip_structure:
        if not (a.graph_dir and a.assign_dirs):
            ap.error("--graph-dir and --assign-dirs are needed for the structural checks (or --skip-structure)")
        print("PART 1: structure"); structure(a)
    if a.runs_root:
        print("\nPART 2: trained runs"); trained(a)
    bad = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    for name, _, detail in bad:
        print(f"  FAIL {name}  {detail}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
