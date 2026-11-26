#!/usr/bin/env python3
"""Future work, point 2: what do EXISTING checkpoints tell us about patch resolution?

Runs on trained HSNet/N2P checkpoints only (inference, no training):

  predict    test-set softmax of every run, optionally also under N fixed random
             rotations (rotation test) and, for multi-scale runs, the per-surface
             resolution weights (gates).
  ensemble   Do different patch resolutions carry complementary information?
             Compares ensembles of m models of ONE resolution (different seeds)
             with ensembles of m models of DIFFERENT resolutions, at equal m.
             Cross-k > same-k  =>  resolutions are complementary and a
             multi-scale model is worth building.
  attribute  Which patches carry the class signal, at each resolution?
             - importance of each patch = gradient x input of the true-class logit
               w.r.t. the final patch embeddings (with the global max readout, a
               patch only receives gradient on the channels it wins);
             - deletion curves: remove the top-ranked vs random patches from the
               readout and track the true-class probability (faithfulness);
             - "flip point": fraction of top patches whose removal changes the
               prediction (how localised the signal is);
             - geometry: correlation of importance with patch protrusion
               (distance to the surface centre) and patch size;
             - cross-scale agreement: patch importance projected onto the nodes,
               Spearman correlation between the maps of every pair of resolutions.
  report     report.md with every table.

Each stage caches its outputs under --out, so the script is resumable.

Example (Jean-Zay, see jeanzay/README_future_work.md):
  python tools/ensemble_attribution.py \
      --runs-root $SCRATCH/fw_ckpts/runs/shrec19_recheck \
      --include 'shrec19_protein_geom_k\\d+_gvp_h256' \
      --data-root $WORK --out $WORK/protein_classification/runs/fw_analysis/shrec19_protein \
      --rotations 5
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import classify_hierarchical_gnn as C  # noqa: E402

RUN_RX = re.compile(r"^(?P<ds>shrec19_protein|shrec19_species|shrec25)_(?P<fg>[a-z0-9]+)_k(?P<k>\d+)_"
                    r"(?:(?P<arch>flat\d)_)?(?P<bb>[a-z0-9_]+?)_h(?P<h>\d+)_s(?P<seed>\d+)$")
FRACS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75]


# --------------------------------------------------------------------------- #
# discovery / data
# --------------------------------------------------------------------------- #
def find_runs(roots, include, exclude):
    runs = []
    for root in roots:
        for fr in sorted(Path(root).rglob("final_results.json")):
            d = fr.parent
            if not (d / "best_model.pt").exists():
                continue
            if include and not re.search(include, d.name):
                continue
            if exclude and re.search(exclude, d.name):
                continue
            runs.append(d)
    return runs


def run_identity(run_dir, cfg):
    """(family, k, seed): family groups runs that differ only by k and seed."""
    m = RUN_RX.match(cfg.get("run_name") or run_dir.name)
    multi = cfg.get("multi_assign_dirs")
    if m:
        fam = f"{m['ds']}_{m['fg']}_{(m['arch'] + '_') if m['arch'] else ''}{m['bb']}_h{m['h']}"
        k = int(m["k"])
        seed = int(m["seed"])
    else:
        k_m = re.search(r"_k(\d+)", Path(cfg.get("assign_dir", "")).name)
        fam = re.sub(r"_s\d+$", "", run_dir.name)
        fam = re.sub(r"_k\d+", "", fam)
        k = int(k_m.group(1)) if k_m else -1
        seed = int(cfg.get("seed", 0))
    if multi:
        ks = [int(re.search(r"_k(\d+)", Path(a).name).group(1)) for a in multi]
        fam = re.sub(r"_s\d+$", "", run_dir.name)
        k = "+".join(map(str, ks))
    if cfg.get("invariant"):
        fam += "_invariant"
    return fam, k, seed


def remap(path, data_root, is_file):
    """Point a path recorded on another machine/disk at --data-root."""
    if path is None or data_root is None:
        return path
    p = Path(path)
    cand = Path(data_root) / (Path(p.parent.name) / p.name if is_file else p.name)
    return str(cand) if cand.exists() else str(p)


def test_dataset(cfg, data_root):
    gdir = remap(cfg["graph_dir"], data_root, False)
    csv = remap(cfg.get("test_labels_csv"), data_root, True)
    labels, _ = C.load_labels(gdir, "test", csv)
    kw = dict(feature_mode=cfg.get("feature_mode", "all"),
              use_bidirectional_edges=bool(cfg.get("use_bidirectional_edges", False)))
    if cfg.get("multi_assign_dirs"):
        dirs = [remap(a, data_root, False) for a in cfg["multi_assign_dirs"]]
        return C.MultiScalePatchDataset(gdir, dirs, "test", labels, **kw), C.collate_multi
    adir = remap(cfg["assign_dir"], data_root, False)
    return C.HierPatchDataset(gdir, adir, "test", labels, **kw), C.collate


def load_model(run_dir, cfg, device):
    model = C.build_model(cfg, cfg["in_dim"], cfg["num_classes"])
    state = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # checkpoints trained before the (unused) SupCon projection head existed lack it
    bad = [k for k in missing if not k.startswith("proj_head.")] + list(unexpected)
    if bad:
        raise RuntimeError(f"{run_dir.name}: checkpoint/model mismatch: {bad[:6]}")
    return model.to(device).eval()


def to_dev(batch, device):
    bn, bp, gpid, y = batch
    if isinstance(bp, list):
        return bn.to(device), [b.to(device) for b in bp], [g.to(device) for g in gpid], y.to(device)
    return bn.to(device), bp.to(device), gpid.to(device), y.to(device)


def metrics(labels, probs):
    from sklearn.metrics import accuracy_score, f1_score
    pred = probs.argmax(1)
    out = {"accuracy": float(accuracy_score(labels, pred)),
           "f1_weighted": float(f1_score(labels, pred, average="weighted", zero_division=0))}
    out.update(C.present_class_metrics(labels, pred))
    return out


def ms(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return f"{np.mean(v):.4f} ± {np.std(v):.4f}" if v else "-"


# --------------------------------------------------------------------------- #
# stage 1: predictions
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_run(run_dir, cfg, args, device):
    ds, col = test_dataset(cfg, args.data_root)
    if args.max_test_graphs:
        ds.items = ds.items[:args.max_test_graphs]
    ids = np.array([str(it[0]) for it in ds.items])
    model = load_model(run_dir, cfg, device)
    multi_gate = bool(cfg.get("multi_assign_dirs")) and cfg.get("fusion", "gate") == "gate"

    def run(dataset):
        probs, labs, gates = [], [], []
        for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=col):
            bn, bp, gpid, y = to_dev(batch, device)
            if multi_gate:
                logits, g = model(bn, bp, gpid, return_gates=True)
                gates.append(g.cpu().numpy())
            else:
                logits = model(bn, bp, gpid)
            probs.append(torch.softmax(logits, 1).cpu().numpy().astype(np.float32))
            labs.append(y.cpu().numpy())
        return np.concatenate(probs), np.concatenate(labs), (np.concatenate(gates) if gates else None)

    probs, labels, gates = run(ds)
    out = {"ids": ids, "labels": labels, "probs": probs}
    if gates is not None:
        out["gates"] = gates
    if args.rotations > 0 and cfg.get("feature_mode") in C.FEATURE_LAYOUT:
        out["probs_rot"] = np.stack([run(C.RotatedView(ds, cfg["feature_mode"], seed=1000 + r))[0]
                                     for r in range(args.rotations)])
    return out


def stage_predict(runs, args, device):
    pdir = Path(args.out) / "preds"
    pdir.mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(runs):
        f = pdir / f"{d.name}.npz"
        if f.exists():
            continue
        cfg = json.load(open(d / "final_results.json"))["config"]
        try:
            out = predict_run(d, cfg, args, device)
        except Exception as e:  # keep going: one broken run must not stop the study
            print(f"  !! {d.name}: {e}", flush=True)
            continue
        np.savez_compressed(f, **out)
        print(f"  [{i+1}/{len(runs)}] {d.name}: test n={len(out['labels'])} "
              f"acc={metrics(out['labels'], out['probs'])['accuracy']:.4f}", flush=True)


def load_preds(runs, args):
    """family -> k -> seed -> dict(npz arrays, cfg)."""
    tab = defaultdict(lambda: defaultdict(dict))
    for d in runs:
        f = Path(args.out) / "preds" / f"{d.name}.npz"
        if not f.exists():
            continue
        cfg = json.load(open(d / "final_results.json"))["config"]
        fam, k, seed = run_identity(d, cfg)
        z = dict(np.load(f, allow_pickle=False))
        z["cfg"] = cfg
        z["run_dir"] = d
        tab[fam][k][seed] = z
    return tab


def aligned(models):
    """Common test ids (sorted) and each model's probs on them."""
    common = sorted(set.intersection(*[set(m["ids"].tolist()) for m in models]))
    out = []
    for m in models:
        pos = {g: i for i, g in enumerate(m["ids"].tolist())}
        idx = np.array([pos[g] for g in common])
        out.append((m["probs"][idx], m["labels"][idx]))
    return common, out


# --------------------------------------------------------------------------- #
# stage 2: ensembles
# --------------------------------------------------------------------------- #
def stage_ensemble(tab, args):
    rng = np.random.default_rng(0)
    res = {}
    for fam, byk in sorted(tab.items()):
        ks = sorted(k for k in byk if isinstance(k, int))
        if len(ks) < 2:
            continue
        flat = [(k, s, byk[k][s]) for k in ks for s in sorted(byk[k])]
        _, al = aligned([m for _, _, m in flat])
        labels = al[0][1]
        P = {(k, s): al[i][0] for i, (k, s, _) in enumerate(flat)}
        seeds = {k: sorted(byk[k]) for k in ks}
        m = min(args.ensemble_size, len(ks), min(len(v) for v in seeds.values()))

        def ens(keys):
            return metrics(labels, np.mean([P[x] for x in keys], axis=0))

        single = {k: [metrics(labels, P[(k, s)])["f1_weighted"] for s in seeds[k]] for k in ks}
        same = {}
        for k in ks:
            combos = list(itertools.combinations(seeds[k], m))
            if len(combos) > args.draws:
                combos = [tuple(rng.choice(seeds[k], m, replace=False)) for _ in range(args.draws)]
            same[k] = [ens([(k, s) for s in c])["f1_weighted"] for c in combos]
        cross = []
        for _ in range(args.draws):
            kk = rng.choice(ks, m, replace=False)
            cross.append(ens([(int(k), int(rng.choice(seeds[int(k)]))) for k in kk])["f1_weighted"])
        # disagreement between two models: same k (different seeds) vs different k
        def dis(a, b):
            return float((P[a].argmax(1) != P[b].argmax(1)).mean())
        same_pairs = [dis((k, a), (k, b)) for k in ks for a, b in itertools.combinations(seeds[k], 2)]
        cross_pairs = [dis((k1, a), (k2, b)) for k1, k2 in itertools.combinations(ks, 2)
                       for a in seeds[k1] for b in seeds[k2] if a == b]   # same seed, different k
        best_k = max(ks, key=lambda k: np.mean(same[k]))
        res[fam] = {
            "ks": [int(k) for k in ks], "ensemble_size": int(m), "n_test": len(labels),
            "single_f1w": {str(k): [float(np.mean(v)), float(np.std(v))] for k, v in single.items()},
            "same_k_f1w": {str(k): [float(np.mean(v)), float(np.std(v))] for k, v in same.items()},
            "cross_k_f1w": [float(np.mean(cross)), float(np.std(cross))],
            "best_same_k": str(best_k),
            "cross_minus_best_same": float(np.mean(cross) - np.mean(same[best_k])),
            "disagreement_same_k": float(np.mean(same_pairs)) if same_pairs else None,
            "disagreement_cross_k": float(np.mean(cross_pairs)) if cross_pairs else None,
            "all_seeds_best_k": ens([(best_k, s) for s in seeds[best_k]]),
            "all_models_all_k": ens(list(P)),
        }
        print(f"  {fam}: same-k(best k={best_k}) {np.mean(same[best_k]):.4f}  cross-k {np.mean(cross):.4f}  "
              f"(m={m})", flush=True)
    json.dump(res, open(Path(args.out) / "ensemble.json", "w"), indent=2)
    return res


# --------------------------------------------------------------------------- #
# stage 3: patch attribution
# --------------------------------------------------------------------------- #
def spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def attribute_run(run_dir, cfg, args, device):
    if cfg.get("multi_assign_dirs") or cfg.get("node_readout") or cfg.get("flat") \
            or cfg.get("readout", "max") != "max":
        return None   # defined for the single-scale max-readout model
    ds, col = test_dataset(cfg, args.data_root)
    if args.attr_max_graphs:
        ds.items = ds.items[:args.attr_max_graphs]
    model = load_model(run_dir, cfg, device)
    xyz_sl, _ = C.xyz_nor_slices(cfg.get("feature_mode", "all"))
    rng = np.random.default_rng(0)
    curves_top, curves_rand, flips, rho_prot, rho_size, node_maps = [], [], [], [], [], {}
    gi = 0
    for batch in DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=col):
        bn, bp, gpid, y = to_dev(batch, device)
        with torch.enable_grad():
            logits, ph = model(bn, bp, gpid, return_patch=True)
            ph.retain_grad()
            logits.gather(1, y.unsqueeze(1)).sum().backward()
        imp_all = (ph.grad * ph).sum(1).detach()
        ph = ph.detach()
        pc = bp.batch_num_nodes().tolist()
        nc = bn.batch_num_nodes().tolist()
        p_off = np.cumsum([0] + pc)
        n_off = np.cumsum([0] + nc)
        pid_local = bn.ndata["pid"]
        xyz = bn.ndata["attr"][:, xyz_sl]
        for b in range(len(pc)):
            P_ = pc[b]
            emb = ph[p_off[b]:p_off[b + 1]]
            imp = imp_all[p_off[b]:p_off[b + 1]].cpu().numpy()
            yb = int(y[b])
            pred0 = int(logits[b].argmax())
            order_top = np.argsort(-imp)
            order_rand = rng.permutation(P_)

            def prob_after(order):
                ps = []
                for f in FRACS:
                    r = int(math.ceil(f * P_)) if f > 0 else 0
                    keep = torch.ones(P_, dtype=torch.bool, device=emb.device)
                    keep[torch.as_tensor(order[:min(r, P_ - 1)], device=emb.device)] = False
                    with torch.no_grad():
                        ps.append(float(torch.softmax(model.head(emb[keep].max(0).values[None]), 1)[0, yb]))
                return ps
            curves_top.append(prob_after(order_top))
            curves_rand.append(prob_after(order_rand))
            # flip point: smallest number of top patches whose removal changes the prediction
            flip = 1.0
            for r in range(1, P_):
                keep = torch.ones(P_, dtype=torch.bool, device=emb.device)
                keep[torch.as_tensor(order_top[:r], device=emb.device)] = False
                with torch.no_grad():
                    if int(model.head(emb[keep].max(0).values[None]).argmax()) != pred0:
                        flip = r / P_
                        break
            flips.append(flip)
            # geometry of important patches
            nodes = slice(n_off[b], n_off[b + 1])
            pl = pid_local[nodes].long()
            rad = xyz[nodes].norm(dim=1)
            cnt = torch.zeros(P_, device=rad.device).index_add_(0, pl, torch.ones_like(rad))
            prot = (torch.zeros(P_, device=rad.device).index_add_(0, pl, rad) / cnt.clamp(min=1)).cpu().numpy()
            rho_prot.append(spearman(imp, prot))
            rho_size.append(spearman(imp, cnt.cpu().numpy()))
            # node map: rank-normalised patch importance on every node of the surface
            ranks = np.empty(P_)
            ranks[order_top[::-1]] = np.linspace(0, 1, P_)
            node_maps[str(ds.items[gi][0])] = ranks[pl.cpu().numpy()].astype(np.float16)
            gi += 1
        model.zero_grad(set_to_none=True)
    ct, cr = np.array(curves_top), np.array(curves_rand)
    return {
        "n_graphs": len(flips),
        "fracs": FRACS,
        "prob_true_remove_top": ct.mean(0).tolist(),
        "prob_true_remove_random": cr.mean(0).tolist(),
        "aopc_top": float((ct[:, :1] - ct[:, 1:]).mean()),
        "aopc_random": float((cr[:, :1] - cr[:, 1:]).mean()),
        "flip_fraction_median": float(np.median(flips)),
        "flip_fraction_q25_q75": [float(np.quantile(flips, .25)), float(np.quantile(flips, .75))],
        "spearman_importance_protrusion": float(np.nanmean(rho_prot)),
        "spearman_importance_patch_size": float(np.nanmean(rho_size)),
    }, node_maps


def stage_attribute(tab, args, device):
    adir = Path(args.out) / "attribution"
    adir.mkdir(parents=True, exist_ok=True)
    res = {}
    for fam, byk in sorted(tab.items()):
        maps = {}
        for k in sorted(kk for kk in byk if isinstance(kk, int)):
            seeds = sorted(byk[k]) if args.attr_all_seeds else \
                [args.attr_seed if args.attr_seed in byk[k] else sorted(byk[k])[0]]
            for s in seeds:
                m = byk[k][s]
                tag = f"{fam}_k{k}_s{s}"
                jf, nf = adir / f"{tag}.json", adir / f"{tag}_nodemaps.npz"
                if jf.exists() and nf.exists():
                    r = json.load(open(jf))
                    z = np.load(nf, allow_pickle=False)
                    nm = {g: z[g] for g in z.files}
                else:
                    out = attribute_run(m["run_dir"], m["cfg"], args, device)
                    if out is None:
                        continue
                    r, nm = out
                    json.dump(r, open(jf, "w"), indent=2)
                    np.savez_compressed(nf, **nm)
                res.setdefault(fam, {})[f"k{k}_s{s}"] = r
                maps.setdefault(k, nm)
                print(f"  {tag}: AOPC top {r['aopc_top']:.3f} vs random {r['aopc_random']:.3f}  "
                      f"flip at {r['flip_fraction_median']:.2f} of patches", flush=True)
        # cross-scale agreement of node maps (same graphs, different k)
        ks = sorted(maps)
        if len(ks) > 1:
            common = sorted(set.intersection(*[set(maps[k]) for k in ks]))
            mat = {}
            for a, b in itertools.combinations(ks, 2):
                mat[f"{a}-{b}"] = float(np.nanmean([spearman(maps[a][g].astype(np.float32),
                                                             maps[b][g].astype(np.float32)) for g in common]))
            res.setdefault(fam, {})["cross_scale_spearman"] = mat
    json.dump(res, open(Path(args.out) / "attribution.json", "w"), indent=2)
    return res


# --------------------------------------------------------------------------- #
# stage 4: report (also: rotation test, re-scoring, multi-scale gates)
# --------------------------------------------------------------------------- #
def relabel_scores(tab, csv_path):
    import csv
    new = {r["protein_id"].replace("_pc", ""): int(r["class_id"]) for r in csv.DictReader(open(csv_path))}
    out = {}
    for fam, byk in tab.items():
        for k, bys in byk.items():
            rows = []
            for s, m in bys.items():
                ids = m["ids"].tolist()
                keep = [i for i, g in enumerate(ids) if g in new]
                if not keep:
                    continue
                lab = np.array([new[ids[i]] for i in keep])
                pr = m["probs"][keep]
                if pr.shape[1] <= lab.max():
                    continue
                rows.append(metrics(lab, pr) | {"n": len(keep)})
            if rows:
                out[f"{fam} k{k}"] = rows
    return out


def stage_report(tab, ens, attr, args):
    L = [f"# Future work, point 2: analysis of existing checkpoints\n",
         f"runs root: `{', '.join(map(str, args.runs_root))}`  include: `{args.include}`\n"]
    L.append("## Single models (test set)\n")
    L.append("| family | k | n runs | F1w | acc | macro-F1 (present classes) |")
    L.append("|---|---|---|---|---|---|")
    for fam, byk in sorted(tab.items()):
        for k in sorted(byk, key=str):
            ms_ = [metrics(m["labels"], m["probs"]) for m in byk[k].values()]
            L.append(f"| {fam} | {k} | {len(ms_)} | {ms([x['f1_weighted'] for x in ms_])} | "
                     f"{ms([x['accuracy'] for x in ms_])} | {ms([x['f1_macro_present'] for x in ms_])} |")
    if ens:
        L.append("\n## Do resolutions carry complementary information? (ensembles of equal size)\n")
        L.append("Same-k = m seeds of one resolution; cross-k = m different resolutions (one model each). "
                 "Cross-k above the best same-k means the resolutions are complementary.\n")
        L.append("| family | m | best same-k (k) | cross-k | cross − best same | disagreement same-k / cross-k |")
        L.append("|---|---|---|---|---|---|")
        for fam, r in ens.items():
            bs = r["same_k_f1w"][r["best_same_k"]]
            L.append(f"| {fam} | {r['ensemble_size']} | {bs[0]:.4f} ± {bs[1]:.4f} (k={r['best_same_k']}) | "
                     f"{r['cross_k_f1w'][0]:.4f} ± {r['cross_k_f1w'][1]:.4f} | {r['cross_minus_best_same']:+.4f} | "
                     f"{r['disagreement_same_k']:.3f} / {r['disagreement_cross_k']:.3f} |")
        L.append("\nSame-k ensemble F1w per k: " + "; ".join(
            f"**{fam}** " + ", ".join(f"k{k} {v[0]:.4f}" for k, v in r["same_k_f1w"].items())
            for fam, r in ens.items()))
    rot = [(fam, k, m) for fam, byk in tab.items() for k, bys in byk.items() for m in bys.values()
           if "probs_rot" in m]
    if rot:
        L.append("\n## Rotation test (each test surface under a fixed random rotation)\n")
        L.append("| family | k | F1w unrotated | F1w rotated (mean of rotations) | drop |")
        L.append("|---|---|---|---|---|")
        grp = defaultdict(list)
        for fam, k, m in rot:
            u = metrics(m["labels"], m["probs"])["f1_weighted"]
            r = np.mean([metrics(m["labels"], p)["f1_weighted"] for p in m["probs_rot"]])
            grp[(fam, k)].append((u, r))
        for (fam, k), v in sorted(grp.items(), key=str):
            u, r = np.mean([a for a, _ in v]), np.mean([b for _, b in v])
            L.append(f"| {fam} | {k} | {u:.4f} | {r:.4f} | {r - u:+.4f} |")
    if args.relabel_test_csv:
        rs = relabel_scores(tab, args.relabel_test_csv)
        if rs:
            L.append(f"\n## Re-scored on `{Path(args.relabel_test_csv).name}`\n")
            L.append("| family / k | n test | acc | F1w | macro-F1 (present) | bal. acc (present) |")
            L.append("|---|---|---|---|---|---|")
            for key, rows in sorted(rs.items()):
                L.append(f"| {key} | {rows[0]['n']} | {ms([r['accuracy'] for r in rows])} | "
                         f"{ms([r['f1_weighted'] for r in rows])} | {ms([r['f1_macro_present'] for r in rows])} | "
                         f"{ms([r['balanced_accuracy_present'] for r in rows])} |")
    gates = [(fam, k, m) for fam, byk in tab.items() for k, bys in byk.items() for m in bys.values()
             if "gates" in m]
    if gates:
        L.append("\n## Multi-scale models: resolution weights (mean gate) per class\n")
        for fam, k, _ in sorted({(f, k, None) for f, k, _ in gates}, key=str):
            ms_runs = [m for f, kk, m in gates if f == fam and kk == k]
            G = np.concatenate([m["gates"] for m in ms_runs])
            Y = np.concatenate([m["labels"] for m in ms_runs])
            scales = str(k).split("+")
            L.append(f"**{fam}** (scales k = {', '.join(scales)})\n")
            L.append("| class | n | " + " | ".join(f"k{s}" for s in scales) + " |")
            L.append("|---|---|" + "---|" * len(scales))
            L.append("| all | " + str(len(Y)) + " | " + " | ".join(f"{v:.3f}" for v in G.mean(0)) + " |")
            for c in sorted(set(Y.tolist())):
                g = G[Y == c].mean(0)
                L.append(f"| {c} | {(Y == c).sum()} | " + " | ".join(f"{v:.3f}" for v in g) + " |")
    if attr:
        L.append("\n## Which patches carry the class signal?\n")
        L.append("AOPC = mean drop of the true-class probability when the top patches are removed "
                 "(higher = the ranking finds the important patches); random removal is the baseline. "
                 "Flip = fraction of top patches whose removal changes the prediction (smaller = more localised).\n")
        L.append("| family | run | graphs | AOPC top | AOPC random | flip median (IQR) | ρ(importance, protrusion) | ρ(importance, size) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for fam, rr in attr.items():
            for run, r in rr.items():
                if run == "cross_scale_spearman":
                    continue
                q = r["flip_fraction_q25_q75"]
                L.append(f"| {fam} | {run} | {r['n_graphs']} | {r['aopc_top']:.3f} | {r['aopc_random']:.3f} | "
                         f"{r['flip_fraction_median']:.2f} ({q[0]:.2f}–{q[1]:.2f}) | "
                         f"{r['spearman_importance_protrusion']:+.2f} | {r['spearman_importance_patch_size']:+.2f} |")
        for fam, rr in attr.items():
            if "cross_scale_spearman" in rr:
                L.append(f"\nCross-scale agreement of node importance maps, **{fam}** (Spearman): " +
                         ", ".join(f"k{p}: {v:.2f}" for p, v in rr["cross_scale_spearman"].items()))
    (Path(args.out) / "report.md").write_text("\n".join(L) + "\n")
    print(f"report -> {Path(args.out) / 'report.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", nargs="+", required=True, help="dir(s) searched for run dirs")
    ap.add_argument("--include", default=r"gvp_h256", help="regex on run-dir names to keep")
    ap.add_argument("--exclude", default=None, help="regex on run-dir names to drop")
    ap.add_argument("--data-root", default=None,
                    help="re-point graph/patch/label paths recorded in the configs to this dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", nargs="+", default=["predict", "ensemble", "attribute", "report"],
                    choices=["predict", "ensemble", "attribute", "report"])
    ap.add_argument("--rotations", type=int, default=0, help="rotation test: N fixed random rotations")
    ap.add_argument("--relabel-test-csv", default=None,
                    help="also score every run on these test labels (e.g. SHREC2025 updated labels)")
    ap.add_argument("--ensemble-size", type=int, default=5)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--attr-seed", type=int, default=1, help="seed attributed per k")
    ap.add_argument("--attr-all-seeds", action="store_true")
    ap.add_argument("--attr-max-graphs", type=int, default=0, help="0 = whole test set")
    ap.add_argument("--max-test-graphs", type=int, default=0,
                    help="debug only: predict on the first N test graphs (0 = all)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    runs = find_runs(args.runs_root, args.include, args.exclude)
    print(f"{len(runs)} runs with a checkpoint under {args.runs_root} (include={args.include!r})")
    if not runs:
        sys.exit("no runs found: check --runs-root / --include")
    if "predict" in args.stages:
        stage_predict(runs, args, device)
    tab = load_preds(runs, args)
    ens = stage_ensemble(tab, args) if "ensemble" in args.stages else \
        (json.load(open(Path(args.out) / "ensemble.json")) if (Path(args.out) / "ensemble.json").exists() else {})
    attr = stage_attribute(tab, args, device) if "attribute" in args.stages else \
        (json.load(open(Path(args.out) / "attribution.json")) if (Path(args.out) / "attribution.json").exists() else {})
    if "report" in args.stages:
        stage_report(tab, ens, attr, args)


if __name__ == "__main__":
    main()
