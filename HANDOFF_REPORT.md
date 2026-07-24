# Protein Surface Classification — Project Handoff Report

**Repository:** `protein_classification/`  ·  **Active branch:** `flatgnn-optuna-and-fixes`
**Remote:** `github.com:osamagharbi/protein_classification`
**Last updated:** 2026-07-24 (at launch of the SHREC19 backbone study)
**Cluster:** IDRIS Jean-Zay (`acu@v100`), code + data under `$WORK`.

This document is a standalone reference for anyone picking up the project. It
covers what we build, what has been established, what is running now, and what
remains. Read the **TL;DR** and **§8 (current study)** first if you only have
five minutes.

---

## 0. TL;DR

We classify **protein molecular surfaces** with a compact **two-level
(hierarchical) graph neural network** we call **HSNet** (a.k.a. HierGIN). Each
surface → a 1,000-node graph → node-level GNN → pooled into *k* geodesic patch
super-nodes → patch-level GNN → readout → MLP. It matches state of the art on
SHREC 2019 (~0.985 acc / 0.997 mAP at protein level) at ~2.8 M params.

**Right now:** a 3,960-run benchmark study is running on Jean-Zay t3
(`runs/shrec19_hsnet/`, jobs `177513–177525`), sweeping 11 backbones × 3 hidden
dims × 10 seeds × 6 patch counts × 2 feature groups on SHREC 2019 protein.

**Deliverable:** the HSNet paper (`paper_hiergin_shrec19/`, compiles, 18 pp.);
the study fills its ablation/robustness tables.

---

## 1. Scientific goal

Protein function is largely decided at the **molecular surface** (solvent-excluded
surface). The surface is a *deformable* 3D shape — the same protein visits many
conformations, each a slightly different mesh with the same identity. The task:
classify/retrieve surfaces by protein identity (and by orthologous group /
"species"). This is the setting of the SHREC protein-surface tracks (2019, 2021,
2025). Our thesis: a **hierarchical patch-pooling GNN** on a compact surface
graph captures both fine local detail and coarse global layout, and is
competitive with far heavier retrieval pipelines.

---

## 2. The model — HSNet / HierGIN

Implemented in [`classify_hierarchical_gnn.py`](classify_hierarchical_gnn.py)
(class `HierarchicalGNN`). Pipeline:

1. **Surface → graph.** Raw mesh (10k–50k verts) reduced to **N=1,000 nodes** by
   farthest-point sampling; edges = kept mesh edges + kNN edges. Node features:
   either `coords` (xyz, 3-D) or enriched descriptors (see §4).
2. **Node-level GNN** — `L_node = 3` layers over the 1,000-node graph. Backbone
   is a **plug-in** (11 supported, §6). Residual + LayerNorm + ELU + dropout.
3. **Patch pooling** — group the 1,000 nodes into **k patches** (FPS seeds +
   nearest-seed Voronoi = approximate geodesic-Voronoi partition, precomputed).
   Each patch → one super-node via **attention pooling**.
4. **Patch-level GNN** — `L_patch = 2` layers over the k-patch graph (kNN-8).
5. **Read-out + head** — global **max** read-out → 2-layer MLP → class logits.

**Winning recipe (held fixed across the study):** GIN backbone, `hidden=512`,
`node_layers=3`, `patch_layers=2`, `pool=attn`, `readout=max`, `batch=64`,
`epochs=150`, `patience=30`, `lr=5e-4` cosine + 5-epoch warmup,
`label_smoothing=0.1`, class weights `inverse_sqrt`, **no augmentation**,
patch-backbone `gin`. ~2.8 M params (2.05 M for GCN variant, up to 4.6 M for
sgformer/h512).

> **Loss note:** the loss is plain cross-entropy; `inverse_sqrt` is the *class
> weighting* (`w_c ∝ 1/√count_c`) fed into it — a gentle imbalance correction
> that beat both uniform and full-inverse weighting.

---

## 3. Datasets (under `$WORK` on Jean-Zay, `~/datasets/` locally)

| dataset | classes | split | node feats | notes |
|---|---|---|---|---|
| **SHREC 2019** | 17 (protein), 54/53 (species) | protein 4223/1058; species 4195/1049 | 3-D coords, or 16-D enriched | main benchmark; pure-geometry meshes (no chemistry) |
| **SHREC 2025** | 99 (88 official non-empty) | internal train/val/test CSVs | 8-D base (potential+normals+xyz), 14-D geom, 21-D ri | has electrostatic potential; conformation-rich |
| **SHREC 2021** | 144 train / 65 test (disjoint!) | retrieval track (GitHub rea1991) | 1000-node | **on hold** — non-separable at this resolution (§10) |

**Directory layout (SHREC19, the study's data), all at `$WORK`:**
- `ssl_graphs_shared_proteinsplit/` — base graphs (xyz), **train/test split at the
  protein level** so conformers don't leak across the split.
- `ssl_graphs_shared_proteinsplit_geom_enriched/` — same graphs + 13 geometric
  descriptors (16-D `x`).
- `ssl_graphs_shared_proteinsplit_patchassign_k{8,16,32,64,128,256}/` — patch
  assignments per k. **k8 must be built once** (SHREC19 shipped 16–256):
  ```
  python build_patchassign_k64.py --graph-dir $WORK/ssl_graphs_shared_proteinsplit \
    --out-dir $WORK/ssl_graphs_shared_proteinsplit_patchassign_k8 --k 8 --workers 16
  ```
- `graphs_pyg_shared_proteins/`, `graphs_pyg_shared_species/` — label CSVs
  (task = which CSV; graphs are shared).

**Feature groups (the study's two arms):**
- `geom` = base graphs + `--feature-mode coords` (3-D, pure geometry).
- `enriched` = geom-enriched graphs + `--feature-mode all` (SHREC19, 16-D) /
  `geom_14d_raw` (SHREC25, 14-D).

The registry mapping (dataset, feature-group, patch-k) → concrete paths lives in
`grid_hiergin.py:DATASETS`.

---

## 4. Codebase map

| file | role |
|---|---|
| [`classify_hierarchical_gnn.py`](classify_hierarchical_gnn.py) | **the model + training loop.** All backbones, HierarchicalGNN, run_epoch, mAP/TPR/TNR, per-run `final_results.json`. |
| [`classify_gnn.py`](classify_gnn.py) | shared utilities: dataset loaders, `compute_metrics` (incl. TPR/TNR), class weights, focal loss. |
| [`grid_hiergin.py`](grid_hiergin.py) | **study engine.** Modes: `plan`, `check`, `run` (task-id → config → train), `agg` (one cell), `agg-study` (master). Dataset/feature/patch registry. |
| [`run_study.sh`](run_study.sh) | **the launcher.** All axes in one config block; accounting + 640-GPU guard + per-cell preflight + manifest + chained master-agg. `DRY_RUN=1` previews. |
| [`slurm_grid_hiergin.slurm`](slurm_grid_hiergin.slurm) | array body: 1 GPU/task, reads the cell from env vars. |
| [`build_patchassign_k64.py`](build_patchassign_k64.py) | builds patch-assignment dirs for any k. |
| `optuna_search_flatgnn.py`, `slurm_flatgnn_*.slurm`, `submit_flatgnn_search.sh` | **separate prior workstream:** flat-GNN Optuna HPO for the SHREC25 benchmark paper (§7). |

**Every run writes one self-contained `final_results.json`:** full config,
`n_params`, `train_time_sec`, `sec_per_epoch`, `epochs_run`, `best_epoch`,
per-epoch `history`, and `final_test_metrics` = accuracy, balanced_accuracy,
f1_macro, **f1_weighted**, tpr_macro, tnr_macro, **mAP** (one-vs-rest AP over
softmax scores, ProCLR-style protocol).

---

## 5. Studies — what is done, running, planned

### 5.1 DONE — SHREC19 regularization + patch-count ablations (local GPU)
- 16-run SHREC19 regularization study (baseline / LS / LS+aug / LS+SupCon ×
  protein/species × GIN/GAT). **Verdict:** LS-only best on protein, LS+SupCon
  best on species; **augmentation collapses everything** (SO(3) rotation vs a
  coords-only model with no built-in rotation invariance).
- **Patch-count ablation** (k ∈ {16,32,64,128,256}, both tasks): protein has a
  broad plateau, **k=64 is the mAP optimum (0.9973)**; species prefers fewer
  patches (k=16 best), and **k=256 collapses** (~4 nodes/patch). This justified
  k=64 as the global default and is written into the paper (§7).
- **Backbone-agnostic study** (GIN/GAT/GCN/SAGE): all strong at protein level;
  GIN best overall → the two-level design (not one operator) carries the result.

### 5.2 DONE — SHREC25 geometry runs (local)
- coords-only: 73.3 % acc (low balanced-acc). `geom_14d`: **86.1 % acc / 0.704
  bal-acc / 0.697 F1m** — matches the historical 0.8646 benchmark.

### 5.3 DONE — single-run validation of the new backbones on Jean-Zay
- **gvp** (new equivariant backbone, §6) @ h256/k64/geom: **mAP 0.9931, F1w
  0.9839**, 23 min, 745 K params — competitive, confirms it learns.
- **sgformer** @ h512/k64/geom: **mAP 0.9935, F1w 0.9802**, 82 min (113 ep,
  early-stop), 4.6 M params — the slowest backbone, ~45 s/epoch.

### 5.4 RUNNING — the big SHREC19 backbone study (§8)

### 5.5 PLANNED (see §10)
Species task + SHREC25 as extra cells; SHREC21 revisit; non-technical paper tasks.

---

## 6. Backbones

11 dataset-agnostic backbones in the study:
`gin sage gcn gat gatv2 pna msg_gin gvp sgformer gt gps`.

- **`gvp`** is **new** — a self-contained equivariant Geometric Vector Perceptron
  conv we implemented (`GVP`, `GVPConvLayer`): reads xyz from the feature slice,
  does GVP-style scalar/vector message passing per layer, returns SO(3)-invariant
  scalars, drops into the DGL scalar pipeline unchanged. Smoke- and cluster-tested.
- **`sgformer`/`gt`/`gps`** = global-attention transformers → the **slow tail**
  (~80–115 min/run vs ~10–40 min for the local ops). They run a per-graph Python
  attention loop (a known optimization target, §10).
- **Excluded:** `ri_gin`, `lrf_gin` — need rotation-invariant edge features that
  SHREC19 graphs don't carry (only the SHREC25 `_ri` dataset has them).

Backbone is a plug-in at both levels; the study varies only the **node** backbone
and keeps **patch-backbone = gin**.

---

## 7. Papers (local, gitignored: `paper_*`)

- **`paper_hiergin_shrec19/`** — the **HSNet paper** (main deliverable). Compiles
  clean (~18 pp., `pdflatex ×3`). Sections 01–09 + appendix. Reports SHREC19
  (protein + species) vs published methods (ProCLR, GF-GAT, descriptor methods)
  and SHREC25. **All technical TODOs resolved** (exact param counts, corrected
  splits, weighted-F1 comparison, SSL results removed per instruction, appendix
  full-metric table, backbone-agnostic table, patch-count subsection).
  **`TODO.md` deferred (non-technical) items:** method-name macro `\M`
  (currently `HSNet`), authors/affiliation, architecture figure (placeholder),
  SHREC25 official-participant comparison, ProCLR venue verification, optional
  SHREC2021 future-work note.
- **`paper_gnn_shrec25/`** — a separate **flat-GNN vs HierGIN benchmark** paper;
  driven by the `optuna_search_flatgnn.py` cluster HPO (per-backbone TPE search
  over feature sets on the 21-D `ssl_graphs_1k_geom_enriched_ri` dataset).

Reference PDFs (FAG-MAE, ProCLR, SHREC2021 benchmark) are in the paper dir.

---

## 8. CURRENT STUDY — SHREC19 backbone benchmark (running now)

**What:** `runs/shrec19_hsnet/` — the full grid, launched via `./run_study.sh`.

| axis | values | n |
|---|---|---|
| dataset | shrec19_protein | 1 |
| feature group | geom, enriched | 2 |
| patch-k | 8 16 32 64 128 256 | 6 |
| backbone | (the 11) | 11 |
| hidden | 128 256 512 | 3 |
| seed | 1…10 | 10 |

→ **12 cells × 330 = 3,960 runs.** t3, `CONCURRENCY=40` (peak ~480 GPU ≤ 640),
`TIME=20:00:00`. Jobs **`177513–177524`** (one array/cell) + **`177525`**
(master-agg, dependency-held). ~2,300 GPU-hours estimated.

**Monitor:**
```bash
find runs/shrec19_hsnet -name final_results.json | wc -l          # done / 3960
squeue -u $USER                                                    # running/pending
for d in runs/shrec19_hsnet/*/; do echo "$(basename $d): $(find $d/runs -name final_results.json|wc -l)/330"; done
```

**Aggregate (auto at the end, or anytime for partial results):**
```bash
python grid_hiergin.py --mode agg-study --study-root runs/shrec19_hsnet
```
→ `master_summary.csv` (one row/run) + `master_summary.json` (mean±std over seeds
per dataset/feature/patch-k/backbone/hidden) + per-cell `summary.*`.

**Download just the summaries** (few files):
```bash
rsync -avP <jz>:'$WORK/protein_classification/runs/shrec19_hsnet/*summary*' ./
```

**Scheduling note:** `TIME=20:00:00` is safe but conservative; because every task
is a short 1-GPU job (≤~2 h), a tighter `TIME` (e.g. `04:00:00`) would backfill
and start much sooner. Re-launch with `TIME=04:00:00 ./run_study.sh` if queue
waits are long (see §9 for the trade-off).

---

## 9. Infrastructure & how to launch (Jean-Zay)

- **Env:** `module load pytorch-gpu/py3/2.8.0` + `source $WORK/myvenv/bin/activate`
  (needs torch, dgl, scikit-learn, tqdm, pandas). Account `acu@v100`,
  `--constraint v100-32g`, `--gres gpu:1`, `--hint nomultithread`.
- **QoS:** `qos_gpu-t3` = 20 h wall, up to **640 GPU/user** → the study runs here.
  `qos_gpu-dev` = 2 h, **32 GPU, 10 running-or-pending jobs** → **spot-checks only**
  (it *rejects* large arrays with `QOSMaxSubmitJobPerUserLimit`, and its 2 h wall
  would kill the transformer backbones). Also `acu@a100` (A100, ~2× faster,
  smaller pool) if you want speed — no code change, just account/constraint.
- **Launch:** `git pull` → build k8 → `DRY_RUN=1 ./run_study.sh` → `./run_study.sh`.
  The launcher refuses to submit if peak > 640 GPU or any cell's data is missing.
- **One GPU per task, `%CONCURRENCY` throttle** (riconv-style), **never DDP**.

---

## 10. Known issues, lessons, gotchas

1. **Augmentation collapses coords-only models.** SO(3) rotation makes the 3-D
   coordinate model much worse (no built-in rotation invariance). We train
   without augmentation. If you want rotation robustness, use an equivariant
   backbone (`gvp`) or RI features, not random rotations.
2. **`qos_gpu-dev` cannot host the study.** 10-job submit cap rejects 330-task
   arrays; 2 h wall kills sgformer/gt/gps. Use t3.
3. **Patch count matters.** k in 16–64 is the sweet spot; **k=256 degenerates**
   (~4 nodes/patch, coarse graph as noisy as the fine one). k=64 default.
4. **Slow tail = transformers** (`sgformer/gt/gps`), ~80–115 min. Optimization
   target: their **per-graph Python attention loop** could be batched with a
   block-diagonal mask. AMP (fp16) would also help ~1.3–2× but **alters numerics**
   — we kept **fp32** for reproducibility. Only results-neutral speedups are in
   (`persistent_workers`, `prefetch_factor`).
5. **SHREC21 is non-separable at 1,000 nodes.** Closed-set CE collapses to the
   majority class across every variant (native-144, merged-209, test-only-65,
   3-class). Root cause: ~2 % downsampling vs ProCLR's ~4,122 nodes; conformers of
   one protein vary more than distinct small domains at coarse resolution.
   **To revisit:** regenerate at ~4,000 nodes, and/or use **metric learning
   (prototypical networks)** given the disjoint train/test class split.
6. **mAP protocol caveat.** Our mAP is one-vs-rest AP from classifier softmax —
   close but not identical to the retrieval dissimilarity-matrix mAP of
   ProCLR/GF-GAT. Stated honestly in the paper.
7. **Label numbering must be consistent** across train/test CSVs — an earlier
   train/test class-numbering mismatch tanked test accuracy to ~0.02 until fixed
   (see `MEMORY.md` → `ri-hiernet-fails-to-generalize`).

Persistent project memory lives in
`~/.claude/projects/-home-ogharbi/memory/MEMORY.md` (SHREC-specific findings).

---

## 11. Future work / next steps

**Immediate (once the study finishes):**
1. `agg-study` → pull `master_summary.*`; build the backbone × hidden × patch-k ×
   feature tables (mean±std over 10 seeds) into `paper_hiergin_shrec19/`.
2. Decide the headline backbone/hidden/patch-k from the aggregated mAP/F1w.

**Extend the study (same launcher, edit axes):**
3. Add the **species** task: `DATASETS="shrec19_protein shrec19_species"`.
4. Add **SHREC25** cells (geom + enriched/ri) — confirm its patch dirs
   (`ssl_graphs_1k_patchassign_k{8,32,64,128}`) exist on `$WORK`.

**Paper (non-technical, deliberately deferred — see `TODO.md`):**
5. Final method name (update the single `\M` macro), authors/affiliation,
   architecture figure, SHREC25 official-participant comparison table, ProCLR
   venue check.

**Research directions:**
6. **SHREC21 revisit** at higher graph resolution and/or with prototypical-network
   metric learning (fits the disjoint-class retrieval setting).
7. Batch the transformer attention loop (perf); optionally an AMP arm if a
   speed/repro trade is acceptable (apply uniformly).
8. A100 partition for the slow tail if wall-clock matters.

---

## 12. Key numbers (quick reference)

**Our HSNet on SHREC19 (verified):**
| task | config | Acc | BalAcc | F1m | F1w | mAP |
|---|---|---|---|---|---|---|
| protein | GIN coords LS | 0.9849 | 0.9906 | 0.9916 | **0.9848** | **0.9973** |
| species | GAT coords LS+SupCon | 0.9628 | 0.9478 | 0.9254 | **0.9652** | **0.9719** |

**SHREC19 patch-count (protein, mAP / F1w):** k16 .992/.974 · k32 .994/.987 ·
**k64 .997/.985** · k128 .985/.975 · k256 .979/.945.

**SHREC25:** coords 73.3 % acc · geom_14d **86.1 % / 0.704 bal / 0.697 F1m**.

**Published comparison points:** SHREC19 protein — GF-GAT mAP .986/F1 .978,
ProCLR-GAT .983/.981. SHREC19 species — ProCLR-GAT .962/.966, GF-GAT .957/.956.

---

*End of report. For the exact training/aggregation commands see
[`README_study.md`](README_study.md); for the paper status see
`paper_hiergin_shrec19/TODO.md`.*
