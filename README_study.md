# HSNet benchmark study — Jean-Zay runbook

Max-GPU-parallel study of the hierarchical patch-pooling GNN
(`classify_hierarchical_gnn.py`). Every configuration is one independent
single-GPU training; the whole study launches with **one command**.

## The study
The Cartesian product of these axes:

| axis | default values | count |
|---|---|---|
| dataset | `shrec19_protein` | 1 |
| feature group | `geom`, `enriched` | 2 |
| patch-k | `8 16 32 64 128 256` | 6 |
| backbone | `gin sage gcn gat gatv2 pna msg_gin gvp sgformer gt gps` | 11 |
| hidden dim | `128 256 512` | 3 |
| seed | `1…10` | 10 |

→ **12 cells × 330 runs = 3,960 runs**. A *cell* = (dataset, feature, patch-k)
and is one Slurm array gridding over backbone × hidden × seed.

**Fixed recipe** (`grid_hiergin.py:FIXED`): 3 node / 2 patch layers, attn pool,
max readout, inverse-sqrt class weights, cosine LR 5e-4 + 5-epoch warmup,
label-smoothing 0.1, 150 epochs, patience 30, patch-backbone `gin`.

## Files
| file | role |
|---|---|
| `run_study.sh` | **the launcher** — all axes in one config block; submit / dry-run |
| `grid_hiergin.py` | engine: `plan` · `check` · `run` · `agg` (cell) · `agg-study` (master) |
| `slurm_grid_hiergin.slurm` | array body (1 GPU/task), reads the cell from env |

## Prerequisites (once)
```bash
cd $WORK/protein_classification && git pull

# SHREC19 shipped patch-k 16/32/64/128/256 — build the missing k8:
python build_patchassign_k64.py \
  --graph-dir $WORK/ssl_graphs_shared_proteinsplit \
  --out-dir   $WORK/ssl_graphs_shared_proteinsplit_patchassign_k8 --k 8 --workers 16
```
Data lives under `$WORK` (`DATA_ROOT`): `ssl_graphs_shared_proteinsplit[_geom_enriched]`,
`ssl_graphs_shared_proteinsplit_patchassign_k{8,16,32,64,128,256}`,
`graphs_pyg_shared_{proteins,species}`.

## Launch
```bash
# 1) preview: accounting + per-cell data preflight, submits nothing
DRY_RUN=1 ./run_study.sh

# 2) go
./run_study.sh
```
Peak GPUs ≈ `cells × CONCURRENCY` (must stay ≤ 640). Defaults: `CONCURRENCY=40`
→ ~400 GPUs, `QOS=qos_gpu-t3`, `TIME=20:00:00`. The launcher refuses to submit if
the peak exceeds 640 or any cell's data is missing.

Common overrides (env vars):
```bash
CONCURRENCY=60 ./run_study.sh                     # more GPUs (<=640/cells)
FEATURE_GROUPS="geom" ./run_study.sh              # geom only
DATASETS="shrec19_protein shrec19_species" ./run_study.sh
BACKBONES="gin gvp" ./run_study.sh                # a quick subset
```

## Monitor
```bash
squeue -u $USER
find runs/shrec19_hsnet -name final_results.json | wc -l      # done / 3960
```

## Results
Every run writes ONE self-contained `runs/<tag>/final_results.json`: full config,
`n_params`, `train_time_sec`, `sec_per_epoch`, per-epoch `history`, and
`final_test_metrics` = acc, balanced_accuracy, f1_macro, **f1_weighted**,
tpr_macro, tnr_macro, **mAP**.

A master-aggregation job runs automatically at the end; re-run anytime:
```bash
python grid_hiergin.py --mode agg-study --study-root runs/shrec19_hsnet
```
Produces, at the study root:
- `master_summary.csv` — one row per run (params, time, all metrics)
- `master_summary.json` — mean ± std over seeds, per (dataset, feature, patch-k, backbone, hidden)
- plus a `summary.{csv,json}` inside each cell.

**Download just the summaries:**
```bash
rsync -avP <jz>:'$WORK/protein_classification/runs/shrec19_hsnet/*summary*' ./
```

## Notes
- Backbones are dataset-agnostic; `gvp` is a self-contained equivariant Geometric
  Vector Perceptron conv (SO(3)-invariant outputs, xyz from the feature slice).
  `ri_gin`/`lrf_gin` are excluded (need RI edge features SHREC19 lacks).
- `sgformer/gt/gps` are the slow tail (~80–115 min/run on V100); the fast
  backbones finish in ~10–40 min. `TIME=20:00:00` leaves ample headroom.
- DataLoaders use `persistent_workers`/`prefetch_factor` (overhead-only, results
  unchanged). No AMP — kept fp32 for numerical reproducibility.
- `qos_gpu-dev` is **not** usable for the study: its ~10-job submit cap rejects
  large arrays. Use it only for one-off spot-checks (≤10 tasks).
