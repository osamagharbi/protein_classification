# HSNet backbone × hidden × seed GRID on Jean-Zay

Max-GPU-parallel study of the hierarchical patch-pooling GNN
(`classify_hierarchical_gnn.py`). One SLURM array task = one independent
single-GPU training (the *riconv* pattern — **not** one job spread over 4 GPUs).

## Files
| file | role |
|---|---|
| `grid_hiergin.py` | grid driver — `plan` / `run` / `agg`; dataset+feature+patch registry |
| `slurm_grid_hiergin.slurm` | array job body (1 GPU/task), reads cell from env |
| `submit_grid_hiergin.sh` | **edit the cell, run this** — submits array + chained aggregation |

## What varies vs. what is fixed
- **Grid axes** (Cartesian product → one array per patch-k): `BACKBONES` × `HIDDEN_DIMS` × `SEEDS`.
  - default backbones (11, dataset-agnostic): `gin sage gcn gat gatv2 pna gt gps sgformer msg_gin gvp`
  - default hidden dims: `128 256 512` · default seeds: `1..10` → **330 tasks / patch-k**
- **Cell** (chosen per submission): `DATASET` × `FEATURE_GROUP` × `PATCH_K_LIST`.
  `PATCH_K_LIST` may hold several k (default `8 16 32 64 128 256`); the submit
  script fires one independent 330-task array (+ agg) per k.
- **Fixed recipe** (in `grid_hiergin.py:FIXED`): 3 node / 2 patch layers, attn pool,
  max readout, inverse-sqrt class weights, cosine LR (5e-4) + 5-epoch warmup,
  label-smoothing 0.1, 150 epochs, patience 30, patch-backbone `gin`.

### Backbones
The 11th, `gvp`, is a self-contained equivariant Geometric Vector Perceptron conv
(reads xyz from the feature slice; SO(3)-invariant outputs) — no extra data files.
`ri_gin`/`lrf_gin` are still excluded: they need RI edge features SHREC19 lacks.

### patch-k = 8 needs building once (SHREC19 shipped 16/32/64/128/256)
```bash
python build_patchassign_k64.py \
  --graph-dir $WORK/ssl_graphs_shared_proteinsplit \
  --out-dir   $WORK/ssl_graphs_shared_proteinsplit_patchassign_k8 \
  --k 8 --workers 16
```
(SHREC25 already has k8.) The planner will error clearly if a patch-k dir is missing.

## Datasets & feature groups (registry in `grid_hiergin.py:DATASETS`)
| dataset | classes | geom (feature-mode) | enriched (feature-mode) |
|---|---|---|---|
| `shrec19_protein` | 17 | `ssl_graphs_shared_proteinsplit` · `coords` (3D) | `..._geom_enriched` · `all` (16D) |
| `shrec19_species` | 54/53 | same graphs, species labels | `..._geom_enriched` · `all` (16D) |
| `shrec25` | 99 | `ssl_graphs_1k` · `coords` (3D) | `ssl_graphs_1k_geom_enriched` · `geom_14d_raw` (14D) |

`PATCH_K` picks `..._patchassign_k{K}` — SHREC19: `16 32 64 128 256`; SHREC25: `8 32 64 128`.

`DATA_ROOT` = the dir that **contains** these graph dirs. On Jean-Zay that is `$WORK`
(same level as `ssl_graphs_1k_geom_enriched_ri`). Upload the SHREC19 dirs there.

## Run it
```bash
# on a Jean-Zay login node, in $WORK/protein_classification
export DATASET=shrec19_protein FEATURE_GROUP=geom
export PATCH_K_LIST="8 16 32 64 128 256"   # one array (+agg) per k
export CONCURRENCY=16                        # = max GPUs your allocation runs at once
./submit_grid_hiergin.sh
```
Preview a cell without submitting:
```bash
python grid_hiergin.py --mode plan --dataset shrec19_protein --feature-group geom \
       --patch-k 64 --data-root $WORK --verbose
```
Sweep another cell: change `DATASET` / `FEATURE_GROUP` / `PATCH_K`, re-run. Each
submission is independent, so several cells can be in the queue at once.

## Outputs — verbose but concentrated
Every run writes ONE self-contained `runs/<tag>/final_results.json` containing:
`config`, `n_params`, `n_params_trainable`, `train_time_sec`, `sec_per_epoch`,
`epochs_run`, `best_epoch`, full per-epoch `history` (the training log), and
`final_test_metrics` = **acc, balanced_accuracy, f1_macro, f1_weighted,
tpr_macro, tnr_macro, mAP**.

The chained agg job produces, at the study root:
- `summary.csv` — one row per run (params, time, all metrics)
- `summary.json` — mean ± std over seeds, grouped by backbone × hidden

**Download just these two** — the SLURM `.out` logs are redundant:
```bash
scp -r <jz>:$WORK/protein_classification/runs/grid_*/summary.csv \
        <jz>:$WORK/protein_classification/runs/grid_*/summary.json ./
```

## Notes
- `mAP` is one-vs-rest AP from softmax scores over present classes (ProCLR-style
  protocol, same as the paper appendix).
- `sgformer`, `gt`, `gps`, `pna` are heavier per epoch — the `--time` budget
  (default 12 h) and per-task isolation absorb this without blocking the fast ones.
- The grid is deterministic (index → config); no shared Optuna journal is needed
  because an exhaustive grid needs no TPE coordination — that is what makes it
  fully parallel across every allotted GPU.
