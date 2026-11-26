# Future work on Jean-Zay: patch resolution, rotation invariance, multi-scale

Branch `future-work`. Everything runs from `$WORK/protein_classification`.
All new model options are opt-in: existing scripts, runs and checkpoints behave as before.

| Point | Question | Code | Compute |
|---|---|---|---|
| 2 | Do patch resolutions carry complementary information? Which patches carry the class signal? | `tools/ensemble_attribution.py` (existing checkpoints, inference only) | 1 GPU job, a few hours |
| 3 | Does an exactly rotation-invariant HiPatchNet hold its accuracy on rotated surfaces (FINAL SHREC2025 release)? | `--invariant`, `--test-rotations` | 60 runs |
| 4 | Does combining resolutions in one model help? | `--multi-assign-dirs`, `--fusion`, `--share-patch-gnn` | 40 runs (SHREC2019) + 20 (SHREC2025) |
| + | Does bidirectional message passing help? | `--use-bidirectional-edges` (existing flag) | 40 runs |

## 0. Get the code

```bash
cd $WORK/protein_classification
git status                      # commit or stash any local edits first
git fetch origin && git checkout future-work && git pull
chmod +x jeanzay/*.sh
./jeanzay/restore_data.sh       # links or extracts every graph folder into $SCRATCH/fw_data
./jeanzay/check_data.sh         # confirms every input is present
```

About `restore_data.sh`:
- **What it does:** it looks in `$WORK`, `$SCRATCH` and `$STORE`. A graph folder that already exists is symlinked; one that exists only as an archive (`<name>.tar[.gz|.zst]`) is extracted. It reads the archive's layout first, so it does not matter how the archive was made.
- **Defaults:** once `$SCRATCH/fw_data` exists, every fw script uses it automatically, so `$WORK` inodes are not used. `DATA_ROOT=...` overrides this.
- **Scratch purge:** `$SCRATCH` deletes files not accessed for 30 days. If that happens, rerun the script; the archives stay in `$WORK`.
- **Missing data:** a dataset it cannot find is reported as `MISSING`. The planner also refuses to submit an arm whose data is missing, so submit only the arms you have, e.g. `ARMS="s19p_inv s19s_inv"`.

## 1. Point 2: analysis of existing checkpoints

```bash
./jeanzay/extract_checkpoints.sh          # streams runs_20260907.tar.zst -> $SCRATCH/fw_ckpts (~5 min)
sbatch jeanzay/fw_analysis.slurm          # -> runs/fw_analysis/{shrec19_protein,shrec19_species,shrec25}/report.md
```

What the report contains:
- **Ensembles:** for each m, it compares m seeds of one resolution against m different resolutions (equal m).
  If cross-k beats the best same-k, the resolutions are complementary, which motivates point 4.
- **Patch attribution:** gradient × input on the final patch embeddings, which is exact for the max readout. Checks:
  - deletion curves (top vs random patches);
  - the fraction of patches that must be removed to flip the prediction;
  - correlation of importance with protrusion and with patch size;
  - cross-scale agreement of the importance maps.
  Per-node importance maps are saved for visualisation in `attribution/*_nodemaps.npz`.
- **Rotation test** of the current models: how much they lose when the test surfaces are rotated.
- **SHREC2025 re-scored** on the organisers' 28 Oct 2025 test labels (the papers used the original 2,311-surface labels).

## 2. Point 3: exact rotation invariance

```bash
SMOKE=1 STUDY=rotation ./jeanzay/submit_fw.sh     # 1 epoch per arm on the dev QoS: checks the whole chain
STUDY=rotation ./jeanzay/submit_fw.sh             # 6 arms x 10 seeds
python jeanzay/fw_grid.py --mode agg --study rotation   # -> runs/fw_rotation/summary.md
```

Arms:

| Arm | Data | Model |
|---|---|---|
| `s19p_inv`, `s19s_inv` | SHREC2019 protein / species | invariant HiPatchNet |
| `s25_inv` | SHREC2025, paper labels | invariant HiPatchNet |
| `s25final_base` | SHREC2025 FINAL labels (98 classes, 2,305 test) | current model |
| `s25final_so3` | SHREC2025 FINAL labels | current model + random-rotation augmentation |
| `s25final_inv` | SHREC2025 FINAL labels | invariant HiPatchNet |

Every run also reports its test metrics under 5 fixed random rotations (`F1w rotated`).

How the invariant model works:
- **Node input:** only invariant channels (potentials, the 6 geometric descriptors, distance to the centre).
- **Geometry:** coordinates enter only as relative geometry, and normals enter as GVP vectors.
- **Patches:** already invariant (FPS distances, geodesic Voronoi).
- **Check:** under rotation the logits change by about 1e-7, against up to 0.3 for the current model.

Why there are no rotated graphs: the FINAL release meshes are rotated copies of our surfaces. For the invariant model, the predictions are therefore the same on our graphs. The current models are scored on rotated versions through `--test-rotations`.

For an official submission to the organisers, build the graphs from the rotated meshes, which gives the same prediction for the invariant model.

## 3. Point 4: multi-scale model

```bash
STUDY=multiscale19 ./jeanzay/submit_fw.sh         # k = 16, 64, 256; gate fusion; shared vs separate patch GNN
python jeanzay/fw_grid.py --mode agg --study multiscale19

# SHREC2025 needs k16 / k256 patch files (about 23k inodes; the builder first checks
# that it reproduces the existing k64 files exactly)
sbatch jeanzay/build_patchassign.slurm
STUDY=multiscale25 ./jeanzay/submit_fw.sh
```

Arm suffixes:
- `_shared`: one patch GNN for all resolutions, nearly the single-scale parameter count (0.68M vs 0.75M).
- `_sep`: one patch GNN per resolution (1.2M).

Compare against the single-scale k64 recheck runs (same seeds 1–10).
To see which resolution each class relies on (gate weights per class), run:

```bash
python tools/ensemble_attribution.py --runs-root runs/fw_multiscale19/runs --include . \
    --stages predict report --out runs/fw_analysis/multiscale19
```

## 4. Bidirectional message passing

The reported models pass messages along each stored edge in one direction only. `edge_index_mesh` stores every undirected edge once, from the lower to the higher node index. As a result, 9.4% (SHREC2019) and 24.3% (SHREC2025) of nodes receive no neighbour message.

```bash
STUDY=edges ./jeanzay/submit_fw.sh     # s19p/s19s/s25 reference + bidirectional edges, and invariant FINAL + bidirectional
python jeanzay/fw_grid.py --mode agg --study edges
```

## Notes

- **Override variables:** `SEEDS="1 2"`, `ARMS="s25final_inv"`, `CONCURRENCY=20`, `DATA_ROOT=...`, `DRY_RUN=1`.
- **Outputs:** in `runs/fw_<study>/runs/<run>/`, three files per run.
- **Run names:** they encode the arm and seed. Finished runs are skipped on resubmission.
- **Seeds:** 1–10, matching `runs/shrec19_recheck` and `runs/shrec_crosseval`, so comparisons with the single-scale reference are paired.
- **SHREC2025 labels:** `data_labels/` holds the FINAL-release and 28 Oct 2025 label sets in our graph ID format. They were built by `tools/make_shrec25_label_sets.py`.
