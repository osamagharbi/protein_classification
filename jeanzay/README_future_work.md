# Future work on Jean-Zay: patch resolution, rotation invariance, multi-scale

Branch `future-work`. Everything runs from `$WORK/protein_classification`.
All new model options are opt-in: existing scripts, runs and checkpoints behave as before.

| Point | Question | Code | Compute |
|---|---|---|---|
| 2 | Do patch resolutions carry complementary information? Which patches carry the class signal? | `tools/ensemble_attribution.py` (existing checkpoints, inference only) | 1 GPU job, a few hours |
| 3 | Does an exactly rotation-invariant HiPatchNet hold its accuracy on rotated surfaces (FINAL SHREC2025 release)? | `--invariant`, `--test-rotations` | 60 runs |
| 4 | Does combining resolutions in one model help? | `--multi-assign-dirs`, `--fusion`, `--share-patch-gnn` | 40 runs (SHREC2019) + 20 (SHREC2025) |
| + | Does bidirectional message passing help? | `--use-bidirectional-edges` (existing flag) | 40 runs |
| 5 | All SHREC2025 experiments on the FINAL release itself (rotated train and test meshes, 98 classes) | `RELEASE=rot` graph build, study `rot25` | 1 CPU job + 10 arms x seeds |

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

### SHREC2025 inputs: rebuild from the original archives

The paper's SHREC2025 folders (`ssl_graphs_1k_geom_enriched_ri`, `ssl_graphs_1k_patchassign_k64`) have been deleted from `$WORK`. One CPU job on `prepost` rebuilds them from the organisers' original, unrotated archives:

```bash
mkdir -p logs/fw && sbatch jeanzay/build_shrec25_graphs.slurm    # log: logs/fw/S25_GRAPHS_<jobid>.out
./jeanzay/check_data.sh                                            # when it finishes
```

What the job does:
1. **Archives:** it reuses `$WORK/datasets/train_set.tar.xz` if the size matches exactly (4,449,169,028 bytes), and otherwise downloads it. `test_set.tar.xz` (1.1 GB) is downloaded to `$SCRATCH/shrec25_orig/archives`.
2. **Graphs:** it runs the original pipeline unchanged (`tools/shrec25_rebuild/`, vendored copies of the original scripts) and uses the paper's split CSVs (`data_labels/shrec25_paper`).
3. **Patch files:** k = 64, 16 and 256, so `build_patchassign.slurm` is not needed for SHREC2025.
4. **Check against the paper graphs:** it compares a 100-graph sample with fingerprints of the paper graphs. Mesh edges and patch assignments must match exactly; floats must agree within 1e-4. **The job stops if anything differs.**
5. **Safety copies:** one tarball per folder in `$STORE/fw_archives/`. After a `$SCRATCH` purge, `restore_data.sh` extracts them again.

Re-running the job is safe: finished graphs are skipped.

**Tested locally:** the rebuild on 7 sampled meshes gave exactly the paper's graphs and k=64 patch files (largest float difference 0).

**Optional quick check first** (2 graphs, about 10 minutes; it catches dependency or download problems early):

```bash
ONLY=105m_1_A_A_model1,0 KS=64 SAVE=0 sbatch jeanzay/build_shrec25_graphs.slurm
```

The full run then continues from these 2 graphs.

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

# SHREC2025: the k16 / k256 patch files come from build_shrec25_graphs.slurm (section 0)
STUDY=multiscale25 ./jeanzay/submit_fw.sh
```

Arm suffixes:
- `_shared`: one patch GNN for all resolutions, 0.69M parameters. The single-scale model trains 0.66M; its reported 0.76M includes a 99k SupCon head that is never used.
- `_sep`: one patch GNN per resolution, 1.22M parameters.
- `s25_k16`, `s25_k256` (multiscale25 only): single-resolution references. SHREC2025 otherwise only has k=64 runs.

### Is the multi-scale model implemented correctly?

`tools/verify_multiscale.py` answers this in two parts.

**Part 1, structure.** It runs on real graphs with untrained models and takes about a minute on the login node:

```bash
python tools/verify_multiscale.py --graph-dir $SCRATCH/fw_data/ssl_graphs_1k_geom_enriched_ri \
    --assign-dirs $SCRATCH/fw_data/ssl_graphs_1k_patchassign_k{16,64,256} --n 32
```

Each resolution must receive exactly the inputs of a single-scale run at that k. With one resolution, the model must give the single-scale model's logits; with three, each branch must equal a single-scale model, and the fused output must equal `head(sum_s gate_s * readout_s)`. The script also checks:
- gate weights: one per protein and resolution, summing to 1, independent of the rest of the batch;
- gradients reach the gate and every branch;
- rotation invariance, with `--invariant`;
- bidirectional edges.

All 31 checks pass on local data.

**Part 2, trained runs** (test set, a few minutes on one GPU):

```bash
srun --account=acu@v100 --qos=qos_gpu-dev --gres=gpu:1 --cpus-per-task=10 --hint=nomultithread --time=00:30:00 \
    python tools/verify_multiscale.py --skip-structure --runs-root runs/fw_multiscale25/runs --data-root $SCRATCH/fw_data
```

For each run it reports:
- accuracy with the learned gate, with each resolution alone, without each resolution, and with equal weights;
- gate statistics: mean weight per resolution, entropy, and the classes per resolution.

The report goes to `runs/fw_analysis/multiscale_check/report.md`. How to read it:
- **Gate collapse:** one resolution alone ≈ the full model, with low entropy, means the model is single-scale in practice.
- **Gate not helping:** equal weights ≈ the learned gate means the gate adds nothing.

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

## 5. SHREC2025 FINAL release (rotated meshes): every experiment again

Sections 2–4 train on our graphs of the original, unrotated surfaces. The `s25final_*` arms only change the labels and rotate the *test* set. In the FINAL release, the *training* meshes are rotated as well, each surface by its own random rotation. Study `rot25` reruns every SHREC2025 experiment on graphs built from the released rotated meshes: the official protocol.

**Inputs.** The rotated archives are already on Jean-Zay: the Mamba baselines downloaded them to `$SCRATCH/shrec25_final_rot/raw/archives/`, and the job reuses them without downloading again. One CPU job builds `ssl_graphs_1k_geom_enriched_ri_rot` (9,219 train / 2,305 test, 98 classes) and `ssl_graphs_1k_patchassign_rot_k{64,16,256}` under `$SCRATCH/fw_data`:

```bash
mkdir -p logs/fw && RELEASE=rot sbatch jeanzay/build_shrec25_graphs.slurm
grep -E "streaming|built|VERIFIED|DIFF|ready" logs/fw/S25_GRAPHS_<jobid>.out
```

The job checks the result with `fingerprint.py verify --rotated` against the paper fingerprints. Every pipeline step is rotation-invariant except xyz and normals, so it checks node counts, edge counts, and the potentials and 13 geometric descriptors.

A local check on 7 surfaces (4 compared in full, 3 through the job itself) gave these results:
- **Same surfaces:** the same 1,000 sampled vertices and the same mesh edges.
- **A pure rotation:** xyz and normals differ by a rotation, with residual about 1e-7 and angles of 125–176°.
- **Same patches:** the k=64 partitions of the 4 surfaces compared in full are identical (adjusted Rand index 1.000).
- **Node order:** about 1% of nodes swap position, from float32 near-ties in FPS, so exact hashes cannot match. Edge directions are unchanged.

**Arms** (`jeanzay/fw_grid.py`; same recipe, final labels, 5 extra test rotations for single-scale arms):

| Arm | Question |
|---|---|
| `r25_k64` | reference HiPatchNet, k=64, under the official protocol |
| `r25_so3` | + random-rotation augmentation |
| `r25_inv` | exactly rotation-invariant encoder |
| `r25_bidir`, `r25_inv_bidir` | bidirectional message passing, alone and with the invariant encoder |
| `r25_k16`, `r25_k256` | patch resolution |
| `r25_ms_gate_shared`, `r25_ms_gate_sep` | gated multi-scale k = 16/64/256 |
| `r25_flat5` | no patch stage (5 node layers): does the hierarchy still help? |

```bash
SMOKE=1 STUDY=rot25 ./jeanzay/submit_fw.sh                                     # 1 epoch per arm (10 tasks, dev QoS)
QOS=qos_gpu-t3 TIME=02:00:00 SEEDS="1 2 3" STUDY=rot25 ./jeanzay/submit_fw.sh   # 30 runs, ~30-45 min each
python jeanzay/fw_grid.py --mode agg --study rot25 --seeds 1 2 3                # -> runs/fw_rot25/summary.md
```

The analyses work the same way on these runs:

```bash
python tools/ensemble_attribution.py --runs-root runs/fw_rot25/runs --include "r25_k" \
    --stages predict ensemble report --ensemble-size 3 --out runs/fw_analysis/rot25_ensembles
python tools/verify_multiscale.py --skip-structure --runs-root runs/fw_rot25/runs --data-root $SCRATCH/fw_data \
    --out runs/fw_analysis/rot25_multiscale_check/report.md
```

## Notes

- **Override variables:** `SEEDS="1 2"`, `ARMS="s25final_inv"`, `CONCURRENCY=20`, `DATA_ROOT=...`, `DRY_RUN=1`.
- **Outputs:** in `runs/fw_<study>/runs/<run>/`, three files per run.
- **Run names:** they encode the arm and seed. Finished runs are skipped on resubmission.
- **Seeds:** 1–10, matching `runs/shrec19_recheck` and `runs/shrec_crosseval`, so comparisons with the single-scale reference are paired.
- **SHREC2025 labels:** `data_labels/` holds the FINAL-release and 28 Oct 2025 label sets in our graph ID format. They were built by `tools/make_shrec25_label_sets.py`.
