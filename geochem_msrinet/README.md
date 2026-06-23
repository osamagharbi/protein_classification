# GeoChem-MSRINet

Pure-PyTorch implementation of the max-performance architecture proposed in
`MAX_PERFORMANCE_ARCHITECTURE_DESIGN.md`.

The folder is intentionally separate from the existing HierGIN scripts. It uses
the current SHREC25 `.pt` graph tensors as dense point clouds:

- `pos`
- `normals`
- `scalars_norm`
- optional raw 14D geometric features

No DGL, PyG, torch-cluster, torch-scatter, or PyTorch3D dependency is required.
The tradeoff is speed: the current version uses dense `torch.cdist` operations.

## Files

```text
geochem_msrinet/
  data.py        dataset, label CSV loading, augmentation
  ops.py         pure torch FPS, radius query, PPF features
  model.py       GeoChem-MSRINet model
  losses.py      CE, cosine-margin CE, SupCon, prototype memory
  metrics.py     macro-F1/balanced-accuracy helpers
  train.py       training driver
  eval.py        checkpoint evaluation
  smoke_test.py  small forward + rotation-invariance check
```

## Smoke Test

```bash
cd /home/ogharbi/projects/protein_classification
python -m geochem_msrinet.smoke_test --device cpu
```

## Fast Training Sanity Run

```bash
cd /home/ogharbi/projects/protein_classification
python -m geochem_msrinet.train \
  --device cuda:0 \
  --epochs 2 \
  --batch-size 2 \
  --limit-train 64 \
  --limit-val 32 \
  --limit-test 32 \
  --phase geom \
  --run-name geochem_smoke
```

## First Real Run

Start with geometry-only to verify that the multi-scale RI encoder is learning:

```bash
python -m geochem_msrinet.train \
  --device cuda:0 \
  --phase geom \
  --epochs 80 \
  --batch-size 8 \
  --amp \
  --rotate \
  --jitter 0.005 \
  --run-name geochem_msri_geom_1k
```

Then run the curriculum:

```bash
python -m geochem_msrinet.train \
  --device cuda:0 \
  --phase curriculum \
  --geom-epochs 40 \
  --epochs 120 \
  --batch-size 8 \
  --amp \
  --rotate \
  --jitter 0.005 \
  --supcon-weight 0.10 \
  --run-name geochem_msri_curriculum_1k
```

## Evaluate

```bash
python -m geochem_msrinet.eval \
  --checkpoint runs/geochem_msri_curriculum_1k/best_model.pt \
  --split test \
  --batch-size 8
```

## Notes

- The existing dataset has 1,000 FPS points per protein, so the current default
  is the 1k architecture schedule.
- The model code supports larger point counts, but a larger point source must be
  available first.
- `--use-geom-raw` adds the six globally normalized geometry features from the
  successful 14D run.
- `--proto-weight` is implemented but should be tuned later; leave it at `0.0`
  until the geometry and chemistry phases are stable.
