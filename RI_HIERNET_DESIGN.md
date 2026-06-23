# RI-HierNet: A Ground-Up Architecture for SHREC25 Protein Surface Classification

**Design synthesis from the full experimental record (HierGIN ablations, RI/LRF/MSG
backbones, three-level pooling, separability analysis, and the RIConv2 comparison).**

Date: 2026-06-19
Status: Design proposal (not yet implemented)

---

## 1. Why a new architecture (the one decisive finding)

Every experiment in this project converges on a single conclusion:

> **The bottleneck is architecture, not features.**

The hard evidence:

| Method | Features | Macro-F1 | Bal.Acc |
|---|---|---|---|
| HierGIN 8D | xyz+normals+Pot+NormPot | 0.798 | 0.800 |
| HierGIN 14D (global norm) — *our best* | 8D + 6 geometric | **0.808** | 0.812 |
| HierGIN 14D + RI-GIN | global norm | 0.793 | 0.807 |
| HierGIN 8D + LRF-GIN | precomputed PCA frames | 0.766 | 0.776 |
| HierGIN 14D + MSG-GIN (multi-k) | global norm | ~0.07 (failing) | ~0.09 |
| HierGIN 14D + 3-level | 1000→128→32 | 0.016 | 0.023 |
| **RIConv2 RUN1** | **xyz + normals only** | **0.838** | **0.813** |
| **RIConv2 RUN2** | xyz + normals + Pot | **0.854** | **0.828** |

RIConv2 using **geometry only — no chemistry at all** — beats our best 14D model by
**+3.0 pp macro-F1**. On the hardest class (61) it scores F1 = 0.887 vs our 0.576.
Our own separability analysis showed the kNN-6 *edge marginals* for pair (61,90) are
near-identical (Sliced-Wasserstein = 0.004), yet RIConv2 separates them cleanly from
the same surface. **The discriminating signal lives in multi-scale local patch geometry
that a kNN-6 graph cannot see.**

So the new architecture is not "more features" or "more layers." It is built to extract
the one thing every HierGIN variant was blind to: **rotation-invariant, multi-scale,
fixed-radius local surface descriptors** — while keeping the two things HierGIN proved
*do* help: **hierarchical pooling (+15.8 pp over flat GNN)** and **electrostatic
potential on top of a strong geometric backbone (+1.6 pp in RIConv2)**.

---

## 2. Evidence base — every design choice traces to a result

| # | Finding | Source | Design consequence |
|---|---|---|---|
| F1 | RIConv2 geom-only (0.838) > HierGIN-14D (0.808) | results table | **Adopt multi-scale ball-query RI grouping as the core.** |
| F2 | kNN-6 edge marginals inseparable (SW=0.004) but surface IS separable | separability §; RIConv2 cls61 F1=0.887 | Signal is in *local groups at fixed 3D radius*, not pairwise edges. Use **ball query, not kNN**. |
| F3 | Hierarchical pooling beats flat GNN +15.8 pp (0.798 vs 0.640) | analysis §9.3 | **Keep a coarse patch-level stage** for global spatial reasoning. |
| F4 | Class signal lives in the *full mesh structure*; coarsened/pooled features fail (~8–10%) | memory: shrec25-class-signal | Never classify from pooled distributions alone; keep fine geometry until late. |
| F5 | Precomputed PCA-LRF frames unstable on flat surfaces → −3.4 pp | LRF-GIN result | **Compute RI features on-the-fly with PPF (no eigendecomposition).** |
| F6 | Multi-k kNN (MSG-GIN) adds redundant connectivity, no new signal → fails | current run, 0.07 F1 | Varying *k* ≠ varying *scale*. Scale must be a **metric radius in 3D**. |
| F7 | Independent FPS for 1000→128→32 collapses (0.016) | three-level result | Multi-level pooling **must be nested**; reuse SA-hierarchy centroids. |
| F8 | Per-protein z-scoring erases size signal −5.6 pp | 14D z-score result | **Global normalization only**; size encoded via raw distances. |
| F9 | Patch-mean scalar re-injection hurts −1.8 to −3.0 pp | injection runs | **Do not re-inject raw scalars** at the patch level. |
| F10 | Potential dominant in HierGIN (−11.8 pp) but only +1.6 pp in RIConv2 | feature ablation | Chemistry is a **secondary refinement on a strong geometric backbone**, not the foundation. |
| F11 | NormalPotential critical for class 61 (−14 pp there) | ablation | Keep NormPot; it resolves a specific confusion. |
| F12 | Class 80 (14 test samples) F1≈0.22–0.24 for *all* methods incl. RIConv2 | per-class table | Data scarcity, not architecture. Needs a **few-shot / metric head**, not more capacity. |
| F13 | RI backbone = representation regularizer (+0.2 pp), not feature enrichment | analysis §9.5 | RI is worth it for *robustness*, gains come from the *multi-scale grouping*, not invariance alone. |
| F14 | FPS-1k preserves separability vs full-res VTK (SVM +0.02 only) | resolution analysis | 1024 points is enough; denser sampling is a minor ablation knob. |

---

## 3. Architecture: RI-HierNet

```
                          INPUT: 1024 surface points
        pos (x,y,z, global-normed)  ·  normals (nx,ny,nz)  ·  chem (Pot, NormPot)
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        │   STAGE A — Multi-Scale RI Set Abstraction (the RIConv2 core) │
        │                                                              │
        │   SA1  r=0.10  1024→512   PPF-RI grouping → MLP → max/mean   │
        │   SA2  r=0.20   512→256   PPF-RI grouping → MLP → max/mean   │
        │   SA3  r=0.40   256→128   PPF-RI grouping → MLP → max/mean   │
        │   SA4  r=0.80   128→ 64   PPF-RI grouping → MLP → max/mean   │
        │                                                              │
        │   FPS centroids are NESTED across layers (solves F7 for free)│
        └──────────────────────────────┬──────────────────────────────┘
                                       │  64 enriched local descriptors (= patches)
        ┌──────────────────────────────┴──────────────────────────────┐
        │   STAGE B — Patch-level GNN (the HierGIN gain, F3)           │
        │   kNN-8 graph over the 64 SA centroids                       │
        │   2× GIN layers  +  edge = relative geometry of centroids    │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        │   STAGE C — Readout + Heads                                  │
        │   global readout = [max ‖ attention-pool]  → 1024-D embed    │
        │   ├─ Linear head      → 99 class logits (CE + label smooth)  │
        │   └─ Projection head  → 128-D  (SupCon on hard pairs +       │
        │                                  prototype memory for rare)  │
        └──────────────────────────────────────────────────────────────┘
```

### 3.1 Input representation

- **1024 points** via FPS from the ~10,676-vertex VTK mesh (default; ablate 2048/4096 per F14).
- Per point: `pos ∈ R³` (centroid→origin, scaled to unit ball — **global per-protein, not z-scored**, F8), `normal ∈ R³`, `chem = [Potential, NormalPotential] ∈ R²`.
- **xyz is kept in raw normalized metric form** so ball-query radii are physically meaningful and protein size survives (F8).
- The 6 extra geometric features (curvatures etc.) from the 14D set are **optional**; the RI-SA layers recompute most local geometry, so they are an ablation knob, not a default.

### 3.2 Stage A — Multi-Scale RI Set Abstraction (core, addresses F1/F2/F5/F6)

This is the part HierGIN never had. Each SA layer:

1. **FPS-downsample** the current point set to `N'` centroids. Because each layer samples
   from the previous layer's points, the centroid hierarchy is **automatically nested**
   (1024⊃512⊃256⊃128⊃64) — this is exactly what the three-level pooling failure (F7) was
   missing, obtained here for free.
2. **Ball query** at fixed radius `r` (metric, in unit-ball coords): gather up to `K=32`
   neighbors within `r` of each centroid. **Fixed radius, not fixed k** — this is the
   distinction MSG-GIN missed (F6). Small-radius layers see bumps/grooves; large-radius
   layers see the global fold.
3. **On-the-fly PPF rotation-invariant features** per (centroid `c`, neighbor `j`) — *no
   PCA, no precomputation* (avoids the LRF-GIN instability, F5):

   ```
   d   = pos_j − pos_c
   φ_1 = ‖d‖                            (distance — encodes size, F8)
   φ_2 = ∠(n_c, d)                      (centroid normal vs displacement)
   φ_3 = ∠(n_j, d)                      (neighbor normal vs displacement)
   φ_4 = ∠(n_c, n_j)                    (normal–normal)
   φ_5 = ‖d‖ / r                        (normalized radial position within scale)
   φ_6 = n_c · (n_j × d̂)                (chirality / handedness — signed triple product)
   ```

   **6 features** (implemented). All are invariant to proper SO(3) rotation + translation
   by construction — they use only **relative** distances and angles, no covariance
   eigendecomposition (the LRF-GIN failure mode). The chirality term is a pseudoscalar:
   invariant under rotation, sign-flipped under reflection — *desirable*, since proteins
   are chiral and a mirror image is a different molecule.

   > **Implementation note.** A 7th candidate, ∠(n_c, d_grp) (angle to the group
   > centre-of-mass), was **dropped**: the centroid sits at the centre of its own ball, so
   > d_grp ≈ 0 and its normalised direction is float-noise — empirically *not*
   > rotation-invariant (Δ≈1.4 vs ~1e-15 for the other six). The numerical SO(3) gate in
   > `ri_hiernet.py` caught this; with it removed the model is invariant to 3.5e-6 (fp32)
   > / 2.8e-9 (fp64).

4. **Relative chemistry** within the group: `Δchem = chem_j − chem_c` (centered, so no raw
   scalar re-injection at coarse scale — respects F9). Concatenate `[φ(7) ‖ Δchem(2) ‖
   h_j_prev]`.
5. **Shared MLP** `(7+2+C) → C'` then **symmetric aggregation** `max ‖ mean` over the K
   neighbors → centroid feature `h_c ∈ R^{C'}`.

**Layer schedule** (radii in unit-ball units):

| Layer | radius r | N→N' | K | out dim C' |
|---|---|---|---|---|
| SA1 | 0.10 | 1024→512 | 32 | 64 |
| SA2 | 0.20 | 512→256 | 32 | 128 |
| SA3 | 0.40 | 256→128 | 32 | 256 |
| SA4 | 0.80 | 128→64 | 32 | 512 |

Radii are **logarithmically spaced** to cover bump-scale through whole-protein scale,
the four-scale structure RIConv2 used to reach 0.838. Tune via grid in ablation.

### 3.3 Stage B — Patch-level GNN (retains the +15.8 pp hierarchy gain, F3)

The 64 SA4 centroids are the "patches" — but unlike HierGIN's Voronoi patches, each one
already carries a **multi-scale RI descriptor**, not a pooled feature average.

- Build a **kNN-8 graph** over the 64 centroid positions.
- **2× GIN layers** with edge features = relative centroid geometry `[‖Δpos‖, ∠(n_a,n_b)]`
  (kept RI). This is the coarse spatial-reasoning stage that lets the model reason about
  *the arrangement of surface regions* (a charged patch next to a groove) — the capability
  flat GNNs lack (F3).
- Patch depth stays at **2** — deeper patch GNN hurt in prior runs (memory:
  hierarchical-gnn-shrec25-negatives, "patchdepth3 hurts").

### 3.4 Stage C — Readout and heads

- **Readout** = concatenate `max-pool ‖ attention-pool` over the 64 patch nodes → ~1024-D
  graph embedding. (meanmax/attn were the best HierGIN readouts.)
- **Classification head**: `Linear(1024→512) → GELU → Dropout(0.3) → Linear(512→99)`.
- **Projection head** (training only): `Linear(1024→128)`, L2-normalized, for the auxiliary
  losses in §4.3.

**Parameter budget**: ~6–9 M (comparable to current HierGIN 8 M, fits batch 64 on the
local GPU; RI-GIN-14D was forced to batch 32 and lost 1.5 pp — RI-HierNet avoids that
because PPF features are cheap and there is no full-mesh edge-MLP blow-up).

---

## 4. Training strategy

### 4.1 Normalization & features (locks in F8, F10, F11)

- **Global per-protein normalization** of positions (centroid→origin, unit ball). **No
  per-protein z-scoring** of scalars (cost us −5.6 pp).
- Potential / NormalPotential standardized with **dataset-global** mean/std.
- **Two-phase feature curriculum** (mirrors RIConv2 RUN1→RUN2, +1.6 pp):
  - **Phase 1 — geometry-only pretrain.** Train Stages A+B+C with `chem` zeroed. This forces
    the RI-SA core to learn the geometric signal that *is* the primary discriminator (F10).
    Target ≈ RIConv2 RUN1 (0.838).
  - **Phase 2 — chemistry fine-tune.** Unfreeze `Δchem` inputs and continue. Potential +
    NormalPotential then refine the confusions they uniquely fix — esp. class 61 (F11).
    Target ≈ RIConv2 RUN2 (0.854) or better.

### 4.2 Optimization (proven settings, unchanged)

| Hyperparameter | Value | Justification |
|---|---|---|
| Optimizer | AdamW | standard |
| LR | 5e-4 | best HierGIN |
| Weight decay | 1e-4 | best HierGIN |
| Schedule | cosine, 5-epoch warmup | beat step LR |
| Batch size | 64 | RI-HierNet fits (vs RI-GIN's forced 32) |
| Epochs / patience | 120 / 30 | SA core converges slower than GIN |
| Dropout | 0.3 | best HierGIN |
| Label smoothing | 0.1 | best HierGIN |
| Class weights | inverse_sqrt | best HierGIN |

### 4.3 Loss (targets the confusion pairs directly)

```
L = L_CE(label_smooth=0.1, weight=inverse_sqrt)
  + λ_sc · L_SupCon(projection head)        # λ_sc ≈ 0.2
  + λ_pp · L_prototype(rare classes)         # λ_pp ≈ 0.1, phase 2 only
```

- **Supervised contrastive (SupCon)** on the 128-D projection: explicitly **pulls apart
  (61,90), (70,56), (86,14)**. Justified because RIConv2 proves these pairs *are* separable
  from geometry (F2) — a contrastive term gives the gradient a direct lever to realize that
  separation rather than hoping CE finds it.
- **Prototype / metric loss for rare classes** (class 80 et al., F12): maintain a running
  class-prototype memory; classify rare-class samples by cosine distance to prototypes. This
  is the only lever for class 80, which is **data-limited, not architecture-limited** — all
  methods including RIConv2 plateau at F1≈0.22–0.24, so capacity won't help; few-shot might.

### 4.4 Augmentation (turns RI invariance into free regularization)

- **Random SO(3) rotation** per sample. The architecture is RI by construction, so this
  costs nothing in expressivity but acts as a strong consistency regularizer and validates
  the invariance empirically.
- **FPS reseed each epoch**: different FPS seed → different 1024-point subsample. Directly
  addresses the "fixed FPS seed" limitation noted in the report and multiplies effective
  data.
- **Point jitter** (Gaussian σ≈0.01 on positions), **random point dropout** (≤10%).
- **No feature mixup** on chemistry — Potential is a sharp discriminator (F10), mixup would
  blur it.

---

## 5. What we explicitly do NOT do (negative results as hard constraints)

| Anti-pattern | Why it's banned | Evidence |
|---|---|---|
| Precompute LRF/PCA local frames | unstable on flat surfaces, uncorrectable | LRF-GIN −3.4 pp |
| Vary *k* in kNN and call it multi-scale | adds redundant edges, no metric scale | MSG-GIN ~0.07 |
| Independent FPS per pooling level | non-nested → incoherent mapping | 3-level 0.016 |
| Per-protein z-score scalars | erases protein-size signal | 14D z-score −5.6 pp |
| Re-inject patch-mean raw scalars | consistently regresses | injection −1.8…−3.0 pp |
| Classify from coarsened/pooled distributions only | class signal is in fine structure | memory F4, ~8–10% |
| Patch GNN depth > 2 | overfits/regresses | memory: patchdepth3 hurts |
| Lead with chemistry as the foundation | geometry is primary when extracted well | RIConv2 RUN1 0.838 |

---

## 6. Expected performance & ablation plan

**Primary hypothesis**: RI-HierNet ≥ RIConv2 RUN2 (macro-F1 ≈ 0.854), because it combines
RIConv2's multi-scale RI core with the patch-level spatial reasoning RIConv2 lacks (+15.8 pp
of which HierGIN already demonstrated over flat models) and the full chemical feature set.

**Realistic milestones:**
1. Phase-1 geometry-only RI-SA (no Stage B) ≈ **0.83–0.84** (reproduce RIConv2 RUN1).
2. + Stage B patch-GNN ≈ **0.84–0.86** (the hierarchy gain on a strong core).
3. + chemistry phase 2 ≈ **0.85–0.87**.
4. + SupCon on hard pairs: target cls-61 F1 from 0.576 → **>0.85**.
5. Class 80 remains the ceiling (data-limited); prototype head is exploratory.

**Ablation grid (in priority order):**
1. **Ball-query radius schedule** {0.1/0.2/0.4/0.8} vs {0.05/0.15/0.3/0.6} vs 3-scale — the
   single most important knob (it *is* the core contribution).
2. **Ball query vs kNN** at matched neighbor count — direct test of F2/F6.
3. **With/without Stage B patch-GNN** — isolates the hierarchy gain on the new core.
4. **PPF-RI vs raw relative xyz** in the SA layer — isolates the invariance contribution.
5. **Geometry-only vs +chem vs two-phase curriculum** — replicates RIConv2 RUN1/RUN2 logic.
6. **SupCon λ_sc** {0, 0.1, 0.2, 0.5} — confusion-pair separation.
7. Point count {1024, 2048, 4096} (F14 says minor).

---

## 7. Implementation roadmap

1. **Data layer**: reuse the existing 1024-point FPS pipeline, but **store raw normalized
   xyz + normals + chem per point** (not the precomputed kNN-6 graph). Ball query and FPS
   run on-the-fly per batch on GPU. *No new precomputed dataset needed* — this sidesteps the
   patchassign-build dependency entirely.
2. **`ri_sa_layer.py`**: FPS + ball-query (`torch_cluster.fps`, `ball_query`) + PPF feature
   computation + shared MLP + symmetric pool. ~150 lines.
3. **`ri_hiernet.py`**: stack 4 SA layers → build kNN-8 patch graph on SA4 centroids → 2×
   DGL GIN → readout → heads. Reuse existing readout/head code from `classify_hierarchical_gnn.py`.
4. **Losses**: add SupCon (`pytorch-metric-learning` or hand-rolled) and prototype memory.
5. **Train script**: extend the current launcher; add `--phase {geom,chem,joint}` and the
   two-phase curriculum. Keep the proven optimizer block verbatim.
6. **Eval**: identical 88-class protocol (`labels=sorted(valid_classes)`), so numbers are
   directly comparable to the report table.

**Critical-path risk**: the PPF feature construction and ball-query batching must be correct
and fast. Validate on a 200-sample subset first; confirm SO(3) rotation of an input leaves
the embedding unchanged (numerical RI check) before any full run.

---

## 8. One-paragraph summary

RI-HierNet replaces HierGIN's blind spot — its single-scale kNN-6 node stage — with the
proven winner from the cross-architecture comparison: **multi-scale, fixed-radius,
rotation-invariant ball-query grouping (PPF features, computed on-the-fly to avoid the
LRF instability that sank LRF-GIN)**. It keeps the two things our own ablations proved
help — **a 2-layer patch-level GNN for coarse spatial reasoning (+15.8 pp over flat)** and
**electrostatic potential layered onto a strong geometric backbone (RIConv2's +1.6 pp)** —
and adds **supervised-contrastive separation of the confusion pairs** (which RIConv2 proved
are geometrically separable) plus a **few-shot prototype head for the data-starved class 80**.
The SA hierarchy's nested FPS centroids give correct multi-level pooling for free, fixing the
three-level failure. Every choice is bound to a measured result, and every known
negative-result trap is encoded as an explicit constraint.
