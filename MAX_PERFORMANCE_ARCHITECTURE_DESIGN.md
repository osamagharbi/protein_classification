# Max-Performance Architecture Design for SHREC25 Protein Surface Classification

Date: 2026-06-19  
Status: implemented prototype and initial diagnostic runs completed on 2026-06-20  
Primary objective: exceed the current HierGIN ceiling by replacing the single-scale kNN graph bottleneck with a multi-scale, rotation-invariant surface encoder.

---

## 1. Executive Decision

The next model should not be another HierGIN variant. The evidence says the main bottleneck is not the scalar feature set, the loss weight, or patch count. The bottleneck is the first representation stage:

- Current HierGIN sees a 1k FPS point set through a sparse kNN-6 graph.
- RI-GIN made the edge messages invariant, but it still used the same single-scale kNN-6 neighborhood.
- LRF-GIN failed because PCA local frames are unstable on flat surface regions.
- Three-level pooling failed because patch assignments were independent rather than nested.
- Patch scalar reinjection failed because the information was already available and the projection disturbed learned patch embeddings.
- RIConv2 geometry-only results show that local multi-scale surface geometry is highly informative when extracted correctly.

The proposed architecture is:

## GeoChem-MSRINet

Full name: **Geometry-Chemistry Multi-Scale Rotation-Invariant Network**.

Core idea:

1. Use raw point-surface data, not a fixed kNN-6 graph, as the primary input.
2. Extract local shape with multi-scale ball-query neighborhoods and on-the-fly point-pair features.
3. Keep chemistry as a local relative field, not as late patch-mean reinjection.
4. Build a nested hierarchy of centroids so multi-level pooling is coherent.
5. Add a patch-token graph transformer for global fold reasoning.
6. Use metric/prototype losses to help rare classes and hard pairs.
7. Use FPS test-time augmentation and multi-seed ensembling for maximum final performance.

Expected target:

- Single model: `0.855-0.875` macro-F1 if the RI multi-scale encoder is implemented correctly.
- 5-seed ensemble with FPS TTA: `0.870+` is the practical target.
- Class 80 may remain the main ceiling because it is data-scarce across all tested methods.

These are targets, not guarantees. The design is the highest-probability path based on the experiments so far.

---

## 2. Problems This Architecture Must Solve

### 2.1 Single-scale graph blindness

HierGIN uses a kNN-6 graph. This captures tiny local adjacency but not metric neighborhoods at different surface scales. RIConv2's advantage appears to come from fixed-radius ball-query grouping at multiple scales.

Design answer:

- Replace kNN-6 node processing with multi-scale radius neighborhoods.
- Use physical radius in normalized protein coordinates, not only neighbor count.

### 2.2 Rotation and frame dependence

Global xyz and normals are frame-dependent. Standard GIN can learn approximate invariance, but it wastes capacity.

Design answer:

- Do not feed raw xyz/normals as ordinary channels to the main encoder.
- Use xyz only for grouping and relative distances.
- Use normals only inside rotation-invariant point-pair features.

### 2.3 PCA local frame instability

LRF-GIN underperformed because local reference frames from PCA become unstable on near-flat surface patches.

Design answer:

- No precomputed PCA/LRF frames.
- Compute point-pair features on the fly from distances, dot products, and cross-product chirality.

### 2.4 Independent pooling failure

The failed 1000->128->32 experiment used independent FPS assignments, so mid-patches and final patches had no clean parent-child relation.

Design answer:

- Use nested FPS set abstraction: 2048->1024->512->256->128->64.
- Every lower-resolution point is sampled from the previous resolution.

### 2.5 Long-tail and hard-pair errors

Cross-entropy alone under-serves rare classes and does not explicitly separate confusable pairs.

Design answer:

- Keep inverse-square-root CE weights.
- Add supervised contrastive loss on hard classes.
- Add a prototype head for rare classes.
- Use a two-stream batch sampler during fine-tuning.

---

## 3. Input Representation

### 3.1 Raw input

Use the full VTK mesh or the current SSL point dataset as source. The model input should be a point set, not a prebuilt graph.

Default point count:

| Mode | Points | Use |
|---|---:|---|
| fast/debug | 1024 | reproduce RIConv2-scale behavior quickly |
| default | 2048 | best single-model tradeoff |
| max-performance | 4096 | final ablation only, with smaller batch |

Recommendation:

- Start with 2048 points.
- If memory is tight, run 1024 first.
- Only test 4096 after 1024/2048 prove the architecture works.

### 3.2 Per-point stored fields

For each sampled point:

| Field | Dim | Use |
|---|---:|---|
| `pos` | 3 | grouping, distances, radius query |
| `normal` | 3 | point-pair invariant geometry |
| `chem` | 2 | Potential, NormalPotential |
| `geom_raw` | 0 or 6 | optional 14D extra geometry, globally normalized |

Default model input uses:

```text
pos      = centered xyz, scaled to unit ball
normal   = unit surface normal
chem     = [Potential, NormalPotential], train-global z-score, clamp [-5, 5]
geom_raw = optional [dist_centroid, dist_anchor_0..3, surf_variation]
```

Important:

- Compute normalization statistics on the training split only.
- Do not per-protein z-score geometric or chemical scalars.
- Do not use patch-mean scalar reinjection.

### 3.3 Feature invariance rule

Raw `pos` and `normal` are not fed directly into the hidden state as ordinary channels. They are used to compute invariant local features:

```text
distance, normalized distance, normal-dot-edge, normal-dot-normal, signed chirality
```

This keeps the representation stable under global SO(3) rotations.

---

## 4. Model Overview

```text
Input point set, N=2048
  pos[2048,3], normal[2048,3], chem[2048,2]
        |
        v
Stage A: Multi-Scale RI Surface Encoder
  MSRI-0: 2048 -> 1024, radii [0.045, 0.075], C=96
  MSRI-1: 1024 ->  512, radii [0.075, 0.140], C=160
  MSRI-2:  512 ->  256, radii [0.140, 0.280], C=256
  MSRI-3:  256 ->  128, radii [0.280, 0.520], C=384
  MSRI-4:  128 ->   64, radii [0.520, 0.900], C=512
        |
        v
Stage B: Patch Token Graph Encoder
  64 tokens, C=512
  kNN-8 + radius graph over centroids
  2 residual GIN blocks
  2 edge-biased transformer blocks
        |
        v
Stage C: Readout
  class token + attention pool + max pool
  graph embedding, C=1536
        |
        v
Stage D: Heads
  CE classifier head
  cosine/prototype head
  projection head for SupCon
```

---

## 5. Stage A: Multi-Scale Rotation-Invariant Surface Encoder

Stage A is the most important part of the model. It replaces the old node-level GIN.

### 5.1 One MSRI block

Each MSRI block performs:

1. FPS downsampling.
2. Ball query around each centroid at two metric radii.
3. Compute invariant pair features for all centroid-neighbor pairs.
4. Encode relative geometry and relative chemistry with a shared MLP.
5. Aggregate neighbors with max and mean pooling.
6. Fuse the two radii with gated residual projection.

### 5.2 Point-pair invariant features

For centroid `i` and neighbor `j`:

```text
d_ij      = pos_j - pos_i
r_ij      = ||d_ij||
e_ij      = d_ij / (r_ij + eps)
n_i       = normal_i
n_j       = normal_j
```

Geometry features:

```text
g1 = r_ij
g2 = r_ij / radius
g3 = dot(n_i, e_ij)
g4 = dot(n_j, e_ij)
g5 = dot(n_i, n_j)
g6 = dot(cross(n_j, n_i), e_ij)
g7 = abs(dot(n_i, e_ij))
g8 = abs(dot(n_j, e_ij))
```

Chemistry features:

```text
c1 = chem_i
c2 = chem_j
c3 = chem_j - chem_i
c4 = abs(chem_j - chem_i)
```

If `geom_raw` is enabled:

```text
u1 = geom_raw_i
u2 = geom_raw_j - geom_raw_i
```

Default pair input dimension:

```text
8 geometry + 8 chemistry/extra hidden inputs
```

In the first block, hidden input is only the point stem. In later blocks, hidden input is the previous centroid feature.

### 5.3 Local pair MLP

For each radius branch:

```text
pair_input = [h_j, ppf_ij, chem_i, chem_j, chem_j - chem_i, abs(chem_j - chem_i)]

MLP:
  Linear(pair_dim -> C/2)
  LayerNorm
  SiLU
  Linear(C/2 -> C)
  LayerNorm
  SiLU
```

Neighbor aggregation:

```text
branch_out = Linear([max_pool(pair_emb), mean_pool(pair_emb)] -> C)
```

Radius fusion:

```text
gate = sigmoid(Linear([branch_small, branch_large] -> 2))
out  = gate_0 * branch_small + gate_1 * branch_large
out  = out + residual_projection(h_centroid_previous)
```

Use dropout `0.10` inside Stage A. Avoid high dropout here because the local geometry signal is fragile.

### 5.4 Default Stage A configuration

For default 2048-point training:

| Block | Input N | Output N | Radii | Max neighbors | Output C |
|---|---:|---:|---|---:|---:|
| MSRI-0 | 2048 | 1024 | `[0.045, 0.075]` | `[24, 32]` | 96 |
| MSRI-1 | 1024 | 512 | `[0.075, 0.140]` | `[32, 40]` | 160 |
| MSRI-2 | 512 | 256 | `[0.140, 0.280]` | `[32, 48]` | 256 |
| MSRI-3 | 256 | 128 | `[0.280, 0.520]` | `[48, 64]` | 384 |
| MSRI-4 | 128 | 64 | `[0.520, 0.900]` | `[64, 96]` | 512 |

For 1024-point training:

| Block | Input N | Output N | Radii | Max neighbors | Output C |
|---|---:|---:|---|---:|---:|
| MSRI-0 | 1024 | 512 | `[0.060, 0.100]` | `[24, 32]` | 96 |
| MSRI-1 | 512 | 256 | `[0.100, 0.200]` | `[32, 40]` | 160 |
| MSRI-2 | 256 | 128 | `[0.200, 0.400]` | `[40, 64]` | 256 |
| MSRI-3 | 128 | 64 | `[0.400, 0.800]` | `[64, 96]` | 512 |

Use the 1024 version first as the sanity implementation. Move to 2048 after it beats HierGIN.

### 5.5 Empty-neighborhood fallback

Ball query can produce too few neighbors on sparse regions. Required rule:

```text
if neighbors_in_radius < min_neighbors:
    add nearest neighbors until min_neighbors is reached
```

Recommended `min_neighbors = 8`.

This avoids dead centroids and unstable gradients.

---

## 6. Stage B: Patch Token Graph Encoder

Stage A outputs 64 meaningful surface patch descriptors. Stage B reasons over their spatial arrangement.

This keeps the one thing HierGIN clearly proved: a patch-level hierarchy is useful.

### 6.1 Patch graph construction

Nodes:

```text
64 Stage-A centroids
```

Edges:

```text
knn_k = 8
plus radius edges at r = 0.75
remove duplicate edges
add self loops
```

Edge features between patch tokens `a` and `b`:

```text
e1 = ||pos_b - pos_a||
e2 = e1 / 0.75
e3 = dot(normal_a, unit(pos_b - pos_a))
e4 = dot(normal_b, unit(pos_b - pos_a))
e5 = dot(normal_a, normal_b)
e6 = dot(cross(normal_b, normal_a), unit(pos_b - pos_a))
```

Patch normals can be inherited from the nearest original centroid normal or computed as mean normal over the group.

### 6.2 Patch encoder depth

Use 4 patch blocks total:

```text
Patch-GIN block 1, dim 512
Patch-GIN block 2, dim 512
Edge-biased Transformer block 1, dim 512, heads 8
Edge-biased Transformer block 2, dim 512, heads 8
```

Why this hybrid:

- GIN is robust and was the best local graph backbone in existing ablations.
- Transformer attention can capture long-range relationships between surface regions.
- Keeping only 2 transformer layers controls overfitting on 9k training proteins.

### 6.3 Patch-GIN block

```text
message = MLP([h_src, edge_feat] -> 512)
aggregate = sum
update = LayerNorm(h_old + DropPath(message_sum))
ffn = Linear(512 -> 1024) -> GELU -> Dropout(0.15) -> Linear(1024 -> 512)
output = LayerNorm(update + DropPath(ffn))
```

### 6.4 Edge-biased transformer block

Attention logits:

```text
attn_ij = (Q_i dot K_j) / sqrt(d_head) + edge_bias(edge_feat_ij)
```

Where:

```text
edge_bias: MLP(6 -> heads)
heads = 8
d_model = 512
ffn_dim = 2048
dropout = 0.15
drop_path = 0.10
```

Use masked attention over patch graph edges first. Full 64-token attention can be an ablation if overfitting is controlled.

---

## 7. Stage C: Readout

Use three readout paths:

```text
r_cls  = learned class token after transformer
r_attn = attention-pool over 64 patch tokens
r_max  = max-pool over 64 patch tokens
graph_embedding = concat(r_cls, r_attn, r_max)  # 1536 dim
```

Attention pooling:

```text
score_i = Linear(tanh(Linear(h_i))) -> scalar
weights = softmax(score_i over patches)
r_attn = sum_i weights_i * h_i
```

Do not use raw patch-mean scalar reinjection. Chemistry has already entered locally inside Stage A.

---

## 8. Stage D: Heads

### 8.1 Main classifier head

```text
MLP head:
  Linear(1536 -> 768)
  LayerNorm
  GELU
  Dropout(0.30)
  Linear(768 -> 99)
```

### 8.2 Cosine classifier head

Use a normalized classifier for long-tail stability:

```text
z = Linear(1536 -> 512)
z = L2Norm(z)
W = L2Norm(class_weights)
cos_logits = scale * z @ W.T
scale = 30
margin = 0.15 during training
```

At inference:

```text
final_logits = 0.65 * mlp_logits + 0.35 * cos_logits
```

Tune this blend on validation only.

### 8.3 Projection head for SupCon

```text
proj = Linear(1536 -> 512) -> GELU -> Linear(512 -> 128)
proj = L2Norm(proj)
```

Use only during training.

### 8.4 Prototype memory head

Maintain exponential-moving-average class prototypes:

```text
prototype_c = EMA(mean z for class c)
prototype_logits = temperature * cosine(z, prototype_c)
```

Use stronger prototype blending for rare classes:

```text
if train_count[class] < 20:
    final_logits = 0.50 * mlp + 0.25 * cosine + 0.25 * prototype
else:
    final_logits = 0.65 * mlp + 0.35 * cosine
```

This targets class 80 and other few-shot classes.

---

## 9. Loss Function

Total loss:

```text
L = L_ce
  + 0.30 * L_cosine_margin
  + 0.15 * L_supcon
  + 0.10 * L_proto
  + 0.05 * L_rotation_consistency
```

### 9.1 Class-weighted cross entropy

Use the best existing setting:

```text
class_weight[c] = 1 / sqrt(train_count[c])
normalize weights to mean 1
label_smoothing = 0.10
```

### 9.2 Cosine-margin loss

Apply margin only to the true class:

```text
cos_y = cos_y - margin
margin = 0.15
scale = 30
```

Use the same inverse-square-root class weights.

### 9.3 Supervised contrastive loss

Use projection head output `proj`.

Batch strategy must ensure positives:

- Natural samples: 75 percent of batch.
- Hard/rare samples: 25 percent of batch.
- Hard classes: `61, 90, 70, 56, 86, 14, 80`.

SupCon temperature:

```text
tau = 0.07
lambda_supcon = 0.15
```

Raise to `0.25` only in final hard-pair fine-tuning.

### 9.4 Prototype loss

For rare classes:

```text
L_proto = CE(prototype_logits, y)
lambda_proto = 0.10
```

Apply to all classes, but up-weight rare-class samples by `2x`.

### 9.5 Rotation consistency loss

Create two random rotations of the same protein in a batch. Because the model is supposed to be RI, embeddings should match:

```text
L_rotation_consistency = 1 - cosine(z_view1, z_view2)
```

Weight:

```text
lambda_rot = 0.05
```

If this loss does not decrease quickly, the RI implementation is wrong.

---

## 10. Training Strategy

### 10.1 Four-phase training

#### Phase 0: geometry self-check, 3 epochs

Purpose: validate implementation before spending GPU time.

Run on 200-500 samples.

Checks:

- Forward pass non-NaN.
- Every centroid has at least 8 neighbors after fallback.
- Rotating input changes logits by less than `1e-4` before dropout, or embeddings by cosine distance less than `1e-3`.
- Overfit 200 samples to near-perfect train F1.

Do not proceed until this passes.

#### Phase 1: geometry-only supervised pretraining

Purpose: force the architecture to learn the signal RIConv2 geometry-only exploits.

Settings:

```text
chem inputs zeroed
geom_raw disabled
epochs = 80
batch_size = 32 for 2048 points, grad_accum = 2
loss = CE + cosine-margin + rotation-consistency
```

Expected:

```text
macro-F1 should approach or exceed 0.83
```

If it cannot beat HierGIN-8D, the architecture implementation is not extracting multi-scale geometry correctly.

#### Phase 2: chemistry fine-tuning

Purpose: add Potential and NormalPotential after the geometry backbone is strong.

Settings:

```text
initialize from Phase 1
enable chem inputs
epochs = 60
lr = 2e-4
loss = CE + cosine-margin + SupCon + rotation-consistency
```

Expected:

```text
+1 to +2 macro-F1 points over geometry-only
class 61 should improve
```

#### Phase 3: hard-pair and rare-class fine-tuning

Purpose: focus on macro-F1 failures.

Settings:

```text
epochs = 30
lr = 5e-5
batch = two-stream: 75 percent natural, 25 percent hard/rare
lambda_supcon = 0.25
lambda_proto = 0.15
freeze MSRI-0 and MSRI-1 for first 10 epochs
```

Target classes:

```text
hard_pairs = [(61,90), (70,56), (86,14)]
rare_focus = [80] plus classes with train_count < 20
```

#### Phase 4: SWA/EMA polish

Purpose: improve generalization and calibration.

Settings:

```text
SWA over last 10 checkpoints
EMA decay = 0.999
temperature scaling on validation logits
```

Use the better of EMA and SWA on validation macro-F1.

### 10.2 Optimizer

Default:

```text
optimizer = AdamW
base_lr = 5e-4 for from-scratch
fine_tune_lr = 2e-4
hard_tune_lr = 5e-5
weight_decay = 1e-4
betas = (0.9, 0.999)
eps = 1e-8
```

Schedule:

```text
5 epoch linear warmup
cosine decay
minimum lr = 1e-6
```

Gradient:

```text
clip_grad_norm = 1.0
mixed_precision = true
gradient_accumulation = 2 for 2048 points
gradient_accumulation = 4 for 4096 points
```

### 10.3 Regularization

```text
Stage A dropout = 0.10
Patch encoder dropout = 0.15
Classifier dropout = 0.30
DropPath = 0.10
Label smoothing = 0.10
```

Do not use high dropout in Stage A. It can destroy local geometric details.

### 10.4 Augmentation

Training augmentations:

| Augmentation | Value | Why |
|---|---:|---|
| random SO(3) rotation | always | verifies/regularizes RI behavior |
| FPS reseed | every epoch | reduces fixed-subset overfit |
| point jitter | sigma `0.005-0.010` | robustness |
| point dropout | max `10%` | robustness |
| normal noise | sigma `0.01`, renormalize | robustness |
| scalar noise | sigma `0.02` after z-score | mild chem robustness |

Avoid:

- mixup on chemistry.
- heavy coordinate scaling.
- per-protein scalar z-scoring.

---

## 11. Inference Strategy for Maximum Performance

### 11.1 Single-model inference

Use FPS test-time augmentation:

```text
num_fps_views = 8
points_per_view = 2048
logits = mean logits over views
```

Do not use rotation TTA as a main tool; the model should already be rotation invariant. Use one rotation check only as a diagnostic.

### 11.2 Ensemble

For maximum performance:

```text
seeds = [11, 22, 33, 44, 55]
models = 5 independent trainings
views_per_model = 8
final_logits = mean over 40 logits
```

Optional diversity:

```text
3 models at 2048 points
2 models at 1024 points with slightly different radius schedule
```

Recommended ensemble blend:

```text
final = 0.70 * GeoChem-MSRINet ensemble
      + 0.30 * best RIConv2-style geometry-only reproduction, if implemented separately
```

Only use this if validation confirms complementarity.

### 11.3 Calibration

Use validation temperature scaling:

```text
temperature in [0.8, 1.5]
choose by validation NLL, verify macro-F1 does not drop
```

For rare classes, tune prototype-logit blend on validation macro-F1.

---

## 12. Implementation Plan in This Repository

Recommended new files:

```text
models/
  msri_surface_encoder.py
  patch_token_encoder.py
  geochem_msrinet.py
  losses.py

train_geochem_msrinet.py
eval_geochem_msrinet.py
```

If keeping the repo flat:

```text
classify_geochem_msrinet.py
```

### 12.1 Dataset loader

Create a dataset class that returns:

```python
{
    "pos": FloatTensor[N, 3],
    "normal": FloatTensor[N, 3],
    "chem": FloatTensor[N, 2],
    "geom_raw": Optional[FloatTensor[N, 6]],
    "label": LongTensor[]
}
```

Do not return DGL graphs for Stage A. Build radius neighborhoods inside the model.

### 12.2 Core operations

Preferred operations:

```text
fps: torch_cluster.fps or pytorch3d.ops.sample_farthest_points
ball query: torch_cluster.radius or pytorch3d.ops.ball_query
scatter: torch_scatter
```

Fallback:

- For debugging, use `torch.cdist` per graph.
- For full training, avoid all-pairs distance on 4096 points unless memory permits.

### 12.3 Model forward shape contract

For 2048 default:

```text
Input:
  pos       [B, 2048, 3]
  normal    [B, 2048, 3]
  chem      [B, 2048, 2]

After MSRI-0:
  pos       [B, 1024, 3]
  normal    [B, 1024, 3]
  hidden    [B, 1024, 96]

After MSRI-1:
  hidden    [B, 512, 160]

After MSRI-2:
  hidden    [B, 256, 256]

After MSRI-3:
  hidden    [B, 128, 384]

After MSRI-4:
  hidden    [B, 64, 512]

Patch encoder:
  hidden    [B, 64, 512]

Readout:
  graph_emb [B, 1536]

Heads:
  logits    [B, 99]
  proj      [B, 128]
```

### 12.4 Pseudocode

```python
class GeoChemMSRINet(nn.Module):
    def __init__(self):
        self.sa0 = MSRIBlock(2,   96,  n_out=1024, radii=(0.045, 0.075), k=(24, 32))
        self.sa1 = MSRIBlock(96,  160, n_out=512,  radii=(0.075, 0.140), k=(32, 40))
        self.sa2 = MSRIBlock(160, 256, n_out=256,  radii=(0.140, 0.280), k=(32, 48))
        self.sa3 = MSRIBlock(256, 384, n_out=128,  radii=(0.280, 0.520), k=(48, 64))
        self.sa4 = MSRIBlock(384, 512, n_out=64,   radii=(0.520, 0.900), k=(64, 96))

        self.patch_gin = PatchGIN(dim=512, layers=2, edge_dim=6)
        self.patch_attn = EdgeBiasedTransformer(dim=512, heads=8, layers=2, edge_dim=6)

        self.readout = ClassTokenAttnMaxReadout(dim=512)
        self.mlp_head = MLP(1536, 768, 99)
        self.cos_head = CosineClassifier(512, 99, scale=30, margin=0.15)
        self.proj_head = ProjectionHead(1536, 128)

    def forward(self, pos, normal, chem, geom_raw=None, phase="joint"):
        if phase == "geometry":
            chem = torch.zeros_like(chem)

        h = chem
        pos, normal, h = self.sa0(pos, normal, h, chem, geom_raw)
        pos, normal, h = self.sa1(pos, normal, h, chem=None, geom_raw=None)
        pos, normal, h = self.sa2(pos, normal, h, chem=None, geom_raw=None)
        pos, normal, h = self.sa3(pos, normal, h, chem=None, geom_raw=None)
        pos, normal, h = self.sa4(pos, normal, h, chem=None, geom_raw=None)

        patch_edges, patch_edge_feat = build_patch_graph(pos, normal, k=8, radius=0.75)
        h = self.patch_gin(h, patch_edges, patch_edge_feat)
        h = self.patch_attn(h, patch_edges, patch_edge_feat)

        graph_emb = self.readout(h)
        logits_mlp = self.mlp_head(graph_emb)
        logits_cos = self.cos_head(graph_emb)
        proj = self.proj_head(graph_emb)

        logits = 0.65 * logits_mlp + 0.35 * logits_cos
        return logits, proj, graph_emb
```

---

## 13. Default Training Config

```yaml
model:
  name: geochem_msrinet
  num_classes: 99
  points: 2048
  use_geom_raw: false
  stage_a:
    type: multi_scale_ri_set_abstraction
    blocks:
      - {n_out: 1024, radii: [0.045, 0.075], max_neighbors: [24, 32], out_dim: 96}
      - {n_out: 512,  radii: [0.075, 0.140], max_neighbors: [32, 40], out_dim: 160}
      - {n_out: 256,  radii: [0.140, 0.280], max_neighbors: [32, 48], out_dim: 256}
      - {n_out: 128,  radii: [0.280, 0.520], max_neighbors: [48, 64], out_dim: 384}
      - {n_out: 64,   radii: [0.520, 0.900], max_neighbors: [64, 96], out_dim: 512}
  patch_encoder:
    gin_layers: 2
    transformer_layers: 2
    dim: 512
    heads: 8
    ffn_dim: 2048
    edge_dim: 6
    dropout: 0.15
    drop_path: 0.10
  readout:
    type: class_token_attention_max
    out_dim: 1536
  heads:
    mlp_hidden: 768
    cosine_dim: 512
    projection_dim: 128

training:
  optimizer: adamw
  lr: 5.0e-4
  fine_tune_lr: 2.0e-4
  hard_tune_lr: 5.0e-5
  weight_decay: 1.0e-4
  warmup_epochs: 5
  schedule: cosine
  min_lr: 1.0e-6
  batch_size: 32
  grad_accum: 2
  amp: true
  clip_grad_norm: 1.0
  label_smoothing: 0.10
  class_weight: inverse_sqrt

loss:
  ce_weight: 1.0
  cosine_margin_weight: 0.30
  supcon_weight: 0.15
  prototype_weight: 0.10
  rotation_consistency_weight: 0.05
  supcon_temperature: 0.07
  cosine_scale: 30
  cosine_margin: 0.15

augmentation:
  random_rotation: true
  fps_reseed_each_epoch: true
  point_jitter_sigma: 0.0075
  point_dropout_max: 0.10
  normal_noise_sigma: 0.01
  scalar_noise_sigma: 0.02

inference:
  fps_tta_views: 8
  temperature_scaling: true
  ensemble_seeds: [11, 22, 33, 44, 55]
```

---

## 14. Ablation Plan

Run ablations in this order. Stop early if an ablation cannot beat HierGIN-8D.

### 14.1 Sanity ablations

| Experiment | Purpose | Expected |
|---|---|---|
| 1024 points, geometry only | verify RIConv2-like signal | >= 0.83 macro-F1 |
| 2048 points, geometry only | test resolution gain | +0 to +1.5 pp |
| no rotation consistency | confirm regularizer value | small drop |
| raw xyz GIN baseline on same points | verify architecture gap | below MSRI |

### 14.2 Core architecture ablations

| Experiment | Purpose |
|---|---|
| ball query vs kNN with same K | prove metric radius is needed |
| one radius vs two radii per block | prove multi-scale branch helps |
| remove patch transformer | test global attention value |
| remove patch GIN | test local patch graph value |
| full attention vs graph-masked attention | choose patch transformer mode |

### 14.3 Feature and loss ablations

| Experiment | Purpose |
|---|---|
| geometry only | reproduce RIConv2 RUN1 behavior |
| geometry + Potential only | quantify Potential gain |
| geometry + Potential + NormalPotential | default chemical setting |
| add 6 raw geometry features | test 14D complementarity |
| remove SupCon | measure hard-pair separation |
| remove prototype head | measure rare-class effect |
| no cosine head | test long-tail classifier |

### 14.4 Performance ablations

| Experiment | Purpose |
|---|---|
| 1 FPS view vs 8 views | quantify TTA gain |
| 1 seed vs 5 seeds | quantify ensemble gain |
| 1024/2048 mixed ensemble | ensemble diversity |

---

## 15. Failure Diagnostics

If performance is weak, debug in this order:

1. **Rotation invariance check fails**  
   The PPF implementation is wrong or raw frame-dependent features leaked into hidden states.

2. **Geometry-only phase below 0.80 macro-F1**  
   Ball-query radii are wrong, neighborhoods are empty, or point sampling is broken.

3. **Geometry-only good but chemistry hurts**  
   Chemical normalization is wrong, per-protein z-scoring leaked in, or scalar noise is too high.

4. **Train F1 high but validation macro-F1 low**  
   Reduce patch transformer depth, increase dropout only in Stage B/head, keep Stage A dropout low.

5. **Hard pairs still confused**  
   Increase SupCon weight to `0.25`, add hard-pair sampler, inspect class 61/90 embeddings.

6. **Class 80 remains bad**  
   This may be data-limited. Tune prototype blending and TTA, but do not overfit the whole architecture to class 80.

---

## 16. Negative-Result Guardrails

Do not reintroduce these ideas unless there is a new reason:

| Avoid | Reason |
|---|---|
| precomputed PCA/LRF frames | LRF-GIN failed; flat regions make frames unstable |
| kNN-only multi-scale | previous multi-k GIN did not expose the right signal |
| independent FPS levels | caused the three-level collapse |
| patch-mean scalar reinjection | degraded 14D runs |
| per-protein scalar z-scoring | erased useful scale information |
| very deep patch GNN | likely overfits; keep patch reasoning compact |
| raw xyz/normals as hidden channels | breaks rotation invariance |

---

## 17. Surface-Only vs External Sequence Features

The default architecture is surface-only and comparable to the existing report.

If external sequence information is allowed, the highest-probability extra gain is an ESM-style sequence embedding branch:

```text
ESM embedding -> MLP(1280 -> 512) -> concat with graph_embedding -> classifier
```

But this changes the problem setting. Do not mix this into the main surface-only benchmark unless the evaluation rules allow it.

---

## 18. Final Recommendation

Implement the 1024-point version first, not the full 2048/4096 version. The make-or-break experiment is:

```text
GeoChem-MSRINet, 1024 points, geometry-only, no chemistry, no prototype head.
```

If that run cannot approach RIConv2 geometry-only performance, the local RI multi-scale encoder is not implemented correctly. If it works, then add chemistry, SupCon, prototype memory, 2048 points, FPS TTA, and finally a 5-seed ensemble.

This architecture directly attacks the bottleneck all recent experiments exposed: HierGIN could reason over patches, but its node stage never learned the multi-scale local surface descriptors that separate the hard families. GeoChem-MSRINet makes that signal the foundation of the model.

---

## 19. Prototype Implementation and Results, 2026-06-20

This section records what was actually implemented and tested after the initial blueprint.

### 19.1 Implemented files

A separate prototype implementation was added under:

```text
geochem_msrinet/
  data.py        dataset, label CSV loading, augmentation
  ops.py         pure PyTorch FPS, radius query, kNN, PPF features
  model.py       GeoChem-MSRINet model
  losses.py      CE, cosine-margin CE, SupCon, prototype memory
  metrics.py     macro-F1/balanced-accuracy helpers
  train.py       training driver
  eval.py        checkpoint evaluation
  smoke_test.py  forward and rotation-invariance smoke check
```

Important implementation constraint:

- The local environment did not expose DGL, PyG, torch-cluster, torch-scatter, or PyTorch3D.
- The prototype therefore uses pure PyTorch operations, especially dense `torch.cdist`.
- This makes the model runnable and easy to inspect, but slower and less memory-efficient than the intended CUDA-kernel implementation.

### 19.2 Basic correctness checks

The synthetic smoke test passed:

```text
logits shape: (2, 99)
projection shape: (2, 128)
rotation max |delta logits|: ~8.9e-07
```

A tiny real-data smoke run also completed successfully after fixing two implementation details:

- label CSV lookup now defaults to `ssl_graphs_1k_geom_enriched_ri`;
- unseen classes in tiny smoke subsets use neutral CE weight instead of zero, avoiding validation-loss `NaN`.

### 19.3 First full run: failed curriculum collapse

Run:

```text
runs/geochem_msri_curriculum_1k_bs16_acc2_20260620_032436
```

Configuration:

```text
phase = curriculum
geometry-only until epoch 40
batch_size = 16
grad_accum = 2
amp = true
lr = 5e-4
dropout = 0.30
logit_blend = 0.65
cosine_weight = 0.30
supcon_weight = 0.10
label_smoothing = 0.10
random rotation + scalar/normal noise
```

Result:

| Split | Accuracy | Balanced Acc. | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| validation | 0.2223 | 0.0114 | 0.0041 | 0.0808 |
| test | 0.2224 | 0.0114 | 0.0041 | 0.0809 |

The run early-stopped at epoch 31, before chemistry was enabled. Best epoch was epoch 1.

Diagnosis:

- This was not a label-loading issue.
- The model predicted class `8` for every train, validation, and test sample.
- Class `8` represents about 22.2 percent of all three splits:

```text
train class 8: 1643 / 7395 = 0.2222
val   class 8:  411 / 1849 = 0.2223
test  class 8:  514 / 2311 = 0.2224
```

An all-class-8 predictor exactly reproduces the observed macro-F1 and accuracy.

### 19.4 Label and matching audit

The label pipeline was checked after the collapse.

Findings:

- `train_split.csv` and `val_split.csv` in `ssl_graphs_1k_geom_enriched_ri` are byte-identical to the base `ssl_graphs_1k` split files.
- Train, validation, and test IDs have zero overlap.
- CSV IDs resolve to existing `.pt` files:

```text
train = 7395
val   = 1849
test  = 2311
```

- Class IDs are already zero-indexed and cover `0..98` in train.
- Sampled tensor `meta["source_file"]` matches the `.pt` filename after canonicalizing VTK chain separators from `:` to `_`.
- The `.pt` tensors do not contain internal label fields, so the CSV split files are the source of truth.

Conclusion:

```text
The 0.004 macro-F1 result was majority-class collapse, not a label-matching bug.
```

### 19.5 Overfit probes

Several small overfit tests were run to determine whether the implementation can learn at all.

#### Single fixed batch

Setup:

```text
8 samples
8 classes: [8, 14, 56, 61, 70, 86, 90, 92]
chemistry enabled
MLP logits only, logit_blend = 1.0
cosine loss off
SupCon off
lr = 3e-4
no augmentation
```

Result:

```text
step 050: accuracy = 1.000
```

Conclusion:

```text
Forward/backward, labels, and classifier path are capable of memorization.
```

#### Balanced 32-sample probe

Setup:

```text
32 samples
8 classes, 4 samples per class
chemistry enabled
MLP logits only
cosine loss off
SupCon off
lr = 3e-4
no augmentation
```

Result:

```text
best observed accuracy: ~0.875
best observed macro-F1: ~0.833
```

Conclusion:

```text
The model can learn a small balanced multi-batch subset, but it is sensitive to optimization settings.
```

### 19.6 Training-script correction

To address class-frequency collapse, the training driver was extended with:

```text
--balanced-sampler
--samples-per-epoch
```

This uses inverse-frequency sampling over the training set. The correction is intentionally separate from CE class weights, so sampling and loss weighting can be ablated independently.

### 19.7 Corrected full diagnostic run

Run:

```text
runs/geochem_chem_balanced_mlp_lr3e4_20260620_123402
```

Configuration:

```text
phase = chem
chemistry enabled from epoch 1
batch_size = 16
grad_accum = 2
amp = true
lr = 3e-4
warmup_epochs = 3
dropout = 0.10
patch_gin_layers = 1
patch_transformer_layers = 1
logit_blend = 1.0
cosine_weight = 0
supcon_weight = 0
label_smoothing = 0
balanced_sampler = true
samples_per_epoch = 7395
rotation augmentation disabled
```

Validation trajectory:

```text
epoch 001: val macro-F1 = 0.0311
epoch 010: val macro-F1 = 0.5707
epoch 023: val macro-F1 = 0.6635
epoch 042: val macro-F1 = 0.7207
epoch 050: val macro-F1 = 0.7392
epoch 056: val macro-F1 = 0.7649
epoch 076: early stopping
```

Final selected checkpoint:

```text
best_epoch = 56
best_val_macro_f1 = 0.7649
```

Metrics:

| Split | Accuracy | Balanced Acc. | Macro-F1 | Weighted-F1 | Loss |
|---|---:|---:|---:|---:|---:|
| validation | 0.7577 | 0.7978 | 0.7649 | 0.7578 | 0.9655 |
| test | 0.7559 | 0.7577 | 0.7336 | 0.7546 | 0.9008 |

This corrected run proves the implementation can train on the full split and avoid majority-class collapse.

### 19.8 Position relative to existing results

| Method | Macro-F1 | Comment |
|---|---:|---|
| RIConv2 + xyz/normals/Potential | 0.854 | current external target |
| RIConv2 geometry-only | 0.838 | target for correct multi-scale RI geometry |
| HierGIN 14D global norm | 0.808 | best in-repo mature baseline |
| HierGIN 8D | 0.798 | strong original baseline |
| HierGIN 14D + RI-GIN | 0.793 | mature RI edge-message baseline |
| LRF-GIN 8D | 0.766 | unstable-frame baseline |
| GeoChem-MSRINet corrected diagnostic, validation | 0.765 | best validation from this prototype |
| GeoChem-MSRINet corrected diagnostic, test | 0.734 | current prototype test result |
| Flat GNN best | ~0.640 | weaker non-hierarchical baseline |
| GeoChem-MSRINet first curriculum run | 0.004 | majority-class collapse |
| MSG-GIN / three-level failed runs | ~0.07 / 0.016 | failed ablations |

Interpretation:

- The prototype is now clearly above weak/failed baselines.
- It is still below all serious HierGIN and RIConv2 results.
- The current result is a diagnostic success, not a max-performance success.
- The gap to HierGIN 8D is about `6.4` macro-F1 points on test.
- The gap to HierGIN 14D is about `7.4` macro-F1 points on test.
- The gap to RIConv2 + Potential is about `12.0` macro-F1 points.

### 19.9 Updated interpretation

The original architecture hypothesis is not validated yet.

What was validated:

- The label pipeline is correct.
- The model is rotation-invariant in the synthetic smoke test.
- The model can memorize a fixed batch.
- The model can learn the full split under conservative chemistry-first balanced training.

What failed:

- The proposed geometry-first curriculum collapsed before chemistry was enabled.
- The initial cosine/SupCon/logit-blend objective was too unstable for the prototype.
- The pure-PyTorch implementation is memory-heavy and slow, limiting batch size and ablation speed.
- The current multi-scale RI encoder does not yet reproduce the RIConv2 geometry-only signal.

Most likely causes of underperformance:

1. **Geometry-only representation is too weak in this implementation.**  
   The full successful diagnostic needed chemistry from epoch 1. This contradicts the desired RIConv2-like behavior where geometry alone should approach `0.83`.

2. **Balanced sampling helps macro-F1 but changes training distribution.**  
   It prevents class-8 collapse, but may hurt calibration and frequent-class accuracy.

3. **The patch encoder may be overfitting or adding noise.**  
   The corrected run needed reduced patch depth (`1` GIN + `1` transformer) and lower dropout.

4. **The pure PyTorch radius-query implementation may not match the intended optimized set-abstraction behavior.**  
   Dense `cdist`, duplicate padding, and deterministic FPS may create different optimization dynamics from a production PointNet++/RIConv2-style implementation.

5. **The original loss stack was too aggressive too early.**  
   Cosine logits, cosine-margin CE, SupCon, label smoothing, and geometry-only training together encouraged collapse.

### 19.10 Updated next experiments

Do not rerun the original curriculum configuration.

Recommended next ablations, in order:

| Priority | Experiment | Purpose |
|---:|---|---|
| 1 | corrected run + `--use-geom-raw` | test whether the successful 14D global geometry features recover part of HierGIN-14D performance |
| 2 | corrected run with `logit_blend=0.9`, `cosine_weight=0.05` | reintroduce metric classifier gently |
| 3 | corrected run with light rotation only, no scalar/noise jitter | test regularization without destabilizing training |
| 4 | geometry-only with balanced sampler, MLP-only logits, `lr=3e-4` | isolate whether geometry can learn when collapse is controlled |
| 5 | no balanced sampler, but class-balanced batch sampler | reduce distribution shift while preserving macro-F1 pressure |
| 6 | remove patch transformer entirely | test whether global attention is hurting generalization |
| 7 | one-radius vs two-radius MSRI blocks | confirm whether the multi-scale branch contributes |
| 8 | optimized torch-cluster/PyTorch3D radius-query implementation | test whether the pure-PyTorch grouping is a bottleneck |

Updated recommendation:

```text
Treat GeoChem-MSRINet as an active prototype, not the final architecture.
The next target is not 0.85 yet; the next target is to exceed HierGIN-8D
under the corrected stable training recipe.
```
