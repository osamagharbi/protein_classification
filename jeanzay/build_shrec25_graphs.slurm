#!/bin/bash
#SBATCH --job-name=S25_GRAPHS
#SBATCH --account=acu@v100
#SBATCH --partition=prepost
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=logs/fw/%x_%j.out
#SBATCH --error=logs/fw/%x_%j.err
#
# Rebuild the paper's SHREC2025 inputs (the folders the HiPatch/HSNet runs used, now deleted from $WORK)
# from the organisers' ORIGINAL, unrotated archives:
#   $DATA_ROOT/ssl_graphs_1k_geom_enriched_ri      9,244 train + 2,311 test graphs, paper split CSVs
#   $DATA_ROOT/ssl_graphs_1k_patchassign_k{64,16,256}
# then checks a 100-graph sample against fingerprints of the paper graphs (exact mesh edges and patch
# assignments; floats within 1e-4) and stops if anything differs.
#
# CPU job on prepost (internet access; acu@v100 rejects CPU-only jobs on the GPU partitions).
# Resumable: resubmit after a timeout and finished graphs are skipped.
#
#   mkdir -p logs/fw && sbatch jeanzay/build_shrec25_graphs.slurm
#   KS="64" SAVE=0 sbatch jeanzay/build_shrec25_graphs.slurm      # k64 only, no safety tarballs
#   TRAIN_ARCHIVE=... TEST_ARCHIVE=...                              # archives somewhere else
#   ONLY=id1,id2 ...                                                # smoke test on a few graph ids
#
# RELEASE=rot: the FINAL release (Sept 2026) instead: rotated meshes train_set_rot.tar.xz / test_set_rot.tar.xz
# (reused from $SCRATCH/shrec25_final_rot/raw/archives, where the Mamba baselines downloaded them), final
# labels (98 classes) -> $DATA_ROOT/ssl_graphs_1k_geom_enriched_ri_rot + ssl_graphs_1k_patchassign_rot_k{64,16,256}.
# Checked graph by graph against the paper graphs in $DATA_ROOT (tools/shrec25_rebuild/compare_rotated.py):
# shared vertices, one rotation mapping them (residual), mesh edges, k64 patches (ARI).
#   mkdir -p logs/fw && RELEASE=rot sbatch jeanzay/build_shrec25_graphs.slurm
set -euo pipefail

module purge
module load pytorch-gpu/py3/2.8.0
source "$WORK/myvenv/bin/activate"

PROJECT="${PROJECT:-$WORK/protein_classification}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH/fw_data}"
ARCH_DIR="${ARCH_DIR:-$SCRATCH/shrec25_orig/archives}"   # downloaded archives go here
KS="${KS:-64 16 256}"                                     # 64 first: it is checked against the paper files
SAVE="${SAVE:-${STORE:+$STORE/fw_archives}}"             # tarball copies that survive the $SCRATCH purge
URL=https://shrec2025.drugdesign.fr/files
RELEASE="${RELEASE:-paper}"
case "$RELEASE" in
  paper) G=ssl_graphs_1k_geom_enriched_ri;     P=ssl_graphs_1k_patchassign
         TRAIN_NAME=train_set.tar.xz;          TRAIN_SIZE=4449169028
         TEST_NAME=test_set.tar.xz;            TEST_SIZE=1114399932 ;;
  rot)   G=ssl_graphs_1k_geom_enriched_ri_rot; P=ssl_graphs_1k_patchassign_rot
         TRAIN_NAME=train_set_rot.tar.xz;      TRAIN_SIZE=5943586380
         TEST_NAME=test_set_rot.tar.xz;        TEST_SIZE=1484868888 ;;
  *) echo "RELEASE must be paper or rot"; exit 1 ;;
esac
echo "release: $RELEASE -> $G, ${P}_k*"
REF_G="$DATA_ROOT/ssl_graphs_1k_geom_enriched_ri"; REF_P="$DATA_ROOT/ssl_graphs_1k_patchassign_k64"
if [ "$RELEASE" = rot ] && { [ ! -d "$REF_G/train_set" ] || [ ! -d "$REF_P/train_set" ]; }; then
  echo "!! the rotated graphs are checked against the paper graphs, but $REF_G or $REF_P is missing:"
  echo "   ./jeanzay/restore_data.sh (tarballs in \$STORE/fw_archives) or sbatch jeanzay/build_shrec25_graphs.slurm first"
  exit 1
fi
NW="${SLURM_CPUS_PER_TASK:-16}"
cd "$PROJECT"
mkdir -p "$DATA_ROOT" "$ARCH_DIR"

echo "== 1/5 dependencies"
python -c "import vtk, trimesh" 2>/dev/null || pip install --no-cache-dir "vtk==9.5.2" "trimesh==4.11.2"
python -c "import vtk, trimesh, torch, numpy, scipy; print('vtk', vtk.__version__, '| trimesh', trimesh.__version__, \
'| torch', torch.__version__, '| numpy', numpy.__version__, '| scipy', scipy.__version__)"

echo "== 2/5 archives ($RELEASE release)"
fetch() {   # fetch <name> <exact size in bytes>: reuse a copy with the exact size, else download
  local name=$1 size=$2 c
  for c in "$ARCH_DIR/$name" "$SCRATCH/shrec25_final_rot/raw/archives/$name" "$WORK/datasets/$name" \
           "$WORK/$name" "$SCRATCH/$name"; do
    if [ -f "$c" ] && [ "$(stat -c %s "$c")" = "$size" ]; then echo "  using $c" >&2; echo "$c"; return 0; fi
  done
  echo "  downloading $URL/$name -> $ARCH_DIR" >&2
  wget -q -c -O "$ARCH_DIR/$name" "$URL/$name" >&2
  [ "$(stat -c %s "$ARCH_DIR/$name")" = "$size" ] || { echo "  !! $name has the wrong size" >&2; return 1; }
  echo "$ARCH_DIR/$name"
}
TRAIN="${TRAIN_ARCHIVE:-$(fetch "$TRAIN_NAME" "$TRAIN_SIZE")}"
TEST="${TEST_ARCHIVE:-$(fetch "$TEST_NAME" "$TEST_SIZE")}"
echo "  train: $TRAIN"; echo "  test:  $TEST"
ONLY="${ONLY:-}"
SMOKE_ARGS=(); VERIFY_ARGS=()
[ -n "$ONLY" ] && { SMOKE_ARGS=(--only "$ONLY"); VERIFY_ARGS+=(--present-only); }

echo "== 3/5 graphs -> $DATA_ROOT/$G"
python -u tools/shrec25_rebuild/build_shrec25_paper_graphs.py \
    --release "$RELEASE" --train-archive "$TRAIN" --test-archive "$TEST" --out "$DATA_ROOT" -j "$NW" "${SMOKE_ARGS[@]}"
if [ "$RELEASE" = rot ]; then   # every graph id in both builds; with ONLY, just those graphs
  python -u tools/shrec25_rebuild/compare_rotated.py --ref "$REF_G" --rot "$DATA_ROOT/$G" --workers "$NW" \
      --out "$DATA_ROOT/$G/checks/compare_graphs.json"
else
  python -u tools/shrec25_rebuild/fingerprint.py verify --graphs "$DATA_ROOT/$G" \
      --ref tools/shrec25_rebuild/fingerprints_paper.json "${VERIFY_ARGS[@]}"
fi

echo "== 4/5 patch assignments k = $KS"
for k in $KS; do
  python -u tools/build_patchassign.py --input "$DATA_ROOT/$G" --out "$DATA_ROOT/${P}_k$k" \
      --num-patches "$k" --workers "$NW"
done
case " $KS " in *" 64 "*)
  if [ "$RELEASE" = rot ]; then
    python -u tools/shrec25_rebuild/compare_rotated.py --ref "$REF_G" --rot "$DATA_ROOT/$G" --workers "$NW" \
        --ref-patches "$REF_P" --rot-patches "$DATA_ROOT/${P}_k64" --out "$DATA_ROOT/$G/checks/compare_patches_k64.json"
  else
    python -u tools/shrec25_rebuild/fingerprint.py verify --graphs "$DATA_ROOT/$G" --patches "$DATA_ROOT/${P}_k64" \
        --ref tools/shrec25_rebuild/fingerprints_paper.json "${VERIFY_ARGS[@]}"
  fi ;;
esac

echo "== 5/5 counts and safety copies"
for d in "$G" $(for k in $KS; do echo "${P}_k$k"; done); do
  printf "  %-36s train %5d  test %5d\n" "$d" "$(ls "$DATA_ROOT/$d/train_set" | wc -l)" "$(ls "$DATA_ROOT/$d/test_set" | wc -l)"
done
if [ -n "$ONLY" ]; then
  echo "  smoke run (ONLY set): no safety copies written"
elif [ -n "$SAVE" ] && [ "$SAVE" != 0 ]; then
  mkdir -p "$SAVE"
  for d in "$G" $(for k in $KS; do echo "${P}_k$k"; done); do
    if [ -f "$SAVE/$d.tar" ]; then echo "  kept     $SAVE/$d.tar"; continue; fi
    tar -cf "$SAVE/$d.tar.part" -C "$DATA_ROOT" "$d" && mv "$SAVE/$d.tar.part" "$SAVE/$d.tar" \
      && echo "  saved    $SAVE/$d.tar" || echo "  !! could not write $SAVE/$d.tar (the data in $DATA_ROOT is fine)"
  done
  echo "  after a \$SCRATCH purge, ./jeanzay/restore_data.sh extracts these tarballs again"
fi
[ -n "$ONLY" ] && echo "smoke run OK: rerun without ONLY for the full build" || echo "SHREC2025 $RELEASE inputs ready in $DATA_ROOT"
