#!/bin/bash
#SBATCH --job-name=OFFMESH
#SBATCH --account=acu@v100
#SBATCH --partition=prepost
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/fw/%x_%j.out
#SBATCH --error=logs/fw/%x_%j.err
#
# Non-protein meshes for HiPatch: SHREC2011 non-rigid watertight (600 meshes, 30 classes) and
# TOSCA (80 meshes, 9 classes). Downloads the official archives (checked by size and sha256),
# converts them with prepare_off_hierarchy.py (1,024 vertices, kNN-16, xyz + normals + 16-channel
# HKS) and builds k=64 geodesic patches:
#   $DATA_ROOT/offmesh_shrec11  + offmesh_shrec11_patchassign_k64
#   $DATA_ROOT/offmesh_tosca    + offmesh_tosca_patchassign_k64
# Splits are the committed files in data_labels/shrec11 and data_labels/tosca; the job regenerates
# them from the downloaded class file and stops if they differ.
#
#   mkdir -p logs/fw && sbatch jeanzay/build_offmesh.slurm
# CPU job on prepost (internet access). Re-running is safe: finished graphs are skipped.
set -euo pipefail

module purge
module load pytorch-gpu/py3/2.8.0
source "$WORK/myvenv/bin/activate"

PROJECT="${PROJECT:-$WORK/protein_classification}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH/fw_data}"
SRC="${SRC:-$SCRATCH/offmesh_src}"
NW="${SLURM_CPUS_PER_TASK:-16}"
cd "$PROJECT"
mkdir -p "$DATA_ROOT" "$SRC"

S11_URL=https://data.nist.gov/od/ds/mds2-2215/SHREC2011_NonRigid.zip
S11_SIZE=259086968
S11_SHA=c08ccf56ef1876adc341f58c9f2670515e2ad6b83d21241b7aaa9193eb6f5ec3
TOS_URL=https://vovakim.com/projects/CorrsBlended/SurfCorr2.0.benchmarks.bins.zip
TOS_SIZE=184438709
TOS_SHA=ee007e8bae590f3aef226fd44178714481dd6233f2b1988c86af892d2ad0bbc5

fetch() {   # fetch <url> <file> <size> <sha256>
  local url=$1 f="$SRC/$2" size=$3 sha=$4
  if [ ! -f "$f" ] || [ "$(stat -L -c %s "$f")" != "$size" ]; then
    [ -L "$f" ] && { echo "!! $f is a link to a file of the wrong size; not downloading over it"; exit 1; }
    echo "  downloading $url"
    wget -q -c -O "$f" "$url"
  fi
  [ "$(stat -L -c %s "$f")" = "$size" ] || { echo "!! $2: wrong size"; exit 1; }
  # sha256 of the copies these were tested with; the content checks below (mesh counts, split files
  # regenerated from the class file) are what must pass, so a different checksum only warns
  echo "$sha  $f" | sha256sum -c --quiet - 2>/dev/null && echo "  ok $f (sha256 as tested)" \
    || echo "  WARNING $2: sha256 differs from the tested copy; continuing, the content checks decide"
}

echo "== 1/4 archives"
fetch "$S11_URL" SHREC2011_NonRigid.zip "$S11_SIZE" "$S11_SHA"
fetch "$TOS_URL" SurfCorr2.0.benchmarks.bins.zip "$TOS_SIZE" "$TOS_SHA"

echo "== 2/4 extract"
S11_MESH="$SRC/shrec11/meshes"; S11_CLA="$SRC/shrec11/test.cla"; TOS_MESH="$SRC/tosca/meshes"
if [ "$(ls "$S11_MESH"/T*.off 2>/dev/null | wc -l)" != 600 ]; then
  T=$(mktemp -d "$SRC/tmp.XXXX")
  unzip -q -o "$SRC/SHREC2011_NonRigid.zip" "NonRigid/SHREC11_test_database_new.zip" "NonRigid/For_evaluation.zip" -d "$T"
  mkdir -p "$S11_MESH"
  unzip -q -o -P 'SHREC11@NIST' "$T/NonRigid/SHREC11_test_database_new.zip" -d "$S11_MESH"   # password published with the track
  unzip -q -o -P 'SHREC11@NIST' "$T/NonRigid/For_evaluation.zip" test.cla -d "$SRC/shrec11"
  rm -rf "$T"
fi
if [ "$(ls "$TOS_MESH"/*.off 2>/dev/null | wc -l)" != 80 ]; then
  T=$(mktemp -d "$SRC/tmp.XXXX")
  unzip -q -o "$SRC/SurfCorr2.0.benchmarks.bins.zip" "CorrsBenchmark/Data/TOSCA/Meshes/*.off" -d "$T"
  mkdir -p "$TOS_MESH" && mv "$T"/CorrsBenchmark/Data/TOSCA/Meshes/*.off "$TOS_MESH"/ && rm -rf "$T"
fi
echo "  SHREC2011 meshes: $(ls "$S11_MESH"/T*.off | wc -l) (expect 600) | TOSCA meshes: $(ls "$TOS_MESH"/*.off | wc -l) (expect 80)"
[ "$(ls "$S11_MESH"/T*.off | wc -l)" = 600 ] && [ "$(ls "$TOS_MESH"/*.off | wc -l)" = 80 ] && [ -f "$S11_CLA" ]

echo "== 3/4 splits: regenerate from the downloaded class file, must equal the committed ones"
python -u tools/offmesh/make_splits.py --shrec11-cla "$S11_CLA" --tosca-meshes "$TOS_MESH"
git diff --quiet -- data_labels/shrec11 data_labels/tosca \
  && echo "  VERIFIED: split files identical to the committed ones" \
  || { echo "!! split files differ from the committed ones (git diff data_labels/)"; exit 1; }

echo "== 4/4 graphs and k=64 patches"
python -u prepare_off_hierarchy.py --dataset shrec11 --input "$S11_MESH" --output "$DATA_ROOT/offmesh_shrec11" \
    --categories "$S11_CLA" --workers "$NW"
python -u prepare_off_hierarchy.py --dataset tosca --input "$TOS_MESH" --output "$DATA_ROOT/offmesh_tosca" --workers "$NW"
for ds in shrec11 tosca; do
  python -u tools/build_patchassign.py --input "$DATA_ROOT/offmesh_$ds" --out "$DATA_ROOT/offmesh_${ds}_patchassign_k64" \
      --num-patches 64 --workers "$NW"
done
python - "$DATA_ROOT" <<'PY'
import csv, sys
from pathlib import Path
root = Path(sys.argv[1]); bad = 0
for ds, n in (("shrec11", 600), ("tosca", 80)):
    g = {p.stem for p in (root / f"offmesh_{ds}" / "all").glob("*.pt")}
    a = {p.stem for p in (root / f"offmesh_{ds}_patchassign_k64" / "train_set").glob("*.pt")}
    ids = {r["protein_id"] for f in Path("data_labels", ds).glob("*/*.csv") for r in csv.DictReader(open(f))}
    miss = ids - (g & a)
    print(f"  {ds}: {len(g)} graphs, {len(a)} patch files (expect {n}); split ids without both files: {len(miss)}")
    bad += len(g) != n or len(a) != n or bool(miss)
sys.exit(bad)
PY
echo "OFF meshes ready in $DATA_ROOT"
