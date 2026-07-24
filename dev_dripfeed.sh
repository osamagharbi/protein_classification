#!/bin/bash
# ===========================================================================
# dev_dripfeed.sh — run a big grid study on qos_gpu-dev DESPITE its ~10-job
# submit cap, by drip-feeding small submissions and keeping <= MAX_INFLIGHT
# tasks in flight at any moment.
#
# Each "unit" = one (patch-k, backbone, hidden) submitted as an array over the
# seeds (default 10 tasks). Before each unit the script waits until there is
# room under MAX_INFLIGHT, then submits (NO_AGG, no per-cell agg job). Run it in
# tmux/screen — it is a light sleep+poll loop (no compute) and just calls
# squeue/sbatch periodically, which is fine on a login node.
#
# Usage (edit the cell, then run inside tmux):
#   STUDY=shrec19_bench BACKBONE_SET=safe \
#   DATASET=shrec19_protein FEATURE_GROUP=geom \
#   PATCH_K_LIST="8 16 32 64 128 256" \
#   ./dev_dripfeed.sh
#
# When it finishes submitting everything, aggregate once at the end:
#   for K in $PATCH_K_LIST; do
#     python grid_hiergin.py --mode agg --dataset $DATASET \
#       --feature-group $FEATURE_GROUP --patch-k $K \
#       --grid-root runs/$STUDY/${DATASET}_${FEATURE_GROUP}_k$K ; done
# ===========================================================================
set -uo pipefail

# ------------------------------- the cell ----------------------------------
STUDY="${STUDY:-shrec19_bench}"
DATASET="${DATASET:-shrec19_protein}"
FEATURE_GROUP="${FEATURE_GROUP:-geom}"
PATCH_K_LIST="${PATCH_K_LIST:-8 16 32 64 128 256}"
HIDDEN_DIMS="${HIDDEN_DIMS:-128 256 512}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"

# backbones: BACKBONE_SET picks a group, or override BACKBONES directly
BACKBONE_SET="${BACKBONE_SET:-safe}"
if [ -z "${BACKBONES:-}" ]; then
  case "$BACKBONE_SET" in
    safe)  BACKBONES="gin sage gcn gat gatv2 pna msg_gin gvp" ;;
    risky) BACKBONES="sgformer gt gps" ;;
    all)   BACKBONES="gin sage gcn gat gatv2 pna msg_gin gvp sgformer gt gps" ;;
    *) echo "bad BACKBONE_SET=$BACKBONE_SET"; exit 1 ;;
  esac
fi

# ------------------------------- dev limits --------------------------------
MAX_INFLIGHT="${MAX_INFLIGHT:-10}"   # dev cap on running+pending tasks (jobs)
POLL="${POLL:-60}"                    # seconds between slot checks
TIME="${TIME:-02:00:00}"             # dev wall cap
PROJECT="${PROJECT:-$WORK/protein_classification}"
# ---------------------------------------------------------------------------

cd "$PROJECT"
UNIT=$(echo $SEEDS | wc -w)          # tasks submitted per unit (= #seeds)
PREFIX="HGRID_${DATASET}_${FEATURE_GROUP}"

if [ "$UNIT" -gt "$MAX_INFLIGHT" ]; then
  echo "!! a unit is $UNIT tasks (#seeds) but MAX_INFLIGHT=$MAX_INFLIGHT — reduce SEEDS."
  exit 1
fi

# count MY running+pending dev tasks for this study (array elements expanded with -r)
inflight() { squeue -u "$USER" -r -h -t RUNNING,PENDING -o "%j" 2>/dev/null | grep -c "^${PREFIX}"; }

# build the full unit list
UNITS=()
for K in $PATCH_K_LIST; do
  for BB in $BACKBONES; do
    for H in $HIDDEN_DIMS; do
      UNITS+=("$K|$BB|$H")
    done
  done
done
TOTAL=${#UNITS[@]}
echo "drip-feed: study=$STUDY $DATASET/$FEATURE_GROUP  set=$BACKBONE_SET"
echo "units=$TOTAL  (each $UNIT seed-tasks)  MAX_INFLIGHT=$MAX_INFLIGHT  poll=${POLL}s"
echo

i=0
for U in "${UNITS[@]}"; do
  K="${U%%|*}"; rest="${U#*|}"; BB="${rest%%|*}"; H="${rest##*|}"
  i=$((i+1))

  # wait for a slot: keep total in flight <= MAX_INFLIGHT - UNIT
  while [ "$(inflight)" -gt "$(( MAX_INFLIGHT - UNIT ))" ]; do
    sleep "$POLL"
  done

  echo "[$i/$TOTAL] submit k=$K bb=$BB h=$H  (inflight=$(inflight))"
  STUDY="$STUDY" BACKBONE_SET="$BACKBONE_SET" BACKBONES="$BB" \
  HIDDEN_DIMS="$H" SEEDS="$SEEDS" PATCH_K_LIST="$K" \
  DATASET="$DATASET" FEATURE_GROUP="$FEATURE_GROUP" \
  QOS=qos_gpu-dev TIME="$TIME" CONCURRENCY="$UNIT" NO_AGG=1 SERIALIZE=0 \
  ./submit_grid_hiergin.sh 2>&1 | sed -n 's/^  array/    array/p'

  if [ $? -ne 0 ]; then
    echo "!! submission failed at unit $i (k=$K bb=$BB h=$H). Stopping so you can inspect."
    exit 1
  fi
  sleep 3
done

echo
echo "=== all $TOTAL units submitted. Let dev drain, then aggregate:"
echo "for K in $PATCH_K_LIST; do python grid_hiergin.py --mode agg \\"
echo "  --dataset $DATASET --feature-group $FEATURE_GROUP --patch-k \$K \\"
echo "  --grid-root runs/$STUDY/${DATASET}_${FEATURE_GROUP}_k\$K ; done"
