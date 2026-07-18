#!/bin/bash
# ============================================================================
# run_flat_vs_hier_matched.sh
#
# Matched flat-GNN vs HierGIN comparison on the 5 INTERSECTION backbones
# {gin, gcn, gat, sage, sgformer}. Isolates the architecture effect
# (hierarchical patch-pooling ON vs OFF) at an otherwise identical config.
#
# HierGIN side already complete (runs/hier_bb_*_cosine). This script fills the
# MISSING flat-GNN side, one run at a time.
#
# Design guarantees:
#   * SEQUENTIAL  — bash waits for each training to finish; never 2 GPU jobs.
#   * GPU-SAFE    — gates on free VRAM before each launch (no OOM from a stray job).
#   * NOT UNDERUSED — bs=64 / h=512 (matches HierGIN; ~4-6 GB of 12 GB), workers=8.
#   * RESUMABLE   — skips a backbone whose final_results.json already exists (FORCE=1 to redo).
#   * OBSERVABLE  — live STATUS.txt, per-run logs, accumulating results.tsv.
#
# Matched config (identical to hier_bb_*_cosine except two documented caveats):
#   data=ssl_graphs_1k · features=4D coords_potential (Potential_norm + xyz)
#   h=512 · node-layers=3 · ce loss · inverse_sqrt weights · AdamW+cosine
#   lr=5e-4 · wd=1e-4 · 80 epochs · seed=42
#   CAVEAT A: HierGIN has a 5-epoch LR warmup; flat does not.
#   CAVEAT B: flat clips Potential to +/-5 sigma; HierGIN coords_potential does not
#             (affects <1% outlier nodes on 1 of 4 features).
# ============================================================================
set -uo pipefail   # deliberately NOT -e: keep going if one run fails, and log it.

PY=/home/ogharbi/miniconda3/envs/graphmae/bin/python3
FLAT=/home/ogharbi/projects/protein_classification/classify_gnn.py
DATA=/home/ogharbi/datasets/SHREC25/ssl_graphs_1k
RUNS=/home/ogharbi/projects/protein_classification/runs

# ---- matched hyperparameters -------------------------------------------------
HID=512; NLAYERS=3; EPOCHS=80; LR=5e-4; WD=1e-4; SEED=42; WORKERS=8
FEAT="0 5 6 7"                       # coords_potential = Potential_norm + xyz
BACKBONES=(gin gcn gat sage sgformer)
declare -A BS=( [gin]=64 [gcn]=64 [gat]=64 [sage]=64 [sgformer]=64 )
GPU_FREE_MIN_MIB=9000                # require >=9 GB free VRAM before launching
FORCE=${FORCE:-0}                    # FORCE=1 to re-run even if results exist

# ---- orchestration workspace -------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
ORCH="$RUNS/flat_vs_hier_matched_$TS"
mkdir -p "$ORCH"
STATUS="$ORCH/STATUS.txt"
RESULTS="$ORCH/results.tsv"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

wait_for_gpu() {
    while true; do
        local free
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
        if [ "${free:-0}" -ge "$GPU_FREE_MIN_MIB" ]; then
            log "GPU ready (${free} MiB free)"; return 0
        fi
        log "GPU busy (${free} MiB free < ${GPU_FREE_MIN_MIB}); waiting 30s..."
        sleep 30
    done
}

metrics_of() {  # $1 = final_results.json -> "f1_macro accuracy"
    "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]))['final_test_metrics'];print(d['f1_macro'],d['accuracy'])" "$1" 2>/dev/null
}

# ---- preflight ---------------------------------------------------------------
log "=== PREFLIGHT ==="
[ -x "$PY" ]                  || { log "FATAL: python not found: $PY"; exit 1; }
[ -f "$FLAT" ]               || { log "FATAL: flat script not found: $FLAT"; exit 1; }
[ -d "$DATA/train_set" ]     || { log "FATAL: train data missing: $DATA/train_set"; exit 1; }
[ -f "$DATA/train_split.csv" ] || { log "FATAL: train_split.csv missing"; exit 1; }
[ -f "$DATA/val_split.csv" ]   || { log "FATAL: val_split.csv missing"; exit 1; }
log "preflight OK — data=$DATA  python=$PY"
log "orchestration dir: $ORCH"
log "backbones: ${BACKBONES[*]}"
log "config: h$HID ${NLAYERS}L 4D(coords_potential) ${EPOCHS}ep lr$LR wd$WD seed$SEED workers$WORKERS"
log "observe with:  tail -f $STATUS   (and per-run logs in $ORCH)"

printf 'backbone\tarch\trun_name\tf1_macro_raw\taccuracy\tstatus\n' > "$RESULTS"

# ---- record the pre-existing HierGIN side for side-by-side -------------------
for bb in "${BACKBONES[@]}"; do
    hr="$RUNS/hier_bb_${bb}_cosine/final_results.json"
    if [ -f "$hr" ]; then
        read -r f1 acc <<< "$(metrics_of "$hr")"
        printf '%s\thier\thier_bb_%s_cosine\t%s\t%s\tPREEXISTING\n' "$bb" "$bb" "$f1" "$acc" >> "$RESULTS"
    else
        printf '%s\thier\thier_bb_%s_cosine\tNA\tNA\tMISSING\n' "$bb" "$bb" >> "$RESULTS"
        log "WARN: expected HierGIN run missing: $hr"
    fi
done

# ---- run the flat side sequentially -----------------------------------------
i=0; n=${#BACKBONES[@]}
for bb in "${BACKBONES[@]}"; do
    i=$((i+1))
    rn="flat_bb_${bb}_4d_matched"
    bs=${BS[$bb]}
    logf="$ORCH/${rn}.log"
    fr="$RUNS/$rn/final_results.json"

    if [ "$FORCE" != "1" ] && [ -f "$fr" ]; then
        read -r f1 acc <<< "$(metrics_of "$fr")"
        printf '%s\tflat\t%s\t%s\t%s\tSKIPPED_EXISTS\n' "$bb" "$rn" "$f1" "$acc" >> "$RESULTS"
        log "--- [$i/$n] FLAT $bb : results exist, skipping (FORCE=1 to redo) ---"
        continue
    fi

    log "--- [$i/$n] FLAT $bb  (bs=$bs) -> $rn ---"
    wait_for_gpu
    log "launching $rn  (log: $logf)"
    "$PY" "$FLAT" \
        --data-dir "$DATA" \
        --model "$bb" \
        --keep-features $FEAT \
        --hidden-dim $HID --num-layers $NLAYERS \
        --class-weight-method inverse_sqrt --loss-type ce \
        --batch-size "$bs" --num-workers $WORKERS \
        --epochs $EPOCHS --lr $LR --weight-decay $WD \
        --seed $SEED \
        --run-name "$rn" --output-dir "$RUNS" --device 0 \
        > "$logf" 2>&1
    rc=$?

    if [ $rc -ne 0 ]; then
        printf '%s\tflat\t%s\tNA\tNA\tFAILED_rc%d\n' "$bb" "$rn" "$rc" >> "$RESULTS"
        log "FLAT $bb FAILED (rc=$rc) — tail of log:"
        tail -n 15 "$logf" | tee -a "$STATUS"
        continue
    fi

    if [ -f "$fr" ]; then
        read -r f1 acc <<< "$(metrics_of "$fr")"
        printf '%s\tflat\t%s\t%s\t%s\tDONE\n' "$bb" "$rn" "$f1" "$acc" >> "$RESULTS"
        log "FLAT $bb DONE  f1_macro(raw)=$f1  acc=$acc"
    else
        printf '%s\tflat\t%s\tNA\tNA\tNO_RESULTS\n' "$bb" "$rn" >> "$RESULTS"
        log "FLAT $bb finished (rc=0) but no final_results.json at $fr"
    fi
done

# ---- final side-by-side ------------------------------------------------------
log "=== ALL RUNS COMPLETE ==="
log "results file: $RESULTS"
{
  echo
  echo "flat vs HierGIN (raw f1_macro / accuracy, matched 4D config):"
  column -t -s $'\t' "$RESULTS"
} | tee -a "$STATUS"
