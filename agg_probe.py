#!/usr/bin/env python3
"""
Collect every probe_checkpoint.py output (one final_results_corrected_test.json
per completed task, under --out-root) into a single CSV with the ORIGINAL
(buggy-test-set) metrics and the CORRECTED-test metrics side by side, plus the
delta, so the paper's affected tables can be updated directly.

Usage:
  python agg_probe.py --out-root runs/shrec19_probe_corrected_test/results \
                       --csv master_summary_corrected_test.csv
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
           "tpr_macro", "tnr_macro", "mAP"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True, help="dir of *.json from probe_checkpoint.py")
    ap.add_argument("--csv", default="master_summary_corrected_test.csv")
    args = ap.parse_args()

    rows = []
    for fp in sorted(Path(args.out_root).glob("*.json")):
        d = json.load(open(fp))
        cfg = d["config"]
        orig = d["orig_final_test_metrics"]
        new = d["corrected_test_metrics"]
        row = {
            "tag": fp.stem,
            "n_test_corrected": d["n_test_corrected"],
        }
        for k in METRICS:
            row[f"{k}_orig"] = orig.get(k)
            row[f"{k}_corrected"] = new.get(k)
            if isinstance(orig.get(k), (int, float)) and isinstance(new.get(k), (int, float)):
                row[f"{k}_delta"] = new[k] - orig[k]
            else:
                row[f"{k}_delta"] = None
        rows.append(row)

    if not rows:
        print(f"[agg-probe] no *.json found under {args.out_root}")
        return

    fieldnames = ["tag", "n_test_corrected"] + [
        f"{k}_{s}" for k in METRICS for s in ("orig", "corrected", "delta")
    ]
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[agg-probe] {len(rows)} rows -> {args.csv}")

    biggest = sorted(rows, key=lambda r: abs(r["mAP_delta"] or 0), reverse=True)[:10]
    print("\nLargest |mAP delta| (corrected - orig):")
    for r in biggest:
        print(f"  {r['tag']:55s}  mAP {r['mAP_orig']:.4f} -> {r['mAP_corrected']:.4f}  "
              f"(delta {r['mAP_delta']:+.4f})")


if __name__ == "__main__":
    main()
