"""Metric helpers without a hard sklearn dependency."""

from __future__ import annotations

import numpy as np


def classification_metrics(y_true, y_pred, labels=None) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if labels is None:
        labels = sorted(set(int(x) for x in y_true.tolist()))
    labels = list(labels)

    accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0
    f1s = []
    recalls = []
    weighted_sum = 0.0
    total_support = 0
    for c in labels:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        support = int((y_true == c).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1s.append(f1)
        recalls.append(recall)
        weighted_sum += support * f1
        total_support += support

    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "f1_macro": float(np.mean(f1s)) if f1s else 0.0,
        "f1_weighted": float(weighted_sum / total_support) if total_support else 0.0,
        "n_classes": len(labels),
    }
