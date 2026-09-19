"""M4 — SP3 predictive metrics.

All models are scored on probabilities; the 0.5 threshold converts to a class
label. ROC-AUC is reported as NaN when a test block contains a single class.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """Accuracy, precision, recall, F1, ROC-AUC and MCC for one prediction set."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(auc),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
