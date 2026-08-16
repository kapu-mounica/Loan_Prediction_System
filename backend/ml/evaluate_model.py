"""Binary classification evaluation (positive class = "Approved").

Computes the same metric set as the TypeScript engine (src/ml/metrics.ts):
accuracy, precision, recall, F1-score, ROC-AUC and the confusion matrix, on
the held-out test set.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   proba: np.ndarray) -> Dict:
    """Return the full metric dict for a fitted estimator on a test set.

    ``proba`` is the precomputed predicted probability of the positive class
    (``model.predict_proba(X_test)[:, 1]``) so callers can reuse it for the
    ROC-AUC / selection logic without recomputation.
    """
    y_hat = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_hat, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_test, y_hat)), 4),
        "precision": round(float(precision_score(y_test, y_hat, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_hat, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_hat, zero_division=0)), 4),
        "rocAuc": round(float(roc_auc_score(y_test, proba)), 4),
        "confusionMatrix": [int(tn), int(fp), int(fn), int(tp)],
        "n": int(len(y_test)),
    }


def describe_model(name: str, metrics: Dict, train_accuracy: float) -> str:
    tn, fp, fn, tp = metrics["confusionMatrix"]
    return (
        f"{name}: acc={metrics['accuracy'] * 100:.2f}% "
        f"prec={metrics['precision'] * 100:.2f}% "
        f"rec={metrics['recall'] * 100:.2f}% "
        f"f1={metrics['f1'] * 100:.2f}% auc={metrics['rocAuc'] * 100:.2f}% "
        f"train_acc={train_accuracy * 100:.2f}% cm=[[{tn},{fp}],[{fn},{tp}]]"
    )
