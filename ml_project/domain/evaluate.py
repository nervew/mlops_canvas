"""Model evaluation utilities."""
from __future__ import annotations

from typing import Any, Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_model(model: Any, X_test, y_test) -> Dict[str, float | None]:
    """Return common classification metrics."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary"),
        "recall": recall_score(y_test, y_pred, average="binary"),
        "f1": f1_score(y_test, y_pred, average="binary"),
    }
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["auc"] = roc_auc_score(y_test, y_proba)
    except AttributeError:
        metrics["auc"] = None
    return metrics
