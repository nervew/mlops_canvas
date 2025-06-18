from __future__ import annotations

import pandas as pd
from typing import Any
from sklearn.metrics import accuracy_score, r2_score


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str = "classification",
) -> float:
    """Return evaluation metric based on task type."""
    if task_type == "classification":
        y_pred = model.predict(X_test)
        return float(accuracy_score(y_test, y_pred))
    else:
        y_pred = model.predict(X_test)
        return float(r2_score(y_test, y_pred))
