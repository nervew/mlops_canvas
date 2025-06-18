from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from typing import Any


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: str = "classification",
) -> Any:
    """Train a model according to the task type."""
    if task_type == "classification":
        base_model = RandomForestClassifier(random_state=42)
    else:
        base_model = RandomForestRegressor(random_state=42)

    grid = GridSearchCV(
        base_model,
        param_grid={"n_estimators": [50, 100], "max_depth": [None, 5]},
        cv=3,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)
    return best_model
