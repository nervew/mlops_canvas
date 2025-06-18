"""Model training utilities."""
from __future__ import annotations

from typing import Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[RandomForestClassifier, dict]:
    """Train RandomForest with hyperparameter search."""
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    param_dist = {
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "n_estimators": [50, 100, 200],
    }
    search = RandomizedSearchCV(
        model,
        param_dist,
        n_iter=5,
        cv=3,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    return best_model, search.best_params_
