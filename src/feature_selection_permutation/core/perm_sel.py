from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

from feature_selection_permutation.ports.selector import IFeatureSelector


class PermutationSelector(IFeatureSelector):
    """Final check using permutation importance."""

    def __init__(self, tol: float = 0.0) -> None:
        self.tol = tol
        self.model = RandomForestClassifier(random_state=0)
        self.selected_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PermutationSelector":
        self.model.fit(X, y)
        result = permutation_importance(self.model, X, y, n_repeats=5, random_state=0)
        importances = result.importances_mean
        self.selected_cols = [
            c for c, imp in zip(X.columns, importances) if imp > self.tol
        ]
        if not self.selected_cols:
            self.selected_cols = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_cols]
