from __future__ import annotations

from typing import List

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

from feature_selection.ports.selector import IFeatureSelector


class PermutationSelector(IFeatureSelector):
    """Remove features with negligible permutation importance."""

    def __init__(self, tol: float = 0.0) -> None:
        self.tol = tol
        self.model = RandomForestClassifier(random_state=0)
        self.selected_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PermutationSelector":
        self.model.fit(X, y)
        result = permutation_importance(self.model, X, y, n_repeats=5, random_state=0)
        self.selected_cols = [
            col
            for col, imp in zip(X.columns, result.importances_mean)
            if imp > self.tol
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_cols]
