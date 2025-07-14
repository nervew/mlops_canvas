from __future__ import annotations

from typing import List

import pandas as pd
from abess.linear import abessMultinomial

from feature_selection.ports.selector import IFeatureSelector


class AbessSelector(IFeatureSelector):
    """Selection based on ABESS algorithm."""

    def __init__(self) -> None:
        self.model = abessMultinomial()
        self.selected_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AbessSelector":
        self.model.fit(X, y)
        non_zero = (self.model.coef_ != 0).any(axis=1)
        self.selected_cols = X.columns[non_zero].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_cols]
