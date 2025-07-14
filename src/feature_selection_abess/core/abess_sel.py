from __future__ import annotations

from typing import Optional

import pandas as pd
from abess.linear import MultinomialRegression

from feature_selection_abess.ports.selector import IFeatureSelector


class AbessSelector(IFeatureSelector):
    """Feature selection using ABESS logistic regression."""

    def __init__(self, support_size: Optional[int] = None) -> None:
        self.support_size = support_size
        self.model: Optional[MultinomialRegression] = None
        self.selected_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AbessSelector":
        self.model = MultinomialRegression(support_size=self.support_size)
        self.model.fit(X, y)
        support = (self.model.coef_ != 0).any(axis=1)
        self.selected_cols = list(X.columns[support])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_cols]
