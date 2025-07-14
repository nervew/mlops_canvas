from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from feature_selection.ports.selector import IFeatureSelector


class FilterRoughFS(IFeatureSelector):
    """Initial filtering using simple statistics."""

    def __init__(self, corr_threshold: float = 0.9) -> None:
        self.corr_threshold = corr_threshold
        self.selected_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FilterRoughFS":
        df = X.copy()
        # remove constant columns
        constant_selector = VarianceThreshold(threshold=0.0)
        constant_selector.fit(df)
        df = df[df.columns[constant_selector.get_support()]]

        # remove highly correlated features
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [
            col for col in upper.columns if any(upper[col] > self.corr_threshold)
        ]
        self.selected_cols = [c for c in df.columns if c not in to_drop]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_cols]
