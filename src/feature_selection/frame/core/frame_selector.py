from __future__ import annotations

from typing import List

import pandas as pd
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

from feature_selection.ports.selector import IFeatureSelector


class FrameSelector(IFeatureSelector):
    """Hybrid forward selection followed by RFE."""

    def __init__(self, forward_k: int = 3, final_k: int = 2) -> None:
        self.forward_k = forward_k
        self.final_k = final_k
        self.selected_cols: List[str] = []
        self.estimator = LogisticRegression(max_iter=200)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FrameSelector":
        n_forward = max(1, min(self.forward_k, X.shape[1] - 1))
        sfs = SequentialFeatureSelector(
            self.estimator,
            n_features_to_select=n_forward,
            direction="forward",
        )
        sfs.fit(X, y)
        cols = X.columns[sfs.get_support()].tolist()
        n_final = max(1, min(self.final_k, len(cols)))
        rfe = RFE(self.estimator, n_features_to_select=n_final)
        rfe.fit(X[cols], y)
        self.selected_cols = [c for c, keep in zip(cols, rfe.support_) if keep]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_cols]
