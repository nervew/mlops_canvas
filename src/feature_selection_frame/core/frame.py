from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SequentialFeatureSelector

from feature_selection_frame.ports.selector import IFeatureSelector


class FrameSelector(IFeatureSelector):
    """Hybrid forward selection followed by RFE."""

    def __init__(self, n_features: Optional[int] = None) -> None:
        self.n_features = n_features
        self.forward_sel: Optional[SequentialFeatureSelector] = None
        self.rfe_sel: Optional[RFE] = None
        self.selected_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FrameSelector":
        estimator = RandomForestClassifier(n_estimators=50, random_state=0)
        max_feats = X.shape[1] - 1 if X.shape[1] > 1 else 1
        n_forward = min(self.n_features or max_feats, max_feats)
        self.forward_sel = SequentialFeatureSelector(
            estimator,
            n_features_to_select=n_forward,
            direction="forward",
        )
        X_fwd = self.forward_sel.fit_transform(X, y)
        cols_fwd = X.columns[self.forward_sel.get_support()]

        self.rfe_sel = RFE(
            estimator,
            n_features_to_select=min(self.n_features or len(cols_fwd), len(cols_fwd)),
        )
        self.rfe_sel.fit(X_fwd, y)
        support = self.rfe_sel.get_support()
        self.selected_cols = list(cols_fwd[support])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.forward_sel is not None
        df = X[self.forward_sel.feature_names_in_[self.forward_sel.get_support()]]
        df = df[self.selected_cols]
        return df
