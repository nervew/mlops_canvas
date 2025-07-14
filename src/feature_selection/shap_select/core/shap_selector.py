from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

from feature_selection.ports.selector import IFeatureSelector


class ShapSelector(IFeatureSelector):
    """Select features based on mean absolute SHAP values."""

    def __init__(self, top_n: int = 2) -> None:
        self.top_n = top_n
        self.model = RandomForestClassifier(random_state=0)
        self.selected_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ShapSelector":
        self.model.fit(X, y)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        arr = np.array(shap_values)
        if arr.ndim == 3:
            vals = np.abs(arr).mean(axis=0).mean(axis=1)
        else:
            vals = np.abs(arr).mean(axis=0)
        ranked = np.argsort(vals)[::-1][: self.top_n]
        self.selected_cols = X.columns[ranked].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_cols]
