from __future__ import annotations

import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

from feature_selection_shap.ports.selector import IFeatureSelector


class ShapSelector(IFeatureSelector):
    """Feature selection using shap-select on a GradientBoosting model."""

    def __init__(self, threshold: float = 0.01) -> None:
        self.threshold = threshold
        self.model = RandomForestClassifier(n_estimators=100, random_state=0)
        self.selected_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ShapSelector":
        self.model.fit(X, y)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_array = sum(abs(sv) for sv in shap_values) / len(shap_values)
        else:
            shap_array = abs(shap_values)
        mean_abs = shap_array.mean(axis=0)
        if mean_abs.ndim > 1:
            mean_abs = mean_abs.mean(axis=1)
        self.selected_cols = [
            col for col, val in zip(X.columns, mean_abs) if val > self.threshold
        ]
        if not self.selected_cols:
            self.selected_cols = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_cols]
