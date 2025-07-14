from __future__ import annotations

import pandas as pd

from feature_selection_filter.ports.selector import IFeatureSelector


class RoughSetSelector(IFeatureSelector):
    """Placeholder Rough Set selector that keeps all columns."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RoughSetSelector":
        self.cols = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.cols]
