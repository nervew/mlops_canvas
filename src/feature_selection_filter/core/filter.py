from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

from feature_selection_filter.ports.selector import IFeatureSelector
from feature_selection_filter.adapters.rough_set import RoughSetSelector


class FilterSelector(IFeatureSelector):
    """Initial filtering step using variance, correlation and ANOVA."""

    def __init__(
        self, k_best: Optional[int] = None, corr_threshold: float = 0.9
    ) -> None:
        self.k_best = k_best
        self.corr_threshold = corr_threshold
        self.var_sel = VarianceThreshold()
        self.kbest_sel: Optional[SelectKBest] = None
        self.rough_sel = RoughSetSelector()
        self.selected_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FilterSelector":
        data = self.var_sel.fit_transform(X)
        cols = X.columns[self.var_sel.get_support()].tolist()
        df = pd.DataFrame(data, columns=cols)

        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [
            column
            for column in upper.columns
            if any(upper[column] > self.corr_threshold)
        ]
        df = df.drop(columns=to_drop)

        k = self.k_best or max(1, int(len(df.columns) / 2))
        self.kbest_sel = SelectKBest(score_func=f_classif, k=min(k, len(df.columns)))
        self.kbest_sel.fit(df, y)
        df = df[df.columns[self.kbest_sel.get_support()]]
        self.rough_sel.fit(df, y)
        self.selected_cols = self.rough_sel.cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.kbest_sel is not None
        data = self.var_sel.transform(X)
        df = pd.DataFrame(data, columns=X.columns[self.var_sel.get_support()])
        df = df.drop(
            columns=[c for c in df.columns if c not in self.kbest_sel.feature_names_in_]
        )
        df = df[self.selected_cols]
        return df
