# /.../m06__feature_selection/step02_filtering/core/filter_roughfs.py

from __future__ import annotations
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from ..ports.selector import IFeatureSelector


class FilterRoughFS(IFeatureSelector):
    """
    Filtro inicial simple:
      1. Elimina columnas de varianza 0.
      2. Elimina una de cada par de columnas con |corr| > corr_threshold.
    Compatible con Scikit-learn (fit / transform).
    """

    def __init__(self, corr_threshold: float = 0.9) -> None:
        self.corr_threshold = corr_threshold
        self.selected_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FilterRoughFS:
        df = X.copy()
        # 1) Elimina constantes
        vt = VarianceThreshold(threshold=0.0)
        vt.fit(df)
        df = df[df.columns[vt.get_support()]]

        # 2) Elimina correladas
        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > self.corr_threshold)]
        self.selected_cols = [c for c in df.columns if c not in to_drop]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_cols:
            raise RuntimeError("Ejecuta primero fit() antes de transform().")
        return X[self.selected_cols]


def filter_partitions(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_backtest: pd.DataFrame,
    y_train: pd.Series | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Pipeline]:
    """
    Ajusta FilterRoughFS usando X_train (+ y_train),
    luego aplica la misma transformación a las tres particiones.
    Devuelve (X_train_f, X_test_f, X_backtest_f, pipeline).
    """
    selector = FilterRoughFS(corr_threshold=0.9)
    pipe = Pipeline([("rough_filter", selector)])

    pipe.fit(X_train, y_train)

    X_tr = pipe.transform(X_train)
    X_te = pipe.transform(X_test)
    X_ba = pipe.transform(X_backtest)

    return X_tr, X_te, X_ba, pipe
