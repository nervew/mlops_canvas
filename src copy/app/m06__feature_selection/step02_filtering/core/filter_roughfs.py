# /.../m06__feature_selection/step02_filtering/core/filter_roughfs.py

from __future__ import annotations
from typing     import List, Tuple

import numpy       as np
import pandas      as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..ports.selector import IFeatureSelector


class FilterRoughFS(IFeatureSelector):
    """
    1) Elimina columnas constantes (varianza == 0).
    2) Entre pares con |corr| > corr_threshold deja solo una.
    """
    def __init__(self, corr_threshold: float = 0.9) -> None:
        self.corr_threshold = corr_threshold
        self.selected_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FilterRoughFS:
        # 1) Quitar constantes
        df = X.loc[:, X.std() > 0]
        # 2) Quitar correladas
        corr    = df.corr().abs()
        upper   = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > self.corr_threshold)]
        self.selected_cols = [c for c in df.columns if c not in to_drop]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_cols:
            raise RuntimeError("Primero ejecuta fit()")
        return X[self.selected_cols]


class RoughWithDatesTransformer(BaseEstimator, TransformerMixin):
    """
    1) Separa numéricas vs no-numéricas.
    2) Aplica FilterRoughFS sobre lo numérico.
    3) Reconstruye el DataFrame con todas las columnas.
    """
    def __init__(self, corr_threshold: float = 0.9):
        self.corr_threshold = corr_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        # columnas numéricas y no-numéricas
        self.num_cols     = X.select_dtypes(include=[np.number]).columns.tolist()
        self.non_num_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        # ajusta el filtro
        self.filter = FilterRoughFS(corr_threshold=self.corr_threshold)
        self.filter.fit(X[self.num_cols], y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # filtra solo numéricas
        X_num = self.filter.transform(X[self.num_cols])
        # mantiene intactas las no-numéricas
        X_non = X[self.non_num_cols]
        # concatena y reordena
        X_all = pd.concat([X_num, X_non], axis=1)
        order = self.filter.selected_cols + self.non_num_cols
        return X_all[order]


def filter_partitions(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_back: pd.DataFrame,
    y_train: pd.Series | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RoughWithDatesTransformer]:
    """
    Ajusta y devuelve un RoughWithDatesTransformer ya fitteado,
    junto con las tres particiones transformadas.
    """
    transformer = RoughWithDatesTransformer(corr_threshold=0.9)
    transformer.fit(X_train, y_train)

    X_tr = transformer.transform(X_train)
    X_te = transformer.transform(X_test)
    X_ba = transformer.transform(X_back)

    return X_tr, X_te, X_ba, transformer
