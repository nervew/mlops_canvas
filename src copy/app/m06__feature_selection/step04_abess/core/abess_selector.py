# src/app/m06__feature_selection/step04_abess/core/abess_selector.py

from __future__ import annotations
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LassoCV

from ..ports.selector import IFeatureSelector


class AbessSelector(IFeatureSelector):
    """
    Selección de variables con ABESS (o LassoCV si no está abess).
    mode="regression": usa abess.linear.LinearRegression si está disponible,
                       si no, LassoCV(cv=5).
    mode="classification": usa abess.linear.abessMultinomial.
    """
    def __init__(self, mode: str = "regression") -> None:
        self.mode = mode
        if mode == "regression":
            try:
                from abess.linear import LinearRegression as AbessLinearRegression
                self.model: BaseEstimator = AbessLinearRegression()
            except ImportError:
                self.model = LassoCV(cv=5)
        elif mode == "classification":
            from abess.linear import abessMultinomial
            self.model = abessMultinomial()
        else:
            raise ValueError("mode debe ser 'regression' o 'classification'.")
        self.selected_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AbessSelector":
        self.model.fit(X, y)
        coef = getattr(self.model, "coef_", None)
        if coef is None:
            raise RuntimeError("El modelo no tiene atributo coef_.")
        # coef puede ser 1d (regresión) o 2d (clasificación)
        mask = (
            (coef != 0)
            if coef.ndim == 1
            else (coef != 0).any(axis=0)
        )
        self.selected_cols = X.columns[mask].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_cols:
            raise RuntimeError("Ejecuta primero .fit() o .fit_transform().")
        return X[self.selected_cols]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)


def abess_partitions(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_backtest: pd.DataFrame,
    y_train: pd.Series,
    *,
    mode: str = "regression",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, AbessSelector]:
    """
    1) Separa columnas numéricas y no-numéricas de X_train.
    2) Ajusta AbessSelector solo sobre columnas numéricas de train.
    3) Transforma train/test/backtest numéricas.
    4) Vuelve a pegar las columnas no-numéricas intactas.
    5) Reordena: primero las seleccionadas, luego las no-numéricas.
    """
    # 1) Detectar numéricas vs no-numéricas
    num_cols     = X_train.select_dtypes(include=[np.number]).columns.tolist()
    non_num_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    # 2) Ajustar selector sobre lo numérico
    selector = AbessSelector(mode=mode)
    X_tr_num = selector.fit_transform(X_train[num_cols], y_train)
    X_te_num = selector.transform(X_test[num_cols])
    X_ba_num = selector.transform(X_backtest[num_cols])

    # 3) Reensamblar con las columnas no-numéricas
    X_tr = pd.concat([X_tr_num, X_train[non_num_cols]], axis=1)
    X_te = pd.concat([X_te_num, X_test[non_num_cols]],  axis=1)
    X_ba = pd.concat([X_ba_num, X_backtest[non_num_cols]], axis=1)

    # 4) Reordenar: seleccionadas + no-numéricas (en orden original)
    final_order = selector.selected_cols + non_num_cols
    X_tr = X_tr[final_order]
    X_te = X_te[final_order]
    X_ba = X_ba[final_order]

    return X_tr, X_te, X_ba, selector
