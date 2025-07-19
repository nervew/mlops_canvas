# src/app/m06__feature_selection/step04_abess/core/abess_selector.py

from __future__ import annotations
from typing import List, Tuple
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
            else (coef != 0).any(axis=1)
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
    Ajusta AbessSelector sobre X_train/y_train y aplica la misma selección
    a X_test y X_backtest.

    Retorna: (X_train_sel, X_test_sel, X_backtest_sel, selector_ajustado)
    """
    selector = AbessSelector(mode=mode)
    X_tr = selector.fit_transform(X_train, y_train)
    X_te = selector.transform(X_test)
    X_ba = selector.transform(X_backtest)
    return X_tr, X_te, X_ba, selector
