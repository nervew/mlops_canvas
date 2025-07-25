# /.../m06__feature_selection/step03_frame/core/frame_selector.py

from __future__ import annotations
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.base import BaseEstimator

from ..ports.selector import IFeatureSelector


class FrameSelector(IFeatureSelector):
    """
    Selección híbrida de características sobre datos numéricos:
      1) Forward selection (SequentialFeatureSelector) hasta forward_k.
      2) RFE (Recursive Feature Elimination) hasta final_k.
    """
    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        forward_k: int = 3,
        final_k: int = 2,
    ) -> None:
        from sklearn.linear_model import LogisticRegression

        self.estimator = estimator or LogisticRegression(max_iter=200)
        self.forward_k = forward_k
        self.final_k = final_k
        self.selected_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FrameSelector":
        # 1) Forward selection
        n_forward = max(1, min(self.forward_k, X.shape[1] - 1))
        sfs = SequentialFeatureSelector(
            self.estimator,
            n_features_to_select=n_forward,
            direction="forward",
        )
        sfs.fit(X, y)
        cols_forward = X.columns[sfs.get_support()].tolist()

        # 2) RFE sobre ese subconjunto
        n_final = max(1, min(self.final_k, len(cols_forward)))
        rfe = RFE(self.estimator, n_features_to_select=n_final)
        rfe.fit(X[cols_forward], y)
        self.selected_cols = [
            feat for feat, keep in zip(cols_forward, rfe.support_) if keep
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_cols:
            raise RuntimeError("Ejecuta primero fit() o fit_transform()")
        return X[self.selected_cols]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)


def frame_partitions(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_backtest: pd.DataFrame,
    y_train: pd.Series,
    *,
    estimator: Optional[BaseEstimator] = None,
    forward_k: int = 3,
    final_k: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FrameSelector]:
    """
    1) Separa columnas numéricas y no-numéricas de X_train.
    2) Ajusta FrameSelector sólo sobre las columnas numéricas de train.
    3) Transforma train/test/backtest numéricas con el selector.
    4) Vuelve a pegar las columnas no-numéricas intactas.
    5) Reordena: primero las features seleccionadas, luego las no-numéricas.
    """
    # 1) detecta numéricas vs no-numéricas
    num_cols     = X_train.select_dtypes(include=[np.number]).columns.tolist()
    non_num_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    # 2) crea y ajusta selector sobre datos numéricos
    selector = FrameSelector(
        estimator=estimator,
        forward_k=forward_k,
        final_k=final_k,
    )
    # fit_transform actúa sólo sobre numéricas
    X_tr_num = selector.fit_transform(X_train[num_cols], y_train)
    X_te_num = selector.transform(X_test[num_cols])
    X_ba_num = selector.transform(X_backtest[num_cols])

    # 3) reensambla con las no-numéricas
    X_tr = pd.concat([X_tr_num, X_train[non_num_cols]], axis=1)
    X_te = pd.concat([X_te_num, X_test[non_num_cols]],  axis=1)
    X_ba = pd.concat([X_ba_num, X_backtest[non_num_cols]], axis=1)

    # 4) reordena columnas: seleccionadas + no-numéricas
    new_order = selector.selected_cols + non_num_cols
    X_tr = X_tr[new_order]
    X_te = X_te[new_order]
    X_ba = X_ba[new_order]

    # 5) devuelve también el selector (fitted) para inspección
    return X_tr, X_te, X_ba, selector
