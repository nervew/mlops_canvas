# /Workspace/Users/jorgee.lopez@adres.gov.co/mlops_canvas/src/app/
#    m06__feature_selection/step03_frame/core/frame_selector.py

from __future__ import annotations

from typing import List, Tuple, Optional
import pandas as pd
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.base import BaseEstimator
from ..ports.selector import IFeatureSelector


class FrameSelector(IFeatureSelector):
    """
    Selección híbrida de características:
      1) SequentialFeatureSelector (forward) para quedarnos con `forward_k` features.
      2) RFE (Recursive Feature Elimination) para refinar a `final_k` features.

    Parámetros:
      - estimator: cualquier estimador compatible con fit()/predict().
        Si es None, por defecto LogisticRegression(max_iter=200).
      - forward_k: número de features en el paso forward.
      - final_k: número final de features tras RFE.
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
        # Paso 1: selección forward
        n_forward = max(1, min(self.forward_k, X.shape[1] - 1))
        sfs = SequentialFeatureSelector(
            self.estimator,
            n_features_to_select=n_forward,
            direction="forward",
        )
        sfs.fit(X, y)
        cols_forward = X.columns[sfs.get_support()].tolist()

        # Paso 2: RFE sobre ese subconjunto
        n_final = max(1, min(self.final_k, len(cols_forward)))
        rfe = RFE(self.estimator, n_features_to_select=n_final)
        rfe.fit(X[cols_forward], y)
        self.selected_cols = [
            feat for feat, keep in zip(cols_forward, rfe.support_) if keep
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_cols:
            raise RuntimeError("Debe ejecutar primero .fit() o .fit_transform()")
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
    Ajusta FrameSelector con X_train/y_train y aplica idéntica transformación
    a X_train, X_test y X_backtest.

    Parámetros extra (opcionalmente los pasas al selector):
      - estimator: estimador a usar (por defecto LogisticRegression).
      - forward_k, final_k: configuran los pasos de selección.

    Retorna:
      X_train_sel, X_test_sel, X_backtest_sel, selector_ajustado
    """
    selector = FrameSelector(estimator=estimator, forward_k=forward_k, final_k=final_k)
    X_tr = selector.fit_transform(X_train, y_train)
    X_te = selector.transform(X_test)
    X_ba = selector.transform(X_backtest)
    return X_tr, X_te, X_ba, selector
