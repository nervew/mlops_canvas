from __future__ import annotations
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ..ports.selector import IFeatureSelector


class ShapSelector(IFeatureSelector):
    """
    Selector basado en valores SHAP (importancia interpretativa de features).
    Mantiene columnas no-numéricas y aplica SHAP solo a numéricas.
    """
    def __init__(
        self,
        top_n: int = 5,
        model: Optional[BaseEstimator] = None,
        task: str = "classification",
        random_state: int = 0
    ) -> None:
        self.top_n = top_n
        self.task = task
        # Modelo flexible (clasificación o regresión)
        if model:
            self.model = model
        else:
            if task == "classification":
                self.model = RandomForestClassifier(random_state=random_state)
            elif task == "regression":
                self.model = RandomForestRegressor(random_state=random_state)
            else:
                raise ValueError("task debe ser 'classification' o 'regression'.")
        self.selected_cols: List[str] = []
        self.num_cols: List[str] = []
        self.non_num_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ShapSelector:
        # 1) Detectar columnas numéricas y no-numéricas
        self.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.non_num_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # 2) Ajustar el modelo solo en numéricas
        X_num = X[self.num_cols]
        self.model.fit(X_num, y)

        # 3) Calcular valores SHAP
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_num)

        # 4) Unificar array y calcular importancia media absoluta
        arr = np.array(shap_values)
        if arr.ndim == 3:
            # multiclass: (n_classes, n_samples, n_features)
            imp = np.abs(arr).mean(axis=0).mean(axis=1)
        else:
            # regresión o binaria: (n_samples, n_features)
            imp = np.abs(arr).mean(axis=0)

        # 5) Seleccionar top_n indices y nombres
        idxs = np.argsort(imp)[::-1][:self.top_n]
        self.selected_cols = [self.num_cols[i] for i in idxs]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_cols:
            raise RuntimeError("Debe ejecutar primero .fit() o .fit_transform().")
        # 1) Extraer seleccionadas numéricas
        X_num_sel = X[self.selected_cols]
        # 2) Volver a pegar no-numéricas intactas
        X_non = X[self.non_num_cols]
        # 3) Reorden: primero seleccionadas, luego no-numéricas (orden original)
        cols_order = self.selected_cols + self.non_num_cols
        return pd.concat([X_num_sel, X_non], axis=1)[cols_order]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)


def shap_partitions(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_backtest: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = 5,
    task: str = "classification",
    random_state: int = 0,
    model: Optional[BaseEstimator] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ShapSelector]:
    """
    Ejecuta selección SHAP entrenando en train y aplicando la misma transformación
    a test y backtest. Mantiene columnas no numéricas intactas.

    Retorna:
      X_train_sel, X_test_sel, X_backtest_sel, selector_ajustado
    """
    selector = ShapSelector(
        top_n=top_n, task=task, random_state=random_state, model=model
    )
    X_tr_sel = selector.fit_transform(X_train, y_train)
    X_te_sel = selector.transform(X_test)
    X_ba_sel = selector.transform(X_backtest)
    return X_tr_sel, X_te_sel, X_ba_sel, selector
