# src/app/m06__feature_selection/step06_permutation/core/permutation_selector.py

from __future__ import annotations
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ..ports.selector import IFeatureSelector


class PermutationSelector(IFeatureSelector):
    """
    Selector basado en importancia por permutación,
    separando automáticamente numéricas y no-numéricas.
    """
    def __init__(
        self,
        tol: float = 0.0,
        model: Optional[BaseEstimator] = None,
        task: str = "classification",
        scoring: Optional[str] = None,
        n_repeats: int = 5,
        random_state: int = 0
    ) -> None:
        # umbral para retener features
        self.tol = tol
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.task = task

        # Modelo base
        if model:
            self.model = model
        else:
            if task == "classification":
                self.model = RandomForestClassifier(random_state=random_state)
            elif task == "regression":
                self.model = RandomForestRegressor(random_state=random_state)
            else:
                raise ValueError("task debe ser 'classification' o 'regression'.")

        # Se rellenarán en fit()
        self.num_cols: List[str] = []
        self.non_num_cols: List[str] = []
        self.selected_cols: List[str] = []
        self.importances: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PermutationSelector:
        # 1) Separa numéricas y no-numéricas
        self.num_cols     = X.select_dtypes(include=[np.number]).columns.tolist()
        self.non_num_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # 2) Ajusta el modelo solo con las numéricas
        X_num = X[self.num_cols]
        self.model.fit(X_num, y)

        # 3) Calcula importancia por permutación
        result = permutation_importance(
            self.model, X_num, y,
            scoring=self.scoring,
            n_repeats=self.n_repeats,
            random_state=self.random_state
        )

        # 4) Guarda importancias medias en un Series
        self.importances = pd.Series(result.importances_mean, index=self.num_cols)

        # 5) Selecciona las columnas numéricas cuya importancia > tol
        self.selected_cols = [
            c for c, imp in self.importances.items() if imp > self.tol
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_cols:
            raise RuntimeError("Debe ejecutar primero .fit() o .fit_transform().")
        # 1) Extrae numéricas seleccionadas
        X_num_sel = X[self.selected_cols]
        # 2) Pega no-numéricas intactas
        X_non = X[self.non_num_cols]
        # 3) Reordena: primero las seleccionadas, luego las no-numéricas
        cols = self.selected_cols + self.non_num_cols
        return pd.concat([X_num_sel, X_non], axis=1)[cols]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)


def permutation_partitions(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_backtest: pd.DataFrame,
    y_train: pd.Series,
    tol: float = 0.0,
    task: str = "classification",
    scoring: Optional[str] = None,
    n_repeats: int = 5,
    random_state: int = 0,
    model: Optional[BaseEstimator] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PermutationSelector]:
    """
    1) Crea PermutationSelector (fitted) separando numéricas/no-numéricas.
    2) Aplica fit_transform a train y transform a test/backtest.
    3) Devuelve las particiones reducidas y el selector.
    """
    selector = PermutationSelector(
        tol=tol,
        task=task,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        model=model
    )
    X_tr = selector.fit_transform(X_train, y_train)
    X_te = selector.transform(X_test)
    X_ba = selector.transform(X_backtest)
    return X_tr, X_te, X_ba, selector
