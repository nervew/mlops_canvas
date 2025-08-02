# search_model/automl_base.py

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class AutoMLBase(ABC):
    """Clase base para cualquier wrapper AutoML."""

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.model: Any = None
        self.best_params: Dict[str, Any] = {}
        self.metrics: Dict[str, float] = {}
        self.model_ranking: Any = None

    # ---------- Métodos que cada wrapper debe implementar ----------
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None: ...

    @abstractmethod
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray | None: ...

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def get_best_model(self) -> Any: ...

    @abstractmethod
    def get_model_ranking(self) -> Any: ...

    @abstractmethod
    def get_best_params(self) -> Dict[str, Any]: ...
    # ----------------------------------------------------------------

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Evalúa el modelo:
          • Clasificación → ROC AUC, Accuracy, F1 Score
          • Regresión     → R2, MAE, MSE
        La tarea se infiere: si el modelo devuelve predict_proba → clasificación,
        en caso contrario → regresión.
        """
        y_pred = self.predict(X_test)

        # Intentar obtener probabilidades.
        try:
            y_pred_proba = self.predict_proba(X_test)
        except Exception:
            y_pred_proba = None

        # ---- Detección de tarea ------------------------------------
        is_regression = y_pred_proba is None
        # ------------------------------------------------------------

        if is_regression:
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2  = r2_score(y_test, y_pred)
            self.metrics = {"MAE": mae, "MSE": mse, "R2": r2}
        else:
            # Calcular ROC AUC
            if y_pred_proba.ndim == 1:
                y_score = y_pred_proba
            elif y_pred_proba.shape[1] == 2:
                y_score = y_pred_proba[:, 1]
            else:
                y_score = y_pred_proba
            roc_auc = roc_auc_score(
                y_test,
                y_score,
                multi_class="ovr" if len(np.unique(y_test)) > 2 else "raise"
            )
            acc = accuracy_score(y_test, y_pred)
            f1  = f1_score(y_test, y_pred, average="weighted")
            self.metrics = {"ROC AUC": roc_auc, "Accuracy": acc, "F1 Score": f1}
