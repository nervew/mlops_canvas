from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


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
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def get_best_model(self) -> Any: ...

    @abstractmethod
    def get_model_ranking(self) -> Any: ...

    @abstractmethod
    def get_best_params(self) -> Dict[str, Any]: ...
    # ----------------------------------------------------------------

    # Evaluación genérica para cualquier problema de clasificación
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        y_pred_proba = self.predict_proba(X_test)
        y_pred = self.predict(X_test)

        # ── Manejar forma 1-D ó 2-D según devuelva el modelo ──────────
        if y_pred_proba.ndim == 1:                       # proba binaria en 1-D
            y_score = y_pred_proba
        elif len(np.unique(y_test)) == 2:                # binario 2-D
            y_score = y_pred_proba[:, 1]
        else:                                            # multiclase
            y_score = y_pred_proba
        # -------------------------------------------------------------

        roc_auc = roc_auc_score(
            y_test,
            y_score,
            multi_class="ovr" if len(np.unique(y_test)) > 2 else "raise"
        )
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        self.metrics = {"ROC AUC": roc_auc, "Accuracy": acc, "F1 Score": f1}
