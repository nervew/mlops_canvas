import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    from flaml import AutoML
except ImportError:  # pragma: no cover
    print("FLAML no está instalado. Instálalo usando `pip install flaml`.")
    sys.exit(1)

from .automl_base import AutoMLBase


class FLAMLWrapper(AutoMLBase):
    """Wrapper sencillo para FLAML."""

    _NAME_MAP: Dict[str, str] = {
        "lgbm": "LightGBM",
        "xgb_limitdepth": "XGBoost (Limit Depth)",
        "xgboost": "XGBoost",
        "rf": "Random Forest",
        "extra_tree": "Extra Trees",
        "lrl1": "Logistic Regression (L1)",
        "sgd": "Stochastic Gradient Descent",
    }

    def __init__(
        self,
        time_budget: int = 360,
        metric: Optional[str] = None,
        verbose: int = 0,
        log_file: str = "flaml.log",
        preprocess: bool = False,
    ):
        super().__init__("FLAML")
        self.time_budget = time_budget
        self.metric_override = metric
        self.verbose = verbose
        self.log_file = log_file
        self.preprocess = preprocess
        self.automl = AutoML()

    # ------------------------------------------------------------------
    #  ENTRENAMIENTO
    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, np.ndarray, List[Any]],
        X_test: pd.DataFrame,
        y_test: Union[pd.Series, np.ndarray, List[Any]],
    ) -> None:
        """
        X_train y X_test deben ser SIEMPRE DataFrames con los mismos nombres de columnas.
        y_train/y_test pueden ser Series, arrays o listas.
        """
        # Validaciones mínimas
        if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
            raise ValueError("X_train y X_test deben ser DataFrames de pandas para evitar problemas de feature names.")
        if len(X_train) == 0:
            raise ValueError("❌ X_train no puede estar vacío.")
        if len(y_train) == 0:
            raise ValueError("❌ y_train no puede estar vacío.")
        if not all(X_train.columns == X_test.columns):
            raise ValueError("Las columnas de X_train y X_test deben coincidir exactamente en orden y nombre.")

        # Determinar métrica
        y_tr = y_train.values if isinstance(y_train, pd.Series) else y_train
        metric = (
            self.metric_override
            if self.metric_override
            else ("roc_auc" if np.unique(y_tr).size == 2 else "roc_auc_ovr")
        )

        settings: Dict[str, Any] = {
            "task": "classification",
            "time_budget": self.time_budget,
            "eval_method": "holdout",
            "metric": metric,
            "verbose": self.verbose,
            "log_file_name": self.log_file,
            "skip_transform": self.preprocess,
            "model_history": True,
        }

        # Entrenar (con DataFrames, sin .values)
        self.automl.fit(X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, **settings)

        # Modelo ganador
        best_est_name = self.automl.best_estimator
        self.model = (
            self._NAME_MAP.get(best_est_name, best_est_name),
            self.automl.model,
        )
        self.best_params = dict(self.automl.best_config)

        # Ranking
        records: List[Dict[str, Any]] = []
        for est in self.automl.estimator_list:
            best_md = self.automl.best_model_for_estimator(est)
            if best_md is None:
                continue
            train_score = 1 - self.automl.best_loss_per_estimator[est]
            try:
                # Debe recibir un DataFrame con mismas columnas que X_train
                test_score = best_md.score(X_test, y_test)
            except Exception:
                test_score = np.nan
            records.append(
                {
                    "estimator_name": self._NAME_MAP.get(est, est),
                    "metric_train": train_score,
                    "metric_test": test_score,
                    "hyperparameters": self.automl.best_config_per_estimator[est],
                }
            )

        self.model_ranking = (
            pd.DataFrame(records)
            .sort_values("metric_test", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    #  INFERENCIA
    # ------------------------------------------------------------------
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        if not isinstance(X_test, pd.DataFrame):
            raise ValueError("X_test debe ser un DataFrame con los mismos nombres de columnas usados en entrenamiento.")
        return self.automl.predict_proba(X_test)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if not isinstance(X_test, pd.DataFrame):
            raise ValueError("X_test debe ser un DataFrame con los mismos nombres de columnas usados en entrenamiento.")
        return self.automl.predict(X_test)

    # ------------------------------------------------------------------
    #  GETTERS
    # ------------------------------------------------------------------
    def get_best_model(self) -> Any:
        return self.model

    def get_model_ranking(self) -> pd.DataFrame:
        return self.model_ranking.copy()

    def get_best_params(self) -> Dict[str, Any]:
        return dict(self.best_params)
