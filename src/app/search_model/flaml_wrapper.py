import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score            # ⬅️ NUEVO

try:
    from flaml import AutoML
except ImportError:                                   # pragma: no cover
    print("FLAML no está instalado. Instálalo usando `pip install flaml`.")
    sys.exit(1)

from .automl_base import AutoMLBase


class FLAMLWrapper(AutoMLBase):
    """Wrapper sencillo para FLAML (clasificación)."""

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

    # ------------------------------------------------------------ #
    #  ENTRENAMIENTO
    # ------------------------------------------------------------ #
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:

        # --- Validaciones mínimas ---
        if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
            raise ValueError("X_train y X_test deben ser DataFrames (pandas).")
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("X_train / y_train no pueden estar vacíos.")
        if not all(X_train.columns == X_test.columns):
            raise ValueError("Las columnas de X_train y X_test deben coincidir exactamente.")

        # --- Métrica base para FLAML ---
        metric = (
            self.metric_override
            if self.metric_override
            else ("roc_auc" if y_train.nunique() == 2 else "roc_auc_ovr")
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

        self.automl.fit(X_train=X_train, y_train=y_train,
                        X_val=X_test,  y_val=y_test, **settings)

        # --- Mejor modelo global ---
        best_est_name = self.automl.best_estimator
        self.model = (
            self._NAME_MAP.get(best_est_name, best_est_name),
            self.automl.model,
        )
        self.best_params = dict(self.automl.best_config)

        # -------------------------------------------------------- #
        #  RANKING  (ahora con métrica calculada explícitamente)
        # -------------------------------------------------------- #
        records: List[Dict[str, Any]] = []
        n_classes = y_train.nunique()

        for est in self.automl.estimator_list:
            best_md = self.automl.best_model_for_estimator(est)
            if best_md is None:
                continue

            train_score = 1 - self.automl.best_loss_per_estimator[est]

            # --- test_score robusto ---
            try:
                # FLAML devuelve modelos ya ajustados; si no, re-ajusta rápido
                if not hasattr(best_md, "predict"):
                    best_md.fit(X_train, y_train)

                # Preferimos AUC para coherencia
                y_prob = best_md.predict_proba(X_test)
                if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                    y_score = y_prob[:, 1] if n_classes == 2 else y_prob
                    test_score = roc_auc_score(
                        y_test,
                        y_score,
                        multi_class="ovr" if n_classes > 2 else "raise",
                    )
                else:
                    y_pred = best_md.predict(X_test)
                    test_score = accuracy_score(y_test, y_pred)

            except Exception:                                 # ← si algo falla
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

    # ------------------------------------------------------------ #
    #  INFERENCIA
    # ------------------------------------------------------------ #
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        if not isinstance(X_test, pd.DataFrame):
            raise ValueError("X_test debe ser DataFrame.")
        return self.automl.predict_proba(X_test)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if not isinstance(X_test, pd.DataFrame):
            raise ValueError("X_test debe ser DataFrame.")
        return self.automl.predict(X_test)

    # ------------- getters -------------
    def get_best_model(self) -> Any:
        return self.model

    def get_model_ranking(self) -> pd.DataFrame:
        return self.model_ranking.copy()

    def get_best_params(self) -> Dict[str, Any]:
        return dict(self.best_params)
