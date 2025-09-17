# search_model/flaml_wrapper.py
from __future__ import annotations

from typing import Any
from pathlib import Path
import types
import sys
import importlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error

from .automl_base import AutoMLBase
from .export import get_log_path


def _ensure_safe_mlflow_import() -> None:
    """
    Si existe un módulo 'mlflow' sin la API esperada (p.ej. sin active_run),
    lo reemplaza por un stub para que FLAML no falle al intentar loguear.
    Debe ejecutarse ANTES de importar FLAML.
    """
    try:
        m = importlib.import_module("mlflow")
        # Si no tiene active_run, lo consideramos inválido.
        if not hasattr(m, "active_run"):
            raise ImportError("mlflow sin 'active_run'")
    except Exception:
        dummy = types.ModuleType("mlflow")
        # No-op helpers
        def _none(*args, **kwargs): 
            return None
        def _ctx(*args, **kwargs):
            class _C:
                def __enter__(self): return self
                def __exit__(self, exc_type, exc, tb): return False
            return _C()
        # API mínima que FLAML podría tocar
        dummy.active_run = _none
        dummy.start_run  = _ctx
        dummy.log_metric = _none
        dummy.log_param  = _none
        dummy.set_tag    = _none
        sys.modules["mlflow"] = dummy


# Parchear antes del import de FLAML
_ensure_safe_mlflow_import()
from flaml import AutoML  # noqa: E402


class FLAMLWrapper(AutoMLBase):
    _NAME_MAP = {
        "lgbm": "LightGBM",
        "xgboost": "XGBoost",
        "rf": "Random Forest",
        "extra_tree": "Extra Trees",
        "catboost": "CatBoost",
        "linear": "Linear",
    }

    def __init__(
        self,
        task: str = "auto",
        time_budget: int = 60,
        metric: str | None = None,
        verbose: int = 0,
        log_file: str | None = None,
        preprocess: bool = False,
    ) -> None:
        super().__init__("FLAML")
        self.task = task
        self.time_budget = time_budget
        self.metric_override = metric
        self.verbose = verbose

        # Ruta de log absoluta y con directorio creado
        self.log_file = (
            get_log_path() if log_file is None
            else get_log_path(Path(log_file).name)
        )

        self.preprocess = preprocess
        self.automl = AutoML()
        self.raw_estimator: Any = None  # Para exportación ONNX

    # ------------------------------------------------------------------ #
    # Métodos internos
    # ------------------------------------------------------------------ #
    def _infer_task(self, y: pd.Series) -> str:
        if self.task != "auto":
            return self.task
        if y.dtype.kind in "biu" and y.nunique() <= 15:
            return "classification"
        return "regression"

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        task = self._infer_task(y_train)
        metric = (
            self.metric_override
            or ("roc_auc" if task == "classification" and y_train.nunique() == 2 else
                "roc_auc_ovr" if task == "classification" else
                "mae")
        )

        settings = {
            "task": task,
            "time_budget": self.time_budget,
            "eval_method": "holdout",
            "metric": metric,
            "verbose": self.verbose,
            "log_file_name": self.log_file,
            "skip_transform": self.preprocess,
            "model_history": True,
        }

        # Entrena FLAML
        self.automl.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            **settings
        )

        # Resultados
        self.raw_estimator = self.automl.model.estimator
        self.best_params = dict(self.automl.best_config)

        # Ranking
        records = []
        for est, loss in self.automl.best_loss_per_estimator.items():
            metric_value = (1 - loss) if task == "classification" else loss
            records.append({
                "estimator_name": self._NAME_MAP.get(est, est),
                "metric": metric_value
            })

        self.model_ranking = (
            pd.DataFrame(records)
              .sort_values("metric", ascending=(task != "classification"))
              .reset_index(drop=True)
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.automl.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        return getattr(self.automl, "predict_proba", lambda _: None)(X)

    def get_best_model(self) -> Any:
        return self.raw_estimator

    def get_best_params(self) -> dict[str, Any]:
        return self.best_params

    def get_model_ranking(self) -> pd.DataFrame:
        return self.model_ranking
