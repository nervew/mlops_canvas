# search_model/flaml_wrapper.py

from typing import Any
import numpy as np
import pandas as pd
from flaml import AutoML
from .automl_base import AutoMLBase
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error

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
        log_file: str = "mlops_canvas/logs/flaml.log",
        preprocess: bool = False,
    ) -> None:
        super().__init__("FLAML")
        self.task = task
        self.time_budget = time_budget
        self.metric_override = metric
        self.verbose = verbose
        self.log_file = log_file
        self.preprocess = preprocess
        self.automl = AutoML()
        self.raw_estimator: Any = None  # Para exportación ONNX

    def _infer_task(self, y: pd.Series) -> str:
        if self.task != "auto":
            return self.task
        if y.dtype.kind in "biu" and y.nunique() <= 15:
            return "classification"
        return "regression"

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        task = self._infer_task(y_train)
        if task == "classification":
            metric = self.metric_override or (
                "roc_auc" if y_train.nunique() == 2 else "roc_auc_ovr"
            )
        else:
            metric = self.metric_override or "mae"

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

        # Entrena el AutoML de FLAML (incluye transformaciones internas)
        self.automl.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            **settings
        )

        # Guardar el estimador puro para exportación ONNX
        self.raw_estimator = self.automl.model.estimator

        # Guardar los mejores hiperparámetros
        self.best_params = dict(self.automl.best_config)

        # Construir ranking de modelos probados
        records = []
        for est, loss in self.automl.best_loss_per_estimator.items():
            # En clasificación, loss = 1 - score; en regresión, loss = error
            metric_value = (1 - loss) if task == "classification" else loss
            records.append({
                "estimator_name": self._NAME_MAP.get(est, est),
                "metric": metric_value
            })

        ascending = task != "classification"
        self.model_ranking = (
            pd.DataFrame(records)
              .sort_values("metric", ascending=ascending)
              .reset_index(drop=True)
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Usa la lógica interna de FLAML (transformaciones + predictor)
        return self.automl.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        if hasattr(self.automl, "predict_proba"):
            return self.automl.predict_proba(X)
        return None

    def get_best_model(self) -> Any:
        # Devuelve el estimador puro de LightGBM (o similar) para exportar
        return self.raw_estimator

    def get_best_params(self) -> dict[str, Any]:
        return self.best_params

    def get_model_ranking(self) -> pd.DataFrame:
        return self.model_ranking
