# -*- coding: utf-8 -*-
# search_model/flaml_wrapper.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from flaml import AutoML


def _metric_name(task: str, metric: Optional[str]) -> str:
    """Normaliza el nombre de métrica aceptado por FLAML."""
    if metric:
        m = metric.lower()
        if task == "regression":
            if m in {"mae", "l1"}:
                return "mae"
            if m in {"mse", "l2"}:
                return "mse"
            if m in {"rmse"}:
                return "rmse"
            if m in {"r2", "r2score", "r^2"}:
                return "r2"
            return m
        else:
            if m in {"logloss", "cross_entropy"}:
                return "log_loss"
            if m in {"auc", "roc_auc"}:
                return "roc_auc"
            if m in {"f1", "f1score"}:
                return "f1"
            if m in {"acc", "accuracy"}:
                return "accuracy"
            return m
    return "mae" if task == "regression" else "log_loss"


@dataclass
class AutoMLResult:
    name: str
    metrics: Dict[str, float]


class FLAMLWrapper:
    """Capa delgada sobre FLAML que **no** usa MLflow en ningún caso."""

    def __init__(
        self,
        task: str = "regression",
        metric: Optional[str] = None,
        time_budget: int = 300,
        estimator_list: Optional[List[str]] = None,
        eval_method: str = "cv",              # por defecto CV si NO hay X_val/y_val
        log_file: Optional[str] = None,       # ruta de log de FLAML (archivo)
        verbose: int = 1,
    ) -> None:
        self.task = task
        self.metric = _metric_name(task, metric)
        self.time_budget = time_budget
        self.estimator_list = estimator_list or [
            "lgbm", "rf", "catboost", "xgboost", "extra_tree", "xgb_limitdepth"
        ]
        self.eval_method = eval_method
        self.log_file = log_file
        self.verbose = verbose

        self.automl = AutoML()
        # Evitar logs de pruebas intermedias
        try:
            self.automl._log_trial = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        except Exception:
            pass

        self._result: Optional[AutoMLResult] = None

    # ---- API pública ----
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "FLAMLWrapper":
        # Base de argumentos
        fit_kwargs: Dict[str, Any] = dict(
            task=self.task,
            metric=self.metric,
            time_budget=self.time_budget,
            estimator_list=self.estimator_list,
            eval_method=self.eval_method,
            verbose=self.verbose,
        )
        # FLAML usa 'log_file_name' (no 'log_file')
        if self.log_file:
            fit_kwargs["log_file_name"] = self.log_file

        # Si traen validación explícita → FLAML exige 'holdout'
        if X_val is not None and y_val is not None:
            fit_kwargs["X_val"] = X_val
            fit_kwargs["y_val"] = y_val
            fit_kwargs["eval_method"] = "holdout"

        self.automl.fit(X_train=X_train, y_train=y_train, **fit_kwargs)

        best_est = self.automl.best_estimator
        self._result = AutoMLResult(name=str(best_est), metrics={})
        return self

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evalúa el mejor modelo con métricas básicas (regresión o clasificación)."""
        if self.task == "regression":
            y_pred = self.predict(X_test)
            mae = float(np.mean(np.abs(y_test - y_pred)))
            rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
            y_bar = float(np.mean(y_test))
            ss_res = float(np.sum((y_test - y_pred) ** 2))
            ss_tot = float(np.sum((y_test - y_bar) ** 2)) or 1.0
            metrics = {"mae": mae, "rmse": rmse, "r2": 1.0 - ss_res / ss_tot}
        else:
            if hasattr(self.automl, "predict_proba"):
                proba = self.automl.predict_proba(X_test)
                eps = 1e-15
                proba = np.clip(proba, eps, 1 - eps)
                if proba.ndim == 1 or proba.shape[1] == 1:
                    p1 = proba.ravel()
                    yb = y_test.astype(int).to_numpy()
                    logloss = float(-np.mean(yb * np.log(p1) + (1 - yb) * np.log(1 - p1)))
                else:
                    yb = y_test.astype(int).to_numpy()
                    ll = -np.log(proba[np.arange(len(yb)), yb])
                    logloss = float(np.mean(ll))
                metrics = {"log_loss": logloss}
            else:
                y_pred = self.predict(X_test)
                acc = float(np.mean(y_pred == y_test))
                metrics = {"accuracy": acc}

        if self._result:
            self._result.metrics = metrics
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.automl.predict(X))

    # ---- accesores usados por pipeline.py ----
    @property
    def name(self) -> str:
        return self._result.name if self._result else str(self.automl.best_estimator)

    @property
    def metrics(self) -> Dict[str, float]:
        return self._result.metrics if self._result else {}

    def get_best_model(self):
        return self.automl.model

    def get_model_ranking(self) -> pd.DataFrame:
        """Devuelve ranking (DataFrame) de los mejores modelos de FLAML."""
        rows: List[Dict[str, Any]] = []
        try:
            br = self.automl.best_result
            rows.append(
                {
                    "estimator": self.automl.best_estimator,
                    "metric_val": br.get("val_loss", np.nan),
                    "train_time": br.get("train_time", np.nan),
                    "config": br.get("config", {}),
                }
            )
        except Exception:
            pass
        return pd.DataFrame(rows)
