# flaml_wrapper.py

import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from .automl_base import AutoMLBase

class FLAMLWrapper(AutoMLBase):
    _NAME_MAP = {
        "lgbm": "LightGBM",
        "xgboost": "XGBoost",
        "rf": "Random Forest",
        "extra_tree": "Extra Trees",
        "catboost": "CatBoost",
        "linear": "Linear",
        "arima": "ARIMA",
        "prophet": "Prophet",
        "lstm": "LSTM",
    }

    def __init__(
        self,
        task: str = "auto",         # "auto", "classification" o "regression"
        time_budget: int = 360,
        metric: str | None = None,
        verbose: int = 0,
        log_file: str = "flaml.log",
        preprocess: bool = False,
    ):
        super().__init__("FLAML")
        self.task = task
        self.time_budget = time_budget
        self.metric_override = metric
        self.verbose = verbose
        self.log_file = log_file
        self.preprocess = preprocess
        self.automl = AutoML()

    def _infer_task(self, y: pd.Series) -> str:
        # Si es auto, decide la tarea automáticamente
        if self.task != "auto":
            return self.task
        if y.dtype.kind in "biu" and y.nunique() <= 15:
            return "classification"
        return "regression"

    def fit(self, X_train, y_train, X_test, y_test) -> None:
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

        self.automl.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            **settings
        )

        # Guardar el mejor modelo y sus hiperparámetros
        best_est_name = self.automl.best_estimator
        self.model = (
            self._NAME_MAP.get(best_est_name, best_est_name),
            self.automl.model,
        )
        self.best_params = dict(self.automl.best_config)

        # Ranking de modelos
        records = []
        for est in self.automl.estimator_list:
            best_md = self.automl.best_model_for_estimator(est)
            if best_md is None:
                continue

            train_score = 1 - self.automl.best_loss_per_estimator[est]
            try:
                if not hasattr(best_md, "predict"):
                    best_md.fit(X_train, y_train)

                if task == "classification":
                    y_prob = best_md.predict_proba(X_test)
                    if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                        y_score = (
                            y_prob[:, 1]
                            if y_train.nunique() == 2
                            else y_prob
                        )
                        test_score = roc_auc_score(
                            y_test,
                            y_score,
                            multi_class="ovr"
                            if y_train.nunique() > 2
                            else "raise",
                        )
                    else:
                        y_pred = best_md.predict(X_test)
                        test_score = accuracy_score(y_test, y_pred)
                else:  # regression
                    y_pred = best_md.predict(X_test)
                    # Puedes cambiar aquí por r2_score, mse, etc.
                    test_score = mean_absolute_error(y_test, y_pred)
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
            .sort_values(
                "metric_test", ascending=True if task == "regression" else False
            )
            .reset_index(drop=True)
        )

    def predict(self, X):
        return self.automl.predict(X)

    def predict_proba(self, X):
        if hasattr(self.automl, "predict_proba"):
            return self.automl.predict_proba(X)
        return None

    def get_best_model(self):
        return self.model

    def get_best_params(self):
        return self.best_params

    def get_model_ranking(self):
        return self.model_ranking
