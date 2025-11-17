from __future__ import annotations

from typing import Any, Dict, Tuple

import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score


class HyperparameterTuner:
    def __init__(self, estimator_name: str, n_trials: int = 10, metric: str = "roc_auc", task: str = "classification", random_state: int | None = None) -> None:
        self.estimator_name = estimator_name
        self.n_trials = n_trials
        self.metric = metric
        self.task = task
        self.random_state = random_state

    def _search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        if self.estimator_name == "lgbm":
            return {
                "num_leaves": trial.suggest_int("num_leaves", 16, 64),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            }
        if self.estimator_name == "xgboost":
            return {
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            }
        if self.estimator_name == "rf":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 150),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
            }
        return {}

    def tune(self, pipeline, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, Any], float]:
        def objective(trial: optuna.Trial) -> float:
            params = self._search_space(trial)
            estimator = pipeline.named_steps["model"]
            for key, value in params.items():
                setattr(estimator, key, value)
            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=3,
                scoring=self.metric,
                n_jobs=1,
            )
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params, study.best_value
