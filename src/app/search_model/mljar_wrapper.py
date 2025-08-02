# search_model/mljar_wrapper.py

import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
from supervised.automl import AutoML
from .automl_base import AutoMLBase

class MLJARWrapper(AutoMLBase):
    def __init__(self):
        super().__init__("MLJAR")
        self.automl = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        if len(np.unique(y_train)) == 2:
            model_type = 'binary_classification'
            eval_metric = 'auc'
        else:
            model_type = 'multiclass_classification'
            eval_metric = 'f1'

        self.automl = AutoML(
            mode="Compete",
            algorithms=["CatBoost", "Xgboost", "LightGBM", "Random Forest", "Extra Trees", "Neural Network"],
            total_time_limit=10,
            ml_task=model_type,
            eval_metric=eval_metric,
            random_state=42
        )
        self.automl.fit(X_train, y_train)
        self.model = self.automl._best_model
        self.best_params = self.model.get_params()
        leaderboard = pd.DataFrame(self.automl.report()["leaderboard"])
        self.model_ranking = leaderboard.rename(
            columns={"model_name": "estimator_name", "metric_value": "metric_test"}
        )

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.automl.predict_proba(X_test)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.automl.predict(X_test)

    def get_best_model(self) -> Any:
        return self.model

    def get_model_ranking(self):
        return self.model_ranking

    def get_best_params(self) -> Dict:
        return self.best_params
