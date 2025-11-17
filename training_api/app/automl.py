from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class AutoMLResult:
    estimator: str
    config: Dict[str, Any]
    metric: str
    task: str
    best_score: float


class AutoMLRunner:
    def __init__(
        self,
        time_budget: int = 60,
        metric: Optional[str] = None,
        task: str = "auto",
        random_state: Optional[int] = None,
    ) -> None:
        self.time_budget = time_budget
        self.metric = metric
        self.task = task
        self.random_state = random_state

    @staticmethod
    def detect_task_type(y: pd.Series) -> Tuple[str, str]:
        unique = y.dropna().unique()
        if y.dtype == object or len(unique) <= 20:
            return "classification", "roc_auc"
        return "regression", "rmse"

    @staticmethod
    def build_preprocessor(df: pd.DataFrame) -> Tuple[Pipeline, List[str], List[str]]:
        cat_cols = [c for c in df.columns if df[c].dtype == object]
        num_cols = [c for c in df.columns if c not in cat_cols]

        numeric = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("categorical", categorical, cat_cols),
                ("numeric", numeric, num_cols),
            ]
        )
        return preprocessor, cat_cols, num_cols

    def run(self, X: pd.DataFrame, y: pd.Series) -> Tuple[AutoMLResult, Pipeline]:
        detected_task, default_metric = self.detect_task_type(y)
        task = self.task if self.task != "auto" else detected_task
        metric = self.metric or default_metric

        preprocessor, _, _ = self.build_preprocessor(X)
        automl = AutoML()
        automl.fit(
            X_train=X,
            y_train=y,
            task=task,
            metric=metric,
            preprocess=False,
            time_budget=self.time_budget,
            ensemble=False,
            seed=self.random_state,
        )

        model = automl.model
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        result = AutoMLResult(
            estimator=automl.best_estimator,
            config=json.loads(json.dumps(automl.best_config)),
            metric=metric,
            task=task,
            best_score=automl.best_loss if task == "regression" else automl.best_loss,
        )
        return result, pipeline
