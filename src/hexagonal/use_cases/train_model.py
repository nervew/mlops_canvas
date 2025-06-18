from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from ..ports.data_repository import DataRepository
from ..ports.model_repository import ModelRepository
from app.split_dataset.robust_data_splitter import RobustDataSplitter


@dataclass
class TrainResult:
    model: RandomForestClassifier
    accuracy: float


class TrainModelUseCase:
    """Orchestrates the ML pipeline using hexagonal architecture."""

    def __init__(self, data_repo: DataRepository, model_repo: ModelRepository) -> None:
        self.data_repo = data_repo
        self.model_repo = model_repo

    def execute(self) -> TrainResult:
        df = self.data_repo.load()
        train_df, test_df = self._split(df)
        train_df, test_df = self._feature_engineering(train_df, test_df)

        X_train = train_df.drop(columns=["target"])
        y_train = train_df["target"]
        X_test = test_df.drop(columns=["target"])
        y_test = test_df["target"]

        X_train_sel, X_test_sel = self._feature_selection(X_train, y_train, X_test)
        model = self._train_model(X_train_sel, y_train)
        acc = self._evaluate(model, X_test_sel, y_test)

        self.model_repo.save(model)
        return TrainResult(model=model, accuracy=acc)

    def _split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        splitter = RobustDataSplitter(
            df,
            split_method="time",
            time_column="date",
            target_column="target",
            train_size=0.6,
            test_size=0.4,
            backtest_size=0.0,
        )
        train_df, test_df, _ = splitter.split_data()
        return train_df, test_df

    def _feature_engineering(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        def transform(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["day"] = df["date"].dt.day
            df = df.drop(columns=["date"])
            df = pd.get_dummies(df, columns=["category"], drop_first=False)
            return df

        return transform(train_df), transform(test_df)

    def _feature_selection(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        selector = SelectKBest(score_func=f_classif, k=min(8, X_train.shape[1]))
        selector.fit(X_train, y_train)
        return selector.transform(X_train), selector.transform(X_test)

    def _train_model(self, X_train: np.ndarray, y_train: pd.Series) -> RandomForestClassifier:
        model = RandomForestClassifier(random_state=42)
        grid = GridSearchCV(model, param_grid={"n_estimators": [50, 100], "max_depth": [None, 5]}, cv=3)
        grid.fit(X_train, y_train)
        best_model: RandomForestClassifier = grid.best_estimator_
        best_model.fit(X_train, y_train)
        return best_model

    def _evaluate(self, model: RandomForestClassifier, X_test: np.ndarray, y_test: pd.Series) -> float:
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)
