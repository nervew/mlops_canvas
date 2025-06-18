from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ports.data_repository import DataRepository
from ..ports.model_repository import ModelRepository
from steps.data_split.split import split_data
from steps.feature_engineering.basic import engineer_features
from steps.feature_selection.univariate import select_features
from steps.model_training.train import train_model
from steps.model_evaluation.metrics import evaluate_model


@dataclass
class TrainResult:
    model: Any
    score: float


class TrainModelUseCase:
    """Orchestrates the ML pipeline using modular steps."""

    def __init__(
        self,
        data_repo: DataRepository,
        model_repo: ModelRepository,
        task_type: str = "classification",
    ) -> None:
        self.data_repo = data_repo
        self.model_repo = model_repo
        self.task_type = task_type

    def execute(self) -> TrainResult:
        df = self.data_repo.load()
        train_df, test_df = split_data(
            df, target_column="target", split_method="time", time_column="date"
        )
        train_df, test_df = engineer_features(train_df, test_df)

        X_train = train_df.drop(columns=["target"])
        y_train = train_df["target"]
        X_test = test_df.drop(columns=["target"])
        y_test = test_df["target"]

        X_train_sel, X_test_sel = select_features(X_train, y_train, X_test)
        model = train_model(X_train_sel, y_train, self.task_type)
        score = evaluate_model(model, X_test_sel, y_test, self.task_type)

        self.model_repo.save(model)
        return TrainResult(model=model, score=score)
