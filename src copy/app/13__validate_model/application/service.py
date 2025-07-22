import pandas as pd
from sklearn.base import BaseEstimator
from ..infrastructure.evaluator import evaluate
from ..domain.metrics import ValidationMetrics


def run(model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> ValidationMetrics:
    acc = evaluate(model, X, y)
    return ValidationMetrics(accuracy=acc)
