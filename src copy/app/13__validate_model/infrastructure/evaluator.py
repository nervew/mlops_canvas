import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score


def evaluate(model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> float:
    preds = model.predict(X)
    return float(accuracy_score(y, preds))
