import pandas as pd
from sklearn.base import BaseEstimator
from ..infrastructure.predictor import predict
from ..domain.predictions import Predictions


def run(model: BaseEstimator, X: pd.DataFrame) -> Predictions:
    values = predict(model, X)
    return Predictions(values=values)
