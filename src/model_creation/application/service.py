import pandas as pd
from sklearn.base import BaseEstimator
from ..infrastructure.onnx_exporter import export_model
from ..domain.model import TrainedModel


def run(model: BaseEstimator, X: pd.DataFrame, path: str = "model.onnx") -> TrainedModel:
    export_model(model, X.shape[1], path)
    return TrainedModel(model=model)
