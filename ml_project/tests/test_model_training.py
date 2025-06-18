import numpy as np
from sklearn.datasets import make_classification
from ml_project.domain.model import train_model


def test_train_model_runs():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model, params = train_model(X, y)
    assert hasattr(model, "predict")
    assert isinstance(params, dict)
