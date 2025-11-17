import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from inference_api.app.main import PredictResponse


def test_predict_response_model():
    resp = PredictResponse(predictions=[1, 0], probabilities=None, model_version="v1")
    assert resp.model_version == "v1"
