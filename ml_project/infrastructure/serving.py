"""Simple FastAPI service for inference."""
from __future__ import annotations

from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel

from .model_persistence import load_model
from ..domain.data_processing import clean_data
from ..domain.features import add_domain_features

app = FastAPI()
model = load_model("models/model.pkl")
EXPECTED_FEATURES = list(getattr(model, "feature_names_in_", []))


class PredictionRequest(BaseModel):
    features: dict


@app.post("/predict")
def predict(req: PredictionRequest):
    df = pd.DataFrame([req.features])
    df = clean_data(df)
    df = add_domain_features(df)
    df = pd.get_dummies(df)
    if EXPECTED_FEATURES:
        df = df.reindex(columns=EXPECTED_FEATURES, fill_value=0)
    pred = model.predict(df)
    output = {"prediction": pred.tolist()}
    try:
        proba = model.predict_proba(df).tolist()
        output["probability"] = proba
    except AttributeError:
        pass
    return output
