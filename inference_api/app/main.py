from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from joblib import load
from pydantic import BaseModel, Field

MODEL_DIR = Path("/models")
logger = logging.getLogger("inference_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Inference API", version="1.0.0")


class PredictResponse(BaseModel):
    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None
    model_version: Optional[str] = None


def _latest_model() -> Path:
    candidates = sorted(MODEL_DIR.glob("*.pkl"))
    if not candidates:
        raise HTTPException(status_code=500, detail="No model available")
    return candidates[-1]


def _load(file: Optional[UploadFile], records: Optional[list]) -> pd.DataFrame:
    if file is None and records is None:
        raise HTTPException(status_code=400, detail="Provide file or records")
    if file:
        content = file.file.read()
        df = pd.read_csv(io.BytesIO(content))
    else:
        df = pd.DataFrame(records)
    return df


@app.get("/healthz")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    return_proba: bool = Field(False, description="Return probabilities"),
    file: Optional[UploadFile] = File(None),
    records: Optional[List[Dict[str, Any]]] = None,
) -> PredictResponse:
    df = _load(file, records)
    model_path = _latest_model()
    pipeline = load(model_path)

    preds = pipeline.predict(df)
    probas = None
    if return_proba and hasattr(pipeline, "predict_proba"):
        try:
            probas = pipeline.predict_proba(df).tolist()
        except Exception:
            probas = None

    return PredictResponse(
        predictions=preds.tolist(),
        probabilities=probas,
        model_version=model_path.name,
    )
