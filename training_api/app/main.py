from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .automl import AutoMLRunner
from .io import load_dataset
from .storage import save_model
from .tuning import HyperparameterTuner

logger = logging.getLogger("training_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Training API", version="1.0.0")


class TrainResponse(BaseModel):
    estimator: str
    metric: str
    task: str
    best_score: float
    tuning_score: float
    model_path: str
    metadata_path: str
    features: List[str]


@app.get("/status")
def status() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/train", response_model=TrainResponse)
async def train(
    target: str = Field(..., description="Target column"),
    metric: Optional[str] = Field(None, description="Metric override"),
    task_type: str = Field("auto", description="classification/regression/auto"),
    time_budget: int = Field(60, description="Seconds for FLAML"),
    optuna_trials: int = Field(10, description="Optuna trials"),
    random_state: Optional[int] = Field(None, description="Random seed"),
    file: Optional[UploadFile] = File(None),
    records: Optional[List[Dict[str, Any]]] = None,
) -> TrainResponse:
    df, y = load_dataset(file, records, target)

    automl = AutoMLRunner(time_budget=time_budget, metric=metric, task=task_type, random_state=random_state)
    auto_result, pipeline = automl.run(df, y)

    tuner = HyperparameterTuner(
        estimator_name=auto_result.estimator,
        n_trials=optuna_trials,
        metric=auto_result.metric,
        task=auto_result.task,
        random_state=random_state,
    )
    best_params, tuning_score = tuner.tune(pipeline, df, y)
    estimator = pipeline.named_steps["model"]
    for key, value in best_params.items():
        setattr(estimator, key, value)

    pipeline.fit(df, y)
    metadata = {
        "estimator": auto_result.estimator,
        "metric": auto_result.metric,
        "task": auto_result.task,
        "best_config": auto_result.config,
        "best_params": best_params,
        "features": list(df.columns),
    }
    paths = save_model(pipeline, metadata)

    return TrainResponse(
        estimator=auto_result.estimator,
        metric=auto_result.metric,
        task=auto_result.task,
        best_score=auto_result.best_score,
        tuning_score=tuning_score,
        model_path=paths["model_path"],
        metadata_path=paths["metadata_path"],
        features=list(df.columns),
    )
