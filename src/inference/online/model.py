"""Model loading utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import mlflow
import numpy as np


class InferenceModel(Protocol):
    def predict_proba(self, features: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class LocalModel:
    model: Any

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features)[:, 1]


def load_model(path: str | None = None) -> InferenceModel:
    if path:
        model_path = Path(path)
        model = joblib.load(model_path)
        return LocalModel(model)
    model = mlflow.pyfunc.load_model("models:/mlops-canvas/Production")
    return LocalModel(model)
