"""Inference service orchestrating model predictions."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, ClassVar, Optional

import numpy as np

from .model import InferenceModel
from .schema import PredictionRequest, PredictionResponse


@dataclass(slots=True)
class InferenceService:
    _model: InferenceModel
    _instance: ClassVar[Optional["InferenceService"]] = None
    _lock: ClassVar[Lock] = Lock()

    @classmethod
    def initialize(cls, model: InferenceModel) -> None:
        with cls._lock:
            cls._instance = cls(model)

    @classmethod
    def instance(cls) -> Optional["InferenceService"]:
        return cls._instance

    @classmethod
    def shutdown(cls) -> None:
        with cls._lock:
            cls._instance = None

    def predict(self, payload: PredictionRequest) -> PredictionResponse:
        features = np.array([payload.features])
        score = self._model.predict_proba(features)[0]
        return PredictionResponse(score=float(score), label=int(score >= 0.5))
