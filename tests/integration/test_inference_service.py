from __future__ import annotations

import numpy as np
import pytest

from src.inference.online.inference import InferenceService
from src.inference.online.schema import PredictionRequest


class DummyModel:
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), 0.7)


def setup_module() -> None:
    InferenceService.initialize(DummyModel())


def teardown_module() -> None:
    InferenceService.shutdown()


def test_predict_returns_response() -> None:
    service = InferenceService.instance()
    assert service is not None
    response = service.predict(PredictionRequest(features=[0.1, 0.2]))
    assert response.score == pytest.approx(0.7)
    assert response.label == 1
