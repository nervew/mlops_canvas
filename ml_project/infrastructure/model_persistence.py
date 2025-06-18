"""Model persistence utilities."""
from __future__ import annotations

import joblib
from pathlib import Path
from typing import Any

from sklearn.dummy import DummyClassifier


def save_model(model: Any, path: str) -> None:
    """Serialize model to the given path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    """Load model from disk if available, otherwise return a stub model."""
    path_obj = Path(path)
    if path_obj.exists():
        return joblib.load(path)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit([[0]], [0])
    return dummy
