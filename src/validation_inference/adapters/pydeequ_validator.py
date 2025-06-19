from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd

from validation_inference.ports.validator import IInferenceValidator


class PyDeequValidator(IInferenceValidator):
    """Placeholder for PyDeequ validations."""

    def __init__(
        self, inference_column: str, target_column: str, thresholds: Dict[str, Any]
    ):
        self.inference_column = inference_column
        self.target_column = target_column
        self.thresholds = thresholds

    def validate(self, data: pd.DataFrame, target: str) -> Tuple[bool, Dict[str, Any]]:
        missing = data[self.inference_column].isna().mean()
        result = {
            "missing_ratio": float(missing),
            "schema": True,
            "drift": missing > float(self.thresholds.get("missing", 0.05)),
        }
        valid = not result["drift"]
        return valid, result
