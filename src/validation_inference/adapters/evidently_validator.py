from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd

from validation_inference.ports.validator import IInferenceValidator


class EvidentlyValidator(IInferenceValidator):
    """Placeholder implementation for Evidently-based drift detection."""

    def __init__(
        self, inference_column: str, target_column: str, thresholds: Dict[str, Any]
    ):
        self.inference_column = inference_column
        self.target_column = target_column
        self.thresholds = thresholds

    def validate(self, data: pd.DataFrame, target: str) -> Tuple[bool, Dict[str, Any]]:
        diff = abs(data[self.inference_column].std() - data[self.target_column].std())
        result = {
            "drift": diff > float(self.thresholds.get("std_diff", 0.1)),
            "std_diff": float(diff),
            "schema": True,
        }
        valid = not result["drift"]
        return valid, result
