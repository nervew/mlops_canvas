from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema

from validation_inference.ports.validator import IInferenceValidator


class GreatExpectationsValidator(IInferenceValidator):
    """Simple schema and drift validation using Pandera as a stand-in."""

    def __init__(
        self, inference_column: str, target_column: str, thresholds: Dict[str, Any]
    ):
        self.inference_column = inference_column
        self.target_column = target_column
        self.thresholds = thresholds
        self.schema = DataFrameSchema(
            {
                inference_column: Column(float, nullable=False),
                target_column: Column(float, nullable=False),
            }
        )

    def validate(self, data: pd.DataFrame, target: str) -> Tuple[bool, Dict[str, Any]]:
        result: Dict[str, Any] = {"schema": True, "drift": False}
        try:
            self.schema.validate(
                data[[self.inference_column, self.target_column]], lazy=True
            )
        except pa.errors.SchemaErrors as exc:
            result["schema"] = False
            result["schema_errors"] = exc.failure_cases.to_dict(orient="records")

        diff = abs(data[self.inference_column].mean() - data[self.target_column].mean())
        result["mean_diff"] = float(diff)
        if diff > float(self.thresholds.get("mean_diff", 0.1)):
            result["drift"] = True

        valid = result["schema"] and not result["drift"]
        return valid, result
