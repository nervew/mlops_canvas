from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema

from data_ingestion_validation.ports.validator import IIngestionValidator


class GreatExpectationsValidator(IIngestionValidator):
    """Simplified schema validation using Pandera as a stand-in for Great Expectations."""

    def __init__(self, ingestion_columns: List[str], target_column: str, thresholds: Dict[str, Any]):
        self.ingestion_columns = ingestion_columns
        self.target_column = target_column
        self.thresholds = thresholds
        columns = {col: Column(pa.String, nullable=False) for col in ingestion_columns}
        columns[target_column] = Column(pa.String, nullable=False)
        self.schema = DataFrameSchema(columns)

    def validate(self, data: pd.DataFrame, target: str) -> Tuple[bool, Dict[str, Any]]:
        result: Dict[str, Any] = {"schema": True}
        try:
            self.schema.validate(data[self.ingestion_columns + [self.target_column]], lazy=True)
        except pa.errors.SchemaErrors as exc:
            result["schema"] = False
            result["schema_errors"] = exc.failure_cases.to_dict(orient="records")

        missing_rate = data[self.ingestion_columns].isna().mean().max()
        result["missing_rate"] = float(missing_rate)
        result["missing_ok"] = missing_rate <= float(self.thresholds.get("missing", 0.05))
        valid = result["schema"] and result["missing_ok"]
        return valid, result
