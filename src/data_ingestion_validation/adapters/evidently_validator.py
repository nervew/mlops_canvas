from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from data_ingestion_validation.ports.validator import IIngestionValidator


class EvidentlyValidator(IIngestionValidator):
    """Placeholder for Evidently-based anomaly detection."""

    def __init__(self, ingestion_columns: List[str], target_column: str, thresholds: Dict[str, Any]):
        self.ingestion_columns = ingestion_columns
        self.target_column = target_column
        self.thresholds = thresholds

    def validate(self, data: pd.DataFrame, target: str) -> Tuple[bool, Dict[str, Any]]:
        drift_score = data[self.ingestion_columns].std().sum()
        result = {
            "drift_score": float(drift_score),
            "drift_ok": drift_score <= float(self.thresholds.get("drift", 1.0)),
            "schema": True,
        }
        valid = result["drift_ok"]
        return valid, result
