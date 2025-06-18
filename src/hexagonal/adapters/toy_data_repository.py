from __future__ import annotations

import pandas as pd

from ..ports.data_repository import DataRepository
from steps.data_ingestion.toy_generator import generate_data


class ToyDataRepository(DataRepository):
    """Generates a synthetic dataset using the step utilities."""

    def __init__(self, n_samples: int = 500, task_type: str = "classification") -> None:
        self.n_samples = n_samples
        self.task_type = task_type

    def load(self) -> pd.DataFrame:
        return generate_data(self.n_samples, self.task_type)
