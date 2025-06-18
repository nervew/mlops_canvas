import pandas as pd
import numpy as np
from ..ports.data_repository import DataRepository


class ToyDataRepository(DataRepository):
    """Generates a synthetic dataset similar to the example pipeline."""

    def __init__(self, n_samples: int = 500) -> None:
        self.n_samples = n_samples

    def load(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=self.n_samples, freq="D"),
            "category": rng.choice(["A", "B", "C"], size=self.n_samples),
            "num1": rng.normal(size=self.n_samples),
            "num2": rng.uniform(0, 100, size=self.n_samples),
            "target": rng.integers(0, 2, size=self.n_samples),
        })
        return df
