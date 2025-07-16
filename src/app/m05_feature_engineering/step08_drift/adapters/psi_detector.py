from __future__ import annotations

import numpy as np
import pandas as pd

from ..ports import IDriftDetector


class PSIDriftDetector(IDriftDetector):
    def __init__(self, bins: int = 10) -> None:
        self.bins = bins

    def _psi(self, base: pd.Series, new: pd.Series) -> float:
        quantiles = np.linspace(0, 1, self.bins + 1)
        cuts = np.quantile(base, quantiles)
        base_counts, _ = np.histogram(base, bins=cuts)
        new_counts, _ = np.histogram(new, bins=cuts)
        base_props = base_counts / len(base)
        new_props = new_counts / len(new)
        return float(
            np.sum(
                (base_props - new_props)
                * np.log((base_props + 1e-8) / (new_props + 1e-8))
            )
        )

    def compute(self, reference: pd.DataFrame, new: pd.DataFrame) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for col in reference.select_dtypes(include="number").columns:
            metrics[col] = self._psi(reference[col], new[col])
        return metrics