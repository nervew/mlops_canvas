from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class IDriftDetector(ABC):
    @abstractmethod
    def compute(self, reference: pd.DataFrame, new: pd.DataFrame) -> dict[str, float]:
        raise NotImplementedError