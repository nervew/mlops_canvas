from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class IMetricsExporter(ABC):
    @abstractmethod
    def export(self, df: pd.DataFrame, path: str) -> None:
        raise NotImplementedError