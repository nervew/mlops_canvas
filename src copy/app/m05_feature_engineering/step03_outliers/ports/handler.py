# step03_outliers/ports/handler.py
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd

class IOutlierHandler(ABC):

    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
