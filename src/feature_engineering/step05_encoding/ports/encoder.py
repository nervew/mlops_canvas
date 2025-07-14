from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class ICategoricalEncoder(ABC):
    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
