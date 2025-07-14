from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import TransformerMixin


class IFeatureSelector(ABC, TransformerMixin):
    """Interface for feature selection steps."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "IFeatureSelector":
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
