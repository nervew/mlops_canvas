from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class IFeatureSelector(ABC):
    """Interface for feature selection steps."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "IFeatureSelector":
        """Fit selector using X and y."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform X by selecting features."""
        raise NotImplementedError

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit to data, then transform."""
        self.fit(X, y)
        return self.transform(X)
