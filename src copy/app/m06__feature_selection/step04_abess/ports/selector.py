# src/app/m06__feature_selection/step04_abess/ports/selector.py

from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd

class IFeatureSelector(ABC):
    """Interfaz base para selectores de variables."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> IFeatureSelector:
        ...

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ...
