# src/app/m06__feature_selection/step05_shap_select/ports/selector.py

from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd

class IFeatureSelector(ABC):
    """Interfaz abstracta para selectores de características."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> IFeatureSelector:
        ...

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ...
