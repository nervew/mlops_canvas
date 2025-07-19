from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd

class IFeatureSelector(ABC):
    """Interfaz base para todos los selectores de features."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> IFeatureSelector:
        """
        Ajusta el selector a X (y opcionalmente a y).
        Debe devolver self.
        """
        ...

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma un DataFrame X usando la lógica aprendida en fit().
        """
        ...
