# fs_base.py
from abc import ABC, abstractmethod
import pandas as pd

class FeatureSelector(ABC):
    """Interfaz común para todos los selectores de features."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y=None):
        """Ajusta el selector con los datos X (y opcionalmente y)."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforma los datos X y devuelve el subconjunto de features seleccionadas."""
        pass

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Ajusta y transforma en un solo paso."""
        self.fit(X, y)
        return self.transform(X)