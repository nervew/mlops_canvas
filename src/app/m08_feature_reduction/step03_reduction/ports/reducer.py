# m08_feature_reduction/step03_reduction/ports/reducer.py

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class IReducer(ABC):
    @abstractmethod
    def fit(self, X_train: pd.DataFrame) -> None:
        """Ajusta el reductor solo con X_train."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforma X (puede devolver array o DataFrame)."""
        pass

    @abstractmethod
    def explained_variance_ratio(self) -> np.ndarray | None:
        """Retorna la varianza explicada (o None si está desactivado)."""
        pass
