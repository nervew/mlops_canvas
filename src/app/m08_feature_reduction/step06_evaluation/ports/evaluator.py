from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd

class IEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, float]:
        """
        Evalúa el modelo usando X_val, y_val y retorna un diccionario de métricas.
        """
        pass