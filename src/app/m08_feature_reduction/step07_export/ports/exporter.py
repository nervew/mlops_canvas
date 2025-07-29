# m08_feature_reduction/step07_export/ports/exporter.py

from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd
import numpy as np

class IExporter(ABC):
    @abstractmethod
    def export_all(
        self,
        transformer: Any,
        shap_values: np.ndarray,
        shap_summary: np.ndarray,
        dfs: Dict[str, pd.DataFrame]
    ) -> None:
        """
        Serializa el pipeline final, guarda:
         - shap_summary (importancia global)
         - gráfico SHAP con shap_values
         - dataframes procesados
        """
        pass
