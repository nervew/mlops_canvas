import numpy as np
import pandas as pd
from typing import Tuple, Any

class IShapExplainer:
    """
    Interfaz para un explicador SHAP agnóstico.
    explain() debe devolver:
      - shap_values: np.ndarray, shape=(n_samples, n_features o n_components)
      - shap_summary: np.ndarray, shape=(n_features o n_components,)
    """
    def explain(
        self,
        reducer: Any,
        X_train: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
