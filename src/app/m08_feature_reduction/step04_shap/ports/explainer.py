# m08_feature_reduction/step04_shap/ports/explainer.py

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class IShapExplainer(ABC):
    @abstractmethod
    def explain(
        self,
        reducer,             # objeto con método .transform(X)
        X_train: pd.DataFrame,
    ) -> tuple[
        np.ndarray,          # shap_values (sample × components)
        np.ndarray           # shap_summary (importance por componente)
    ]:
        """
        Calcula SHAP values y summary:
        - Devolver shap_values y shap_summary.
        """
        pass
