# m08_feature_reduction/step04_shap/adapters/shap_explainer.py

import shap
import numpy as np
import pandas as pd

from ..ports.explainer import IShapExplainer

class ShapExplainer(IShapExplainer):
    """
    Calcula SHAP values de forma agnóstica (KernelExplainer).
    """

    def __init__(
        self,
        explainer_type: str = "kernel",
        background_size: int = 100,
        sample_size: int = 100,
        random_state: int = 42,
    ) -> None:
        self.explainer_type   = explainer_type
        self.background_size  = background_size
        self.sample_size      = sample_size
        self.random_state     = random_state

    def explain(
        self,
        reducer,
        X_train: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Fijamos semilla para reproducibilidad
        np.random.seed(self.random_state)

        # Seleccionamos background y sample
        background = shap.sample(X_train, self.background_size, random_state=self.random_state)
        sample     = shap.sample(X_train, self.sample_size, random_state=self.random_state)

        # Elegimos KernelExplainer (agnóstico)
        explainer = shap.KernelExplainer(
            reducer.transform,
            background
        )

        # Calculamos shap_values
        shap_values = explainer.shap_values(sample)

        # Resumimos importancia global (mean absolute)
        shap_summary = np.abs(shap_values).mean(axis=0)

        return shap_values, shap_summary
