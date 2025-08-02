import shap
import numpy as np
import pandas as pd
from ..ports.explainer import IShapExplainer

class ShapExplainer(IShapExplainer):
    """
    Explicador agnóstico basado en SHAP.
    • explainer_type: "kernel" o "permutation"
    • background_size: nº de filas para background
    • sample_size: nº de filas para cálculo de SHAP
    • random_state: semilla reproducible
    """

    def __init__(
        self,
        explainer_type: str = "kernel",
        background_size: int = 100,
        sample_size: int = 100,
        random_state: int = 42,
    ):
        self.explainer_type = explainer_type.lower()
        self.background_size = background_size
        self.sample_size = sample_size
        self.random_state = random_state

    def _build_explainer(self, model_fn, background: pd.DataFrame):
        if self.explainer_type == "permutation":
            return shap.PermutationExplainer(model_fn, background)
        return shap.KernelExplainer(model_fn, background)

    def explain(self, reducer, X_train: pd.DataFrame):
        # 1) Background y sample
        background = shap.sample(
            X_train, self.background_size, random_state=self.random_state
        )
        sample = shap.sample(
            X_train, self.sample_size, random_state=self.random_state
        )

        # 2) Crear explainer
        explainer = self._build_explainer(reducer.transform, background)

        # 3) Calcular shap_values
        shap_values = explainer.shap_values(sample)
        arr = np.asarray(shap_values[0] if isinstance(shap_values, list) else shap_values)

        # 4) Importancia global
        summary = np.mean(np.abs(arr), axis=0)

        return arr, summary
