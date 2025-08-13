# m08_feature_reduction/step04_shap/adapters/shap_explainer.py
from __future__ import annotations
import numpy as np
import pandas as pd
import shap
from ..ports.explainer import IShapExplainer

class ShapExplainer(IShapExplainer):
    """
    Explicador SUPERVISADO sobre la función de predicción del MODELO.
    - explainer_type: "kernel" o "permutation"
    - background_size: nº de filas para background
    - sample_size: nº de filas para cálculo de SHAP
    - random_state: semilla reproducible
    Uso: explain(predict_fn, X_train_df)
      * predict_fn: callable que recibe pd.DataFrame (crudo) y retorna y_pred (1D).
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

    def _build(self, model_fn, background_X: np.ndarray):
        if self.explainer_type == "permutation":
            return shap.PermutationExplainer(model_fn, background_X)
        return shap.KernelExplainer(model_fn, background_X)

    def explain(self, predict_fn, X_train: pd.DataFrame):
        # 1) Muestras: trabajamos en ESPACIO CRUDO para que predict_fn aplique todo
        bg_df = shap.sample(X_train, self.background_size, random_state=self.random_state)
        sm_df = shap.sample(X_train, self.sample_size, random_state=self.random_state)

        # 2) Adaptador: SHAP llamará con ndarray; lo convertimos a DataFrame con mismas columnas
        cols = list(X_train.columns)

        def _wrapped_fn(x_nd: np.ndarray):
            df = pd.DataFrame(x_nd, columns=cols)
            y_pred = predict_fn(df)  # 1D
            return y_pred

        explainer = self._build(_wrapped_fn, bg_df.values)

        # 3) SHAP values
        shap_values = explainer.shap_values(sm_df.values)
        arr = np.asarray(shap_values[0] if isinstance(shap_values, list) else shap_values)
        # 4) Importancia global (media abs)
        summary = np.mean(np.abs(arr), axis=0)
        return arr, summary
