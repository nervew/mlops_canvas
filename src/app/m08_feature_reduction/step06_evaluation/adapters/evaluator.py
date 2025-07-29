from typing import Any, Dict
import pandas as pd
from sklearn.metrics import mean_squared_error

from ..ports.evaluator import IEvaluator

class Evaluator(IEvaluator):
    """
    Evalúa bien:
      - Si el objeto tiene .predict(), usa ese predict vs y_val.
      - Si no, pero tiene 'reducer' en named_steps (un PCA), calcula
        MSE de reconstrucción: ||X - inverse_transform(transform(X))||^2.
      - Si no tiene predict ni reducer, retorna métricas vacías.
    """
    def evaluate(
        self,
        model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, float]:
        # 1) Ruta estimador con predict()
        if hasattr(model, "predict"):
            preds = model.predict(X_val)
            return {"mse": mean_squared_error(y_val, preds)}

        # 2) Ruta reducción: buscar PCA en el pipeline
        if hasattr(model, "named_steps") and "reducer" in model.named_steps:
            pca = model.named_steps["reducer"]
            if not hasattr(pca, "inverse_transform"):
                raise ValueError("El reductor no soporta inverse_transform para reconstrucción.")
            X = X_val.values if hasattr(X_val, "values") else X_val
            reduced = pca.transform(X)
            recon = pca.inverse_transform(reduced)
            mse = mean_squared_error(X, recon)
            return {"reconstruction_mse": mse}

        # 3) Sin predict ni reductor => ningun cálculo
        return {}