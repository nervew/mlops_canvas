# m08_feature_reduction/step06_evaluation/adapters/evaluator.py
from __future__ import annotations
from typing import Callable, Dict
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# En sklearn >= 1.4: usar root_mean_squared_error; en otras versiones, fallback.
try:
    from sklearn.metrics import root_mean_squared_error  # sklearn >= 1.4
    _HAS_RMSE_FN = True
except Exception:
    root_mean_squared_error = None
    _HAS_RMSE_FN = False


class Evaluator:
    """
    Evaluación SUPERVISADA:
    - Recibe predict_fn(X_df_crudo) -> y_pred (1D).
    - Calcula MAE, MSE, RMSE, R2 sin usar el parámetro 'squared'.
    """
    def evaluate(self,
                 predict_fn: Callable[[pd.DataFrame], np.ndarray],
                 X_val: pd.DataFrame,
                 y_val: pd.Series) -> Dict[str, float]:
        if X_val is None or y_val is None or len(X_val) == 0 or len(y_val) == 0:
            return {}

        y_hat = np.asarray(predict_fn(X_val)).ravel()
        y_true = np.asarray(y_val).ravel()

        mse = float(mean_squared_error(y_true, y_hat))
        if _HAS_RMSE_FN:
            rmse = float(root_mean_squared_error(y_true, y_hat))
        else:
            rmse = float(np.sqrt(mse))

        return {
            "mae": float(mean_absolute_error(y_true, y_hat)),
            "mse": mse,
            "rmse": rmse,
            "r2": float(r2_score(y_true, y_hat)),
        }
