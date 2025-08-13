from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class IQRClipper(BaseEstimator, TransformerMixin):
    """
    Capping de outliers con IQR en columnas numéricas.
    - Guarda feature_names_in_ para que ColumnTransformer pueda propagar nombres.
    - Implementa get_feature_names_out -> devuelve los mismos nombres de entrada.
    """
    def __init__(self, factor: float = 1.5) -> None:
        self.factor = float(factor)
        self.numeric_cols_: list[str] = []
        self.bounds_: dict[str, tuple[float, float]] = {}
        self.feature_names_in_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "IQRClipper":
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)

        self.numeric_cols_ = list(X.select_dtypes(include=["number"]).columns)
        self.bounds_ = {}
        for col in self.numeric_cols_:
            q1, q3 = pd.Series(X[col]).quantile([0.25, 0.75])
            iqr = q3 - q1
            self.bounds_[col] = (float(q1 - self.factor * iqr),
                                 float(q3 + self.factor * iqr))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_ or None)
        X = X.copy()
        for col, (low, up) in self.bounds_.items():
            if col in X.columns:
                X[col] = pd.Series(X[col]).clip(low, up)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return np.asarray(list(input_features), dtype=object)
