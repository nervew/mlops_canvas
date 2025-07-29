from __future__ import annotations
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from ..ports.handler import IOutlierHandler


class IQRHandler(IOutlierHandler, BaseEstimator, TransformerMixin):
    """Capping de outliers con método IQR."""

    def __init__(self, factor: float = 1.5) -> None:
        self.factor = factor
        self.bounds: dict[str, tuple[float, float]] = {}
        self.numeric_cols: list[str] = []

    def fit(self, df: pd.DataFrame, y=None) -> "IQRHandler":
        self.numeric_cols = df.select_dtypes(include="number").columns.tolist()
        self.bounds = {}
        for col in self.numeric_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.bounds[col] = (
                q1 - self.factor * iqr,
                q3 + self.factor * iqr,
            )
        self.fitted_ = True  # <-- atributo "_"
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("IQRHandler: llama primero a fit().")
        df = df.copy()
        for col, (low, up) in self.bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(low, up)
        return df

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)
