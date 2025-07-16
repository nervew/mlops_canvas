# step03_outliers/adapters/iqr_handler.py
from __future__ import annotations
import pandas as pd
from ..ports.handler import IOutlierHandler

class IQRHandler(IOutlierHandler):
    """Capping de Outliers usando método del IQR."""

    def __init__(self, factor: float = 1.5) -> None:
        self.factor = factor
        self.bounds: dict[str, tuple[float, float]] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - self.factor * iqr
            upper = q3 + self.factor * iqr
            self.bounds[col] = (lower, upper)
            df[col] = df[col].clip(lower, upper)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, (lower, upper) in self.bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lower, upper)
        return df
