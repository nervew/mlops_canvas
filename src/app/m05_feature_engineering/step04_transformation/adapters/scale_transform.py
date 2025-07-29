from __future__ import annotations
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, StandardScaler
from ..ports import ITransformer


class StandardScaleTransformer(ITransformer, BaseEstimator, TransformerMixin):
    """Yeo-Johnson + escalado estándar."""

    def __init__(self) -> None:
        self.power  = PowerTransformer(method="yeo-johnson")
        self.scaler = StandardScaler()
        self.num_cols: list[str] = []

    def fit(self, df: pd.DataFrame, y=None) -> "StandardScaleTransformer":
        self.num_cols = df.select_dtypes(include="number").columns.tolist()
        if self.num_cols:
            self.power.fit(df[self.num_cols])
            self.scaler.fit(self.power.transform(df[self.num_cols]))
        self.fitted_ = True  # <-- atributo "_"
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("StandardScaleTransformer: llama primero a fit().")
        df = df.copy()
        if self.num_cols:
            tmp = self.power.transform(df[self.num_cols])
            df[self.num_cols] = self.scaler.transform(tmp)
        return df

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)
