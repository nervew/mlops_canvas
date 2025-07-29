from __future__ import annotations
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from ..ports import IImputer


class SimpleImputerAdapter(IImputer, BaseEstimator, TransformerMixin):
    """Imputación mediana para columnas numéricas."""

    def __init__(self) -> None:
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.numeric_cols: list[str] = []

    def fit(self, df: pd.DataFrame, y=None) -> "SimpleImputerAdapter":
        self.numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if self.numeric_cols:
            self.numeric_imputer.fit(df[self.numeric_cols])
        self.fitted_ = True  # <-- atributo con "_" que marca el estimator como fitted :contentReference[oaicite:0]{index=0}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("SimpleImputerAdapter: llama primero a fit().")
        df = df.copy()
        if self.numeric_cols:
            df[self.numeric_cols] = self.numeric_imputer.transform(df[self.numeric_cols])
        return df

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)
