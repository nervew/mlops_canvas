from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer

from ..ports import IImputer


class SimpleImputerAdapter(IImputer):
    """Simple median imputation for numeric columns."""

    def __init__(self) -> None:
        self.numeric_imputer = SimpleImputer(strategy="median")

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = self.numeric_imputer.fit_transform(df[numeric_cols])
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = self.numeric_imputer.transform(df[numeric_cols])
        return df