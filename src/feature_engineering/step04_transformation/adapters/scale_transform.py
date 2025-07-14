from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler

from ..ports import ITransformer


class StandardScaleTransformer(ITransformer):
    """Applies Yeo-Johnson and then standard scaling."""

    def __init__(self) -> None:
        self.power = PowerTransformer(method="yeo-johnson")
        self.scaler = StandardScaler()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = self.power.fit_transform(df[num_cols])
        df[num_cols] = self.scaler.fit_transform(df[num_cols])
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = self.power.transform(df[num_cols])
        df[num_cols] = self.scaler.transform(df[num_cols])
        return df
