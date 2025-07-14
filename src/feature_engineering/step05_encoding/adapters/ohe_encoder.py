from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from ..ports import ICategoricalEncoder


class OneHotEncoderAdapter(ICategoricalEncoder):
    def __init__(self) -> None:
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.columns: list[str] = []
        self.feature_names: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cat_cols = df.select_dtypes(include="object").columns
        self.columns = list(cat_cols)
        transformed = self.encoder.fit_transform(df[cat_cols])
        self.feature_names = list(self.encoder.get_feature_names_out(cat_cols))
        df = df.drop(columns=cat_cols)
        df[self.feature_names] = transformed
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cat_cols = [c for c in self.columns if c in df.columns]
        transformed = self.encoder.transform(df[cat_cols])
        df = df.drop(columns=cat_cols)
        df[self.feature_names] = transformed
        return df
