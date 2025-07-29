from __future__ import annotations
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from ..ports import ICategoricalEncoder


class OneHotEncoderAdapter(ICategoricalEncoder, BaseEstimator, TransformerMixin):
    """One-Hot robusto (handle_unknown='ignore')."""

    def __init__(self) -> None:
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.cat_cols: list[str] = []
        self.feature_names: list[str] = []

    def fit(self, df: pd.DataFrame, y=None) -> "OneHotEncoderAdapter":
        self.cat_cols = df.select_dtypes(include="object").columns.tolist()
        if self.cat_cols:
            self.encoder.fit(df[self.cat_cols])
            self.feature_names = list(self.encoder.get_feature_names_out(self.cat_cols))
        self.fitted_ = True  # <-- atributo "_"
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("OneHotEncoderAdapter: llama primero a fit().")
        df = df.copy()
        if self.cat_cols:
            missing = [c for c in self.cat_cols if c not in df.columns]
            for c in missing:
                df[c] = pd.NA
            enc = self.encoder.transform(df[self.cat_cols])
            df = df.drop(columns=self.cat_cols)
            df[self.feature_names] = enc
        return df

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)
