from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from ..ports import IFeatureGenerator


class PolynomialFeatureGenerator(IFeatureGenerator):
    def __init__(self, degree: int = 2) -> None:
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.feature_names: list[str] = []
        self.numeric_cols: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.numeric_cols = df.select_dtypes(include="number").columns.tolist()
        poly_data = self.poly.fit_transform(df[self.numeric_cols])
        self.feature_names = list(self.poly.get_feature_names_out(self.numeric_cols))
        df = df.drop(columns=self.numeric_cols)
        df[self.feature_names] = poly_data
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        poly_data = self.poly.transform(df[self.numeric_cols])
        df = df.drop(columns=self.numeric_cols)
        df[self.feature_names] = poly_data
        return df
