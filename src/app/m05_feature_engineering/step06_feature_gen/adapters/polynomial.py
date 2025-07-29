# step06_feature_gen/adapters/polynomial.py

from __future__ import annotations
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from ..ports import IFeatureGenerator


class PolynomialFeatureGenerator(IFeatureGenerator, BaseEstimator, TransformerMixin):
    """Genera features polinomiales para columnas numéricas."""

    def __init__(self, degree: int = 2) -> None:
        self.degree = degree
        self.poly: PolynomialFeatures | None = None
        self.num_cols: list[str] = []
        self.feature_names: list[str] = []
        # Marcar fitted
        self.fitted_: bool = False

    def fit(self, df: pd.DataFrame, y=None) -> "PolynomialFeatureGenerator":
        self.num_cols = df.select_dtypes(include="number").columns.tolist()
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        if self.num_cols:
            self.poly.fit(df[self.num_cols])
            self.feature_names = list(self.poly.get_feature_names_out(self.num_cols))
        else:
            self.feature_names = []
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("PolynomialFeatureGenerator: llama primero a fit().")
        df = df.copy()
        if self.num_cols:
            poly_data = self.poly.transform(df[self.num_cols])
            df = df.drop(columns=self.num_cols)
            df[self.feature_names] = poly_data
        return df

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        # Implementación explícita para satisfacer el ABC de IFeatureGenerator
        return self.fit(df, y).transform(df)
