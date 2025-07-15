from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class DateDecomposer(BaseEstimator, TransformerMixin):
    """Convierte columnas datetime en features numéricas simples."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        out = pd.DataFrame(index=X.index)
        for col in X.columns:
            out[f"{col}_year"] = X[col].dt.year
            out[f"{col}_month"] = X[col].dt.month
            out[f"{col}_day"] = X[col].dt.day
        return out
