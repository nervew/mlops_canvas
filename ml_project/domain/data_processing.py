"""Data cleaning and preprocessing utilities."""
from __future__ import annotations

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and impute missing values."""
    df = df.drop_duplicates().copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            median = df[col].median()
            df[col].fillna(median, inplace=True)
        else:
            df[col].fillna("Unknown", inplace=True)
    for col in df.select_dtypes(include="number").columns:
        mean, std = df[col].mean(), df[col].std()
        cap = mean + 3 * std
        floor = mean - 3 * std
        df[col] = df[col].clip(lower=floor, upper=cap)
    return df
