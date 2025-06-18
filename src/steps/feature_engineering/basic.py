from __future__ import annotations

from typing import Tuple
import pandas as pd


def engineer_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Basic time feature extraction and one-hot encoding."""

    def transform(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "date" in df.columns:
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["day"] = df["date"].dt.day
            df = df.drop(columns=["date"])
        if "category" in df.columns:
            df = pd.get_dummies(df, columns=["category"], drop_first=False)
        return df

    return transform(train_df), transform(test_df)
