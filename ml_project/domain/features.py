"""Feature engineering utilities."""
from __future__ import annotations

import pandas as pd


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple domain features."""
    df = df.copy()
    if "income" in df.columns and "expenses" in df.columns:
        df["savings_rate"] = (
            df["income"] - df["expenses"]
        ) / df["income"].replace({0: 1})
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"])
        df["day_of_week"] = dt.dt.weekday
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df
