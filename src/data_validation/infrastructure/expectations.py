"""Simple validation utilities for the Iris dataset."""

import pandas as pd


def validate_schema(df: pd.DataFrame) -> bool:
    """Check that target column exists and has no nulls."""
    if "target" not in df.columns:
        return False
    return df["target"].notnull().all()
