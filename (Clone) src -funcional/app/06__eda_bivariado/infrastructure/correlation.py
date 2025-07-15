import pandas as pd


def generate_report(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(numeric_only=True)
