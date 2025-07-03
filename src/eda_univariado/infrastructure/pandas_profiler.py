import pandas as pd


def generate_report(df: pd.DataFrame) -> pd.DataFrame:
    """Genera estadísticos descriptivos básicos."""
    return df.describe(include="all")
