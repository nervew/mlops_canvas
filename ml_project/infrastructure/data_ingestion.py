"""Data ingestion utilities using DataSource adapters."""
from __future__ import annotations

import pandas as pd
from typing import Optional

from ..adapters.data_source_interface import DataSource


def ingest_data(source: DataSource) -> pd.DataFrame:
    """Fetch data from the given data source."""
    df = source.fetch_data()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Data source must return a pandas DataFrame")
    return df
