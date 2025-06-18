from __future__ import annotations

import pandas as pd
import pyodbc

from ingestion.ports import IDataIngestionPort


class SqlServerAdapter(IDataIngestionPort):
    """Adapter to fetch data from Microsoft SQL Server."""

    def __init__(self, conn_str: str) -> None:
        self.conn_str = conn_str

    def execute_query(self, sql: str) -> pd.DataFrame:
        with pyodbc.connect(self.conn_str) as conn:
            return pd.read_sql(sql, conn)
