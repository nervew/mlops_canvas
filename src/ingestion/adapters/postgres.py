from __future__ import annotations

import pandas as pd
import psycopg2

from ingestion.ports import IDataIngestionPort


class PostgresAdapter(IDataIngestionPort):
    """Adapter to fetch data from PostgreSQL."""

    def __init__(self, conn_str: str) -> None:
        self.conn_str = conn_str

    def execute_query(self, sql: str) -> pd.DataFrame:
        with psycopg2.connect(self.conn_str) as conn:
            return pd.read_sql(sql, conn)
