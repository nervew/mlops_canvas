from typing import Optional
import pandas as pd
import sqlite3

from .data_source_interface import DataSource

class SQLiteDataSource(DataSource):
    """Data source fetching data from a SQLite database."""

    def __init__(self, db_path: str, query: str) -> None:
        self.db_path = db_path
        self.query = query

    def fetch_data(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(self.query, conn)
        finally:
            conn.close()
        if df.empty:
            raise ValueError("SQLiteDataSource: query returned no rows")
        return df
