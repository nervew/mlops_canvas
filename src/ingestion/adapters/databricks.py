from __future__ import annotations

import pandas as pd
from databricks import sql

from ingestion.ports import IDataIngestionPort


class DatabricksSqlAdapter(IDataIngestionPort):
    """Adapter for Databricks SQL warehouses."""

    def __init__(self, server_hostname: str, http_path: str, access_token: str) -> None:
        self.server_hostname = server_hostname
        self.http_path = http_path
        self.access_token = access_token

    def execute_query(self, sql_query: str) -> pd.DataFrame:
        with sql.connect(
            server_hostname=self.server_hostname,
            http_path=self.http_path,
            access_token=self.access_token,
        ) as connection:
            return pd.read_sql(sql_query, connection)
