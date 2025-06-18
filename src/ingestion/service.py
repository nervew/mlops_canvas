from __future__ import annotations

from typing import Any, Dict

from ingestion.ports import IDataIngestionPort
from ingestion.adapters.sql_server import SqlServerAdapter
from ingestion.adapters.postgres import PostgresAdapter
from ingestion.adapters.databricks import DatabricksSqlAdapter


class ConnectorFactory:
    """Factory that returns ingestion adapters."""

    _mapping = {
        "sql_server": SqlServerAdapter,
        "postgres": PostgresAdapter,
        "databricks": DatabricksSqlAdapter,
    }

    @staticmethod
    def get(connector_type: str, config: Dict[str, Any]) -> IDataIngestionPort:
        if connector_type not in ConnectorFactory._mapping:
            raise ValueError(f"Unsupported connector type: {connector_type}")
        adapter_cls = ConnectorFactory._mapping[connector_type]
        return adapter_cls(**config)


class IngestionService:
    """Service orchestrating the ingestion process."""

    def __init__(self, connector: IDataIngestionPort) -> None:
        self.connector = connector

    def ingest(self, sql: str) -> Any:
        data = self.connector.execute_query(sql)
        return data
