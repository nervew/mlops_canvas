from __future__ import annotations
from ..domain.entities import DataSet
from .connectors import (
    SparkAdapter,
    PostgresAdapter,
    SqlServerAdapter,
    DatabricksAdapter,
)

_ADAPTERS: dict[str, type] = {
    "spark":      SparkAdapter,
    "postgres":   PostgresAdapter,
    "sqlserver":  SqlServerAdapter,
    "databricks": DatabricksAdapter,
}

class IngestionRepository:
    def __init__(self, cfg: dict) -> None:
        db_type = cfg["db_type"].lower()
        Adapter = _ADAPTERS.get(db_type)
        if Adapter is None:
            raise ValueError(f"Unknown db_type: {db_type}")
        # Spark usa 'catalog'; RDBMS/JDBC usan 'conn_str'
        if db_type == "spark":
            self.adapter = Adapter(cfg.get("catalog"))
        else:
            self.adapter = Adapter(cfg.get("conn_str", ""))
        self.cfg = cfg

    def ingest(self, sql: str) -> DataSet:
        self.adapter.connect()
        df = self.adapter.run_query(sql)
        return DataSet(data=df)
