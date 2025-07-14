from typing import Any
import pandas as pd
import sqlalchemy as sa
from pyspark.sql import SparkSession
from ..domain.ports import IDataSource

# ------------------------------------------------------------------ #
# Conectores concretos
# ------------------------------------------------------------------ #
class SparkAdapter(IDataSource):
    """Obtiene datos vía Spark SQL."""
    def __init__(self, _: str | None = None) -> None:
        self.spark: SparkSession | None = None

    def connect(self) -> None:
        self.spark = SparkSession.builder.getOrCreate()

    def run_query(self, sql: str) -> pd.DataFrame:
        if not self.spark:
            raise ConnectionError("SparkSession no inicializada")
        return self.spark.sql(sql).toPandas()


class PostgresAdapter(IDataSource):
    def __init__(self, conn_str: str) -> None:
        self.conn_str = conn_str
        self.engine: sa.Engine | None = None

    def connect(self) -> None:
        self.engine = sa.create_engine(self.conn_str)

    def run_query(self, sql: str) -> pd.DataFrame:
        if not self.engine:
            raise ConnectionError("Not connected to Postgres")
        with self.engine.connect() as conn:
            return pd.read_sql(sql, conn)


class SqlServerAdapter(IDataSource):
    def __init__(self, conn_str: str) -> None:
        self.conn_str = conn_str
        self.engine: sa.Engine | None = None

    def connect(self) -> None:
        self.engine = sa.create_engine(self.conn_str)

    def run_query(self, sql: str) -> pd.DataFrame:
        if not self.engine:
            raise ConnectionError("Not connected to SQL Server")
        with self.engine.connect() as conn:
            return pd.read_sql(sql, conn)
