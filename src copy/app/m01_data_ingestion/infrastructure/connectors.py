import pandas as pd
import sqlalchemy as sa
from pyspark.sql import SparkSession
from ..domain.ports import IDataSource

class SparkAdapter(IDataSource):
    def __init__(self, catalog: str = "hive_metastore") -> None:
        self.catalog = catalog
        self.spark: SparkSession | None = None

    def connect(self) -> None:
        self.spark = SparkSession.builder.getOrCreate()
        self.spark.sql(f"USE CATALOG {self.catalog}")  # Aquí corriges el problema.

    def run_query(self, sql: str) -> pd.DataFrame:
        return self.spark.sql(sql).toPandas()

class PostgresAdapter(IDataSource):
    def __init__(self, conn_str: str) -> None:
        self.conn_str = conn_str

    def connect(self) -> None:
        self.engine = sa.create_engine(self.conn_str)

    def run_query(self, sql: str) -> pd.DataFrame:
        with self.engine.connect() as conn:
            return pd.read_sql(sql, conn)

class SqlServerAdapter(IDataSource):
    def __init__(self, conn_str: str) -> None:
        self.conn_str = conn_str

    def connect(self) -> None:
        self.engine = sa.create_engine(self.conn_str)

    def run_query(self, sql: str) -> pd.DataFrame:
        with self.engine.connect() as conn:
            return pd.read_sql(sql, conn)
