from __future__ import annotations
import pandas as pd
from pyspark.sql import SparkSession

class SparkAdapter:
    def __init__(self, catalog: str | None = None) -> None:
        self.catalog = catalog
        self.spark: SparkSession | None = None

    def connect(self) -> None:
        self.spark = SparkSession.builder.getOrCreate()
        if self.catalog:
            self.spark.sql(f"USE CATALOG {self.catalog}")

    def run_query(self, sql: str) -> pd.DataFrame:
        return self.spark.sql(sql).toPandas()

class DatabricksAdapter:
    def __init__(self, conn_str: str) -> None:
        self.conn_str = conn_str
        self.spark: SparkSession | None = None

    def connect(self) -> None:
        self.spark = (
            SparkSession.builder
            .config("spark.jars.packages", "com.databricks:databricks-jdbc:2.6.34")
            .getOrCreate()
        )

    def run_query(self, sql: str) -> pd.DataFrame:
        df = (
            self.spark.read
            .format("jdbc")
            .option("url", self.conn_str)
            .option("query", sql)
            .option("driver", "com.databricks.client.jdbc.Driver")
            .load()
        )
        return df.toPandas()

# Stubs para RDBMS si sqlalchemy NO está instalado
try:
    import sqlalchemy as sa

    class PostgresAdapter:
        def __init__(self, conn_str: str) -> None:
            self.conn_str = conn_str
            self.engine: sa.Engine | None = None

        def connect(self) -> None:
            self.engine = sa.create_engine(self.conn_str)

        def run_query(self, sql: str) -> pd.DataFrame:
            with self.engine.connect() as conn:
                return pd.read_sql(sql, conn)

    class SqlServerAdapter(PostgresAdapter):
        ...
except ImportError:
    class PostgresAdapter:
        def __init__(self, *_: object, **__: object) -> None:
            raise ModuleNotFoundError("sqlalchemy is required for PostgresAdapter")

    class SqlServerAdapter:
        def __init__(self, *_: object, **__: object) -> None:
            raise ModuleNotFoundError("sqlalchemy is required for SqlServerAdapter")
