import pandas as pd
import sqlalchemy as sa
from pyspark.sql import SparkSession
from ..domain.ports import IDataSource

class SparkAdapter(IDataSource):
    def __init__(self, catalog: str = None) -> None:
        self.catalog = catalog
        self.spark: SparkSession | None = None

    def connect(self) -> None:
        self.spark = SparkSession.builder.getOrCreate()
        if self.catalog:
            self.spark.sql(f"USE CATALOG {self.catalog}")

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

class DatabricksAdapter(IDataSource):
    def __init__(self, conn_str: str) -> None:
        self.conn_str = conn_str
        self.spark: SparkSession | None = None

    def connect(self) -> None:
        # Asegúrate de tener el paquete JDBC de Databricks en tu sesión Spark
        # Por ejemplo: spark.jars.packages = "com.databricks:databricks-jdbc:2.6.34"
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

