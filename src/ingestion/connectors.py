from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
import sqlalchemy as sa
from pyspark.sql import SparkSession

class ConnectorPort(ABC):
    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def execute_query(self, sql: str) -> Any:
        pass

class SqlServerAdapter(ConnectorPort):
    def __init__(self, conn_str: str) -> None:
        self.conn_str = conn_str
        self.engine = None

    def connect(self) -> None:
        self.engine = sa.create_engine(self.conn_str)

    def execute_query(self, sql: str) -> pd.DataFrame:
        with self.engine.connect() as connection:
            return pd.read_sql(sql, connection)

class PostgresAdapter(ConnectorPort):
    def __init__(self, conn_str: str) -> None:
        self.conn_str = conn_str
        self.engine = None

    def connect(self) -> None:
        self.engine = sa.create_engine(self.conn_str)

    def execute_query(self, sql: str) -> pd.DataFrame:
        with self.engine.connect() as connection:
            return pd.read_sql(sql, connection)

class SparkAdapter(ConnectorPort):
    def __init__(self) -> None:
        self.spark = None

    def connect(self) -> None:
        self.spark = SparkSession.builder.getOrCreate()

    def execute_query(self, sql: str) -> pd.DataFrame:
        df_spark = self.spark.sql(sql)
        return df_spark.toPandas()
