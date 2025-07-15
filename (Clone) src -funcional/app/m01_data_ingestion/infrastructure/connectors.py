import pandas as pd
import sqlalchemy as sa
from pyspark.sql import SparkSession

def spark_connector(query: str) -> pd.DataFrame:
    spark = SparkSession.builder.getOrCreate()
    spark.sql("USE CATALOG hive_metastore") 
    return spark.sql(query).toPandas()

def postgres_connector(conn_str: str, query: str) -> pd.DataFrame:
    engine = sa.create_engine(conn_str)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

def sqlserver_connector(conn_str: str, query: str) -> pd.DataFrame:
    engine = sa.create_engine(conn_str)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

def get_connector(cfg: dict):
    db_type = cfg["db_type"].lower()
    conn_str = cfg.get("conn_str", "")

    if db_type == "spark":
        return lambda q: spark_connector(q)
    elif db_type == "postgres":
        return lambda q: postgres_connector(conn_str, q)
    elif db_type == "sqlserver":
        return lambda q: sqlserver_connector(conn_str, q)
    else:
        raise ValueError(f"db_type '{db_type}' no soportado.")
