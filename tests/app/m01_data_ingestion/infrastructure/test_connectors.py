from types import SimpleNamespace
import pandas as pd

import app.m01_data_ingestion.infrastructure.connectors as connectors

from types import SimpleNamespace
import pandas as pd

import app.m01_data_ingestion.infrastructure.connectors as connectors


class DummySession:
    def __init__(self):
        self.queries = []

    def sql(self, query):
        self.queries.append(query)
        if query.startswith("USE"):
            return self
        return SimpleNamespace(toPandas=lambda: pd.DataFrame({"a": [1]}))


class DummyBuilder:
    def __init__(self):
        self._session = DummySession()

    def getOrCreate(self):
        return self._session

    def config(self, *args, **kwargs):
        return self


def test_spark_adapter(monkeypatch):
    dummy_builder = DummyBuilder()
    monkeypatch.setattr(connectors.SparkSession, "builder", dummy_builder)
    adapter = connectors.SparkAdapter(catalog="cat")
    adapter.connect()
    assert adapter.spark is dummy_builder._session
    df = adapter.run_query("SELECT 1")
    assert isinstance(df, pd.DataFrame)
    assert dummy_builder._session.queries[0] == "USE CATALOG cat"
    assert dummy_builder._session.queries[1] == "SELECT 1"


class DummyEngine:
    def connect(self):
        class Ctx:
            def __enter__(self):
                return "conn"

            def __exit__(self, *args):
                return False

        return Ctx()


def test_postgres_adapter(monkeypatch):
    monkeypatch.setattr(connectors.sa, "create_engine", lambda conn: DummyEngine())
    monkeypatch.setattr(connectors.pd, "read_sql", lambda sql, conn: pd.DataFrame({"x": [1]}))
    adapter = connectors.PostgresAdapter("db")
    adapter.connect()
    df = adapter.run_query("SELECT 1")
    assert df.iloc[0, 0] == 1
