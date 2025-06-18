import pytest
from ingestion.service import ConnectorFactory
from ingestion.adapters.sql_server import SqlServerAdapter
from ingestion.adapters.postgres import PostgresAdapter
from ingestion.adapters.databricks import DatabricksSqlAdapter


def test_factory_returns_correct_adapter():
    cfg = {"conn_str": "dummy"}
    adapter = ConnectorFactory.get("sql_server", cfg)
    assert isinstance(adapter, SqlServerAdapter)

    adapter = ConnectorFactory.get("postgres", cfg)
    assert isinstance(adapter, PostgresAdapter)

    adapter = ConnectorFactory.get("databricks", {
        "server_hostname": "host",
        "http_path": "path",
        "access_token": "token",
    })
    assert isinstance(adapter, DatabricksSqlAdapter)


def test_factory_invalid_adapter():
    with pytest.raises(ValueError):
        ConnectorFactory.get("unknown", {})
