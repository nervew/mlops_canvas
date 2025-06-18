import sys, os
sys.path.append(os.path.abspath("src"))
from ingestion.connectors import SqlServerAdapter


def test_sqlserver_adapter_connect() -> None:
    adapter = SqlServerAdapter("test_connection")
    adapter.connect()
    assert adapter.connection is not None
