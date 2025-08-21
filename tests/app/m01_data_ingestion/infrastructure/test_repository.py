import pandas as pd
import pytest

import app.m01_data_ingestion.infrastructure.repository as repo_module
from app.m01_data_ingestion.infrastructure.repository import IngestionRepository


class DummyAdapter:
    def __init__(self, param):
        self.param = param
        self.connected = False

    def connect(self):
        self.connected = True

    def run_query(self, sql: str):
        return pd.DataFrame({"x": [1]})


def test_repository_uses_correct_adapter(monkeypatch):
    monkeypatch.setitem(repo_module._ADAPTERS, "spark", DummyAdapter)
    repo = IngestionRepository({"db_type": "spark", "catalog": "cat"})
    result = repo.ingest("SELECT 1")
    assert isinstance(result.data, pd.DataFrame)
    assert result.data.iloc[0, 0] == 1


def test_repository_unknown_type_raises(monkeypatch):
    with pytest.raises(ValueError):
        IngestionRepository({"db_type": "unknown"})
