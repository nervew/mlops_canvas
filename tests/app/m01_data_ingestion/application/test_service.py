import pandas as pd

from app.m01_data_ingestion.application.service import ingest


class DummyRepo:
    def __init__(self, cfg):
        self.cfg = cfg

    def ingest(self, sql: str):
        self.sql = sql
        from app.m01_data_ingestion.domain.entities import DataSet
        return DataSet(pd.DataFrame({"a": [1]}))


def test_ingest_reads_config_and_query(monkeypatch):
    monkeypatch.setattr(
        "app.m01_data_ingestion.application.service.IngestionRepository",
        DummyRepo,
    )
    df = ingest()
    assert isinstance(df, pd.DataFrame)
    assert df.iloc[0, 0] == 1
