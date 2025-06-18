import pandas as pd
from ml_project.adapters.file_source import CSVDataSource
from ml_project.infrastructure.data_ingestion import ingest_data


def test_ingest_data_csv(tmp_path):
    sample = tmp_path / "sample.csv"
    sample.write_text("a,b\n1,x\n2,y\n")
    source = CSVDataSource(str(sample))
    df = ingest_data(source)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
