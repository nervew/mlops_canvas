import pandas as pd

from app.m01_data_ingestion.domain.entities import DataSet


def test_dataset_holds_dataframe(sample_df):
    ds = DataSet(sample_df)
    assert isinstance(ds.data, pd.DataFrame)
    pd.testing.assert_frame_equal(ds.data, sample_df)
