import pytest

from app.m01_data_ingestion.domain.ports import IDataSource


def test_idatasource_is_abstract():
    with pytest.raises(TypeError):
        IDataSource()
