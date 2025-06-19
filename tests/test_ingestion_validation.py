import pandas as pd
import pytest
from unittest import mock

import data_ingestion_validation.validate as dv
from data_ingestion_validation.factory import ValidatorFactory
from data_ingestion_validation.adapters.great_expectations_validator import GreatExpectationsValidator
from data_ingestion_validation.adapters.pandera_validator import PanderaValidator
from data_ingestion_validation.adapters.pydeequ_validator import PyDeequValidator
from data_ingestion_validation.adapters.evidently_validator import EvidentlyValidator


def test_factory_returns_correct_validator():
    cfg = {"ingestion_columns": ["f1"], "target_column": "y", "thresholds": {}}
    assert isinstance(ValidatorFactory.get("great_expectations", cfg), GreatExpectationsValidator)
    assert isinstance(ValidatorFactory.get("pandera", cfg), PanderaValidator)
    assert isinstance(ValidatorFactory.get("pydeequ", cfg), PyDeequValidator)
    assert isinstance(ValidatorFactory.get("evidently", cfg), EvidentlyValidator)


@mock.patch("data_ingestion_validation.validate.ValidatorFactory.get")
@mock.patch("data_ingestion_validation.validate.load_config")
def test_validate_data_ingestion_calls_factory(mock_load_config, mock_get):
    mock_load_config.return_value = {
        "ingestion_columns": ["f1"],
        "target_column": "y",
        "thresholds": {},
        "validator": {"strategy": "great_expectations"},
    }
    dummy_validator = mock.Mock()
    dummy_validator.validate.return_value = (True, {"ok": True})
    mock_get.return_value = dummy_validator

    df = pd.DataFrame({"f1": ["a"], "y": ["b"]})
    valid, report = dv.validate_data_ingestion(df, "y")
    assert valid is True
    assert report["ok"] is True
    mock_get.assert_called_once()
    dummy_validator.validate.assert_called_once_with(df, "y")


def test_factory_invalid_name():
    with pytest.raises(ValueError):
        ValidatorFactory.get("unknown", {})
