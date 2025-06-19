import pandas as pd
import pytest
from unittest import mock

import validation_inference.validate as vi
from validation_inference.factory import ValidatorFactory
from validation_inference.adapters.great_expectations_validator import (
    GreatExpectationsValidator,
)
from validation_inference.adapters.evidently_validator import EvidentlyValidator
from validation_inference.adapters.nannyml_validator import NannyMLValidator
from validation_inference.adapters.pydeequ_validator import PyDeequValidator


def test_factory_returns_correct_validator():
    cfg = {"inference_column": "pred", "target_column": "y", "thresholds": {}}
    assert isinstance(
        ValidatorFactory.get("great_expectations", cfg), GreatExpectationsValidator
    )
    assert isinstance(ValidatorFactory.get("evidently", cfg), EvidentlyValidator)
    assert isinstance(ValidatorFactory.get("nannyml", cfg), NannyMLValidator)
    assert isinstance(ValidatorFactory.get("pydeequ", cfg), PyDeequValidator)


@mock.patch("validation_inference.validate.ValidatorFactory.get")
@mock.patch("validation_inference.validate.load_config")
def test_validate_inference_calls_factory(mock_load_config, mock_get):
    mock_load_config.return_value = {
        "inference_column": "pred",
        "target_column": "y",
        "thresholds": {},
        "validator": {"strategy": "great_expectations"},
    }
    dummy_validator = mock.Mock()
    dummy_validator.validate.return_value = (True, {"ok": True})
    mock_get.return_value = dummy_validator

    df = pd.DataFrame({"pred": [1.0], "y": [1.0]})
    valid, report = vi.validate_inference(df, "y")
    assert valid is True
    assert report["ok"] is True
    mock_get.assert_called_once()
    dummy_validator.validate.assert_called_once_with(df, "y")


def test_factory_invalid_name():
    with pytest.raises(ValueError):
        ValidatorFactory.get("unknown", {})
