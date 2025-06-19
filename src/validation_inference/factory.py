from __future__ import annotations

from typing import Any, Dict, Type

from validation_inference.ports.validator import IInferenceValidator
from validation_inference.adapters.great_expectations_validator import (
    GreatExpectationsValidator,
)
from validation_inference.adapters.evidently_validator import EvidentlyValidator
from validation_inference.adapters.pydeequ_validator import PyDeequValidator
from validation_inference.adapters.nannyml_validator import NannyMLValidator


class ValidatorFactory:
    """Factory to obtain inference validators."""

    _mapping: Dict[str, Type[IInferenceValidator]] = {
        "great_expectations": GreatExpectationsValidator,
        "evidently": EvidentlyValidator,
        "pydeequ": PyDeequValidator,
        "nannyml": NannyMLValidator,
    }

    @staticmethod
    def get(name: str, config: Dict[str, Any]) -> IInferenceValidator:
        if name not in ValidatorFactory._mapping:
            raise ValueError(f"Unsupported validator: {name}")
        adapter_cls = ValidatorFactory._mapping[name]
        return adapter_cls(**config)
