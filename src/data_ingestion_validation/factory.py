from __future__ import annotations

from typing import Any, Dict, Type

from data_ingestion_validation.ports.validator import IIngestionValidator
from data_ingestion_validation.adapters.great_expectations_validator import GreatExpectationsValidator
from data_ingestion_validation.adapters.pandera_validator import PanderaValidator
from data_ingestion_validation.adapters.pydeequ_validator import PyDeequValidator
from data_ingestion_validation.adapters.evidently_validator import EvidentlyValidator


class ValidatorFactory:
    """Factory to obtain ingestion validators."""

    _mapping: Dict[str, Type[IIngestionValidator]] = {
        "great_expectations": GreatExpectationsValidator,
        "pandera": PanderaValidator,
        "pydeequ": PyDeequValidator,
        "evidently": EvidentlyValidator,
    }

    @staticmethod
    def get(name: str, config: Dict[str, Any]) -> IIngestionValidator:
        if name not in ValidatorFactory._mapping:
            raise ValueError(f"Unsupported validator: {name}")
        adapter_cls = ValidatorFactory._mapping[name]
        return adapter_cls(**config)
