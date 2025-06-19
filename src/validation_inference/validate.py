from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import yaml  # type: ignore

from validation_inference.factory import ValidatorFactory


def load_config() -> Dict[str, Any]:
    with open(Path("configuration/config.yml"), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def validate_inference(
    data_raw: pd.DataFrame, target: str
) -> Tuple[bool, Dict[str, Any]]:
    cfg = load_config()
    strategy = cfg.get("validator", {}).get("strategy", "great_expectations")
    adapter_cfg = {
        "inference_column": cfg["inference_column"],
        "target_column": cfg["target_column"],
        "thresholds": cfg.get("thresholds", {}),
    }
    validator = ValidatorFactory.get(strategy, adapter_cfg)
    return validator.validate(data_raw, target)
