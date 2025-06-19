from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml  # type: ignore

from data_ingestion_validation.validate import validate_data_ingestion


if __name__ == "__main__":
    with open(Path("configuration/config.yml"), "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    data = pd.DataFrame({col: ["a", "b", "c"] for col in cfg["ingestion_columns"]})
    data[cfg["target_column"]] = ["x", "y", "z"]

    valid, report = validate_data_ingestion(data, cfg["target_column"])
    print("Validation result:", valid)
    print(report)
