from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml  # type: ignore

from validation_inference.validate import validate_inference


if __name__ == "__main__":
    with open(Path("configuration/config.yml"), "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    df = pd.DataFrame(
        {
            cfg["inference_column"]: [0.1, 0.2, 0.3],
            cfg["target_column"]: [0.0, 0.1, 0.2],
        }
    )

    valid, report = validate_inference(df, cfg["target_column"])
    print("Validation result:", valid)
    print(report)
