from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from ..ports import IMetricsExporter


class JsonMetricsExporter(IMetricsExporter):
    def export(self, df: pd.DataFrame, path: str) -> None:
        stats = {
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "means": df.mean(numeric_only=True).to_dict(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=4)