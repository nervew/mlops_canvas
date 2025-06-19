from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from eda_univariado.core.plots import plot_hist


def detect_outliers_iqr(series: pd.Series, factor: float) -> Dict[str, Any]:
    """Return bounds and count of IQR-based outliers."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    mask = (series < lower) | (series > upper)
    return {
        "lower_bound": lower,
        "upper_bound": upper,
        "num_outliers": int(mask.sum()),
    }


class EDAUnivariado:
    """Perform simple univariate EDA based on a configuration dictionary."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        report: Dict[str, Any] = {"stats": {}, "outliers": {}}
        numeric_cols: List[str] = self.config["variables"].get("numericas", [])
        cat_cols: List[str] = self.config["variables"].get("categoricas", [])
        factor = float(self.config["outliers"].get("factor", 1.5))

        for col in numeric_cols:
            desc = data[col].describe().to_dict()
            report["stats"][col] = desc
            report["outliers"][col] = detect_outliers_iqr(data[col], factor)

            hist_cfg = self.config.get("graficos", {}).get("hist", {})
            if hist_cfg:
                out_dir = Path(self.config["paths"]["output_report"])
                out_dir.mkdir(parents=True, exist_ok=True)
                plot_hist(
                    data[col],
                    out_dir / f"{col}_hist.png",
                    bins=int(hist_cfg.get("bins", 50)),
                    kde=bool(hist_cfg.get("kde", True)),
                )

        for col in cat_cols:
            counts = data[col].value_counts().to_dict()
            report["stats"][col] = counts

        return report
