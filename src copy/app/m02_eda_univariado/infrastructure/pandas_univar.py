from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import sweetviz as sv


# ---------- estadísticas   ----------
def run_univariate_analysis(df: pd.DataFrame) -> dict:
    """Calcula estadísticas básicas + outliers + histograma por columna."""
    report: dict = {}

    for col in df.columns:
        stats: dict = {}
        s = df[col].dropna()

        stats["count"]       = int(len(s))
        stats["missing"]     = int(df[col].isna().sum())
        stats["missing_pct"] = float(df[col].isna().mean())

        if pd.api.types.is_numeric_dtype(s):
            stats |= {                       # merge = Python 3.9+
                "mean":     float(s.mean()),
                "std":      float(s.std()),
                "min":      float(s.min()),
                "q1":       float(s.quantile(0.25)),
                "median":   float(s.median()),
                "q3":       float(s.quantile(0.75)),
                "max":      float(s.max()),
            }
            iqr   = stats["q3"] - stats["q1"]
            lower = stats["q1"] - 1.5 * iqr
            upper = stats["q3"] + 1.5 * iqr
            outliers = s[(s < lower) | (s > upper)]

            counts, bins        = np.histogram(s, bins=10)
            stats["outliers_count"] = int(outliers.size)
            stats["outliers_pct"]   = float(outliers.size / len(s)) if len(s) else 0.0
            stats["hist_bins"]      = bins.tolist()
            stats["hist_counts"]    = counts.tolist()
        else:
            stats["top"]    = s.mode().iloc[0] if not s.mode().empty else None
            stats["unique"] = int(s.nunique())
            stats["freq"]   = int(s.value_counts().iloc[0]) if not s.empty else 0

        report[col] = stats
    return report


# ---------- Sweetviz ----------
def generate_sweetviz_report(
    df: pd.DataFrame,
    output_path: Union[str, Path]
) -> None:
    """
    Genera un reporte Sweetviz y lo guarda en `output_path`.  
    Monkey-patch para evitar el error de np.VisibleDeprecationWarning.
    """
    # —– Monkey-patch para VisibleDeprecationWarning ——
    if not hasattr(np, "VisibleDeprecationWarning"):
        np.VisibleDeprecationWarning = Warning

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = sv.analyze(df)
    report.show_html(str(output_path), open_browser=False)