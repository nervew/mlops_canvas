import pandas as pd
import numpy as np
import json
import os

def run_univariate_analysis(df: pd.DataFrame) -> dict:
    report = {}
    for col in df.columns:
        stats = {}
        s = df[col].dropna()
        stats["count"] = len(s)
        stats["missing"] = int(df[col].isna().sum())
        stats["missing_pct"] = float(df[col].isna().mean())
        if pd.api.types.is_numeric_dtype(s):
            stats["mean"] = float(s.mean())
            stats["std"] = float(s.std())
            stats["min"] = float(s.min())
            stats["q1"] = float(s.quantile(0.25))
            stats["median"] = float(s.median())
            stats["q3"] = float(s.quantile(0.75))
            stats["max"] = float(s.max())
            iqr = stats["q3"] - stats["q1"]
            lower = stats["q1"] - 1.5*iqr
            upper = stats["q3"] + 1.5*iqr
            outliers = s[(s < lower) | (s > upper)]
            stats["outliers_count"] = int(len(outliers))
            stats["outliers_pct"] = float(len(outliers) / len(s)) if len(s) > 0 else 0.0
            counts, bins = np.histogram(s, bins=10)
            stats["hist_bins"] = bins.tolist()
            stats["hist_counts"] = counts.tolist()
        else:
            stats["top"] = s.mode().iloc[0] if not s.mode().empty else None
            stats["unique"] = int(s.nunique())
            stats["freq"] = int(s.value_counts().iloc[0]) if not s.empty else 0
        report[col] = stats
    return report

def save_univariate_report(report: dict, path: str):
    # Asegura que el directorio exista ANTES de guardar
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
