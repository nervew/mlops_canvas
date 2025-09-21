from pathlib import Path
from typing import Union, Dict, Any, List

import json
import numpy as np
import pandas as pd
import sweetviz as sv
from pandas.api.types import is_numeric_dtype, is_bool_dtype
from datetime import datetime, date, time


# ---------- Estadísticas univariadas (robusto a tipos) ----------
def run_univariate_analysis(df: pd.DataFrame) -> dict:
    """
    Calcula estadísticas básicas por columna, incluyendo outliers e histograma
    para columnas numéricas. Maneja booleanos aparte para evitar errores
    de cuantiles sobre bool.
    """
    report: dict = {}

    for col in df.columns:
        stats: dict = {}
        s = df[col].dropna()

        stats["count"]       = int(len(s))
        stats["missing"]     = int(df[col].isna().sum())
        stats["missing_pct"] = float((stats["missing"] / len(df)) * 100) if len(df) else 0.0

        # --- Booleanos: tratarlos como categóricos binarios (evita cuantiles) ---
        if is_bool_dtype(df[col]):
            true_count  = int((s == True).sum())   # noqa: E712
            false_count = int((s == False).sum())  # noqa: E712
            total       = true_count + false_count
            stats.update({
                "dtype": "bool",
                "true_count": true_count,
                "false_count": false_count,
                "true_pct": float(round((true_count / total) * 100, 2)) if total else 0.0,
                "unique": int(s.nunique(dropna=True)),
                "top": (bool(s.mode(dropna=True).iloc[0]) if not s.mode(dropna=True).empty else None),
                "freq": int(s.value_counts(dropna=True).iloc[0]) if not s.empty else 0,
            })

        # --- Numéricos reales (excluye bool) ---
        elif is_numeric_dtype(df[col]):
            stats.update({
                "mean":     float(s.mean()) if len(s) else None,
                "std":      float(s.std()) if len(s) else None,
                "min":      float(s.min()) if len(s) else None,
                "q1":       float(s.quantile(0.25)) if len(s) else None,
                "median":   float(s.median()) if len(s) else None,
                "q3":       float(s.quantile(0.75)) if len(s) else None,
                "max":      float(s.max()) if len(s) else None,
            })
            if len(s) and stats["q1"] is not None and stats["q3"] is not None:
                iqr   = stats["q3"] - stats["q1"]
                lower = stats["q1"] - 1.5 * iqr
                upper = stats["q3"] + 1.5 * iqr
                outliers = s[(s < lower) | (s > upper)]
                stats["outliers_count"] = int(outliers.size)
                stats["outliers_pct"]   = float((outliers.size / len(s)) * 100)
            else:
                stats["outliers_count"] = 0
                stats["outliers_pct"]   = 0.0

            counts, bins = np.histogram(s, bins=10) if len(s) else (np.array([]), np.array([]))
            stats["hist_bins"]   = [float(x) for x in bins.tolist()]
            stats["hist_counts"] = [int(x) for x in counts.tolist()]

        else:
            # Texto, fecha, categórico u “otro”
            mode_vals = s.mode(dropna=True)
            top_val = None if mode_vals.empty else mode_vals.iloc[0]
            top_val = _to_builtin(top_val)
            stats["top"]    = top_val
            stats["unique"] = int(s.nunique(dropna=True))
            freq = s.value_counts(dropna=True)
            stats["freq"]   = int(freq.iloc[0]) if not freq.empty else 0

        report[col] = stats

    return report


# ---------- Resumen de calidad de datos (estilo data quality) ----------
def summarize_data_quality(df: pd.DataFrame, max_duplicate_groups: int = 0) -> Dict[str, Any]:
    """
    Métricas globales de calidad: nulos, distintos/duplicados, tipificación,
    y un head de describe() compatible con múltiples versiones de pandas.
    """
    n_rows, n_cols = df.shape
    types_map: Dict[str, List[str]] = {
        "numeric": [], "string": [], "datetime": [], "boolean": [], "categorical": [], "other": []
    }
    per_col: Dict[str, Any] = {}

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            col_type = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(s):
            col_type = "datetime"
        elif pd.api.types.is_bool_dtype(s):
            col_type = "boolean"
        elif pd.api.types.is_categorical_dtype(s):
            col_type = "categorical"
        elif pd.api.types.is_string_dtype(s) or s.dtype == "object":
            col_type = "string"
        else:
            col_type = "other"

        types_map[col_type].append(col)

        null_count = int(s.isna().sum())
        null_pct   = float((null_count / n_rows) * 100) if n_rows else 0.0
        distinct_count = int(s.nunique(dropna=False))
        unique_pct     = float((distinct_count / n_rows) * 100) if n_rows else 0.0
        duplicate_count = int(n_rows - distinct_count)
        duplicate_pct   = float((duplicate_count / n_rows) * 100) if n_rows else 0.0

        entry = {
            "dtype": str(s.dtype),
            "null_count": null_count,
            "null_pct": _round2(null_pct),
            "distinct_count": distinct_count,
            "unique_pct": _round2(unique_pct),
            "duplicate_count": duplicate_count,
            "duplicate_pct": _round2(duplicate_pct),
        }

        if col_type in {"string", "categorical"}:
            vc = s.value_counts(dropna=True)
            most_frequent_value = (_to_builtin(vc.index[0]) if len(vc) else None)
            most_frequent_count = (int(vc.iloc[0]) if len(vc) else 0)
            entry["num_categories_non_null"] = int(s.nunique(dropna=True))
            entry["most_frequent_category"] = most_frequent_value
            entry["most_frequent_category_count"] = most_frequent_count

        per_col[col] = entry

    # Duplicados a nivel de fila (todas las columnas)
    num_duplicate_rows = int(df.duplicated(keep=False).sum())
    duplicate_groups_preview = []
    num_duplicate_groups = 0
    if max_duplicate_groups > 0 and n_rows > 0:
        vc_rows = df.value_counts(dropna=False)
        dup_groups = vc_rows[vc_rows > 1]
        num_duplicate_groups = int((dup_groups > 1).sum())
        for (row_values, cnt) in dup_groups.head(max_duplicate_groups).items():
            if not isinstance(row_values, tuple):
                row_values = (row_values,)
            duplicate_groups_preview.append({
                "row": [_to_builtin(v) for v in row_values],
                "count": int(cnt)
            })

    # --- describe() compatible con pandas 1.1–1.5 y 2.x ---
    try:
        desc = df.describe(include="all", datetime_is_numeric=True)
    except TypeError:
        desc = df.describe(include="all")

    describe_head = (
        desc.T.head().reset_index().to_dict(orient="list")
        if isinstance(desc, pd.DataFrame) and not desc.empty
        else {}
    )

    return {
        "dataset": {"total_rows": int(n_rows), "total_columns": int(n_cols)},
        "types": {k: v for k, v in types_map.items()},
        "columns": per_col,
        "duplicates": {
            "num_duplicate_rows": int(num_duplicate_rows),
            "num_duplicate_groups": int(num_duplicate_groups),
            "groups_sample": duplicate_groups_preview,
        },
        "describe_head": describe_head,
    }


# ---------- utilidades de serialización ----------
def write_json(payload: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Guarda el dict en JSON (UTF-8) creando carpetas si no existen."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = json_safe(payload)  # convierte todo a tipos serializables
    with p.open("w", encoding="utf-8") as f:
        json.dump(safe_payload, f, ensure_ascii=False, indent=2)


def json_safe(o: Any) -> Any:
    """
    Conversión recursiva a tipos JSON-serializables.
    """
    if o is None or isinstance(o, (str, int, float, bool)):
        return o

    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()

    if isinstance(o, pd.Timestamp):
        if pd.isna(o):
            return None
        return o.isoformat()

    if isinstance(o, (datetime, date, time)):
        return o.isoformat()

    if isinstance(o, np.datetime64):
        try:
            return np.datetime_as_string(o, unit="us")
        except Exception:
            return str(o)

    if isinstance(o, pd.Timedelta):
        return o.isoformat() if hasattr(o, "isoformat") else str(o)
    if isinstance(o, pd.Period):
        try:
            return o.to_timestamp().isoformat()
        except Exception:
            return str(o)
    if isinstance(o, pd.Interval):
        return str(o)

    if isinstance(o, np.ndarray):
        return [json_safe(x) for x in o.tolist()]

    if isinstance(o, (list, tuple, set)):
        return [json_safe(x) for x in o]
    if isinstance(o, dict):
        return {str(json_safe(k)): json_safe(v) for k, v in o.items()}

    if isinstance(o, np.generic):
        return o.item()

    return str(o)


def _to_builtin(x: Any) -> Any:
    return json_safe(x)


def _round2(x: float) -> float:
    try:
        return float(round(x, 2))
    except Exception:
        return x


# ---------- Sweetviz ----------
def generate_sweetviz_report(df: pd.DataFrame, output_path: Union[str, Path]) -> None:
    """
    Genera un reporte Sweetviz y lo guarda en `output_path`.
    Monkey-patch para evitar VisibleDeprecationWarning en ciertos entornos.
    """
    if not hasattr(np, "VisibleDeprecationWarning"):
        np.VisibleDeprecationWarning = Warning

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = sv.analyze(df)
    report.show_html(str(output_path), open_browser=False)
