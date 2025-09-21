from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd

from ..domain.report import UnivariateReport
from ..infrastructure.pandas_univar import (
    run_univariate_analysis,
    generate_sweetviz_report,
    summarize_data_quality,
    write_json,
)

# ---------- paths por defecto ----------
DEFAULT_HTML_NAME = "sweetviz_univariate_report.html"
DEFAULT_JSON_NAME = "univariate_summary.json"


def _project_root() -> Path:
    # .../src/app/m02_eda_univariado/application/run.py -> parents[4] = raíz del repo
    return Path(__file__).resolve().parents[4]


def _default_report_path() -> Path:
    return _project_root() / "output" / "reporte_eda" / DEFAULT_HTML_NAME


def _default_json_path() -> Path:
    return _project_root() / "output" / "reporte_eda" / DEFAULT_JSON_NAME


def _print_formal_summary(dq: dict, univar: dict) -> None:
    """Imprime una tabla formal por columna (dtype, %missing, #distintos, %duplicados, outliers)."""
    import pandas as pd

    dq_cols = pd.DataFrame.from_dict(dq["columns"], orient="index")
    uv = pd.DataFrame.from_dict(univar, orient="index")

    cols_keep = ["dtype", "null_pct", "distinct_count", "duplicate_pct"]
    table = dq_cols.reindex(columns=[c for c in cols_keep if c in dq_cols.columns])

    for c in ["outliers_count", "outliers_pct"]:
        table[c] = uv[c] if c in uv.columns else None

    def _f(x):
        try:
            return float(x)
        except Exception:
            return -1.0

    table = (
        table.assign(_null=table["null_pct"].map(_f),
                     _out=table["outliers_pct"].map(_f))
             .sort_values(by=["_null", "_out"], ascending=False)
             .drop(columns=["_null", "_out"])
    )

    for c in ["null_pct", "duplicate_pct", "outliers_pct"]:
        if c in table.columns:
            table[c] = table[c].apply(lambda v: None if v is None else round(float(v), 2))

    title = "SALIDA FORMAL • Resumen por columna"
    line  = "═" * len(title)
    print(f"\n{line}\n{title}\n{line}")
    print(
        table.rename(columns={
            "dtype": "Tipo",
            "null_pct": "% Missing",
            "distinct_count": "# Distintos",
            "duplicate_pct": "% Duplicados",
            "outliers_count": "Outliers (n)",
            "outliers_pct": "Outliers (%)",
        }).to_string()
    )


def run(
    df: pd.DataFrame,
    reporte_path: Optional[str] = None,
    json_path: Optional[str] = None,
    print_max_rows: int = 20,
    *,
    print_tables: bool = False,            # tablas rápidas (describe, tipos)
    print_formal_summary: bool = True,     # tabla formal (por columna)
    log_artifacts: bool = True,
) -> UnivariateReport:
    """
    Orquesta el EDA univariado:
      1) Estadísticas univariadas.
      2) Calidad de datos.
      3) Sweetviz (HTML).
      4) JSON unificado (univariado + calidad).
      5) Impresiones: (a) rápidas, (b) resumen formal por columna.
    """
    # 1) univariado
    report_dict = run_univariate_analysis(df)

    # 2) calidad de datos
    dq = summarize_data_quality(df, max_duplicate_groups=0)

    # 3) Sweetviz
    destino_html = Path(reporte_path) if reporte_path is not None else _default_report_path()
    generate_sweetviz_report(df, destino_html)

    # 4) JSON unificado
    destino_json = Path(json_path) if json_path is not None else _default_json_path()
    payload = {"univariate_stats": report_dict, "data_quality": dq}
    write_json(payload, destino_json)

    # 5-a) impresiones rápidas (opcional)
    if print_tables:
        print("\n[EDA] Primeras 5 filas de df.describe(include='all').T:")
        desc_head_dict = dq.get("describe_head", {})
        if desc_head_dict:
            print(pd.DataFrame.from_dict(desc_head_dict).head())
        else:
            print("(sin describe_head disponible)")

        per_col = pd.DataFrame.from_dict(dq["columns"], orient="index")
        cols_show = [c for c in ["dtype", "null_pct", "unique_pct", "duplicate_pct", "null_count", "distinct_count"]
                     if c in per_col.columns]
        print("\n[EDA] Calidad por columna (top por % nulos):")
        if len(per_col):
            print(per_col.sort_values("null_pct", ascending=False)[cols_show].head(print_max_rows).to_string())
        else:
            print("(sin columnas)")

        types_series = pd.Series({k: len(v) for k, v in dq["types"].items()})
        print("\n[EDA] Nº de columnas por tipo:")
        print(types_series[types_series > 0].sort_values(ascending=False).to_string())

    # 5-b) resumen formal por columna (principal)
    if print_formal_summary:
        _print_formal_summary(dq, report_dict)

    if log_artifacts:
        print(f"\n[EDA] Artefactos:")
        print(f"  - Sweetviz: {destino_html}")
        print(f"  - Resumen JSON: {destino_json}\n")

    return UnivariateReport(description=report_dict)
