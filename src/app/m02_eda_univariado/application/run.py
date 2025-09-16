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
    # .../src/app/m02_eda_univariado/application/run.py -> parents[4] = raíz del repo (mlops_canvas)
    return Path(__file__).resolve().parents[4]


def _default_report_path() -> Path:
    """…/output/reporte_eda/sweetviz_univariate_report.html"""
    return _project_root() / "output" / "reporte_eda" / DEFAULT_HTML_NAME


def _default_json_path() -> Path:
    """…/output/reporte_eda/univariate_summary.json"""
    return _project_root() / "output" / "reporte_eda" / DEFAULT_JSON_NAME


def run(
    df: pd.DataFrame,
    reporte_path: Optional[str] = None,
    json_path: Optional[str] = None,
    print_max_rows: int = 20,
    *,
    print_tables: bool = True,   # <- NUEVO: imprime tablas-resumen en consola
    log_artifacts: bool = True,  # <- NUEVO: imprime rutas de HTML/JSON
) -> UnivariateReport:
    """
    Orquesta el EDA univariado:

    1) Calcula estadísticas univariadas (manuales).
    2) Calcula métricas de calidad de datos (nulos, únicos, duplicados, tipos).
    3) Genera y guarda Sweetviz (HTML).
    4) Guarda TODO (univariado + calidad) en un único JSON.
    5) (Opcional) Imprime tablas-resumen y rutas de artefactos en terminal.
    """
    # 1) estadísticas “manuales”
    report_dict = run_univariate_analysis(df)

    # 2) calidad de datos
    dq = summarize_data_quality(df, max_duplicate_groups=0)

    # 3) Sweetviz
    destino_html = Path(reporte_path) if reporte_path is not None else _default_report_path()
    generate_sweetviz_report(df, destino_html)

    # 4) Guardar TODO en JSON
    destino_json = Path(json_path) if json_path is not None else _default_json_path()
    payload = {
        "univariate_stats": report_dict,
        "data_quality": dq,
    }
    write_json(payload, destino_json)

    # 5) Impresiones (controladas por flags)
    if print_tables:
        print("\n[EDA] Primeras 5 filas de df.describe(include='all').T:")
        desc_head_dict = dq.get("describe_head", {})
        if desc_head_dict:
            print(pd.DataFrame.from_dict(desc_head_dict).head())
        else:
            print("(sin describe_head disponible)")

        per_col = pd.DataFrame.from_dict(dq["columns"], orient="index")
        cols_show = [c for c in ["dtype", "null_pct", "unique_pct", "duplicate_pct", "null_count", "distinct_count"] if c in per_col.columns]
        print("\n[EDA] Calidad por columna (top por % nulos):")
        if len(per_col):
            # to_string -> impresión “console-friendly” de todo el ancho/filas pedidas
            print(per_col.sort_values("null_pct", ascending=False)[cols_show].head(print_max_rows).to_string())
        else:
            print("(sin columnas)")

        types_series = pd.Series({k: len(v) for k, v in dq["types"].items()})
        print("\n[EDA] Nº de columnas por tipo:")
        print(types_series[types_series > 0].sort_values(ascending=False).to_string())

    if log_artifacts:
        print(f"\n[EDA] Artefactos:")
        print(f"  - Sweetviz: {destino_html}")
        print(f"  - Resumen JSON: {destino_json}\n")

    return UnivariateReport(description=report_dict)
