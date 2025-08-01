from pathlib import Path
from typing import Optional

import pandas as pd

from ..domain.report import UnivariateReport
from ..infrastructure.pandas_univar import (
    run_univariate_analysis,
    generate_sweetviz_report,
)

# ---------- path por defecto ----------
DEFAULT_HTML_NAME = "sweetviz_univariate_report.html"


def _default_report_path() -> Path:
    """
    Devuelve …/output/reporte_eda/sweetviz_univariate_report.html,
    donde “…” es la raíz del proyecto (mlops_canvas).
    """
    # .../src/app/m02_eda_univariado/application/run.py  -> padres[4] = carpeta raíz del repo
    root_dir = Path(__file__).resolve().parents[4]
    return root_dir / "output" / "reporte_eda" / DEFAULT_HTML_NAME


# ---------- API pública ----------
def run(df: pd.DataFrame, reporte_path: Optional[str] = None) -> UnivariateReport:
    """
    Orquesta el EDA univariado:

    1. Calcula estadísticas extendidas (objeto en memoria).
    2. Genera y guarda un reporte Sweetviz en ``output/reporte_eda/`` (o ruta dada).
    3. Imprime las primeras 5 filas de ``df.describe().T``.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset a analizar.
    reporte_path : str | Path | None
        Ruta destino (HTML). Si es ``None`` se usa la ruta por defecto.

    Returns
    -------
    UnivariateReport
        Objeto con el diccionario de estadísticas.
    """
    # 1) estadísticas “manuales”
    report_dict = run_univariate_analysis(df)

    # 2) Sweetviz
    destino = Path(reporte_path) if reporte_path is not None else _default_report_path()
    generate_sweetviz_report(df, destino)

    # 3) Describe
    print("\n Primeras 5 filas de df.describe(include='all').T:")
    print(df.describe(include="all").T.head())

    return UnivariateReport(description=report_dict)
