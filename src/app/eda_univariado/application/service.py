from ..infrastructure.ydata_profiling import generate_html_report
from ..domain.report import UnivariateReport
import pandas as pd

def run(
    df: pd.DataFrame,
    output_path: str = "output/reports/eda_univariado.html",
) -> UnivariateReport:
    """Orquesta el análisis univariado robusto."""

    df = df.copy()

    # Convierte automáticamente a 'category' las columnas con ≤10 valores únicos
    for col in df.columns:
        if df[col].nunique() <= 10:
            df[col] = df[col].astype("category")

    desc = df.describe(include="all")
    html_path = generate_html_report(df, output_path)

    return UnivariateReport(description=desc, html_report_path=html_path)
