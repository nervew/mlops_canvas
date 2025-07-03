from ..infrastructure.pandas_profiler import generate_report
from ..domain.report import UnivariateReport
import pandas as pd


def run(df: pd.DataFrame) -> UnivariateReport:
    """Ejecuta el análisis univariado."""
    desc = generate_report(df)
    return UnivariateReport(description=desc)
