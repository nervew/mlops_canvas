import pandas as pd
from ..infrastructure.pandas_univar import run_univariate_analysis, save_univariate_report
from ..domain.report import UnivariateReport

def run(
    df: pd.DataFrame, 
    reporte_path: str = None
) -> UnivariateReport:
    """
    Orquesta el EDA univariado: calcula estadísticas extendidas, guarda JSON,
    y SOLO imprime las primeras 5 filas de describe().T.
    """
    report = run_univariate_analysis(df)
    if reporte_path is not None:
        save_univariate_report(report, reporte_path)

    # Solo imprime las primeras 5 filas del describe transpuesto
    print(df.describe(include="all").T.head())

    return UnivariateReport(description=report)
