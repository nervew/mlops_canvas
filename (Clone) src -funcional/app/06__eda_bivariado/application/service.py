import pandas as pd
from ..infrastructure.correlation import generate_report
from ..domain.report import BivariateReport


def run(df: pd.DataFrame) -> BivariateReport:
    corr = generate_report(df)
    return BivariateReport(correlation=corr)
