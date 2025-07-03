import pandas as pd
from ..infrastructure.expectations import validate_schema
from ..domain.report import ValidationReport


def run(df: pd.DataFrame) -> ValidationReport:
    success = validate_schema(df)
    stats = {"rows": len(df)}
    return ValidationReport(success=success, statistics=stats)
