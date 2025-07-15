from ..infrastructure.feature_validator import FeatureValidator
from ..domain.report import DataValidationReport
import pandas as pd

def run(df: pd.DataFrame, profile_path="output/validation/data_profile.json",
        reporte_path="output/validation/validation_report.json",
        fit_profile: bool = False) -> DataValidationReport:
    
    validator = FeatureValidator()

    # Fit (aprender perfil) o cargar perfil guardado
    if fit_profile:
        validator.fit(df)
        validator.save_profile(profile_path)
    else:
        validator.load_profile(profile_path)

    # Validar los datos
    result = validator.validate(df, reporte_path=reporte_path)

    return DataValidationReport(valido=result["valido"], detalles=result["detalles"])
