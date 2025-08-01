from pathlib import Path
from ..infrastructure.feature_validator import FeatureValidator
from ..domain.report import DataValidationReport
import pandas as pd

def run(
    df: pd.DataFrame,
    profile_path="output/validation/data_profile.json",
    reporte_path="output/validation/validation_report.json",
    fit_profile: bool = False
) -> DataValidationReport:
    # Sube hasta la raíz del proyecto 'mlops_canvas'
    mlops_canvas_dir = Path(__file__).resolve()
    while mlops_canvas_dir.name != "mlops_canvas" and mlops_canvas_dir.parent != mlops_canvas_dir:
        mlops_canvas_dir = mlops_canvas_dir.parent

    # Si no encontró la carpeta, lanza error
    if mlops_canvas_dir.name != "mlops_canvas":
        raise RuntimeError("No se encontró la carpeta raíz 'mlops_canvas' en la ruta.")

    # Ruta final para los archivos
    profile_path = mlops_canvas_dir / profile_path
    reporte_path = mlops_canvas_dir / reporte_path

    # Crea las carpetas si no existen
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    reporte_path.parent.mkdir(parents=True, exist_ok=True)

    validator = FeatureValidator()

    # Fit (aprender perfil) o cargar perfil guardado
    if fit_profile:
        validator.fit(df)
        validator.save_profile(str(profile_path))
    else:
        validator.load_profile(str(profile_path))

    # Validar los datos
    result = validator.validate(df, reporte_path=str(reporte_path))

    return DataValidationReport(valido=result["valido"], detalles=result["detalles"])
