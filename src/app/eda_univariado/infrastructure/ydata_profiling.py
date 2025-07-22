import os
import pandas as pd
from ydata_profiling import ProfileReport

def generate_html_report(df: pd.DataFrame, output_path: str) -> str:
    """Genera el reporte HTML con YData Profiling (modo minimal para evitar bugs)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    profile = ProfileReport(
        df,
        title="Reporte EDA Univariado",
        explorative=True,
        minimal=True,          # 𝟙  ← Desactiva cálculos que provocan el bug
        correlations=None,     # Opcional: desactiva correlaciones
    )
    profile.to_file(output_path)
    return output_path
