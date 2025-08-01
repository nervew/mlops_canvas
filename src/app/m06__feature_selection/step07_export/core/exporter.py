from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..ports.exporter import IDataExporter

class DataFrameExporter(IDataExporter):
    """
    Implementa IDataExporter para serializar a Parquet:
      - Crea la carpeta si no existe.
      - Usa siempre ruta relativa desde 'src'.
    """
    def __init__(self, base_path: Path | str | None = None) -> None:
        if base_path is None:
            # Sube hasta la raíz del proyecto (mlops_canvas)
            self.export_dir = (
                Path(__file__).resolve().parents[5]  # antes era [4]
                / "data"
                / "processed"
                / "pipeline_selection"
            )
        else:
            self.export_dir = Path(base_path)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        X_backtest: pd.DataFrame,
    ) -> None:
        # Serializamos cada partición con sufijo descriptivo
        X_train.to_parquet(self.export_dir / "X_train_selected.parquet")
        X_test.to_parquet(self.export_dir / "X_test_selected.parquet")
        X_backtest.to_parquet(self.export_dir / "X_backtest_selected.parquet")


def export_partitions(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_backtest: pd.DataFrame,
    base_path: Path | str | None = None,
) -> None:
    """
    Función de conveniencia:
    - Crea un DataFrameExporter.
    - Llama a .export() para guardar las 3 particiones.
    """
    exporter = DataFrameExporter(base_path=base_path)
    exporter.export(X_train, X_test, X_backtest)
