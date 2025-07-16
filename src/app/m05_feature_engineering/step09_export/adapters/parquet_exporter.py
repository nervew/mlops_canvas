from pathlib import Path
import pandas as pd


class ParquetExportAdapter:
    """
    Guarda DataFrames procesados (train, test, backtest) en formato Parquet
    dentro de src/data/processed usando rutas relativas al proyecto.
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        if output_dir is None:
            # …/src/app/m05_feature_engineering/step09_export/adapters → parents[4] = src
            project_root = Path(__file__).resolve().parents[4]
            output_dir = project_root / "data" / "processed"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        backtest_df: pd.DataFrame,
    ) -> None:
        train_df.to_parquet(self.output_dir / "X_train_processed.parquet")
        test_df.to_parquet(self.output_dir / "X_test_processed.parquet")
        backtest_df.to_parquet(self.output_dir / "X_backtest_processed.parquet")
