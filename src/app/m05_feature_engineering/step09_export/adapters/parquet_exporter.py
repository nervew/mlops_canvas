from pathlib import Path
import pandas as pd


class ParquetExportAdapter:
    """
    Guarda DataFrames procesados (train, test, backtest) en formato Parquet
    dentro de src/data/processed usando rutas relativas al proyecto.
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        if output_dir is None:
            project_root = Path(__file__).resolve().parents[5]
            output_dir = project_root / "data" / "processed" / "pipeline_engineering"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _ensure_str_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = out.columns.map(str)
        return out

    def export(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        backtest_df: pd.DataFrame,
    ) -> None:
        train_df = self._ensure_str_columns(train_df)
        test_df = self._ensure_str_columns(test_df)
        backtest_df = self._ensure_str_columns(backtest_df)

        train_df.to_parquet(self.output_dir / "X_train_processed.parquet")
        test_df.to_parquet(self.output_dir / "X_test_processed.parquet")
        backtest_df.to_parquet(self.output_dir / "X_backtest_processed.parquet")
