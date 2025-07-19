# /Workspace/Users/jorgee.lopez@adres.gov.co/mlops_canvas/src/app/
#    m06__feature_selection/step01_import/adapters/parquet_loader.py

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pandas as pd

from ..ports.loader import IDataLoader


class ParquetPartitionLoader2(IDataLoader):
    """Carga X_train, X_test y X_backtest procesados más sus targets."""

    def __init__(
        self,
        base_path: Path | str | None = None,
        filename_train: str = "X_train_processed.parquet",
        filename_test: str = "X_test_processed.parquet",
        filename_backtest: str = "X_backtest_processed.parquet",
    ) -> None:
        # ──────────────────────────────────────────────────────────────
        # Ruta por defecto: …/src/data/processed/pipeline_engineering/
        # (subimos 4 niveles desde adapters → src)
        # ──────────────────────────────────────────────────────────────
        if base_path is None:
            self.base_path = (
                Path(__file__).resolve().parents[4]   # ← 0 adapters, 1 step01_import,
                                                     #    2 m06__feature_selection,
                                                     #    3 app, 4 src
                / "data"
                / "processed"
                / "pipeline_engineering"
            )
        else:
            self.base_path = Path(base_path)

        self.train_file    = self.base_path / filename_train
        self.test_file     = self.base_path / filename_test
        self.backtest_file = self.base_path / filename_backtest

    # ------------------------------------------------------------------
    def load(
        self,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
        pd.Series,
    ]:
        train_df = pd.read_parquet(self.train_file)
        test_df  = pd.read_parquet(self.test_file)
        back_df  = pd.read_parquet(self.backtest_file)

        X_train, y_train = train_df.drop(columns=["target"]), train_df["target"]
        X_test,  y_test  = test_df.drop(columns=["target"]),  test_df["target"]
        X_back,  y_back  = back_df.drop(columns=["target"]), back_df["target"]

        return X_train, X_test, X_back, y_train, y_test, y_back
