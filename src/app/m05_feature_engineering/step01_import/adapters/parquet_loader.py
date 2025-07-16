# /.../m05_feature_engineering/step01_import/adapters/parquet_loader.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from ..ports import IDataLoader


class ParquetPartitionLoader(IDataLoader):
    """Carga tres particiones Parquet y devuelve X_train, X_test, X_backtest y sus targets."""

    def __init__(
        self,
        base_path: Path | str | None = None,
        filename_train: str = "train_df.parquet",
        filename_test: str = "test_df.parquet",
        filename_backtest: str = "backtest_df.parquet",
    ) -> None:
        if base_path is None:
            # Desde src/app/... sube 4 niveles hasta 'src', luego data/raw/partitioned
            self.base_path = (
                Path(__file__).resolve().parents[4]
                / "data"
                / "raw"
                / "partitioned"
            )
        else:
            self.base_path = Path(base_path)

        self.train_file = self.base_path / filename_train
        self.test_file = self.base_path / filename_test
        self.backtest_file = self.base_path / filename_backtest

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
        # Leer particiones
        train_df = pd.read_parquet(self.train_file)
        test_df = pd.read_parquet(self.test_file)
        backtest_df = pd.read_parquet(self.backtest_file)

        # Separar X / y
        X_train, y_train = train_df.drop(columns=["target"]), train_df["target"]
        X_test, y_test = test_df.drop(columns=["target"]), test_df["target"]
        X_backtest, y_backtest = (
            backtest_df.drop(columns=["target"]),
            backtest_df["target"],
        )

        return X_train, X_test, X_backtest, y_train, y_test, y_backtest
