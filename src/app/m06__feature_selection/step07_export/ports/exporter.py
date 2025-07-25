# src/app/m06__feature_selection/step07_export/ports/exporter.py

from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd

class IDataExporter(ABC):
    """Interfaz para exportar DataFrames de train/test/backtest."""

    @abstractmethod
    def export(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        X_backtest: pd.DataFrame,
    ) -> None:
        """Guarda los tres DataFrames en disco."""
        ...
