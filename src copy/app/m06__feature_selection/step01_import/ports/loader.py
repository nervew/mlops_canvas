# /Workspace/Users/jorgee.lopez@adres.gov.co/mlops_canvas/src/app/m06__feature_selection/step01_import/ports/loader.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd

class IDataLoader(ABC):
    """Interface para cargar y dividir datasets en train/test/backtest."""

    @abstractmethod
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
        """Devuelve X_train, X_test, X_backtest, y_train, y_test, y_backtest."""
        raise NotImplementedError
