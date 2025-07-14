from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd


class IDataLoader(ABC):
    """Interface for loading and splitting datasets."""

    @abstractmethod
    def load(
        self,
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """Return X_train, X_test, X_val, y_train, y_test, y_val."""
        raise NotImplementedError
