from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class IDataLoader(ABC):
    """Abstract interface for loading data sources."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return raw data as a DataFrame."""
        raise NotImplementedError
