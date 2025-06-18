from abc import ABC, abstractmethod
from typing import Any
import pandas as pd

class DataSource(ABC):
    """Abstract interface for data sources returning pandas DataFrames."""

    @abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        """Retrieve data and return as DataFrame."""
        raise NotImplementedError
