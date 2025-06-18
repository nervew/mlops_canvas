from abc import ABC, abstractmethod
import pandas as pd


class DataRepository(ABC):
    """Port for loading data for model training."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load raw data as a pandas DataFrame."""
        raise NotImplementedError
