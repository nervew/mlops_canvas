from abc import ABC, abstractmethod
import pandas as pd

class IDataIngestionPort(ABC):
    """Interface for data ingestion adapters."""

    @abstractmethod
    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute the SQL query and return a DataFrame."""
        raise NotImplementedError
