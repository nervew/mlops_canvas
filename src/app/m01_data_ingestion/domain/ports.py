from abc import ABC, abstractmethod
import pandas as pd

class IDataSource(ABC):
    """Puerto de salida: fuente de datos cruda (Spark, SQL, etc.)."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def run_query(self, sql: str) -> pd.DataFrame: ...
