from abc import ABC, abstractmethod
import pandas as pd

class IDataLoader(ABC):
    @abstractmethod
    def load(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pass
