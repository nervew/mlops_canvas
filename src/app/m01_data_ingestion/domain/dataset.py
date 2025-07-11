from dataclasses import dataclass
import pandas as pd

@dataclass
class DataSet:
    """Entidad sencilla que encapsula el dataframe crudo."""
    data: pd.DataFrame
