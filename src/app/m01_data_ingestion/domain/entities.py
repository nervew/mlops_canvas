from dataclasses import dataclass
import pandas as pd

@dataclass
class DataSet:
    """Entidad sencilla que encapsula el DataFrame crudo."""
    data: pd.DataFrame
