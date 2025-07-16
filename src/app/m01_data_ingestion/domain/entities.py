from dataclasses import dataclass
import pandas as pd

@dataclass
class DataSet:
    data: pd.DataFrame
