from dataclasses import dataclass
import pandas as pd

@dataclass
class Dataset:
    data: pd.DataFrame