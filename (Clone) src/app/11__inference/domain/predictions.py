from dataclasses import dataclass
import pandas as pd

@dataclass
class Predictions:
    values: pd.Series
