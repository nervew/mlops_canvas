from dataclasses import dataclass
import pandas as pd

@dataclass
class BivariateReport:
    correlation: pd.DataFrame
