from dataclasses import dataclass
import pandas as pd

@dataclass
class DataSplitResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    backtest_df: pd.DataFrame
