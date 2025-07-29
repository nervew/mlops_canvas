import pandas as pd
from ..ports.loader import IDataLoader

class ParquetPartitionLoader(IDataLoader):
    def __init__(self, base_path: str) -> None:
        self.base_path = base_path

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = pd.read_parquet(f"{self.base_path}/train_df.parquet")
        test  = pd.read_parquet(f"{self.base_path}/test_df.parquet")
        back  = pd.read_parquet(f"{self.base_path}/backtest_df.parquet")
        return train, test, back
