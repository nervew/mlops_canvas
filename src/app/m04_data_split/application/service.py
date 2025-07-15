from ..infrastructure.robust_data_splitter import RobustDataSplitter
from ..domain.result import DataSplitResult
import pandas as pd


def run(
    df: pd.DataFrame,
    split_method: str = "random",
    target_column: str = "target",
    stratify_columns=None,
    train_size=0.6,
    test_size=0.2,
    backtest_size=0.2,
) -> DataSplitResult:
    splitter = RobustDataSplitter(
        df,
        split_method=split_method,
        target_column=target_column,
        stratify_columns=stratify_columns or [],
        train_size=train_size,
        test_size=test_size,
        backtest_size=backtest_size,
    )
    train_df, test_df, backtest_df = splitter.split_data()
    return DataSplitResult(train_df, test_df, backtest_df)
