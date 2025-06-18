from __future__ import annotations

import pandas as pd
from typing import Tuple

from app.split_dataset.robust_data_splitter import RobustDataSplitter


def split_data(
    df: pd.DataFrame,
    target_column: str = "target",
    split_method: str = "time",
    time_column: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset using ``RobustDataSplitter``."""
    splitter = RobustDataSplitter(
        df,
        split_method=split_method,
        time_column=time_column,
        target_column=target_column,
        train_size=0.6,
        test_size=0.4,
        backtest_size=0.0,
    )
    train_df, test_df, _ = splitter.split_data()
    return train_df, test_df
