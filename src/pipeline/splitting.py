from app.split_dataset.robust_data_splitter import RobustDataSplitter

def data_split(df):
    splitter = RobustDataSplitter(
        df,
        split_method="time",
        time_column="date",
        target_column="target",
        train_size=0.6,
        test_size=0.2,
        backtest_size=0.2,
    )
    train_df, test_df, _ = splitter.split_data()
    return train_df, test_df
