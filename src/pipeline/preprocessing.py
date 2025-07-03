import pandas as pd

def _base_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"]   = df["date"].dt.day
    return df.drop(columns=["date"])

def data_engineering(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train = _base_transform(train_df)
    test  = _base_transform(test_df)

    combo = pd.concat([train, test], keys=["train", "test"])
    combo = pd.get_dummies(combo, columns=["category"], drop_first=False)

    train = combo.xs("train")
    test  = combo.xs("test")

    constant_cols = train.columns[train.nunique() <= 1]
    train = train.drop(columns=constant_cols)
    test  = test.drop(columns=constant_cols, errors="ignore")
    test  = test.reindex(columns=train.columns, fill_value=0)
    return train, test
