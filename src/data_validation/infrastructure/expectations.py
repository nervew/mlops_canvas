import pandas as pd
from great_expectations.dataset import PandasDataset


class IrisDataset(PandasDataset):
    pass

def validate_schema(df: pd.DataFrame) -> bool:
    ds = IrisDataset(df)
    ds.expect_column_values_to_not_be_null('target')
    return ds.validate().success
