import pandas as pd
from ml_project.domain.data_processing import clean_data


def test_clean_data_imputes_missing():
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", None, "y"]})
    cleaned = clean_data(df)
    assert cleaned.isna().sum().sum() == 0
