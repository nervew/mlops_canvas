import pandas as pd
import polars as pl
from sklearn.datasets import load_iris

from feature_selection.feature_selection_task import run_feature_selection


def test_pipeline_runs():
    X, y = load_iris(return_X_y=True, as_frame=True)
    df = pl.from_pandas(pd.concat([X, y.rename("target")], axis=1))
    res = run_feature_selection(df, "target", technique="all")
    assert len(res["selected_columns"]) > 0
