from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from ..ports import IDataLoader  # type: ignore


class IrisLoader(IDataLoader):
    """Load iris dataset and split into train/val/test."""

    def load(
        self,
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        iris = load_iris(as_frame=True)
        X = iris.data
        y = iris.target
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        return (
            X_train.reset_index(drop=True),
            X_test.reset_index(drop=True),
            X_val.reset_index(drop=True),
            y_train.reset_index(drop=True),
            y_test.reset_index(drop=True),
            y_val.reset_index(drop=True),
        )
