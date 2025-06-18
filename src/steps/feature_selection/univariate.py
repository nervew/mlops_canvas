from __future__ import annotations

from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply univariate feature selection."""
    selector = SelectKBest(score_func=f_classif, k=min(8, X_train.shape[1]))
    selector.fit(X_train, y_train)
    return selector.transform(X_train), selector.transform(X_test)
