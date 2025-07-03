import pandas as pd
from ..infrastructure.univariate import build_selector
from ..domain.selector import FeatureSelector


def run(X: pd.DataFrame, y: pd.Series) -> FeatureSelector:
    selector = build_selector(X, y)
    return FeatureSelector(selector=selector)
