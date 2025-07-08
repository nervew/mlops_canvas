import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ..infrastructure.simple_search import search
from ..domain.params import HyperParamResult


def run(X: pd.DataFrame, y: pd.Series) -> HyperParamResult:
    model = RandomForestClassifier(random_state=42)
    grid = search(model, X, y)
    return HyperParamResult(grid=grid)
