import pandas as pd
from ..infrastructure.grid_search import search
from ..domain.result import SearchResult


def run(X: pd.DataFrame, y: pd.Series) -> SearchResult:
    grid = search(X, y)
    return SearchResult(grid=grid)
