import pandas as pd
from ..infrastructure.pca_reducer import build_reducer
from ..domain.reducer import FeatureReducer


def run(X: pd.DataFrame, n_components: int = 2) -> FeatureReducer:
    reducer = build_reducer(n_components)
    reducer.fit(X)
    return FeatureReducer(reducer=reducer)
