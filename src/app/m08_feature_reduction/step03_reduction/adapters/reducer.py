from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from ..ports.reducer import IReducer

class Reducer(IReducer):

    def __init__(
        self,
        method: str | None = 'pca',
        n_components: int = 3,
        random_state: int = 42
    ):
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.reducer: PCA | None = None
        self.fitted_ = False

        if self.method == 'pca':
            self.reducer = PCA(
                n_components=self.n_components,
                random_state=self.random_state
            )

    def fit(self, X_train: pd.DataFrame) -> None:
        if self.reducer:
            self.reducer.fit(X_train.values if hasattr(X_train, "values") else X_train)
        self.fitted_ = True

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Reducer: primero fit().")
        if self.reducer:
            return self.reducer.transform(X.values if hasattr(X, "values") else X)
        return X.values if hasattr(X, "values") else X

    def explained_variance_ratio(self) -> np.ndarray | None:
        return self.reducer.explained_variance_ratio_ if self.reducer else None
