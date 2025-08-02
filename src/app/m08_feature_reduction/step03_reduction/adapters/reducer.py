from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

from ..ports.reducer import IReducer

class Reducer(IReducer):
    """
    Adaptador genérico:
      • Si method=None  → pasa-atrás (no cambia dimensión).
      • Si method='pca' → usa sklearn.PCA.
    """

    def __init__(
        self,
        method: str | None = "pca",
        n_components: int | None = 3,
        random_state: int = 42,
    ):
        self.method = method
        if method == "pca":
            self.reducer = PCA(
                n_components=n_components,
                random_state=random_state,
            )
        else:  # None u otro → passthrough
            self.reducer = None

    # ---------------- scikit-learn style ----------------
    def fit(self, X: pd.DataFrame | np.ndarray) -> "Reducer":
        if self.reducer is not None:
            self.reducer.fit(X)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.reducer is not None:
            return self.reducer.transform(X)
        # passthrough → devolver como np.ndarray
        return np.asarray(X)

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.reducer is not None:
            return self.reducer.fit_transform(X)
        return np.asarray(X)
