# pipeline.py
from factory import SelectorFactory
import pandas as pd

class FeatureSelectionPipeline:
    """Pipeline de selección de features con pre y post-procesamiento extensible."""
    def __init__(self, method: str, params: dict):
        self.method = method
        self.params = params

    def pre_process(self, X: pd.DataFrame, y=None):
        # Ejemplo: aquí podrías imputar, codificar categóricas o generar lags
        return X, y

    def post_process(self, X_sel: pd.DataFrame):
        # Ejemplo: persistir lista de features seleccionadas, logging, etc.
        return X_sel

    def run(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_proc, y_proc = self.pre_process(X, y)
        selector = SelectorFactory.create(self.method, **self.params)
        X_sel = selector.fit_transform(X_proc, y_proc)
        return self.post_process(X_sel)