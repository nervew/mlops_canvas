# fs_variance.py
from sklearn.feature_selection import VarianceThreshold
from fs_base import FeatureSelector

class VarianceThresholdSelector(FeatureSelector):
    def __init__(self, threshold: float = 0.0):
        """Elimina features con varianza menor o igual al umbral."""
        self.selector = VarianceThreshold(threshold=threshold)

    def fit(self, X, y=None):
        self.selector.fit(X)

    def transform(self, X):
        mask = self.selector.get_support()
        return X.loc[:, mask]