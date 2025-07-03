# fs_rfe.py
from sklearn.feature_selection import RFE
from fs_base import FeatureSelector

class RFESelector(FeatureSelector):
    def __init__(self, n_features_to_select: int = 5, estimator=None):
        """
        Eliminación recursiva de features.
        estimator: modelo base para evaluar importancias.
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        if estimator is None:
            # Elegir regresor o clasificador según y en fit
            self.dynamic = True
            self.base_classifier = RandomForestClassifier(n_estimators=100)
            self.base_regressor = RandomForestRegressor(n_estimators=100)
            self.selector = None
        else:
            self.dynamic = False
            self.selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
            self.n_features_to_select = n_features_to_select

    def fit(self, X, y):
        # Detectar tipo de tarea si es dinámico
        if self.dynamic:
            from pandas.api.types import is_numeric_dtype
            is_regression = is_numeric_dtype(y) and len(set(y)) != 2
            estimator = self.base_regressor if is_regression else self.base_classifier
            self.selector = RFE(estimator=estimator, n_features_to_select=self.n_features_to_select)
        self.selector.fit(X, y)

    def transform(self, X):
        mask = self.selector.get_support()
        return X.loc[:, mask]