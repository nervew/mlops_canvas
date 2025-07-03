# fs_mutual_info.py
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from fs_base import FeatureSelector

class MutualInfoSelector(FeatureSelector):
    def __init__(self, k: int = 10, task: str = "classification"):
        """
        Selecciona las k mejores features según información mutua.
        task: 'classification' o 'regression'.
        """
        score_func = mutual_info_classif if task == "classification" else mutual_info_regression
        self.selector = SelectKBest(score_func=score_func, k=k)

    def fit(self, X, y):
        self.selector.fit(X, y)

    def transform(self, X):
        mask = self.selector.get_support()
        return X.loc[:, mask]