import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif


def build_selector(X: pd.DataFrame, y: pd.Series) -> SelectKBest:
    selector = SelectKBest(score_func=f_classif, k=min(2, X.shape[1]))
    selector.fit(X, y)
    return selector
