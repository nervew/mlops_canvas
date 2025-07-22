import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def search(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> GridSearchCV:
    grid = GridSearchCV(model, param_grid={"max_depth": [None, 5, 10]}, cv=3)
    grid.fit(X, y)
    return grid
