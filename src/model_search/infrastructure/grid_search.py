import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def search(X: pd.DataFrame, y: pd.Series) -> GridSearchCV:
    model = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(model, param_grid={"n_estimators": [50, 100]}, cv=3)
    grid.fit(X, y)
    return grid
