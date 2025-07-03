from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from .constants import RANDOM_SEED

def search_model(X_train, y_train):
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_SEED),
        param_grid={"n_estimators": [50, 100], "max_depth": [None, 5]},
        cv=3,
    )
    grid.fit(X_train, y_train)
    return grid
