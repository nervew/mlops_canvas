from sklearn.feature_selection import SelectKBest, f_classif

def feature_selection(X_train, y_train, X_test, k_max: int = 8):
    selector = SelectKBest(score_func=f_classif, k=min(k_max, X_train.shape[1]))
    selector.fit(X_train, y_train)
    return selector.transform(X_train), selector.transform(X_test), selector
