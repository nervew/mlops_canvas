import numpy as np
from sklearn.metrics import accuracy_score


def compute_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"accuracy": float(accuracy_score(y_true, y_pred))}
