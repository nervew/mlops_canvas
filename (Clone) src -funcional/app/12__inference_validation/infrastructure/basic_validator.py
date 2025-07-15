import numpy as np


def validate(preds: np.ndarray) -> bool:
    return np.all((preds >= 0) & (preds <= 2))
