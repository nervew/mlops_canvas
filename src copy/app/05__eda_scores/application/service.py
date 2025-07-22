import numpy as np
from ..infrastructure.metrics import compute_scores
from ..domain.report import ScoreReport


def run(y_true: np.ndarray, y_pred: np.ndarray) -> ScoreReport:
    metrics = compute_scores(y_true, y_pred)
    return ScoreReport(metrics=metrics)
