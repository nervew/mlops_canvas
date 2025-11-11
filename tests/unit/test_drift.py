from __future__ import annotations

import numpy as np

from src.monitoring.drift import detect_drift


def test_detect_drift_returns_metrics() -> None:
    expected = np.random.normal(0, 1, 100)
    actual = np.random.normal(0, 1, 100)
    metrics = detect_drift(expected, actual)
    assert set(metrics.keys()) == {"psi", "ks_stat", "ks_pvalue"}
