"""Data drift detection utilities."""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    hist_expected, bin_edges = np.histogram(expected, bins=bins, density=True)
    hist_actual, _ = np.histogram(actual, bins=bin_edges, density=True)
    psi = np.sum((hist_actual - hist_expected) * np.log((hist_actual + 1e-6) / (hist_expected + 1e-6)))
    return float(psi)


def detect_drift(expected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    psi = population_stability_index(expected, actual)
    ks_stat, ks_pvalue = ks_2samp(expected, actual)
    return {"psi": psi, "ks_stat": float(ks_stat), "ks_pvalue": float(ks_pvalue)}
