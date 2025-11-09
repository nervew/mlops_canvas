"""Model quality checks for promotion gates."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QualityThresholds:
    min_auc: float
    max_psi: float


def should_promote(metrics: dict[str, float], thresholds: QualityThresholds) -> bool:
    return metrics.get("auc", 0.0) >= thresholds.min_auc and metrics.get("psi", 1.0) <= thresholds.max_psi
