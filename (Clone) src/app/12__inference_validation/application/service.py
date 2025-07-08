import numpy as np
from ..infrastructure.basic_validator import validate
from ..domain.report import InferenceValidationReport


def run(preds: np.ndarray) -> InferenceValidationReport:
    success = bool(validate(preds))
    stats = {"count": len(preds)}
    return InferenceValidationReport(success=success, statistics=stats)
