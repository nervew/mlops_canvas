from dataclasses import dataclass
from typing import Dict

@dataclass
class InferenceValidationReport:
    success: bool
    statistics: Dict[str, float]
