from dataclasses import dataclass
from typing import Dict

@dataclass
class ValidationReport:
    success: bool
    statistics: Dict[str, float]
