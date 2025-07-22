from dataclasses import dataclass
from typing import Dict

@dataclass
class ScoreReport:
    metrics: Dict[str, float]
