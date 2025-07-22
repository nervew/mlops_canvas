from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DataValidationReport:
    valido: bool
    detalles: Dict[str, Any]
