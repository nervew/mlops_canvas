from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class UnivariateReport:
    description: Dict[str, Any]  # dict serializable (estadísticas, outliers, histo, etc.)
