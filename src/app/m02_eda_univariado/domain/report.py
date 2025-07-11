from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class UnivariateReport:
    """Contenedor serializable para el resumen univariado."""
    description: Dict[str, Any]
