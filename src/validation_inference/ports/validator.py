from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import pandas as pd


class IInferenceValidator(ABC):
    """Interface for inference validation strategies."""

    @abstractmethod
    def validate(self, data: pd.DataFrame, target: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate data and return flag plus details."""
        raise NotImplementedError
