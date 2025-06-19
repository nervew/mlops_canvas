from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class IReportWriter(ABC):
    """Abstract interface for persisting reports."""

    @abstractmethod
    def write(self, report: Dict[str, Any]) -> None:
        """Persist the generated report."""
        raise NotImplementedError
