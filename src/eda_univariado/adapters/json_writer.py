from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict

from eda_univariado.ports.writer import IReportWriter


class JsonWriter(IReportWriter):
    """Write the report dictionary to a JSON file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, report: Dict[str, Any]) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
