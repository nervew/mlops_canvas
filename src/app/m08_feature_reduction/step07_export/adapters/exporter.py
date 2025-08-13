# m08_feature_reduction/step07_export/adapters/exporter.py
from __future__ import annotations
from pathlib import Path

def get_project_root() -> Path:
    d = Path(__file__).resolve()
    while d != d.parent:
        if (d / "src").is_dir():
            return d
        d = d.parent
    raise RuntimeError("No se pudo localizar la raíz del proyecto")

PROJECT_ROOT = get_project_root()
