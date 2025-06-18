from __future__ import annotations

from pathlib import Path
import pandas as pd

from eda_univariado.ports.loader import IDataLoader


class PandasLoader(IDataLoader):
    """Load a CSV file into a DataFrame."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)
