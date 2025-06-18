from pathlib import Path
import pandas as pd

from .data_source_interface import DataSource

class CSVDataSource(DataSource):
    """Data source for reading CSV files."""

    def __init__(self, file_path: str) -> None:
        self.file_path = Path(file_path)

    def fetch_data(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file {self.file_path} not found")
        df = pd.read_csv(self.file_path)
        if df.empty:
            raise ValueError("CSVDataSource: no data loaded (file empty?)")
        return df
