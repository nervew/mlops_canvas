import pandas as pd
import os
from pathlib import Path

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
csv_path = ARTIFACTS_DIR / os.getenv("DATA_CSV_FILE", "data.csv")
parquet_path = ARTIFACTS_DIR / os.getenv("DATA_FILE", "data.parquet")

df = pd.read_csv(csv_path)
df.to_parquet(parquet_path, index=False)

print("Archivo convertido a Parquet:", parquet_path)
