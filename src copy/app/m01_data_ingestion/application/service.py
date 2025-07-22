from pathlib import Path
import json
from ..infrastructure.repository import IngestionRepository
import pandas as pd

def ingest() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[3]
    cfg_path = project_root.parent / "config" / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    sql_path = project_root / cfg["query_file"]
    sql_text = sql_path.read_text(encoding="utf-8")

    repo = IngestionRepository(cfg)
    dataset = repo.ingest(sql_text)
    return dataset.data
