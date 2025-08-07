from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from ..infrastructure.repository import IngestionRepository

def ingest() -> pd.DataFrame:
    # ── IDEM: localiza la raíz real del repositorio ──
    # service.py está en: mlops_canvas/src/app/m01_data_ingestion/application/
    # parents[0]=application,1=m01_data_ingestion,2=app,3=src,4=mlops_canvas
    project_root = Path(__file__).resolve().parents[4]

    # 1) carga config desde <repo_root>/config/config.json
    cfg_path = project_root / "config" / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # 2) lee SQL desde <repo_root>/<query_file>
    sql_path  = project_root / cfg["query_file"]
    sql_text  = sql_path.read_text(encoding="utf-8")

    # 3) ejecuta la consulta y devuelve un DataFrame en memoria
    repo      = IngestionRepository(cfg)
    dataset   = repo.ingest(sql_text)
    return dataset.data
