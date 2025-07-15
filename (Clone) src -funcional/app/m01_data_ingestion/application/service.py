import json
import pandas as pd
from pathlib import Path
from ..infrastructure.connectors import get_connector

def ingest() -> pd.DataFrame:
    """
    Lee configuración y devuelve el DataFrame resultante.
    """
    project_root = Path(__file__).resolve().parents[3] # src/
    cfg_path = project_root.parent / "config" / "config.json"
    cfg = json.loads(cfg_path.read_text())

    sql_file = project_root / cfg["query_file"]
    query = sql_file.read_text()

    connector = get_connector(cfg)
    df = connector(query)

    # Guardar resultado como parquet
    output_dir = project_root.parent / cfg["output_folder"]
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_dir / cfg["output_file"], index=False)

    print(f"✅ Archivo guardado en {output_dir / cfg['output_file']}")
    return df
