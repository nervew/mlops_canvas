from pathlib import Path
import pandas as pd
from ..domain.entities import DataSet
from .connectors import SparkAdapter, PostgresAdapter, SqlServerAdapter

_ADAPTERS = {
    "spark":     SparkAdapter,
    "postgres":  PostgresAdapter,
    "sqlserver": SqlServerAdapter,
}

class IngestionRepository:
    """Orquesta la ingesta usando el adapter apropiado."""

    def __init__(self, cfg: dict) -> None:
        db_type = cfg["db_type"].lower()
        Adapter = _ADAPTERS.get(db_type)
        if Adapter is None:
            raise ValueError(f"db_type no soportado: {db_type}")

        self.adapter = Adapter(cfg.get("conn_str", ""))  # Spark ignora conn_str
        self.cfg = cfg

    # ------------------------------------------------------------------
    def ingest(self, sql: str) -> DataSet:
        self.adapter.connect()
        df = self.adapter.run_query(sql)

        # Guardar en Parquet (output/data/raw/df_raw.parquet)
        project_root = Path(__file__).resolve().parents[3]   # .../src
        out_dir  = project_root / self.cfg["output_folder"] / "data" / "raw"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / self.cfg["output_file"]
        df.to_parquet(out_path, index=False)
        print(f"✅ df_raw guardado en {out_path}", flush=True)

        return DataSet(data=df)
