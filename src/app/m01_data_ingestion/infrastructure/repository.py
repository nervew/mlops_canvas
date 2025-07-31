from pathlib import Path
from ..domain.entities import DataSet
from .connectors import SparkAdapter, PostgresAdapter, SqlServerAdapter, DatabricksAdapter

_ADAPTERS = {
    "spark": SparkAdapter,
    "postgres": PostgresAdapter,
    "sqlserver": SqlServerAdapter,
    "databricks": DatabricksAdapter,
}

class IngestionRepository:
    def __init__(self, cfg: dict) -> None:
        Adapter = _ADAPTERS[cfg["db_type"].lower()]
        
        if cfg["db_type"].lower() == "spark":
            self.adapter = Adapter(cfg.get("catalog"))
        else:
            self.adapter = Adapter(cfg.get("conn_str", ""))
        
        self.cfg = cfg

    def ingest(self, sql: str) -> DataSet:
        self.adapter.connect()
        df = self.adapter.run_query(sql)

        project_root = Path(__file__).resolve().parents[3]
        out_dir = project_root / self.cfg["output_folder"] / "data" / "raw"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / self.cfg["output_file"]
        df.to_parquet(out_path, index=False)

        print(f"✅ Archivo guardado en {out_path}")
        return DataSet(data=df)
