# pipeline/ingestion.py
import os
import json
import numpy as np
import pandas as pd
from .constants import RANDOM_SEED
from typing import Any

# ▸ Opcional: expone un alias para importar los adapters
from ingestion.connectors import PostgresAdapter, SqlServerAdapter, SparkAdapter

_ADAPTERS = {
    "postgres": PostgresAdapter,
    "sqlserver": SqlServerAdapter,
    "spark": SparkAdapter,  # Añadido SparkAdapter
}

# ────────────────────────────────────────────────
# 1) FUNCIÓN DUMMY 100 % INDEPENDIENTE
# ────────────────────────────────────────────────
def data_ingestion_dummy(n_samples: int = 500) -> pd.DataFrame:
    """Genera un DataFrame sintético y reproducible para pruebas."""
    rng = np.random.default_rng(RANDOM_SEED)
    return pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n_samples, freq="D"),
        "category": rng.choice(["A", "B", "C"], size=n_samples),
        "num1": rng.normal(size=n_samples),
        "num2": rng.uniform(0, 100, size=n_samples),
        "target": rng.integers(0, 2, size=n_samples),
    })


# ────────────────────────────────────────────────
# 2) FUNCIÓN DE INGESTA REAL (verificación estricta)
# ────────────────────────────────────────────────
def data_ingestion_db(config_path: str) -> pd.DataFrame:
    import json, os, pandas as pd

    # Carga la configuración
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not cfg.get("use_database", False):
        raise RuntimeError("use_database=false; usa dummy.")

    # Validación mínima de configuración corregida
    required_keys = ["db_type", "query_file"]
    if cfg["db_type"].lower() in ["postgres", "sqlserver"]:
        required_keys.append("conn_str")

    missing_keys = [k for k in required_keys if k not in cfg]
    if missing_keys:
        raise KeyError(f"Faltan claves en config.json: {missing_keys}")

    # Selección de adapter
    Adapter = _ADAPTERS.get(cfg["db_type"].lower())
    if Adapter is None:
        raise ValueError(f"db_type no soportado: {cfg['db_type']}")

    adapter = Adapter(cfg.get("conn_str", "")) if cfg["db_type"].lower() != "spark" else Adapter()
    adapter.connect()

    # Ejecutar la consulta SQL
    with open(cfg["query_file"], "r", encoding="utf-8") as f:
        sql = f.read()

    df = adapter.execute_query(sql)

    # Guardado opcional en Parquet
    out_dir = cfg.get("output_folder", "registros")
    out_file = cfg.get("output_file", "salida.parquet")
    os.makedirs(out_dir, exist_ok=True)
    ruta_parquet = os.path.join(out_dir, out_file)
    df.to_parquet(ruta_parquet, index=False)

    return df

# ────────────────────────────────────────────────
# PEQUEÑA CLI PARA PROBAR RÁPIDO DESDE TERMINAL
# ────────────────────────────────────────────────
if __name__ == "__main__":
    CONFIG = "/Workspace/Users/jorgee.lopez@adres.gov.co/mlops_canvas/config/config.json"

    try:
        df = data_ingestion_db(CONFIG)
        print("Ingesta DB completada. Primeras filas:")
    except RuntimeError:
        df = data_ingestion_dummy()
        print("Generación DUMMY completada. Primeras filas:")

    print(df.head())
