"""Example showing how to use connectors with YAML configs."""

from pathlib import Path
from typing import Any

import yaml

from ingestion.connectors import PostgresAdapter, SqlServerAdapter


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config(Path("configs/connections.yaml"))
    sql_adapter = SqlServerAdapter(cfg["sql_server"]["conn_str"])
    pg_adapter = PostgresAdapter(cfg["postgres"]["conn_str"])

    sql_adapter.connect()
    pg_adapter.connect()

    result1 = sql_adapter.execute_query("SELECT 1")
    result2 = pg_adapter.execute_query("SELECT 2")

    print(result1)
    print(result2)


if __name__ == "__main__":
    main()
