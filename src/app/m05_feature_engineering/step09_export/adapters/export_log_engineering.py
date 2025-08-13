import json
from pathlib import Path

def export_log_engineering(time_cols, num_cols, cat_cols, log_dir: Path):
    """
    Guarda un log con las columnas detectadas en el pipeline de feature engineering.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "log_engineering.json"
    data = {
        "time_cols": time_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"[Log] Configuración de ingeniería exportada a {log_path}")
