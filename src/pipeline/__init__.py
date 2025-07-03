"""
Inicializador del pipeline.
‐ Fuente de datos:
    • data_ingestion_dummy()  ← por defecto (datos sintéticos)
    • data_ingestion_db(CONFIG)  ← descomentar cuando quieras la BD real
"""

from pathlib import Path

# ───────────────── INGESTA ──────────────────
from .ingestion import data_ingestion_dummy, data_ingestion_db

# ───────────────── TRANSFORMACIONES ─────────
from .splitting import data_split
from .preprocessing import data_engineering
from .feat_select import feature_selection

# ───────────────── MODELADO ─────────────────
from .training import search_model
from .export import create_model
from .evaluation import evaluate


# Ruta al archivo de configuración global
CONFIG = Path("/Workspace/Users/jorgee.lopez@adres.gov.co/mlops_canvas/config/config.json")


def run_pipeline() -> None:
    # 1 ─ Ingesta de datos
    df = data_ingestion_dummy()
    # df = data_ingestion_db(CONFIG)  # ← Descomenta para usar la BD real

    # 2 ─ Split temporal y feature engineering
    train_df, test_df = data_split(df)
    train_df, test_df = data_engineering(train_df, test_df)

    # 3 ─ Selección de variables
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test  = test_df.drop(columns=["target"])
    y_test  = test_df["target"]

    X_train_sel, X_test_sel, _ = feature_selection(X_train, y_train, X_test)

    # 4 ─ Búsqueda y entrenamiento de modelo
    grid  = search_model(X_train_sel, y_train)
    model = create_model(grid, X_train_sel, y_train, version="v1")

    # 5 ─ Evaluación
    acc = evaluate(model, X_test_sel, y_test)
    print(f"Test accuracy: {acc:.4f}", flush=True)


if __name__ == "__main__":
    run_pipeline()
