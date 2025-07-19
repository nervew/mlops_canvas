# /Workspace/Users/jorgee.lopez@adres.gov.co/mlops_canvas/src/app/pipeline.py
"""
Pipeline MLOps completo
───────────────────────
Incluye:
1) Ingesta y validación
2) Particionado train/test/backtest
3) Pipeline de Feature Engineering (m05_feature_engineering)
4) AutoML con FLAML
5) Serialización del modelo en joblib y ONNX
"""
from __future__ import annotations

# -------------------------------------------------------------------------
# Configuración de entorno para matplotlib headless
# -------------------------------------------------------------------------
import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

# -------------------------------------------------------------------------
# Dependencias on-the-fly (opcional)
# -------------------------------------------------------------------------
from .m00_instalador import install_requirements
install_requirements()

# ---------------------------- imports estándar ---------------------------
import json
from pathlib import Path
import joblib
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.base import BaseEstimator
# -------------------------------------------------------------------------

# -------------------------- imports de la aplicación ---------------------
from .d_database import generate_synthetic_patient_data
from .m02_eda_univariado import run as eda_univar
from .m03_data_validation.application import service as validate_srv
from .m04_data_split.infrastructure.robust_data_splitter import RobustDataSplitter
from .search_model.flaml_wrapper import FLAMLWrapper

# ---- Pipeline de Feature Engineering -----------------------------------
from .m05_feature_engineering.pipeline_engineering import run_pipeline as fe_run
# -------------------------------------------------------------------------

# --------------------------- configuración global ------------------------
PROJECT_ROOT        = Path(__file__).resolve().parent.parent
MODELO_DIR          = PROJECT_ROOT / "models"
RAW_DIR             = PROJECT_ROOT / "data" / "raw" / "complete"
RAW_PARTITIONED_DIR = PROJECT_ROOT / "data" / "raw" / "partitioned"
PROCESSED_DIR       = PROJECT_ROOT / "data" / "processed" / "pipeline_engineering"

for p in [MODELO_DIR, RAW_DIR, RAW_PARTITIONED_DIR, PROCESSED_DIR]:
    p.mkdir(parents=True, exist_ok=True)
# -------------------------------------------------------------------------


# ========================================================================
# Utilidad para guardar modelo
# ========================================================================
def guardar_modelo(modelo, X_muestra: pd.DataFrame, version: str = "v1") -> None:
    ruta_joblib = MODELO_DIR / f"modelo_{version}.joblib"
    joblib.dump(modelo, ruta_joblib)
    print(f"✅ Modelo guardado en {ruta_joblib}", flush=True)

    ruta_onnx = MODELO_DIR / f"modelo_{version}.onnx"
    if isinstance(modelo, BaseEstimator):
        try:
            initial = [("input", FloatTensorType([None, X_muestra.shape[1]]))]
            modelo_onnx = convert_sklearn(modelo, initial_types=initial)
            ruta_onnx.write_bytes(modelo_onnx.SerializeToString())
            print(f"✅ Modelo ONNX guardado en {ruta_onnx}", flush=True)
        except Exception as exc:
            print(f"⚠️  ONNX no generado: {exc}", flush=True)


# ========================================================================
# Pipeline principal
# ========================================================================
def run() -> None:
    # 1) -------------------- Ingesta --------------------
    df = generate_synthetic_patient_data()
    df.to_parquet(RAW_DIR / "df_raw.parquet")
    print(f"📥 Ingesta completada: {len(df):,} filas")

    # 2) -------------------- EDA + validación -----------
    eda_univar(df)
    if not getattr(validate_srv.run(df, fit_profile=True), "valido", False):
        raise ValueError("Validación de datos fallida")

    # 3) -------------------- Split ----------------------
    splitter = RobustDataSplitter(
        df,
        split_method="time",
        target_column="target",
        time_column="Semana",
        train_size=0.7,
        test_size=0.2,
        backtest_size=0.1,
    )
    train_df, test_df, back_df = splitter.split_data()

    train_df.to_parquet(RAW_PARTITIONED_DIR / "train_df.parquet")
    test_df.to_parquet(RAW_PARTITIONED_DIR / "test_df.parquet")
    back_df.to_parquet(RAW_PARTITIONED_DIR / "backtest_df.parquet")
    print("✅ Particiones raw guardadas en data/raw/partitioned")

    # 4) -------------------- Feature Engineering --------
    print("\n⚙️  Ejecutando pipeline de Feature Engineering …")
    fe_run()  # genera Parquet procesados en data/processed
    print("✅ Feature Engineering finalizado")

    # 5) -------------------- Cargar datos procesados ----
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train_processed.parquet")
    X_test  = pd.read_parquet(PROCESSED_DIR / "X_test_processed.parquet")

    y_train = X_train.pop("target")
    y_test  = X_test.pop("target")

    # 6) -------------------- AutoML FLAML ---------------
    automl = FLAMLWrapper(task="regression", time_budget=300, metric="mae")
    automl.fit(X_train, y_train, X_test, y_test)

    print("\n🏆 Ranking de modelos:\n", automl.get_model_ranking())
    print("🚀 Mejor modelo:", automl.get_best_model()[0])

    # 7) -------------------- Guardar modelo -------------
    guardar_modelo(automl.get_best_model()[1], X_train, version="v1")


# -------------------------------------------------------------------------
if __name__ == "__main__":  # llamada desde consola:  python -m app.pipeline
    run()
