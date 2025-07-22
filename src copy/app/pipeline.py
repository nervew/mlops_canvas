# src/app/run.py
from __future__ import annotations

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import os
from pathlib import Path
import joblib
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.base import BaseEstimator

# 1) Instalador de dependencias
print("▶ [1/8] Instalando requirements (si es necesario)...", flush=True)
from .m00_instalador import install_requirements
install_requirements()
print("✔ [1/8] Requirements instalados.\n", flush=True)

# 2) Fija backend de matplotlib
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

# 3) Imports de aplicación
from .d_database import generate_synthetic_patient_data
from .m02_eda_univariado import run as eda_univar
from .m03_data_validation.application import service as validate_srv
from .m04_data_split.infrastructure.robust_data_splitter import RobustDataSplitter
from .m05_feature_engineering.pipeline_engineering import run_pipeline as fe_run
from .m06__feature_selection import run_pipeline as fs_run
from .search_model.flaml_wrapper import FLAMLWrapper

# 4) Configuración de rutas
PROJECT_ROOT        = Path(__file__).resolve().parent.parent
MODELO_DIR          = PROJECT_ROOT / "models"
RAW_DIR             = PROJECT_ROOT / "data" / "raw" / "complete"
RAW_PARTITIONED_DIR = PROJECT_ROOT / "data" / "raw" / "partitioned"
FE_DIR              = PROJECT_ROOT / "data" / "processed" / "pipeline_engineering"
FS_DIR              = PROJECT_ROOT / "data" / "processed" / "pipeline_selection"

for p in [MODELO_DIR, RAW_DIR, RAW_PARTITIONED_DIR, FE_DIR, FS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def guardar_modelo(modelo, X_muestra: pd.DataFrame, version: str = "v1") -> None:
    ruta_joblib = MODELO_DIR / f"modelo_{version}.joblib"
    joblib.dump(modelo, ruta_joblib)
    print(f"✔ Modelo guardado en {ruta_joblib}", flush=True)

    if isinstance(modelo, BaseEstimator):
        try:
            initial = [("input", FloatTensorType([None, X_muestra.shape[1]]))]
            modelo_onnx = convert_sklearn(modelo, initial_types=initial)
            ruta_onnx = MODELO_DIR / f"modelo_{version}.onnx"
            ruta_onnx.write_bytes(modelo_onnx.SerializeToString())
            print(f"✔ Modelo ONNX guardado en {ruta_onnx}", flush=True)
        except Exception as exc:
            print(f"⚠ ONNX no generado: {exc}", flush=True)

def run() -> None:
    # 5) Ingesta
    print("▶ [2/8] Generando y guardando datos sintéticos...", flush=True)
    df = generate_synthetic_patient_data()
    df.to_parquet(RAW_DIR / "df_raw.parquet")
    print(f"✔ [2/8] Ingesta completada: {len(df):,} filas\n", flush=True)

    # 6) EDA + Validación
    print("▶ [3/8] Ejecutando EDA univariado y validación...", flush=True)
    eda_univar(df)
    validacion = validate_srv.run(df, fit_profile=True)
    if not getattr(validacion, "valido", False):
        raise ValueError("✖ Validación de datos fallida")
    print("✔ [3/8] Data validada correctamente\n", flush=True)

    # 7) Split temporal
    print("▶ [4/8] Particionando datos (train/test/back)...", flush=True)
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
    print("✔ [4/8] Particiones raw guardadas\n", flush=True)

    # 8) Feature Engineering
    print("▶ [5/8] Ejecutando pipeline de Feature Engineering...", flush=True)
    fe_run()
    print("✔ [5/8] Feature Engineering finalizado\n", flush=True)

    # 9) Feature Selection
    print("▶ [6/8] Ejecutando pipeline de Feature Selection...", flush=True)
    X_train_fs, X_test_fs, X_back_fs = fs_run()
    print("✔ [6/8] Feature Selection finalizado\n", flush=True)

    # 10) Preparar para AutoML
    y_train = pd.read_parquet(FE_DIR / "X_train_processed.parquet").pop("target")
    y_test  = pd.read_parquet(FE_DIR / "X_test_processed.parquet").pop("target")

    # 11) AutoML FLAML
    print("▶ [7/8] Ejecutando AutoML con FLAML (regression)...")#, flush=True)
    automl = FLAMLWrapper(task="regression", time_budget=300, metric="mae")
    automl.fit(X_train_fs, y_train, X_test_fs, y_test)
    print("\n🏆 Mejor modelo:", automl.get_best_model()[0])#, "\n", flush=True)

    # 12) Guardar modelo
    print("▶ [8/8] Guardando modelo final...", flush=True)
    guardar_modelo(automl.get_best_model()[1], X_train_fs, version="v1")
    print("✔ [8/8] Pipeline completado exitosamente.", flush=True)

if __name__ == "__main__":
    run()
