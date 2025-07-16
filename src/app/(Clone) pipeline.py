# /Workspace/Users/jorgee.lopez@adres.gov.co/mlops_canvas/src/app/pipeline.py
"""
Pipeline MLOps simplificado
===========================
Carga datos sintéticos, valida, genera particiones (train / test / backtest),
exporta esas particiones a Parquet, ejecuta AutoML con FLAML y
serializa el mejor modelo en joblib y ONNX.
"""
from __future__ import annotations

# -------------------------------------------------------------------------
# Ajustes de entorno para que Matplotlib no abra ventanas
# -------------------------------------------------------------------------
import os
os.environ["MPLBACKEND"] = "Agg"   # backend no interactivo
import matplotlib                 # noqa: E402  (import después del env var)
matplotlib.use("Agg")             # noqa: E402

# -------------------------------------------------------------------------
# Instalación de dependencias declaradas en requirements.txt (opcional)
# -------------------------------------------------------------------------
from .m00_instalador import install_requirements  # noqa: E402
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
# -------------------------------------------------------------------------

# --------------------------- configuración global ------------------------
PROJECT_ROOT        = Path(__file__).resolve().parent.parent
MODELO_DIR          = PROJECT_ROOT / "models"
RAW_DIR             = PROJECT_ROOT / "data" / "raw" / "complete"
RAW_PARTITIONED_DIR = PROJECT_ROOT / "data" / "raw" / "partitioned"

# Crear carpetas si no existen
for p in [MODELO_DIR, RAW_DIR, RAW_PARTITIONED_DIR]:
    p.mkdir(parents=True, exist_ok=True)
# -------------------------------------------------------------------------


# ========================================================================
# Función utilitaria para guardar el modelo
# ========================================================================
def guardar_modelo(modelo, X_muestra: pd.DataFrame, version: str = "v1") -> None:
    """Guarda el modelo en Joblib y, si es posible, en ONNX."""
    ruta_joblib = MODELO_DIR / f"modelo_{version}.joblib"
    joblib.dump(modelo, ruta_joblib)
    print(f"✅ Modelo guardado en {ruta_joblib}", flush=True)

    ruta_onnx = MODELO_DIR / f"modelo_{version}.onnx"
    if isinstance(modelo, BaseEstimator):
        try:
            tipo_inicial = [("input", FloatTensorType([None, X_muestra.shape[1]]))]
            modelo_onnx  = convert_sklearn(modelo, initial_types=tipo_inicial)
            ruta_onnx.write_bytes(modelo_onnx.SerializeToString())
            print(f"✅ Modelo ONNX guardado en {ruta_onnx}", flush=True)
        except Exception as exc:
            print(f"⚠️  No se pudo convertir a ONNX: {exc}", flush=True)
    else:
        print("⚠️  El modelo no es un estimador de Scikit-learn; solo se guarda Joblib.", flush=True)


# ========================================================================
# Pipeline principal
# ========================================================================
def run() -> None:
    # 1) -------------------- INGESTA DE DATOS ----------------------------
    df = generate_synthetic_patient_data()
    print(f"📥 Ingesta completada: {len(df):,} filas – columnas: {list(df.columns)}")
    df.to_parquet(RAW_DIR / "df_raw.parquet")

    # 2) -------------------- EDA UNIVARIADO ------------------------------
    eda_report = eda_univar(df)
    desc_df    = pd.DataFrame(eda_report.description).T
    print("\n🔎 Estadísticas básicas (head):\n", desc_df.head(), "\n")

    # 3) -------------------- VALIDACIÓN DE DATOS -------------------------
    val_report = validate_srv.run(df, fit_profile=True)
    if not getattr(val_report, "valido", False):
        print("❌ La validación falló:", getattr(val_report, "detalles", {}))
        raise ValueError("Los datos no pasan la validación")
    print("✅ Validación exitosa")

    # 4) -------------------- SPLIT DE DATOS ------------------------------
    PARAMS_SPLIT = dict(
        split_method="time",
        target_column="target",
        time_column="Semana",
        train_size=0.7,
        test_size=0.2,
        backtest_size=0.1,
    )
    splitter = RobustDataSplitter(df, **PARAMS_SPLIT)
    train_df, test_df, backtest_df = splitter.split_data()

    print(
        "\n📊 Particiones generadas:"
        f"\n   • Train    : {len(train_df):,} filas"
        f"\n   • Test     : {len(test_df):,} filas"
        f"\n   • Backtest : {len(backtest_df):,} filas",
        flush=True,
    )

    # ---- Guardar particiones en Parquet (CORREGIDO) --------------------
    train_df.to_parquet(RAW_PARTITIONED_DIR / "train_df.parquet")
    test_df.to_parquet(RAW_PARTITIONED_DIR / "test_df.parquet")
    backtest_df.to_parquet(RAW_PARTITIONED_DIR / "backtest_df.parquet")

    # ---- Métricas de estabilidad / balanceo ----------------------------
    metricas = splitter.calculate_metrics()
    print("\n📈 Métricas de estabilidad / balanceo:")
    try:
        print(metricas.round(4).to_markdown())  # requiere tabulate
    except Exception:
        print(json.dumps(metricas.to_dict(orient="index"), indent=2, ensure_ascii=False))

    # 5) -------------------- AutoML con FLAML ----------------------------
    X_train, y_train = train_df.drop(columns=["target"]), train_df["target"]
    X_test,  y_test  = test_df.drop(columns=["target"]),  test_df["target"]

    automl = FLAMLWrapper(
        task="regression",        # o "classification"
        time_budget=300,          # segundos
        metric="mae",             # métrica principal
    )
    automl.fit(X_train, y_train, X_test, y_test)

    print("\n🏆 Ranking de modelos:\n", automl.get_model_ranking())
    print("🚀 Mejor modelo:", automl.get_best_model()[0])
    print("🔧 Parámetros óptimos:", automl.get_best_params())

    # 6) -------------------- Guardado del modelo -------------------------
    guardar_modelo(automl.get_best_model()[1], X_train, version="v1")


# -------------------------------------------------------------------------
if __name__ == "__main__":
    run()
