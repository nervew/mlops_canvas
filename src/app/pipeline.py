"""
Pipeline MLOps simplificado
===========================
"""

from __future__ import annotations

# -------------------------------------------------------------------------
# 0) Instalación de dependencias (opcional)
# -------------------------------------------------------------------------
from .m00_instalador import install_requirements
install_requirements()
# -------------------------------------------------------------------------

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
from .m01_data_ingestion import ingest as ingest_data
from .m02_eda_univariado import run as eda_univar
from .data_validation.application import service as validate_srv
from .m05_data_split.application import service as split_srv
from .m05_data_split.infrastructure.robust_data_splitter import RobustDataSplitter
from .search_model.flaml_wrapper import FLAMLWrapper
# -------------------------------------------------------------------------

# --------------------------- configuración global ------------------------
# Raíz del proyecto = carpeta que contiene /app y /config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELO_DIR   = PROJECT_ROOT / "models"
MODELO_DIR.mkdir(parents=True, exist_ok=True)
# -------------------------------------------------------------------------


# ========================================================================
# Función para guardar el modelo
# ========================================================================
def guardar_modelo(modelo, X_muestra, version: str = "v1") -> None:
    """Guarda el modelo en joblib y, si es posible, en ONNX."""
    ruta_joblib = MODELO_DIR / f"modelo_{version}.joblib"
    joblib.dump(modelo, ruta_joblib)
    print(f"✅ Modelo guardado en {ruta_joblib}", flush=True)

    ruta_onnx = MODELO_DIR / f"modelo_{version}.onnx"
    if isinstance(modelo, BaseEstimator):
        try:
            tipo_inicial = [("input", FloatTensorType([None, X_muestra.shape[1]]))]
            modelo_onnx = convert_sklearn(modelo, initial_types=tipo_inicial)
            ruta_onnx.write_bytes(modelo_onnx.SerializeToString())
            print(f"✅ Modelo ONNX guardado en {ruta_onnx}", flush=True)
        except Exception as exc:
            print(f"⚠️  No se pudo convertir a ONNX: {exc}", flush=True)
    else:
        print("⚠️  El modelo no es un estimador de Scikit-learn; solo se guarda joblib.", flush=True)


# ========================================================================
# Pipeline principal
# ========================================================================
def run() -> None:
    # 1) -------------------- INGESTA DE DATOS ----------------------------
    df = dataset.data
    print(f"📥 Ingesta completada: {len(df):,} filas – columnas: {list(df.columns)}\n")

    # 2) -------------------- EDA UNIVARIADO ------------------------------
    eda_report = eda_univar(df)            # Sweetviz + describe
    desc_df = pd.DataFrame(eda_report.description).T
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
        stratify_columns=[],
        train_size=0.7,
        test_size=0.2,
        backtest_size=0.1,
    )
    splits = split_srv.run(df, **PARAMS_SPLIT)
    train_df = splits.train_df
    test_df  = splits.test_df
    back_df  = splits.backtest_df

    print(
        "\n📊 Particiones generadas:"
        f"\n   • Train    : {len(train_df):,} filas"
        f"\n   • Test     : {len(test_df):,} filas"
        f"\n   • Backtest : {len(back_df):,} filas",
        flush=True,
    )

    # ---- Métricas de estabilidad / balanceo ----------------------------
    splitter = RobustDataSplitter(df, **PARAMS_SPLIT)
    splitter.split_data()
    metricas = splitter.calculate_metrics()

    print("\n📈 Métricas de estabilidad / balanceo:")
    try:
        metricas_df = metricas.round(4)
        print(metricas_df.to_markdown())   # requiere tabulate
    except Exception:
        print(json.dumps(metricas.to_dict(orient="index"), indent=2, ensure_ascii=False))

    # 5) -------------------- AutoML con FLAML ----------------------------
    X_train, y_train = train_df.drop(columns=["target"]), train_df["target"]
    X_test,  y_test  = test_df.drop(columns=["target"]),  test_df["target"]

    automl = FLAMLWrapper(time_budget=300)
    automl.fit(X_train, y_train, X_test, y_test)

    print("\n🏆 Ranking de modelos:\n", automl.get_model_ranking())
    print("🚀 Mejor modelo:", automl.get_best_model()[0])
    print("🔧 Parámetros óptimos:", automl.get_best_params())

    # 6) -------------------- Guardado del modelo -------------------------
    guardar_modelo(automl.get_best_model()[1], X_train, version="v1")


# -------------------------------------------------------------------------
if __name__ == "__main__":
    run()
