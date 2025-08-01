# src/app/run.py
from __future__ import annotations

# -------------------------------------------------------------------------
# 1) Instalador de dependencias
# -------------------------------------------------------------------------
import os, json, warnings, joblib, pandas as pd
from pathlib import Path
print("▶ [1/10] Instalando requirements (si es necesario)...", flush=True)
from .m00_instalador import install_requirements
install_requirements()
print("✔ [1/10] Requirements instalados.\n", flush=True)

# -------------------------------------------------------------------------
# 2) Configuración de warnings
# -------------------------------------------------------------------------
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

# -------------------------------------------------------------------------
# 3) Imports externos
# -------------------------------------------------------------------------
from sklearn.base import BaseEstimator
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# -------------------------------------------------------------------------
# 4) Backend de matplotlib
# -------------------------------------------------------------------------
os.environ["MPLBACKEND"] = "Agg"
import matplotlib; matplotlib.use("Agg")

# -------------------------------------------------------------------------
# 5) Imports de aplicación
# -------------------------------------------------------------------------
from .d_database import generate_synthetic_patient_data
from .m02_eda_univariado import run as eda_univar
from .m03_data_validation.application import service as validate_srv
from .m04_data_split.infrastructure.robust_data_splitter import RobustDataSplitter
from .m05_feature_engineering.pipeline_engineering import run_pipeline as fe_run
from .m06__feature_selection import run_pipeline as fs_run
from .m08_feature_reduction.pipeline_reduction import run_feature_reduction as fr_run
from .search_model.flaml_wrapper import FLAMLWrapper

# -------------------------------------------------------------------------
# 6) Configuración de rutas
# -------------------------------------------------------------------------
PROJECT_ROOT        = Path(__file__).resolve().parent.parent.parent  # Subir un nivel más
MODELO_DIR          = PROJECT_ROOT / "models"
RAW_DIR             = PROJECT_ROOT / "data" / "raw" / "complete"
RAW_PARTITIONED_DIR = PROJECT_ROOT / "data" / "raw" / "partitioned"
FE_DIR              = PROJECT_ROOT / "data" / "processed" / "pipeline_engineering"
FS_DIR              = PROJECT_ROOT / "data" / "processed" / "pipeline_selection"
OUTPUT_DIR          = PROJECT_ROOT / "output"
for p in [MODELO_DIR, RAW_DIR, RAW_PARTITIONED_DIR, FE_DIR, FS_DIR, OUTPUT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 7) Utilidad para guardar modelo
# -------------------------------------------------------------------------
def guardar_modelo(modelo, X_muestra: pd.DataFrame, version="v1"):
    ruta = MODELO_DIR / f"modelo_{version}.joblib"
    joblib.dump(modelo, ruta)
    print(f"✔ Modelo guardado en {ruta}", flush=True)
    if isinstance(modelo, BaseEstimator):
        try:
            onx = convert_sklearn(
                modelo,
                initial_types=[("input", FloatTensorType([None, X_muestra.shape[1]]))]
            )
            (MODELO_DIR / f"modelo_{version}.onnx")\
                .write_bytes(onx.SerializeToString())
            print("✔ Modelo ONNX exportado", flush=True)
        except Exception as exc:
            print(f"⚠ ONNX no generado: {exc}", flush=True)

# -------------------------------------------------------------------------
# 8) Pipeline principal
# -------------------------------------------------------------------------
def run() -> None:
    # 2) Ingesta
    print("▶ [2/10] Generando y guardando datos sintéticos...", flush=True)
    df = generate_synthetic_patient_data()
    df.to_parquet(RAW_DIR / "df_raw.parquet")
    print(f"✔ [2/10] Ingesta completada: {len(df):,} filas\n", flush=True)

    # 3) EDA + Validación
    print("▶ [3/10] Ejecutando EDA univariado y validación...", flush=True)
    eda_univar(df)
    if not validate_srv.run(df, fit_profile=True).valido:
        raise ValueError("✖ Validación fallida")
    print("✔ Datos validados\n", flush=True)

    # 4) Split temporal
    print("▶ [4/10] Particionando datos...", flush=True)
    splitter = RobustDataSplitter(
        df,
        split_method="time",
        target_column="target",
        time_column="Semana",
        train_size=0.7,
        test_size=0.2,
        backtest_size=0.1,
    )
    tr_df, te_df, bk_df = splitter.split_data()
    tr_df.to_parquet(RAW_PARTITIONED_DIR / "train_df.parquet")
    te_df.to_parquet(RAW_PARTITIONED_DIR / "test_df.parquet")
    bk_df.to_parquet(RAW_PARTITIONED_DIR / "backtest_df.parquet")
    print("✔ Particiones guardadas\n", flush=True)

    # 5) Feature Engineering
    print("▶ [5/10] Feature Engineering...", flush=True)
    fe_run()
    print("✔ Feature Engineering finalizado\n", flush=True)

    # 6) Feature Selection (filter → frame)
    print("▶ [6/10] Feature Selection (filter → frame)...", flush=True)
    X_tr_fs, X_te_fs, X_bk_fs, logs = fs_run(
        techniques=["filter", "frame"],
        save_logs=True
    )
    # Mostrar logs como tabla
    print(pd.DataFrame(logs), "\n")

    lista_feats = [c for c in X_tr_fs.columns if c != "target"]
    lista_dir = OUTPUT_DIR / "Lista_feature_final"
    lista_dir.mkdir(parents=True, exist_ok=True)
    with open(lista_dir / "lista_feature_final.json", "w") as fh:
        json.dump(lista_feats, fh, indent=4)
    print("✔ Lista de features exportada\n", flush=True)

    # 7) AutoML FLAML  (posición adelantada, antes de m08)
    print("▶ [7/10] Ejecutando AutoML con FLAML (regression)...", flush=True)
    y_train = pd.read_parquet(FE_DIR / "X_train_processed.parquet").pop("target")
    y_test  = pd.read_parquet(FE_DIR / "X_test_processed.parquet").pop("target")
    automl = FLAMLWrapper(task="regression", time_budget=300, metric="mae")
    automl.fit(X_tr_fs, y_train, X_te_fs, y_test)
    print("\n🏆 Mejor modelo:", automl.get_best_model()[0], "\n", flush=True)

    # 8) Guardar modelo final
    print("▶ [8/10] Guardando modelo final...", flush=True)
    guardar_modelo(automl.get_best_model()[1], X_tr_fs, version="v1")
    print("✔ Modelo final guardado\n", flush=True)

    # 9) Feature Reduction (m08) — último paso, PCA desactivado
    print("▶ [9/10] Ejecutando Feature Reduction (PCA desactivado)...", flush=True)
    transf_inicial = joblib.load(PROJECT_ROOT / "transformers" / "transformador_inicial.joblib")
    fr_run(
        transformer=transf_inicial,
        features=lista_feats,
        params={"method": "pca", "n_components": 3},
        use_reduction=False,        # Sin PCA
        temporal_vars=["Semana"],
        target_var="target",
    )
    print("✔ Artefactos de Feature Reduction exportados\n", flush=True)

    print("✔ [10/10] Pipeline completado exitosamente.", flush=True)


if __name__ == "__main__":
    run()
