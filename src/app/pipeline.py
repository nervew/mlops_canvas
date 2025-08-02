from __future__ import annotations

# -------------------------------------------------------------------------
# 1) Instalador de dependencias
# -------------------------------------------------------------------------
import os
import json
import warnings
import joblib
import pandas as pd
from pathlib import Path

print("[1/10] Instalando dependencias...", flush=True)
from .m00_instalador import install_requirements
install_requirements()
print("[1/10] Dependencias instaladas.\n", flush=True)

# -------------------------------------------------------------------------
# 2) Configuración de warnings
# -------------------------------------------------------------------------
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

# -------------------------------------------------------------------------
# 3) Backend de matplotlib
# -------------------------------------------------------------------------
os.environ["MPLBACKEND"] = "Agg"
import matplotlib  # noqa: E402
matplotlib.use("Agg")  # noqa: E402

# -------------------------------------------------------------------------
# 4) Imports de aplicación
# -------------------------------------------------------------------------
import numpy as np

#numpy ya no trae VisibleDeprecationWarning, SE crea
if not hasattr(np, "VisibleDeprecationWarning"):
    class VisibleDeprecationWarning(Warning):
        """Parche para supervisied.utils.automl_plots"""
        pass
    np.VisibleDeprecationWarning = VisibleDeprecationWarning


from .d_database import generate_synthetic_patient_data
from .m02_eda_univariado import run as eda_univar
from .m03_data_validation.application import service as validate_srv
from .m04_data_split.infrastructure.robust_data_splitter import RobustDataSplitter
from .m05_feature_engineering.pipeline_engineering import run_pipeline as fe_run
from .m06__feature_selection import run_pipeline as fs_run
from .m08_feature_reduction.pipeline_reduction import run_feature_reduction as fr_run

from .search_model.run_model_selector import run_model_selector
from .search_model.export import export_model_onnx
from sklearn.metrics import mean_absolute_error

# -------------------------------------------------------------------------
# 5) Configuración de rutas
# -------------------------------------------------------------------------
PROJECT_ROOT        = Path(__file__).resolve().parent.parent.parent
MODELS_DIR          = PROJECT_ROOT / "models"
RAW_DIR             = PROJECT_ROOT / "data" / "raw" / "complete"
RAW_PARTITIONED_DIR = PROJECT_ROOT / "data" / "raw" / "partitioned"
FE_DIR              = PROJECT_ROOT / "data" / "processed" / "pipeline_engineering"
FS_DIR              = PROJECT_ROOT / "data" / "processed" / "pipeline_selection"
OUTPUT_DIR          = PROJECT_ROOT / "output"
LOG_DIR             = PROJECT_ROOT / "mlops_canvas" / "logs"

for path in [MODELS_DIR, RAW_DIR, RAW_PARTITIONED_DIR, FE_DIR, FS_DIR, OUTPUT_DIR, LOG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 6) Pipeline principal
# -------------------------------------------------------------------------
def run() -> None:
    # [2/10] Ingesta
    print("[2/10] Generando y guardando datos sintéticos...", flush=True)
    df = generate_synthetic_patient_data()
    df.to_parquet(RAW_DIR / "df_raw.parquet")
    print(f"[2/10] Ingesta completada: {len(df):,} filas\n", flush=True)

    # [3/10] EDA univariado + Validación
    print("[3/10] Ejecutando EDA univariado y validación...", flush=True)
    eda_univar(df)
    if not validate_srv.run(df, fit_profile=True).valido:
        raise ValueError("Validación fallida.")
    print("[3/10] Datos validados correctamente.\n", flush=True)

    # [4/10] Particionado temporal
    print("[4/10] Particionando datos de forma temporal...", flush=True)
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
    print("[4/10] Particiones guardadas en disco.\n", flush=True)

    # [5/10] Feature Engineering
    print("[5/10] Ejecutando Feature Engineering...", flush=True)
    fe_run()
    print("[5/10] Feature Engineering completado.\n", flush=True)

    # [6/10] Feature Selection
    print("[6/10] Ejecutando Feature Selection (filter → frame)...", flush=True)
    X_tr_fs, X_te_fs, X_bk_fs, logs = fs_run(
        techniques=["filter", "frame"],
        save_logs=True
    )
    print(pd.DataFrame(logs), "\n", flush=True)
    lista_feats = [c for c in X_tr_fs.columns if c != "target"]
    lista_dir = OUTPUT_DIR / "Lista_feature_final"
    lista_dir.mkdir(parents=True, exist_ok=True)
    with open(lista_dir / "lista_feature_final.json", "w") as fh:
        json.dump(lista_feats, fh, indent=4)
    print("[6/10] Lista de features exportada.\n", flush=True)

    # [7/10] AutoML con FLAML
    print("[7/10] Ejecutando AutoML con FLAML...", flush=True)
    y_train = pd.read_parquet(FE_DIR / "X_train_processed.parquet").pop("target")
    y_test  = pd.read_parquet(FE_DIR / "X_test_processed.parquet").pop("target")
    automl = run_model_selector(
        X_train=X_tr_fs,
        y_train=y_train,
        X_test=X_te_fs,
        y_test=y_test,
        framework="flaml",
        task="regression",
        time_budget=300,
        metric="mae",
        log_file=str(LOG_DIR / "flaml.log"),
    )
    print("[7/10] AutoML finalizado. Ranking de los 5 mejores modelos:", flush=True)
    ranking = automl.get_model_ranking()
    print(ranking.head(5).to_string(index=False), "\n", flush=True)

    # Evaluamos y mostramos métricas del mejor modelo
    automl.evaluate(X_te_fs, y_test)
    print("Mejor modelo:", automl.name, flush=True)
    print("Métricas del mejor modelo:", automl.metrics, "\n", flush=True)

    # [8/10] Exportación a ONNX
    print("[8/10] Exportando modelo final a ONNX...", flush=True)
    export_model_onnx(
        model=automl.get_best_model(),
        X_sample=X_tr_fs,
        output_dir=MODELS_DIR,
        version="v1",
    )
    print("[8/10] Modelo ONNX exportado.\n", flush=True)

    # [9/10] Feature Reduction
    print("[9/10] Ejecutando Feature Reduction...", flush=True)
    transf_inicial = joblib.load(PROJECT_ROOT / "transformers" / "transformador_inicial.joblib")
    # ─── 1) intentar extraer MAE de metrics  ───
    # ─── 1) Intentar extraer MAE (sólo si existe la clave) ───
    mae_base: float | str | None = None
    if isinstance(getattr(automl, "metrics", None), dict):
        mae_base = automl.metrics.get("mae")      # puede ser None

    # ─── 2) Si no hay MAE y es REGRESIÓN, lo calculamos  ───
    if mae_base is None and getattr(automl, "task", "") == "regression":
        try:
            y_pred = automl.predict(X_te_fs)
            mae_base = mean_absolute_error(y_test, y_pred)
        except Exception:           # clasificador sin predict o problema distinto
            mae_base = "n/a"

    # Si sigue sin existir (ej. problema de clasificación), usamos 'n/a'
    if mae_base is None:
        mae_base = "n/a"
    
    fr_run(
        transformer=transf_inicial,
        features=lista_feats,
        params={"method": "pca", "n_components": 3},
        use_reduction=False,
        temporal_vars=["Semana"],
        target_var="target",
        mae_anterior=mae_base,      # ahora puede ser float ó 'n/a'
    )

    print("[9/10] Feature Reduction completado.\n", flush=True)

    # [10/10] Fin
    print("[10/10] Pipeline completado exitosamente.", flush=True)


if __name__ == "__main__":
    run()
