# src/app/pipeline.py
from __future__ import annotations

import os
import json
import warnings
from pathlib import Path
from typing import Tuple

import pandas as pd

# [1] Instalar dependencias antes de importar stack científico
print("[1/10] Instalando dependencias...", flush=True)
from .m00_instalador import install_requirements
install_requirements()
print("[1/10] Dependencias instaladas.\n", flush=True)

# A partir de aquí ya podemos importar con seguridad
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.simplefilter("ignore", ConvergenceWarning)

# Backend “headless” para guardar figuras
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Módulos internos del proyecto
from .m02_eda_univariado import run as eda_univar
from .m03_data_validation.application import service as validate_srv
from .m04_data_split.infrastructure.robust_data_splitter import RobustDataSplitter
from .m05_feature_engineering.pipeline_engineering import run_pipeline as fe_run
from .m06__feature_selection import run_pipeline as fs_run
from .m08_feature_reduction.pipeline_reduction import run_feature_reduction as fr_run
from .search_model.export import export_model_onnx

# ───────────────── Configuración general ─────────────────
PROJECT_ROOT        = Path(__file__).resolve().parent.parent.parent
MODELS_DIR          = PROJECT_ROOT / "models"
RAW_DIR             = PROJECT_ROOT / "data" / "raw" / "complete"
RAW_PARTITIONED_DIR = PROJECT_ROOT / "data" / "raw" / "partitioned"
FE_DIR              = PROJECT_ROOT / "data" / "processed" / "pipeline_engineering"
FS_DIR              = PROJECT_ROOT / "data" / "processed" / "pipeline_selection"
OUTPUT_DIR          = PROJECT_ROOT / "output"
LOG_DIR             = PROJECT_ROOT / "logs"

# Ajusta estas dos si tu dataset cambia
TIME_COLUMN_SOURCE  = "semana"         # como viene en df_raw.parquet
TIME_COLUMN_STD     = "Semana"         # nombre “estándar” que usarán los módulos
TARGET_SOURCE       = "usuarios_eps"   # de tus columnas originales
TARGET_STD          = "target"         # nombre “estándar”

for path in [MODELS_DIR, RAW_DIR, RAW_PARTITIONED_DIR, FE_DIR, FS_DIR, OUTPUT_DIR, LOG_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def _load_raw() -> pd.DataFrame:
    """Lee df_raw.parquet de data/raw/complete y normaliza nombres/columnas clave."""
    in_path = RAW_DIR / "df_raw.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"No existe {in_path}.")
    df = pd.read_parquet(in_path)

    # Normalizar nombres claves (Semana, target)
    cols = {c.lower(): c for c in df.columns}
    if TIME_COLUMN_SOURCE in cols:
        # Si existe en minúsculas, renombrar a estándar
        real = cols[TIME_COLUMN_SOURCE]
        if real != TIME_COLUMN_STD:
            df = df.rename(columns={real: TIME_COLUMN_STD})
    elif TIME_COLUMN_STD not in df.columns:
        raise KeyError(f"Falta columna temporal '{TIME_COLUMN_SOURCE}' o '{TIME_COLUMN_STD}'.")

    if TARGET_SOURCE in cols:
        real = cols[TARGET_SOURCE]
        if TARGET_STD not in df.columns:
            df[TARGET_STD] = df[real]
    elif TARGET_STD not in df.columns:
        # Si no existe, intenta heurística: usa la última columna numérica
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise KeyError(f"No se encontró una columna numérica para usar como {TARGET_STD}.")
        df[TARGET_STD] = df[num_cols[-1]]

    # Tipos
    df[TIME_COLUMN_STD] = pd.to_datetime(df[TIME_COLUMN_STD], errors="coerce")
    if df[TIME_COLUMN_STD].isna().all():
        raise ValueError(f"La columna '{TIME_COLUMN_STD}' no pudo parsearse a datetime.")

    return df


def _plot_target_splits(tr_df: pd.DataFrame, te_df: pd.DataFrame, bk_df: pd.DataFrame, path: Path) -> None:
    for d in (tr_df, te_df, bk_df):
        if TIME_COLUMN_STD in d.columns:
            d[TIME_COLUMN_STD] = pd.to_datetime(d[TIME_COLUMN_STD])
    plt.figure(figsize=(10, 6))
    if TIME_COLUMN_STD in tr_df and TARGET_STD in tr_df:
        plt.plot(tr_df[TIME_COLUMN_STD], tr_df[TARGET_STD], label="Train")
    if TIME_COLUMN_STD in te_df and TARGET_STD in te_df:
        plt.plot(te_df[TIME_COLUMN_STD], te_df[TARGET_STD], label="Test")
    if TIME_COLUMN_STD in bk_df and TARGET_STD in bk_df:
        plt.plot(bk_df[TIME_COLUMN_STD], bk_df[TARGET_STD], label="Backtest")
    plt.xlabel(TIME_COLUMN_STD); plt.ylabel(TARGET_STD); plt.title("Evolución del target por split")
    plt.legend(); plt.tight_layout()
    plt.savefig(path); plt.close()


def _simple_model_search(X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series):
    """Pequeño selector de modelo sin MLflow/FLAML: elige el menor MAE."""
    candidates = {
        "ridge": Ridge(alpha=1.0, random_state=42),
        "rf": RandomForestRegressor(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
        ),
        "lgbm": LGBMRegressor(
            n_estimators=500, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8,
            random_state=42
        ),
        "xgb": XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.8, objective="reg:squarederror", n_jobs=-1, random_state=42
        ),
    }

    results = []
    best_name, best_model, best_mae = None, None, float("inf")
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, pred))
        results.append((name, mae))
        if mae < best_mae:
            best_name, best_model, best_mae = name, model, mae

    print("[7/10] Ranking simple (menor MAE mejor):", flush=True)
    for name, mae in sorted(results, key=lambda x: x[1]):
        print(f"  - {name:>5s} : MAE={mae:.6f}")
    print(f"Mejor modelo: {best_name} | MAE={best_mae:.6f}\n", flush=True)

    # Interfaz “similar” a la que esperaba el resto del pipeline
    class _Best:
        def __init__(self, name, model, mae):
            self.name = name
            self._model = model
            self.metrics = {"mae": mae}
        def get_best_model(self):
            return self._model

    return _Best(best_name, best_model, best_mae)


def run() -> None:
    # [2] Ingesta
    print("[2/10] Cargando datos crudos...", flush=True)
    df = _load_raw()
    print(f"[2/10] Ingesta completada: {len(df):,} filas de {RAW_DIR/'df_raw.parquet'}\n", flush=True)

    # [3] EDA + validación
    print("[3/10] Ejecutando EDA univariado y validación...", flush=True)
    eda_univar(df)
    if not validate_srv.run(df, fit_profile=True).valido:
        raise ValueError("Validación fallida.")
    print("[3/10] Datos validados correctamente.\n", flush=True)

    # [4] Particionado temporal
    print("[4/10] Particionando datos de forma temporal...", flush=True)
    splitter = RobustDataSplitter(
        df, split_method="time", target_column=TARGET_STD, time_column=TIME_COLUMN_STD,
        train_size=0.7, test_size=0.2, backtest_size=0.1,
    )
    tr_df, te_df, bk_df = splitter.split_data()
    tr_df.to_parquet(RAW_PARTITIONED_DIR / "train_df.parquet")
    te_df.to_parquet(RAW_PARTITIONED_DIR / "test_df.parquet")
    bk_df.to_parquet(RAW_PARTITIONED_DIR / "backtest_df.parquet")
    print("[4/10] Particiones guardadas en disco.\n", flush=True)

    metrics = splitter.calculate_metrics()
    metrics_path = LOG_DIR / "split_metrics.csv"
    metrics.to_csv(metrics_path, index=True)
    print(f"[4.1/10] Métricas guardadas en {metrics_path}:\n{metrics}\n", flush=True)

    plot_path = OUTPUT_DIR / "target_splits.png"
    _plot_target_splits(tr_df, te_df, bk_df, plot_path)
    print(f"[4.2/10] Diagrama guardado en {plot_path}\n", flush=True)

    # [5] Feature engineering (usa particiones recién guardadas)
    print("[5/10] Ejecutando Feature Engineering...", flush=True)
    fe_run()
    print("[5/10] Feature Engineering completado.\n", flush=True)

    # [6] Feature selection
    print("[6/10] Ejecutando Feature Selection (filter → frame)...", flush=True)
    X_tr_fs, X_te_fs, X_bk_fs, logs = fs_run(techniques=["filter", "frame"], save_logs=True)
    print(pd.DataFrame(logs), "\n", flush=True)

    # Exportar lista de features (por nombre)
    lista_feats = [c for c in X_tr_fs.columns if c != TARGET_STD]
    lista_dir = OUTPUT_DIR / "Lista_feature_final"
    lista_dir.mkdir(parents=True, exist_ok=True)
    with open(lista_dir / "lista_engineering.json", "w", encoding="utf-8") as fh:
        json.dump(lista_feats, fh, indent=4)
    print("[6/10] Lista de ingeniería exportada: output/Lista_feature_final/lista_engineering.json\n", flush=True)

    # [7] “AutoML” simple sin mlflow
    print("[7/10] Buscando mejor modelo (sin MLflow/FLAML)...", flush=True)
    # El FE guarda X_train_processed/X_test_processed con 'target' dentro.
    X_tr_proc = pd.read_parquet(FE_DIR / "X_train_processed.parquet")
    X_te_proc = pd.read_parquet(FE_DIR / "X_test_processed.parquet")
    y_train = X_tr_proc.pop(TARGET_STD)
    y_test  = X_te_proc.pop(TARGET_STD)

    # Alinear columnas por seguridad
    X_tr_proc, X_te_proc = X_tr_proc.align(X_te_proc, join="left", axis=1, fill_value=0.0)

    automl = _simple_model_search(X_tr_proc, y_train, X_te_proc, y_test)

    # [8] Exportar a ONNX
    print("[8/10] Exportando modelo final a ONNX...", flush=True)
    export_model_onnx(
        model=automl.get_best_model(),
        X_sample=X_tr_proc,           # matriz de entrenamiento (sin target)
        output_dir=MODELS_DIR,
        version="v1",
    )
    print("[8/10] Modelo ONNX exportado.\n", flush=True)

    # [9] Reducción de features (m08) — ahora sí existen particiones y modelo/transformador
    print("[9/10] Ejecutando Feature Reduction...", flush=True)
    res = fr_run(
        transformer_inicial_onnx_path="transformers/transformador_inicial.onnx",
        modelo_onnx_path="models/modelo_v1.onnx",
        feature_names_out_path="logs/feature_names_out.json",
        lista_features_global_path="output/Lista_feature_final/lista_engineering.json",
        lista_features_m08_path="output/Lista_feature_final/lista_reduction.json",
        target_var=TARGET_STD,
        verbose=True,
    )
    print("[9/10] Resumen reducción:")
    print(f"  • lista_engineering_path: {res.get('lista_engineering_path')}")
    print(f"  • lista_reduction_path  : {res.get('lista_reduction_path')}")
    print(f"  • n_selected            : {res.get('n_selected')}")
    print(f"  • temporal_candidates   : {res.get('temporal_candidates')}")
    print(f"  • reduction(head)       : {res.get('reduction_names_head')}\n")
    print("[9/10] Feature Reduction completado.\n", flush=True)

    print("[10/10] Pipeline completado exitosamente.", flush=True)


if __name__ == "__main__":
    run()
