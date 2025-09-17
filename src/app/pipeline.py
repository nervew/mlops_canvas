# src/app/pipeline.py
from __future__ import annotations

import os
import json
import warnings
from pathlib import Path
import pandas as pd
import numpy as np

print("[1/10] Instalando dependencias...", flush=True)
from .m00_instalador import install_requirements
install_requirements()
print("[1/10] Dependencias instaladas.\n", flush=True)

from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

# Backend headless para gráficos
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Módulos del proyecto (sin m08 aún)
from .m02_eda_univariado import run as eda_univar
from .m03_data_validation.application import service as validate_srv
from .m04_data_split.infrastructure.robust_data_splitter import RobustDataSplitter
from .m05_feature_engineering.pipeline_engineering import run_pipeline as fe_run
from .m06__feature_selection import run_pipeline as fs_run
from .search_model.run_model_selector import run_model_selector

# Rutas
PROJECT_ROOT        = Path(__file__).resolve().parent.parent.parent
MODELS_DIR          = PROJECT_ROOT / "models"
RAW_DIR             = PROJECT_ROOT / "data" / "raw" / "complete"
RAW_PARTITIONED_DIR = PROJECT_ROOT / "data" / "raw" / "partitioned"
FE_DIR              = PROJECT_ROOT / "data" / "processed" / "pipeline_engineering"
FS_DIR              = PROJECT_ROOT / "data" / "processed" / "pipeline_selection"
OUTPUT_DIR          = PROJECT_ROOT / "output"
LOG_DIR             = PROJECT_ROOT / "logs"

for p in [MODELS_DIR, RAW_DIR, RAW_PARTITIONED_DIR, FE_DIR, FS_DIR, OUTPUT_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ---------- utilidades ----------
def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df

def _prepare_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fuerza columnas estándar: Semana (datetime) y target (desde usuarios_unicos)."""
    df = _strip_cols(df)

    # Semana
    if "Semana" not in df.columns:
        if "semana" in df.columns:
            df = df.rename(columns={"semana": "Semana"})
        else:
            raise ValueError(f"No existe columna temporal 'Semana' o 'semana'. Columnas: {list(df.columns)}")
    df["Semana"] = pd.to_datetime(df["Semana"], errors="coerce", infer_datetime_format=True)
    if df["Semana"].isna().all():
        raise ValueError("No se pudo convertir 'Semana' a datetime.")
    if df["Semana"].isna().any():
        n = int(df["Semana"].isna().sum())
        print(f"[info] Filas sin fecha válida en 'Semana': {n}. Se eliminarán.", flush=True)
        df = df[df["Semana"].notna()].copy()

    # Target = usuarios_unicos
    if "usuarios_unicos" not in df.columns:
        raise ValueError("No existe la columna 'usuarios_unicos' para usar como target.")
    if "target" not in df.columns:
        df = df.rename(columns={"usuarios_unicos": "target"})
        print("[2.1/10] Target fijada en 'usuarios_unicos' → renombrada a 'target'.", flush=True)
    elif "usuarios_unicos" in df.columns and "target" in df.columns and not df["target"].equals(df["usuarios_unicos"]):
        # coherencia si ya había un 'target' distinto
        df = df.drop(columns=["target"]).rename(columns={"usuarios_unicos": "target"})
        print("[2.1/10] 'target' existente reemplazado por 'usuarios_unicos'.", flush=True)

    return df

# --------------- pipeline ---------------
def run() -> None:
    # [2/10] Ingesta
    print("[2/10] Cargando datos crudos...", flush=True)
    raw_path = RAW_DIR / "df_raw.parquet"
    df = pd.read_parquet(raw_path)
    print(f"[2/10] Ingesta completada: {len(df):,} filas desde {raw_path}", flush=True)

    # Normalizar columnas clave
    df = _prepare_columns(df)

    # [3/10] EDA + Validación
    print("[3/10] EDA univariado y validación...", flush=True)
    try:
        eda_univar(df)
    except TypeError:
        eda_univar(df)
    if not validate_srv.run(df, fit_profile=True).valido:
        raise ValueError("Validación fallida.")
    print("[3/10] Datos validados correctamente.\n", flush=True)

    # [4/10] Split temporal
    print("[4/10] Particionando datos (temporal)...", flush=True)
    splitter = RobustDataSplitter(
        df, split_method="time", target_column="target", time_column="Semana",
        train_size=0.7, test_size=0.2, backtest_size=0.1,
    )
    tr_df, te_df, bk_df = splitter.split_data()
    tr_df.to_parquet(RAW_PARTITIONED_DIR / "train_df.parquet")
    te_df.to_parquet(RAW_PARTITIONED_DIR / "test_df.parquet")
    bk_df.to_parquet(RAW_PARTITIONED_DIR / "backtest_df.parquet")
    print("[4/10] Particiones guardadas.\n", flush=True)

    # [4.1/10] Métricas split
    metrics = splitter.calculate_metrics()
    metrics_path = LOG_DIR / "split_metrics.csv"
    metrics.to_csv(metrics_path, index=True)
    print(f"[4.1/10] Métricas de split:\n{metrics}\n", flush=True)

    # [4.2/10] Gráfico
    for d in (tr_df, te_df, bk_df):
        d["Semana"] = pd.to_datetime(d["Semana"])
    plt.figure(figsize=(10, 6))
    plt.plot(tr_df["Semana"], tr_df["target"], label="Train")
    plt.plot(te_df["Semana"], te_df["target"], label="Test")
    plt.plot(bk_df["Semana"], bk_df["target"], label="Backtest")
    plt.xlabel("Semana"); plt.ylabel("Target"); plt.title("Target por split")
    plt.legend(); plt.tight_layout()
    plot_path = OUTPUT_DIR / "target_splits.png"
    plt.savefig(plot_path); plt.close()
    print(f"[4.2/10] Diagrama guardado en {plot_path}\n", flush=True)

    # [5/10] Feature Engineering
    print("[5/10] Ejecutando Feature Engineering...", flush=True)
    fe_run()
    print("[5/10] Feature Engineering completado.\n", flush=True)

    # [6/10] Feature Selection
    print("[6/10] Ejecutando Feature Selection (filter → frame)...", flush=True)
    X_tr_fs, X_te_fs, X_bk_fs, logs = fs_run(techniques=["filter", "frame"], save_logs=True)
    print(pd.DataFrame(logs), "\n", flush=True)

    # Exporta lista_engineering.json
    lista_dir = OUTPUT_DIR / "Lista_feature_final"
    lista_dir.mkdir(parents=True, exist_ok=True)
    lista_engineering_path = lista_dir / "lista_engineering.json"
    with open(lista_engineering_path, "w", encoding="utf-8") as fh:
        json.dump([c for c in X_tr_fs.columns if c != "target"], fh, indent=4)
    print("[6/10] Lista exportada: output/Lista_feature_final/lista_engineering.json\n", flush=True)

    # [7/10] AutoML con FLAML
    print("[7/10] AutoML (FLAML)...", flush=True)
    y_train = pd.read_parquet(FE_DIR / "X_train_processed.parquet").pop("target")
    y_test  = pd.read_parquet(FE_DIR / "X_test_processed.parquet").pop("target")
    automl = run_model_selector(
        X_train=X_tr_fs, y_train=y_train, X_test=X_te_fs, y_test=y_test,
        framework="flaml", task="regression", time_budget=300, metric="mae",
        log_file=str(LOG_DIR / "flaml.log"),
    )
    ranking = automl.get_model_ranking()
    print("[7/10] Top modelos:\n" + ranking.head(5).to_string(index=False) + "\n", flush=True)
    automl.evaluate(X_te_fs, y_test)
    print("Mejor modelo:", automl.name, flush=True)
    print("Métricas del mejor modelo:", automl.metrics, "\n", flush=True)

    # [8/10] Exportación ONNX
    print("[8/10] Exportando modelo a ONNX...", flush=True)
    from .search_model.export import export_model_onnx
    export_model_onnx(
        model=automl.get_best_model(),
        X_sample=X_tr_fs,
        output_dir=MODELS_DIR,
        version="v1",
    )
    onnx_path = MODELS_DIR / "modelo_v1.onnx"
    print("[8/10] Modelo ONNX exportado.\n", flush=True)

    # [9/10] Feature Reduction (m08) — importar aquí
    from .m08_feature_reduction.pipeline_reduction import run_feature_reduction as fr_run
    print("[9/10] Ejecutando Feature Reduction...", flush=True)
    res = fr_run(
        transformer_inicial_onnx_path="transformers/transformador_inicial.onnx",
        modelo_onnx_path=str(onnx_path),
        feature_names_out_path="logs/feature_names_out.json",
        lista_features_global_path=str(lista_engineering_path),
        lista_features_m08_path="output/Lista_feature_final/lista_reduction.json",
        target_var="target",
        verbose=True,
    )
    print("[9/10] Resumen reducción:")
    print(f"  • lista_engineering_path: {res.get('lista_engineering_path')}")
    print(f"  • lista_reduction_path  : {res.get('lista_reduction_path')}")
    print(f"  • n_selected            : {res.get('n_selected')}")
    print(f"  • temporal_candidates   : {res.get('temporal_candidates')}")
    print(f"  • reduction(head)       : {res.get('reduction_names_head')}\n", flush=True)

    print("[10/10] Pipeline completado exitosamente.", flush=True)

if __name__ == "__main__":
    run()
