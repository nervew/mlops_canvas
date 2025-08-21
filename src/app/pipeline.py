from __future__ import annotations

import os
import json
import warnings
from pathlib import Path
import pandas as pd

print("[1/10] Instalando dependencias...", flush=True)
from .m00_instalador import install_requirements
install_requirements()
print("[1/10] Dependencias instaladas.\n", flush=True)

from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
#from .d_database import generate_synthetic_patient_data
from .m02_eda_univariado import run as eda_univar
from .m03_data_validation.application import service as validate_srv
from .m04_data_split.infrastructure.robust_data_splitter import RobustDataSplitter
from .m05_feature_engineering.pipeline_engineering import run_pipeline as fe_run
from .m06__feature_selection import run_pipeline as fs_run
from .m08_feature_reduction.pipeline_reduction import run_feature_reduction as fr_run
from .search_model.run_model_selector import run_model_selector
from .search_model.export import export_model_onnx
from mlops.mlflow_helper import MLflowHelper

PROJECT_ROOT        = Path(__file__).resolve().parent.parent.parent
MODELS_DIR          = PROJECT_ROOT / "models"
RAW_DIR             = PROJECT_ROOT / "data" / "raw" / "complete"
RAW_PARTITIONED_DIR = PROJECT_ROOT / "data" / "raw" / "partitioned"
FE_DIR              = PROJECT_ROOT / "data" / "processed" / "pipeline_engineering"
FS_DIR              = PROJECT_ROOT / "data" / "processed" / "pipeline_selection"
OUTPUT_DIR          = PROJECT_ROOT / "output"
LOG_DIR             = PROJECT_ROOT / "logs"

for path in [MODELS_DIR, RAW_DIR, RAW_PARTITIONED_DIR, FE_DIR, FS_DIR, OUTPUT_DIR, LOG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

def run() -> None:
    print("[2/10] Generando y guardando datos...", flush=True)
    #df = generate_synthetic_patient_data()
    out_path = RAW_DIR / "df_raw.parquet"
    df = pd.read_parquet(out_path)
    #df.to_parquet(out_path, index=False)
    print(f"[2/10] Ingesta completada: {len(df):,} filas guardadas en {out_path}\n", flush=True)

    print("[3/10] Ejecutando EDA univariado y validación...", flush=True)
    eda_univar(df)
    if not validate_srv.run(df, fit_profile=True).valido:
        raise ValueError("Validación fallida.")
    print("[3/10] Datos validados correctamente.\n", flush=True)

    print("[4/10] Particionando datos de forma temporal...", flush=True)
    splitter = RobustDataSplitter(
        df, split_method="time", target_column="target", time_column="Semana",
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
    print(f"[4.1/10] Métricas guardadas en {metrics_path}:\n", metrics, "\n", flush=True)

    helper = MLflowHelper(
        experiment_name=os.getenv("MLFLOW_EXPERIMENT", "Default"),
        default_tags={"component": "data_split"},
    )
    helper.configure()
    with helper.run(run_name="data_split", tags={"stage": "split"}):
        helper.log_metrics({"train_rows": len(tr_df), "test_rows": len(te_df)})
        helper.log_artifacts(str(metrics_path.parent), artifact_path="split")

    for df_split in (tr_df, te_df, bk_df):
        df_split["Semana"] = pd.to_datetime(df_split["Semana"])
    plt.figure(figsize=(10, 6))
    plt.plot(tr_df["Semana"], tr_df["target"], label="Train")
    plt.plot(te_df["Semana"], te_df["target"], label="Test")
    plt.plot(bk_df["Semana"], bk_df["target"], label="Backtest")
    plt.xlabel("Semana"); plt.ylabel("Target"); plt.title("Evolución de la variable target por split")
    plt.legend(); plt.tight_layout()
    plot_path = OUTPUT_DIR / "target_splits.png"
    plt.savefig(plot_path); plt.close()
    print(f"[4.2/10] Diagrama guardado en {plot_path}\n", flush=True)

    print("[5/10] Ejecutando Feature Engineering...", flush=True)
    fe_run()
    print("[5/10] Feature Engineering completado.\n", flush=True)

    print("[6/10] Ejecutando Feature Selection (filter → frame)...", flush=True)
    X_tr_fs, X_te_fs, X_bk_fs, logs = fs_run(techniques=["filter", "frame"], save_logs=True)
    print(pd.DataFrame(logs), "\n", flush=True)

    # ⟶ CAMBIO: exportamos lista_engineering.json
    lista_feats = [c for c in X_tr_fs.columns if c != "target"]
    lista_dir = OUTPUT_DIR / "Lista_feature_final"
    lista_dir.mkdir(parents=True, exist_ok=True)
    with open(lista_dir / "lista_engineering.json", "w") as fh:
        json.dump(lista_feats, fh, indent=4)
    print("[6/10] Lista de ingeniería exportada: output/Lista_feature_final/lista_engineering.json\n", flush=True)

    print("[7/10] Ejecutando AutoML con FLAML...", flush=True)
    y_train = pd.read_parquet(FE_DIR / "X_train_processed.parquet").pop("target")
    y_test  = pd.read_parquet(FE_DIR / "X_test_processed.parquet").pop("target")
    automl = run_model_selector(
        X_train=X_tr_fs, y_train=y_train, X_test=X_te_fs, y_test=y_test,
        framework="flaml", task="regression", time_budget=300, metric="mae",
        log_file=str(LOG_DIR / "flaml.log"),
    )
    print("[7/10] AutoML finalizado. Ranking de los 5 mejores modelos:", flush=True)
    ranking = automl.get_model_ranking()
    print(ranking.head(5).to_string(index=False), "\n", flush=True)

    automl.evaluate(X_te_fs, y_test)
    print("Mejor modelo:", automl.name, flush=True)
    print("Métricas del mejor modelo:", automl.metrics, "\n", flush=True)

    print("[8/10] Exportando modelo final a ONNX...", flush=True)
    from .search_model.export import export_model_onnx
    export_model_onnx(
        model=automl.get_best_model(),
        X_sample=X_tr_fs,
        output_dir=MODELS_DIR,
        version="v1",
    )
    print("[8/10] Modelo ONNX exportado.\n", flush=True)

    print("[9/10] Ejecutando Feature Reduction...", flush=True)
    res = fr_run(
        transformer_inicial_onnx_path="transformers/transformador_inicial.onnx",
        modelo_onnx_path="models/modelo_v1.onnx",
        feature_names_out_path="logs/feature_names_out.json",
        lista_features_global_path="output/Lista_feature_final/lista_engineering.json",  # ⟶ CAMBIO
        lista_features_m08_path="output/Lista_feature_final/lista_reduction.json",      # lectura previa si existiera
        target_var="target",
        verbose=True,
    )
    # Mostrar en pantalla resultados de m08
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
