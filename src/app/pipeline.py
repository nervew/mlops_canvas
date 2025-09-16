# src/app/pipeline.py
from __future__ import annotations

import os, sys, json, warnings, subprocess
from pathlib import Path
import pandas as pd
import numpy as np

#-------------------- [1/10] DEPENDENCIAS --------------------
print("[1/10] Instalando dependencias...", flush=True)
from .m00_instalador import install_requirements
install_requirements()

# Asegura MLflow si el entorno no lo trae preinstalado
try:
    import mlflow  # noqa: F401
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mlflow>=3.0.0"])
import mlflow
print("[1/10] Dependencias instaladas.\n", flush=True)

# Warnings comunes de sklearn
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

# Backend headless para figuras
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#-------------------- IMPORTS DE TUS MÓDULOS --------------------
from .m02_eda_univariado import run as eda_univar
from .m03_data_validation.application import service as validate_srv
from .m04_data_split.infrastructure.robust_data_splitter import RobustDataSplitter
from .m05_feature_engineering.pipeline_engineering import run_pipeline as fe_run
from .m06__feature_selection import run_pipeline as fs_run
from .m08_feature_reduction.pipeline_reduction import run_feature_reduction as fr_run
from .search_model.run_model_selector import run_model_selector
from .search_model.export import export_model_onnx

#-------------------- RUTAS DEL PROYECTO --------------------
PROJECT_ROOT        = Path(__file__).resolve().parent.parent.parent
MODELS_DIR          = PROJECT_ROOT / "models"
RAW_DIR             = PROJECT_ROOT / "data" / "raw" / "complete"
RAW_PARTITIONED_DIR = PROJECT_ROOT / "data" / "raw" / "partitioned"
FE_DIR              = PROJECT_ROOT / "data" / "processed" / "pipeline_engineering"
FS_DIR              = PROJECT_ROOT / "data" / "processed" / "pipeline_selection"
OUTPUT_DIR          = PROJECT_ROOT / "output"
EDA_DIR             = OUTPUT_DIR / "reporte_eda"              # artefactos EDA (HTML + JSON)
LOG_DIR             = PROJECT_ROOT / "logs"

for path in [MODELS_DIR, RAW_DIR, RAW_PARTITIONED_DIR, FE_DIR, FS_DIR, OUTPUT_DIR, EDA_DIR, LOG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

#-------------------- MLflow: utilidades --------------------
EXPERIMENT_BASENAME = "mlflow_prueba_15"  # nombre de experimento pedido

def _current_dbx_user() -> str | None:
    """Obtiene el usuario de Databricks para formar la ruta absoluta del experimento."""
    if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
        return None
    try:
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate().sql("select current_user()").first()[0]
    except Exception:
        try:
            return dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()  # type: ignore # noqa
        except Exception:
            return None

def _resolve_experiment_path() -> str:
    """
    Databricks exige EXPERIMENTO como RUTA ABSOLUTA del workspace.
    - Si $MLFLOW_EXPERIMENT empieza con '/', lo respetamos.
    - Si no, se usa /Users/<user>/mlflow_prueba_15 o /Shared/mlflow_prueba_15.
    Fuera de Databricks: se usará tracking local (./mlruns) con nombre simple.
    """
    env_val = os.getenv("MLFLOW_EXPERIMENT")
    if env_val and env_val.startswith("/"):
        return env_val
    user = _current_dbx_user()
    return f"/Users/{user}/{EXPERIMENT_BASENAME}" if user else f"/Shared/{EXPERIMENT_BASENAME}"

def _setup_mlflow() -> str:
    """Configura MLflow según entorno y devuelve la ruta/nombre del experimento activo."""
    in_dbr = "DATABRICKS_RUNTIME_VERSION" in os.environ
    if in_dbr:
        exp_path = _resolve_experiment_path()
        mlflow.set_experiment(exp_path)
        return exp_path
    else:
        mlflow.set_tracking_uri("file:./mlruns")
        exp = mlflow.set_experiment(EXPERIMENT_BASENAME)
        return getattr(exp, "name", EXPERIMENT_BASENAME)

def _log_dict_as_json(d: dict, filepath: Path, *, artifact_subdir: str | None = None):
    """Guarda un dict en JSON (conversión numpy→Python) y lo sube a MLflow como artefacto."""
    def to_py(o):
        import numpy as _np
        if isinstance(o, dict): return {k: to_py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [to_py(v) for v in o]
        if isinstance(o, (_np.integer,)): return int(o)
        if isinstance(o, (_np.floating,)): return float(o)
        if isinstance(o, (_np.bool_,)): return bool(o)
        if hasattr(o, "tolist"): return o.tolist()
        return o
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(to_py(d), fh, ensure_ascii=False, indent=2)
    if artifact_subdir:
        mlflow.log_artifact(str(filepath), artifact_path=artifact_subdir)
    else:
        mlflow.log_artifact(str(filepath))

# ---------- helpers EDA (del primer pipeline) ----------
def _print_eda_quality_table(json_summary_path: Path, top: int = 30) -> None:
    """Imprime la tabla de 'calidad por columna' (dtype, %nulos, etc.) usando el JSON del EDA."""
    if not json_summary_path.exists():
        print(f"[3/10] (aviso) No existe {json_summary_path}, no hay tabla para imprimir.")
        return
    with open(json_summary_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    dq = payload.get("data_quality", {})
    per_col = pd.DataFrame.from_dict(dq.get("columns", {}), orient="index")
    cols_show = [c for c in ["dtype", "null_pct", "unique_pct", "duplicate_pct", "null_count", "distinct_count"]
                 if c in per_col.columns]
    print("\n[3/10] Calidad por columna (top por % nulos):")
    print(per_col.sort_values("null_pct", ascending=False)[cols_show].head(top).to_string())

def _fallback_build_summary_json(df: pd.DataFrame, out_json: Path) -> None:
    """Si el EDA no genera JSON, crea un resumen mínimo compatible para la tabla de calidad."""
    d = {}
    for c in df.columns:
        s = df[c]
        d[c] = {
            "dtype": str(s.dtype),
            "null_count": int(s.isna().sum()),
            "null_pct": float(s.isna().mean()*100.0),
            "distinct_count": int(s.nunique(dropna=True)),
            "unique_pct": float((s.nunique(dropna=True)/len(s))*100.0 if len(s) else 0.0),
            "duplicate_pct": float(((len(s)-s.nunique(dropna=False))/len(s))*100.0 if len(s) else 0.0),
        }
    payload = {"data_quality": {"columns": d, "n_rows": int(len(df))}}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

#-------------------- PIPELINE --------------------
def run() -> None:
    # Config MLflow (Databricks: ruta absoluta; local: ./mlruns)
    exp_path_or_name = _setup_mlflow()
    print(f"[MLflow] Experimento activo: {exp_path_or_name}", flush=True)

    # [2/10] Ingesta
    print("[2/10] Cargando datos crudos...", flush=True)
    out_path = RAW_DIR / "df_raw.parquet"
    df = pd.read_parquet(out_path)
    print(f"[2/10] Ingesta completada: {len(df):,} filas desde {out_path}\n", flush=True)

    # [3/10] EDA univariado + VALIDACIÓN (integración del primer pipeline)
    print("[3/10] Ejecutando EDA univariado y validación...", flush=True)
    sweetviz_path     = EDA_DIR / "sweetviz_univariate_report.html"
    json_summary_path = EDA_DIR / "univariate_summary.json"

    # Intento 1: EDA con firma "rica" (reporte_path/json_path/flags)
    try:
        _ = eda_univar(
            df,
            reporte_path=str(sweetviz_path),
            json_path=str(json_summary_path),
            print_max_rows=30,
            print_tables=False,   # no imprimir dentro del módulo
            log_artifacts=False,  # no loguear desde el módulo (lo haremos aquí)
        )
    except TypeError:
        # Intento 2: EDA con firma simple (si el módulo no soporta los kwargs anteriores)
        try:
            _ = eda_univar(df)
        except Exception:
            pass

    # Si no existe JSON, construimos uno mínimo para poder imprimir la tabla
    if not json_summary_path.exists():
        _fallback_build_summary_json(df, json_summary_path)

    # Imprime la misma tabla de calidad por columna que el primer pipeline
    _print_eda_quality_table(json_summary_path, top=30)

    # Validación de datos
    if not validate_srv.run(df, fit_profile=True).valido:
        raise ValueError("Validación fallida.")
    print("[3/10] Datos validados correctamente.\n", flush=True)

    # Run MLflow del EDA (artefactos HTML + JSON)
    with mlflow.start_run(run_name="eda_univariado", tags={"stage": "eda"}):
        if sweetviz_path.exists():
            mlflow.log_artifact(str(sweetviz_path), artifact_path="eda")
        mlflow.log_artifact(str(json_summary_path), artifact_path="eda")

    # [4/10] Split temporal + métricas + gráfico (y run MLflow de split)
    print("[4/10] Particionando datos (temporal)...", flush=True)
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

    # Gráfico de evolución del target por split
    for d in (tr_df, te_df, bk_df):
        d["Semana"] = pd.to_datetime(d["Semana"])
    plt.figure(figsize=(10, 6))
    plt.plot(tr_df["Semana"], tr_df["target"], label="Train")
    plt.plot(te_df["Semana"], te_df["target"], label="Test")
    plt.plot(bk_df["Semana"], bk_df["target"], label="Backtest")
    plt.xlabel("Semana"); plt.ylabel("Target"); plt.title("Evolución de la variable target por split")
    plt.legend(); plt.tight_layout()
    plot_path = OUTPUT_DIR / "target_splits.png"
    plt.savefig(plot_path); plt.close()
    print(f"[4.2/10] Diagrama guardado en {plot_path}\n", flush=True)

    # MLflow: run de "data_split"
    with mlflow.start_run(run_name="data_split", tags={"stage": "split"}):
        mlflow.log_metrics({
            "train_rows": float(len(tr_df)),
            "test_rows":  float(len(te_df)),
            "backtest_rows": float(len(bk_df)),
        })
        # Resumen PSI si existe
        try:
            psi = metrics["psi"] if "psi" in metrics else None
            if psi is not None:
                mlflow.log_metrics({
                    "psi_gt_0.10": float((psi > 0.10).sum()),
                    "psi_max": float(psi.max())
                })
        except Exception:
            pass
        mlflow.log_artifact(str(metrics_path), artifact_path="split")
        mlflow.log_artifact(str(plot_path), artifact_path="split")

    # [5/10] Feature Engineering
    print("[5/10] Ejecutando Feature Engineering...", flush=True)
    fe_run()
    print("[5/10] Feature Engineering completado.\n", flush=True)

    # (opcional informativo, como en tu 2º pipeline): vistazo a las primeras filas
    fe_train_path = FE_DIR / "X_train_processed.parquet"
    if fe_train_path.exists():
        fe_train_df = pd.read_parquet(fe_train_path)
        cols_to_show = [c for c in fe_train_df.columns if c != "Semana"]
        print("[5.1/10] Primeras 5 filas después de Feature Engineering:")
        print(fe_train_df[cols_to_show].head(5).to_string(index=False), "\n", flush=True)

    # [6/10] Feature Selection (filter → frame) y exporta lista_engineering.json
    print("[6/10] Ejecutando Feature Selection (filter → frame)...", flush=True)
    X_tr_fs, X_te_fs, X_bk_fs, logs = fs_run(techniques=["filter", "frame"], save_logs=True)
    print(pd.DataFrame(logs), "\n", flush=True)

    lista_dir = OUTPUT_DIR / "Lista_feature_final"
    lista_dir.mkdir(parents=True, exist_ok=True)
    lista_engineering_path = lista_dir / "lista_engineering.json"
    with open(lista_engineering_path, "w", encoding="utf-8") as fh:
        json.dump([c for c in X_tr_fs.columns if c != "target"], fh, indent=4)
    print("[6/10] Lista de ingeniería exportada: output/Lista_feature_final/lista_engineering.json\n", flush=True)

    # [7/10] AutoML con tu selector (FLAML por detrás)
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

    # Métricas manuales para logueo fiable
    best_estimator = automl.get_best_model()
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_pred = best_estimator.predict(X_te_fs)
    mae  = float(mean_absolute_error(y_test, y_pred))
    mse  = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_test, y_pred))

    # [8/10] Exporta modelo a ONNX
    print("[8/10] Exportando modelo final a ONNX...", flush=True)
    export_model_onnx(
        model=best_estimator,
        X_sample=X_tr_fs,
        output_dir=MODELS_DIR,
        version="v1",
    )
    onnx_path = MODELS_DIR / "modelo_v1.onnx"
    print("[8/10] Modelo ONNX exportado.\n", flush=True)

    # [9/10] Feature Reduction (m08) usando la lista de ingeniería
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
    print(f"  • reduction(head)       : {res.get('reduction_names_head')}\n")
    print("[9/10] Feature Reduction completado.\n", flush=True)

    # [10/10] Run final MLflow: métricas + artefactos clave (sin publicar a registry)
    with mlflow.start_run(run_name="features_and_model", tags={"stage": "fs_fr_automl"}):
        # a) métricas finales
        mlflow.log_metrics({"mae": mae, "mse": mse, "rmse": rmse, "r2": r2})
        # b) parámetros del mejor modelo (si FLAML los expone)
        best_params = dict(getattr(automl, "best_config", {}) or {})
        best_params["best_estimator"] = type(best_estimator).__name__
        mlflow.log_params({k: str(v) for k, v in best_params.items()})
        # c) artefactos relevantes
        mlflow.log_artifact(str(onnx_path), artifact_path="model_onnx")
        mlflow.log_artifact(str(lista_engineering_path), artifact_path="features")
        fe_out = LOG_DIR / "feature_names_out.json"
        if fe_out.exists():
            mlflow.log_artifact(str(fe_out), artifact_path="features")
        mlflow.log_artifact(str(plot_path), artifact_path="split")
        mlflow.log_artifact(str(metrics_path), artifact_path="split")
        # d) summary JSON con rutas y métricas
        summary_path = OUTPUT_DIR / "summary_pipeline.json"
        _log_dict_as_json(
            {
                "experiment": exp_path_or_name,
                "metrics": {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2},
                "best_estimator": type(best_estimator).__name__,
                "artifacts": {
                    "onnx": str(onnx_path),
                    "split_plot": str(plot_path),
                    "split_metrics_csv": str(metrics_path),
                    "lista_engineering_json": str(lista_engineering_path),
                    "lista_reduction_json": res.get("lista_reduction_path"),
                    "eda_html": str(EDA_DIR / "sweetviz_univariate_report.html"),
                    "eda_json": str(EDA_DIR / "univariate_summary.json"),
                },
            },
            summary_path,
            artifact_subdir="summary",
        )

    print("[10/10] Pipeline completado exitosamente.", flush=True)


if __name__ == "__main__":
    run()
