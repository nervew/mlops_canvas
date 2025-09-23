# src/app/pipeline2.py
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

# ─────────────────────────── Configuración adaptable ───────────────────────────
TIME_COLUMN  = "semana"                     # nombre “estándar” temporal
TARGET_ALIAS = "target"                     # alias estándar de target
OLD_TARGET   = "y_usuarios_nuevos_semana"   # nombre antiguo en datasets reales

# Semilla para FIJAR la generación de datos sintéticos
DATA_SEED = 8729

# ────────────────────────────── Rutas de proyecto ──────────────────────────────
# …/mlops_canvas/src/app/pipeline2.py → parents[2] = …/mlops_canvas
PROJECT_ROOT        = Path(__file__).resolve().parents[2]
MODELS_DIR          = PROJECT_ROOT / "models"
RAW_DIR             = PROJECT_ROOT / "data" / "raw" / "complete"
RAW_PARTITIONED_DIR = PROJECT_ROOT / "data" / "raw" / "partitioned"
FE_DIR              = PROJECT_ROOT / "data" / "processed" / "pipeline_engineering"
FS_DIR              = PROJECT_ROOT / "data" / "processed" / "pipeline_selection"
OUTPUT_DIR          = PROJECT_ROOT / "output"
LOG_DIR             = PROJECT_ROOT / "logs"

for p in [MODELS_DIR, RAW_DIR, RAW_PARTITIONED_DIR, FE_DIR, FS_DIR, OUTPUT_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ────────────────────────────── Utilidades de log ─────────────────────────────
def banner(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78 + "\n", flush=True)

def step(n: str, msg: str) -> None:
    print(f"[{n}] {msg}", flush=True)

# ─────────────────────── Reproducibilidad de datos (SEED) ─────────────────────
def _seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

# ─────────────────────────── Instalación (m00) ────────────────────────────────
banner("PIPELINE2 · m00 → m01 → m02 → m03 → m04 → m05 → m06 → m07")
banner("MÓDULO m00 • Instalación de dependencias")
from .m00_instalador.service import install_requirements
install_requirements("requirements.txt")
print("m00 ✓ Dependencias listas\n", flush=True)

# A partir de aquí podemos importar con seguridad
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .m02_eda_univariado import run as m02_run
from .m03_data_validation.application import service as validate_srv
from .m04_data_split.infrastructure.robust_data_splitter import RobustDataSplitter
from .m05_feature_engineering.pipeline_engineering import run_pipeline as fe_run
from .m06__feature_selection import run_pipeline as fs_run

from .search_model.run_model_selector import run_model_selector
from .search_model.export import export_model_onnx

# ─────────────────────────── Helpers de ingesta (m01) ─────────────────────────
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c.lower(): c for c in df.columns}
    if OLD_TARGET.lower() in cols_lower:
        old_real = cols_lower[OLD_TARGET.lower()]
        if TARGET_ALIAS not in df.columns:
            df = df.rename(columns={old_real: TARGET_ALIAS})
        elif old_real != TARGET_ALIAS:
            df = df.drop(columns=[old_real], errors="ignore")
    if TIME_COLUMN.lower() in cols_lower:
        real_time = cols_lower[TIME_COLUMN.lower()]
        if real_time != TIME_COLUMN:
            df = df.rename(columns={real_time: TIME_COLUMN})
    if TIME_COLUMN in df.columns:
        df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
    return df

def _load_dataset() -> pd.DataFrame:
    from .d_database import generate_synthetic_patient_data
    step("1.0", f"Fijando semilla de datos: DATA_SEED={DATA_SEED}")
    _seed_everything(DATA_SEED)

    step("1.1", "Generando datos sintéticos (d_database.generate_synthetic_patient_data)…")
    df = generate_synthetic_patient_data()

    step("1.2", f"Columna temporal detectada: '{TIME_COLUMN}' → datetime OK")
    df = _normalize_columns(df)

    # Forzar target ENTERO (nuevos usuarios)
    if TARGET_ALIAS in df.columns:
        df[TARGET_ALIAS] = (
            pd.to_numeric(df[TARGET_ALIAS], errors="coerce")
              .fillna(0).round().astype(int)
        )

    step("1.3", f"Dataset leído con shape: {len(df)} filas × {df.shape[1]} columnas")
    print(df.head(3).to_string(index=False))
    return df

# ──────────────────────── Predicción robusta (evita mismatch) ─────────────────
def _ensure_same_columns(model, X: pd.DataFrame) -> pd.DataFrame:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        missing = [c for c in names if c not in X.columns]
        for c in missing:
            X[c] = 0.0
        X = X.loc[:, list(names)]
    return X

# ──────────────────────── ONNX: exportación (solo modelo ganador) ─────────────
def _export_onnx_selected_only(sk_model, X_sample: pd.DataFrame, out_dir: Path, version: str = "v1") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"modelo_{version}.onnx"
    try:
        export_model_onnx(sk_model, X_sample, out_dir, version=version)
        if model_path.exists():
            return model_path
    except Exception as e:
        print(f"  [onnx] Helper export_model_onnx falló: {e}")
    try:
        import xgboost as xgb  # noqa
        from onnxmltools.convert import convert_xgboost
        from skl2onnx.common.data_types import FloatTensorType
        onx = convert_xgboost(sk_model, initial_types=[("input", FloatTensorType([None, X_sample.shape[1]]))])
        with open(model_path, "wb") as f:
            f.write(onx.SerializeToString())
        print("  ✓ ONNX exportado con onnxmltools.convert_xgboost")
        return model_path
    except Exception:
        pass
    try:
        import lightgbm as lgb  # noqa
        from onnxmltools.convert import convert_lightgbm
        from skl2onnx.common.data_types import FloatTensorType
        onx = convert_lightgbm(sk_model, initial_types=[("input", FloatTensorType([None, X_sample.shape[1]]))])
        with open(model_path, "wb") as f:
            f.write(onx.SerializeToString())
        print("  ✓ ONNX exportado con onnxmltools.convert_lightgbm")
        return model_path
    except Exception:
        pass
    try:
        from catboost import CatBoostRegressor
        if isinstance(sk_model, CatBoostRegressor):
            sk_model.save_model(str(model_path), format="onnx")
            print("  ✓ ONNX exportado con CatBoost.save_model(format='onnx')")
            return model_path
    except Exception:
        pass
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        for opset in (17, 16, 15, 14):
            try:
                onx = convert_sklearn(
                    sk_model,
                    initial_types=[("input", FloatTensorType([None, X_sample.shape[1]]))],
                    target_opset=opset,
                )
                with open(model_path, "wb") as f:
                    f.write(onx.SerializeToString())
                print(f"  ✓ ONNX exportado con skl2onnx (opset={opset})")
                return model_path
            except Exception as e2:
                print(f"  [onnx] skl2onnx (opset={opset}) falló: {e2}")
    except Exception as e:
        print(f"  [onnx] skl2onnx no disponible/compatible: {e}")
    raise RuntimeError("No fue posible convertir el modelo seleccionado a ONNX.")

# ─────────────────────────────── m07: ONNX + Plot ─────────────────────────────
def _save_features_used(features: List[str]) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    f = LOG_DIR / "model_features_used.json"
    with open(f, "w", encoding="utf-8") as fh:
        json.dump({"expected_n": len(features), "columns": features}, fh, indent=2)
    return f

def _plot_real_vs_pred(
    time_test: pd.Series, y_test: pd.Series, y_pred_test,
    time_back: pd.Series, y_back: pd.Series, y_pred_back,
    out_path: Path,
) -> None:
    """
    Versión robusta: usa Series con índice datetime y ordena por fecha.
    Evita líneas horizontales si matplotlib/pandas colapsan arrays sin índice.
    """
    # Asegurar tipo datetime
    t_test = pd.to_datetime(time_test)
    t_back = pd.to_datetime(time_back)

    # Series indexadas por fecha
    real_t  = pd.Series(pd.to_numeric(y_test, errors="coerce").values, index=t_test)
    real_b  = pd.Series(pd.to_numeric(y_back, errors="coerce").values, index=t_back)
    pred_t  = pd.Series(pd.to_numeric(pd.Series(y_pred_test), errors="coerce").values, index=t_test)
    pred_b  = pd.Series(pd.to_numeric(pd.Series(y_pred_back), errors="coerce").values, index=t_back)

    # Orden cronológico
    real_all = pd.concat([real_t, real_b]).sort_index()
    pred_t = pred_t.sort_index()
    pred_b = pred_b.sort_index()

    plt.figure(figsize=(14, 6))
    plt.plot(real_all.index, real_all.values, marker="o", linestyle="-", label="Real (Test + Backtest)")
    plt.plot(pred_t.index,  pred_t.values,  marker="x", linestyle="--", label="Pronóstico (Test)")
    plt.plot(pred_b.index,  pred_b.values,  marker="x", linestyle="--", label="Pronóstico (Backtest)")
    plt.xlabel("Fecha"); plt.ylabel(TARGET_ALIAS)
    plt.title("Reales vs. Pronóstico (modelo) — Test y Backtest")
    plt.grid(True, alpha=0.25); plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def _print_pred_stats(name: str, y_pred) -> None:
    import numpy as np
    arr = pd.Series(y_pred).astype(float).to_numpy()
    nuniq = int(len(pd.unique(arr.round(6))))
    print(f"  [{name}] min={arr.min():.3f}  max={arr.max():.3f}  std={arr.std():.3f}  n_unique≈{nuniq}")

# ─────────────────────────────────── Main ─────────────────────────────────────
def main() -> None:
    # m01 · Ingesta
    banner("MÓDULO m01 • Ingesta (importación del dataset)")
    df = _load_dataset()

    if TIME_COLUMN in df.columns:
        pre = len(df)
        df = df[df[TIME_COLUMN] >= "2022-01-01"].reset_index(drop=True)
        step("1.4", f"Filtro post-2022: {pre} → {len(df)} filas")

    step("1.5", f"Target = '{TARGET_ALIAS}' | Time = '{TIME_COLUMN}'")
    cols_front = [c for c in [TIME_COLUMN, TARGET_ALIAS] if c in df.columns]
    other_cols = [col for col in df.columns if col not in (TIME_COLUMN, TARGET_ALIAS)]
    print(df[cols_front + other_cols[:6]].head(3).to_string())

    # m02 · EDA
    banner("MÓDULO m02 • EDA univariado"); m02_run(df); print("m02 ✓ EDA OK.", flush=True)

    # m03 · Data validation
    banner("MÓDULO m03 • Data validation")
    res = validate_srv.run(df, fit_profile=True)
    if not getattr(res, "valido", True):
        raise RuntimeError("m03 × Validación fallida")
    print("m03 ✓ Perfil y reporte en output/validation\n", flush=True)

    # m04 · Split temporal
    banner("MÓDULO m04 • Split temporal")
    splitter = RobustDataSplitter(
        df, split_method="time", target_column=TARGET_ALIAS, time_column=TIME_COLUMN,
        train_size=0.7, test_size=0.2, backtest_size=0.1,
    )
    train_df, test_df, back_df = splitter.split_data()
    train_df.to_parquet(RAW_PARTITIONED_DIR / "train_df.parquet")
    test_df.to_parquet(RAW_PARTITIONED_DIR / "test_df.parquet")
    back_df.to_parquet(RAW_PARTITIONED_DIR / "backtest_df.parquet")
    print("m04 ✓ Particiones guardadas en data/raw/partitioned/\n", flush=True)

    # m05 · Feature engineering
    banner("MÓDULO m05 • Feature engineering")
    fe_run()
    print("m05 ✓ Exportados: X_train_processed.parquet, X_test_processed.parquet, X_backtest_processed.parquet", flush=True)
    print("m05.3) También se exporta el transformador inicial ONNX en: transformers/transformador_inicial.onnx\n", flush=True)

    # m06 · Feature selection
    banner("MÓDULO m06 • Feature Selection (filter → frame)")
    X_tr_sel, X_te_sel, X_bk_sel, logs = fs_run(techniques=["filter", "frame"], save_logs=True)
    print("m06 ✓ Artefactos en data/processed/pipeline_selection/\n", flush=True)

    # Preview m06
    try:
        tr_raw = pd.read_parquet(RAW_PARTITIONED_DIR / "train_df.parquet")[[TIME_COLUMN, TARGET_ALIAS]]
        preview = pd.concat([tr_raw.reset_index(drop=True).head(5), X_tr_sel.reset_index(drop=True).head(5)], axis=1)
        print("[6.2] Preview X_train_selected (head 5 con 'semana' y 'target'):")
        print(preview.to_string(index=False))
    except Exception:
        pass

    # m07 · Search + ONNX + Plot
    banner("MÓDULO m07 • Búsqueda de modelo (FLAML) + Predicción ONNX")
    X_train_proc = pd.read_parquet(FE_DIR / "X_train_processed.parquet")
    X_test_proc  = pd.read_parquet(FE_DIR / "X_test_processed.parquet")
    X_back_proc  = pd.read_parquet(FE_DIR / "X_backtest_processed.parquet")

    y_train = X_train_proc.pop(TARGET_ALIAS)
    y_test  = X_test_proc.pop(TARGET_ALIAS)
    y_back  = X_back_proc.pop(TARGET_ALIAS)

    y_train = pd.to_numeric(y_train, errors="coerce").fillna(0).round().astype(int)
    y_test  = pd.to_numeric(y_test,  errors="coerce").fillna(0).round().astype(int)
    y_back  = pd.to_numeric(y_back,  errors="coerce").fillna(0).round().astype(int)

    time_test = pd.read_parquet(RAW_PARTITIONED_DIR / "test_df.parquet")[TIME_COLUMN]
    time_back = pd.read_parquet(RAW_PARTITIONED_DIR / "backtest_df.parquet")[TIME_COLUMN]

    for d in (X_train_proc, X_test_proc, X_back_proc):
        d.drop(columns=[TIME_COLUMN], errors="ignore", inplace=True)

    X_train_proc, X_test_proc = X_train_proc.align(X_test_proc, join="left", axis=1, fill_value=0.0)
    X_train_proc, X_back_proc = X_train_proc.align(X_back_proc, join="left", axis=1, fill_value=0.0)

    step("7.1", f"Shapes → X_train:({X_train_proc.shape[0]}, {X_train_proc.shape[1]}), X_test:({X_test_proc.shape[0]}, {X_test_proc.shape[1]})")

    step("7.2", "Ejecutando FLAML (regresión)…")
    automl = run_model_selector(
        X_train_proc, y_train,
        X_test=X_test_proc, y_test=y_test,
        framework="flaml", task="regression", time_budget=180, metric="mae"
    )
    print(f"  Mejor estimador: {automl.name}")

    best_model = automl.get_best_model()
    sk_model = getattr(best_model, "model", best_model)

    # Métricas (test) sin FutureWarning
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
        Xte_m = _ensure_same_columns(sk_model, X_test_proc.copy())
        y_pred_prev = sk_model.predict(Xte_m)
        metrics_test = {
            "MAE": float(mean_absolute_error(y_test, y_pred_prev)),
            "MSE": float(mean_squared_error(y_test, y_pred_prev)),
            "RMSE": float(root_mean_squared_error(y_test, y_pred_prev)),
            "R2": float(r2_score(y_test, y_pred_prev)),
        }
        print(f"  Métricas (test): {metrics_test}")
    except Exception:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        Xte_m = _ensure_same_columns(sk_model, X_test_proc.copy())
        y_pred_prev = sk_model.predict(Xte_m)
        metrics_test = {
            "MAE": float(mean_absolute_error(y_test, y_pred_prev)),
            "MSE": float(mean_squared_error(y_test, y_pred_prev)),
            "RMSE": float(mean_squared_error(y_test, y_pred_prev) ** 0.5),
            "R2": float(r2_score(y_test, y_pred_prev)),
        }
        print(f"  Métricas (test): {metrics_test}")

    # Ranking y features
    try:
        rk = automl.get_model_ranking()
        (LOG_DIR / "model_search_ranking.csv").parent.mkdir(parents=True, exist_ok=True)
        rk.to_csv(LOG_DIR / "model_search_ranking.csv", index=False)
        print("  Ranking guardado en logs/model_search_ranking.csv")
    except Exception:
        pass
    f_json = _save_features_used(list(X_train_proc.columns))
    print(f"  Features usadas guardadas en {f_json}")

    # Exportar ONNX SOLO del modelo ganador
    step("7.3", "Exportando modelo ganador a ONNX…")
    try:
        model_path = _export_onnx_selected_only(
            sk_model,
            _ensure_same_columns(sk_model, X_train_proc.copy()),
            MODELS_DIR, version="v1"
        )
        print(f"  ✓ Modelo ONNX en {model_path}")
        onnx_ok = True
    except Exception as e:
        onnx_ok = False
        print(f"  × No fue posible convertir a ONNX: {e}")

    # Predicción + gráfica (ahora indexada por fecha)
    step("7.5", "Inferencia + generación de gráfico…")
    try:
        Xte_plot = _ensure_same_columns(sk_model, X_test_proc.copy())
        Xbk_plot = _ensure_same_columns(sk_model, X_back_proc.copy())
        y_pred_test = sk_model.predict(Xte_plot)
        y_pred_back = sk_model.predict(Xbk_plot)

        # DEBUG/chequeo: ¿son constantes?
        _print_pred_stats("pred_test", y_pred_test)
        _print_pred_stats("pred_back", y_pred_back)

        plot_dir = OUTPUT_DIR / "search_model"
        plot_path = plot_dir / "reales_vs_pronostico.png"
        _plot_real_vs_pred(time_test, y_test, y_pred_test, time_back, y_back, y_pred_back, plot_path)
        print(f"  ✓ Gráfica guardada en {plot_path}")
    except Exception as e:
        print(f"  × No se pudo generar la gráfica: {e}")
        if not onnx_ok:
            print("  Sugerencia: revisa compatibilidad ONNX y columnas usadas.")

    print("\nPIPELINE2 · FINALIZADO ✅\n")


if __name__ == "__main__":
    main()
