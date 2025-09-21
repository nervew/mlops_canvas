# src/app/pipeline2.py
from __future__ import annotations

import sys, json
from pathlib import Path

# ───────────── Rutas del proyecto (solo stdlib aquí) ─────────────
APP_DIR       = Path(__file__).resolve().parent          # .../src/app
SRC_DIR       = APP_DIR.parent                           # .../src
PROJECT_ROOT  = SRC_DIR.parent                           # .../mlops_canvas

DATA_PATH         = PROJECT_ROOT / "data" / "raw" / "complete" / "df.parquet"
PARTITIONED_DIR   = PROJECT_ROOT / "data" / "raw" / "partitioned"
VALIDATION_DIR    = PROJECT_ROOT / "output" / "validation"
FE_DIR            = PROJECT_ROOT / "data" / "processed" / "pipeline_engineering"
FS_DIR            = PROJECT_ROOT / "data" / "processed" / "pipeline_selection"
LOGS_DIR          = PROJECT_ROOT / "logs"
OUTPUT_DIR        = PROJECT_ROOT / "output"
MODELS_DIR        = PROJECT_ROOT / "models"
SEARCH_OUT_DIR    = OUTPUT_DIR / "search_model"

# Config clave
TIME_COLUMN  = "semana"                     # variable temporal para split m04 / gráficos
TARGET_ALIAS = "target"                     # alias estándar de target
OLD_TARGET   = "y_usuarios_nuevos_semana"   # si existe, se renombra a 'target'

# sys.path para imports app.*
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ───────────── Utilidades de impresión ─────────────
def banner(t: str) -> None:
    line = "═" * len(t)
    print(f"\n{line}\n{t}\n{line}")

def step(i: str, t: str) -> None:
    print(f"[{i}] {t}")

def _print_preview_with_time_target(df, step_id: str, title: str,
                                    time_col: str = TIME_COLUMN, target_col: str = TARGET_ALIAS,
                                    extra_cols: int = 6) -> None:
    cols = []
    if time_col in df.columns:   cols.append(time_col)
    if target_col in df.columns: cols.append(target_col)
    others = [c for c in df.columns if c not in cols]
    show_cols = cols + others[:max(0, extra_cols - len(cols))]
    step(step_id, title)
    try:
        print(df[show_cols].head(5).to_string())
    except Exception:
        print(df.head(5).to_string())

def _print_selected_joining_time_target(X_sel, fe_dir: Path,
                                        step_id: str, title: str) -> None:
    try:
        import pandas as pd
        xproc_path = fe_dir / "X_train_processed.parquet"
        if not xproc_path.exists():
            step(step_id, f"{title} (no se encontró {xproc_path.name})")
            print(X_sel.head(5).to_string()); return
        X_proc = pd.read_parquet(xproc_path)
        cols_add = [c for c in (TIME_COLUMN, TARGET_ALIAS) if c in X_proc.columns]
        if not cols_add:
            step(step_id, f"{title} (sin 'semana'/'target' en procesado)")
            print(X_sel.head(5).to_string()); return
        try:
            add_part = X_proc.loc[X_sel.index, cols_add]
        except Exception:
            add_part = X_proc[cols_add].iloc[: len(X_sel)].copy()
            add_part.index = X_sel.index[: len(add_part)]
        preview = add_part.join(X_sel, how="left")
        step(step_id, title); print(preview.head(5).to_string())
    except Exception as e:
        step(step_id, f"{title} (no se pudo reconstruir semana/target: {e})")
        print(X_sel.head(5).to_string())

# Conversión robusta de 'semana' a datetime (para filtros/plots)
def _to_datetime_semana(s):
    import numpy as _np
    import pandas as _pd
    if s is None:
        return None
    # Datasets con '17924' etc. = días desde 1970-01-01
    if _np.issubdtype(s.dtype, _np.number) and s.min() > 1000 and s.max() < 100000:
        return _pd.to_datetime(s.astype("int64"), unit="D", origin="1970-01-01")
    return _pd.to_datetime(s, errors="coerce")

# ───────────── Pipeline2 ─────────────
def main() -> None:
    banner("PIPELINE2 • m00 → m01 → m02 → m03 → m04 → m05 → m06 → m07")

    # ===== m00: Instalación (0.x) =====
    banner("MÓDULO m00 • Instalación de dependencias")
    step("0.1", "Instalando/verificando dependencias…")
    from app.m00_instalador import install_requirements
    install_requirements()
    step("0.2", "Dependencias listas ✅")

    # A partir de aquí ya podemos importar paquetes y módulos que requieren dependencias
    import numpy as np
    import pandas as pd
    # Backend headless para gráficos
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from app.m02_eda_univariado import run as m02_run
    from app.m03_data_validation.application import service as validate_srv
    from app.m04_data_split.infrastructure.robust_data_splitter import RobustDataSplitter
    from app.m05_feature_engineering.pipeline_engineering import run_pipeline as fe_run
    from app.m06__feature_selection import run_pipeline as fs_run
    from app.search_model.run_model_selector import run_model_selector
    from app.search_model.export import export_model_onnx, get_log_path

    # ===== m01: Ingesta (1.x) =====
    banner("MÓDULO m01 • Ingesta (importación del dataset)")
    step("1.1", f"Leyendo parquet: {DATA_PATH.relative_to(PROJECT_ROOT)}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el parquet en {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    step("1.2", f"Shape del DataFrame (antes de filtrar): {df.shape[0]} filas × {df.shape[1]} columnas")

    if OLD_TARGET in df.columns and TARGET_ALIAS not in df.columns:
        df = df.rename(columns={OLD_TARGET: TARGET_ALIAS})
        step("1.3", f"Columna renombrada: '{OLD_TARGET}' → '{TARGET_ALIAS}' ✅")
    elif TARGET_ALIAS in df.columns:
        step("1.3", "La columna 'target' ya existe; no se renombra.")
    else:
        step("1.3", f"No se encontró la columna '{OLD_TARGET}'; no se renombra.")

    # --- Corte post-pandemia: quedarnos con datos >= 2022-01-01 ---
    if TIME_COLUMN in df.columns:
        dt_sem = _to_datetime_semana(df[TIME_COLUMN])
        mask = dt_sem >= pd.Timestamp("2022-01-01")
        prev_n = len(df)
        df = df.loc[mask].copy()
        step("1.4", f"Filtro post-pandemia aplicado (desde 2022-01-01): {prev_n:,} → {len(df):,} filas")
    else:
        step("1.4", f"Aviso: no existe la columna temporal '{TIME_COLUMN}'. No se aplicó filtro por fecha.")

    if TARGET_ALIAS in df.columns:
        print("\n[1.5] Vista rápida de 'target' (después del filtro):")
        try:
            print(" - 5 valores:", df[TARGET_ALIAS].head().tolist())
            print(" - Estadísticos básicos:"); print(df[TARGET_ALIAS].describe().to_string())
        except Exception:
            print(" (no se pudo describir 'target')")
    else:
        print("\n[1.5] No hay 'target' para mostrar.")

    # ===== m02: EDA (2.x) =====
    banner("MÓDULO m02 • Análisis Univariado (EDA)")
    step("2.1", "Ejecutando EDA univariado…")
    m02_run(df, print_tables=False, print_formal_summary=True, log_artifacts=True)
    step("2.2", "EDA completado. Artefactos en output/reporte_eda/")

    # ===== m03: Validación (3.x) =====
    banner("MÓDULO m03 • Validación de datos")
    step("3.1", "Creando/cargando perfil y validando contra el perfil…")
    result = validate_srv.run(df, fit_profile=True)
    step("3.2", f"Artefactos en: {VALIDATION_DIR.relative_to(PROJECT_ROOT)}")
    print("       - data_profile.json"); print("       - validation_report.json")
    if hasattr(result, "valido"):
        print(f"       → Resultado global: {'✅ VÁLIDO' if result.valido else '❌ NO VÁLIDO'}")

    # ===== m04: Split temporal (4.x) =====
    banner("MÓDULO m04 • Particionado temporal por 'semana'")
    step("4.1", f"Preparando carpeta de salida: {PARTITIONED_DIR.relative_to(PROJECT_ROOT)}")
    PARTITIONED_DIR.mkdir(parents=True, exist_ok=True); LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if TIME_COLUMN not in df.columns:
        raise KeyError(f"No se encontró la columna temporal '{TIME_COLUMN}' en el DataFrame.")
    step("4.2", "Ejecutando RobustDataSplitter (split_method='time')…")
    splitter = RobustDataSplitter(
        df, split_method="time",
        target_column=TARGET_ALIAS if TARGET_ALIAS in df.columns else None,
        time_column=TIME_COLUMN, train_size=0.7, test_size=0.2, backtest_size=0.1,
    )
    train_df, test_df, backtest_df = splitter.split_data()
    (PARTITIONED_DIR / "train_df.parquet").parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(PARTITIONED_DIR / "train_df.parquet")
    test_df.to_parquet(PARTITIONED_DIR / "test_df.parquet")
    backtest_df.to_parquet(PARTITIONED_DIR / "backtest_df.parquet")
    step("4.3", "Particiones guardadas:")
    print("       -", (PARTITIONED_DIR / "train_df.parquet").relative_to(PROJECT_ROOT))
    print("       -", (PARTITIONED_DIR / "test_df.parquet").relative_to(PROJECT_ROOT))
    print("       -", (PARTITIONED_DIR / "backtest_df.parquet").relative_to(PROJECT_ROOT))
    try:
        metrics = splitter.calculate_metrics()
        mp = LOGS_DIR / "split_metrics.csv"; metrics.to_csv(mp, index=True)
        step("4.4", f"Métricas del split guardadas en: {mp.relative_to(PROJECT_ROOT)}")
    except Exception:
        step("4.4", "El splitter no expuso métricas (se continúa).")

    # ===== m05: Feature Engineering (5.x) =====
    banner("MÓDULO m05 • Feature Engineering")
    step("5.1", "Ejecutando pipeline de ingeniería de variables…"); fe_run()
    step("5.2", f"Artefactos esperados en: {FE_DIR.relative_to(PROJECT_ROOT)}")
    print("       - X_train_processed.parquet")
    print("       - X_test_processed.parquet")
    print("       - X_backtest_processed.parquet")
    print("       - Transformador inicial ONNX:",
          "/Workspace/Users/jorgee.lopez@adres.gov.co/mlops_canvas/transformers/transformador_inicial.onnx")
    try:
        xtrain_proc = pd.read_parquet(FE_DIR / "X_train_processed.parquet")
        _print_preview_with_time_target(xtrain_proc, "5.3", "Preview X_train_processed (head 5 con 'semana' y 'target'):")
    except Exception as e:
        step("5.3", f"No se pudo leer X_train_processed.parquet ({e}).")

    # ===== m06: Feature Selection (6.x) =====
    banner("MÓDULO m06 • Feature Selection (filter → frame)")
    step("6.1", "Ejecutando selección de variables…")
    X_tr_fs, X_te_fs, X_bk_fs, logs = fs_run(techniques=["filter", "frame"], save_logs=True)
    try:
        import pandas as _pd; print(_pd.DataFrame(logs), "\n")
    except Exception:
        pass
    step("6.2", f"Artefactos esperados en: {FS_DIR.relative_to(PROJECT_ROOT)}")
    print("       - X_train_selected.parquet")
    print("       - X_test_selected.parquet")
    print("       - X_backtest_selected.parquet")
    _print_selected_joining_time_target(X_tr_fs, FE_DIR, "6.3", "Preview X_train_selected (head 5 con 'semana' y 'target'):")

    # ===== m07: Search Model (FLAML) + Predicción ONNX =====
    banner("MÓDULO m07 • Búsqueda de modelo (FLAML) + Predicción ONNX")

    # 7.1 Cargar matrices procesadas (FE) para entrenamiento de regresión
    step("7.1", "Cargando X_train/test procesados y separando target…")
    X_tr_proc_full = pd.read_parquet(FE_DIR / "X_train_processed.parquet")
    X_te_proc_full = pd.read_parquet(FE_DIR / "X_test_processed.parquet")
    sem_train = X_tr_proc_full[TIME_COLUMN] if TIME_COLUMN in X_tr_proc_full.columns else None
    sem_test  = X_te_proc_full[TIME_COLUMN] if TIME_COLUMN in X_te_proc_full.columns else None
    y_train = X_tr_proc_full.pop(TARGET_ALIAS)
    y_test  = X_te_proc_full.pop(TARGET_ALIAS)

    # Alinear columnas por seguridad (features)
    X_tr_proc, X_te_proc = X_tr_proc_full.align(X_te_proc_full, join="left", axis=1, fill_value=0.0)
    step("7.1", f"Shapes → X_train:{X_tr_proc.shape}, X_test:{X_te_proc.shape}")

    # 7.2 FLAML (regresión) — exportable a ONNX
    step("7.2", "Ejecutando FLAML (regresión)…")
    automl = run_model_selector(
        X_tr_proc, y_train, X_test=X_te_proc, y_test=y_test,
        framework="flaml", task="regression", time_budget=180,
        metric="mae", log_file=get_log_path("flaml.log")
    )
    metrics_reg = automl.evaluate(X_te_proc, y_test)
    print(f"       Mejor estimador: {automl.name}")
    print(f"       Métricas (test): {metrics_reg}")
    try:
        rank = automl.get_model_ranking()
        if hasattr(rank, "to_csv"):
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            rank_path = LOGS_DIR / "model_search_ranking.csv"
            rank.to_csv(rank_path, index=False)
            print(f"       Ranking guardado en {rank_path.relative_to(PROJECT_ROOT)}")
    except Exception:
        pass

    # 7.3 Exportar modelo a ONNX (campeón regresión) + guardar columnas usadas
    step("7.3", "Exportando modelo ganador a ONNX…")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    export_model_onnx(automl.get_best_model(), X_tr_proc, MODELS_DIR, version="v1")

    # Guardar orden/tamaño de features esperados por el ONNX
    try:
        expected_n = getattr(automl.get_best_model(), "n_features_in_", X_tr_proc.shape[1])
        # quitamos 'semana' si estuviera y recortamos a expected_n
        features_used = [c for c in X_tr_proc.columns if c != TIME_COLUMN][:int(expected_n)]
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        (LOGS_DIR / "model_features_used.json").write_text(
            json.dumps(features_used, indent=2), encoding="utf-8"
        )
        print(f"       Features ONNX guardados en logs/model_features_used.json "
              f"(expected_n={expected_n}, columnas={len(features_used)})")
    except Exception as e:
        print(f"       Aviso: no se pudieron guardar las columnas usadas ({e}).")

    # 7.5 Predicción con ONNX + Gráfica unificada (reales vs pronósticos)
    try:
        step("7.5", "Cargando modelo ONNX y generando pronósticos…")
        import onnxruntime as ort
        import numpy as _np

        onnx_path = MODELS_DIR / "modelo_v1.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo ONNX en {onnx_path}")
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        in_name   = sess.get_inputs()[0].name
        expected_n_onnx = sess.get_inputs()[0].shape[1]  # e.g., 15

        # Cargar lista de columnas usadas por el modelo
        feat_file = LOGS_DIR / "model_features_used.json"
        if feat_file.exists():
            features_used = json.loads(feat_file.read_text(encoding="utf-8"))
        else:
            features_used = [c for c in X_tr_proc.columns if c != TIME_COLUMN][:int(expected_n_onnx)]

        def _prepare(df):
            arr = df.reindex(columns=features_used, fill_value=0.0).to_numpy(dtype=_np.float32, copy=False)
            if expected_n_onnx is not None and arr.shape[1] != expected_n_onnx:
                raise ValueError(f"Input shape mismatch: got {arr.shape[1]} cols, expected {expected_n_onnx}.")
            return arr

        # Cargar backtest procesado (para reales/fechas y features)
        X_bk_proc_full = pd.read_parquet(FE_DIR / "X_backtest_processed.parquet")
        sem_back = X_bk_proc_full[TIME_COLUMN] if TIME_COLUMN in X_bk_proc_full.columns else None
        y_back = X_bk_proc_full.pop(TARGET_ALIAS)

        X_te_inf = _prepare(X_te_proc)
        X_bk_inf = _prepare(X_bk_proc_full)

        # Predicciones ONNX
        y_pred_test = sess.run(None, {in_name: X_te_inf})[0].ravel()
        y_pred_back = sess.run(None, {in_name: X_bk_inf})[0].ravel()

        # Fechas robustas para eje X
        dt_test = _to_datetime_semana(sem_test) if sem_test is not None else None
        dt_back = _to_datetime_semana(sem_back) if sem_back is not None else None

        # Figura única: reales (test+back) vs pronóstico ONNX (test/back)
        plt.figure(figsize=(12, 6))

        # Línea de reales
        if dt_test is not None and dt_back is not None:
            x_real = pd.concat([dt_test.reset_index(drop=True), dt_back.reset_index(drop=True)], ignore_index=True)
        else:
            x_real = pd.RangeIndex(len(y_test) + len(y_back))
        y_real = pd.concat([y_test.reset_index(drop=True), y_back.reset_index(drop=True)], ignore_index=True)
        plt.plot(x_real, y_real, marker='o', linestyle='-', label='Real (Test + Backtest)')

        # Pronósticos
        if dt_test is not None:
            plt.plot(dt_test, y_pred_test, marker='x', linestyle='--', label='Pronóstico ONNX (Test)')
        else:
            plt.plot(range(len(y_pred_test)), y_pred_test, marker='x', linestyle='--', label='Pronóstico ONNX (Test)')

        if dt_back is not None:
            plt.plot(dt_back, y_pred_back, marker='x', linestyle='--', label='Pronóstico ONNX (Backtest)')
        else:
            plt.plot(range(len(y_pred_back)), y_pred_back, marker='x', linestyle='--', label='Pronóstico ONNX (Backtest)')

        plt.xlabel("Fecha" if (dt_test is not None or dt_back is not None) else "Índice")
        plt.ylabel(TARGET_ALIAS)
        plt.title("Reales vs. Pronóstico (modelo ONNX) — Test y Backtest")
        plt.grid(True); plt.legend(); plt.xticks(rotation=45); plt.tight_layout()

        SEARCH_OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig_path = SEARCH_OUT_DIR / "forecast_from_onnx.png"
        plt.savefig(fig_path); plt.close()
        print(f"       Gráfica guardada en: {fig_path.relative_to(PROJECT_ROOT)}")
    except Exception as e:
        print(f"       (7.5) No se pudo generar la gráfica ONNX: {e}")

    banner("PIPELINE2 • FINALIZADO ✅")


if __name__ == "__main__":
    main()
