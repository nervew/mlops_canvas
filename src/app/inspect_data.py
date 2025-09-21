# src/app/inspect_data.py
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

RAW_PARQUET = PROJECT_ROOT / "data" / "raw" / "complete" / "df.parquet"
FE_DIR      = PROJECT_ROOT / "data" / "processed" / "pipeline_engineering"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def banner(t: str) -> None:
    line = "═" * len(t)
    print(f"\n{line}\n{t}\n{line}")

def step(i: str, t: str) -> None:
    print(f"[{i}] {t}")

def main() -> None:
    # (Opcional) asegurar deps si el clúster recién reinició
    from app.m00_instalador import install_requirements
    install_requirements()

    import pandas as pd

    banner("INSPECTOR • Datos para 'search_model'")

    # ── 1) RAW: df.parquet
    banner("1) DataFrame crudo (data/raw/complete/df.parquet)")
    step("1.1", f"Leyendo: {RAW_PARQUET.relative_to(PROJECT_ROOT)}")
    if not RAW_PARQUET.exists():
        raise FileNotFoundError(f"No existe {RAW_PARQUET}")
    df = pd.read_parquet(RAW_PARQUET)
    step("1.2", f"Shape: {df.shape[0]} filas × {df.shape[1]} columnas")
    print("      Columnas (primeras 20):", list(df.columns[:20]))

    # Target: 'target' o 'y_usuarios_nuevos_semana'
    target = "target" if "target" in df.columns else (
        "y_usuarios_nuevos_semana" if "y_usuarios_nuevos_semana" in df.columns else None
    )
    if target:
        step("1.3", f"Target detectada: '{target}'")
        try:
            print(df[target].describe().to_string())
        except Exception:
            print("      (no se pudo describir la target)")
        print("      Head:", df[target].head(5).tolist())
    else:
        step("1.3", "Target no encontrada (ni 'target' ni 'y_usuarios_nuevos_semana').")

    # Columna temporal 'semana'
    time_col = "semana" if "semana" in df.columns else None
    if time_col:
        step("1.4", "Columna temporal 'semana'")
        s = pd.to_datetime(df[time_col], errors="coerce")
        print("      n_no_nulos:", s.notna().sum(), "| n_nulos:", s.isna().sum())
        if s.notna().any():
            s_sorted = s.dropna().sort_values()
            tmin, tmax = s_sorted.min(), s_sorted.max()
            freq = pd.infer_freq(s_sorted.unique())
            print("      rango:", tmin, "→", tmax)
            print("      frecuencia inferida:", freq)
            if freq is None and len(s_sorted) > 1:
                diffs = s_sorted.diff().dropna()
                print("      delta mediana:", diffs.median())
    else:
        step("1.4", "No hay columna 'semana' en df.")

    # Nulos por columna (top 10)
    step("1.5", "Top 10 columnas por % de nulos")
    null_pct = df.isna().mean().sort_values(ascending=False) * 100
    print((null_pct.head(10).round(2)).to_string())

    # ── 2) Procesados de m05 (insumo típico para search_model)
    banner("2) Datasets procesados por m05 (pipeline_engineering)")
    tr_p = FE_DIR / "X_train_processed.parquet"
    te_p = FE_DIR / "X_test_processed.parquet"
    bk_p = FE_DIR / "X_backtest_processed.parquet"

    for tag, p in (("train", tr_p), ("test", te_p), ("backtest", bk_p)):
        if p.exists():
            X = pd.read_parquet(p)
            step(f"2.{ {'train':1,'test':2,'backtest':3}[tag] }",
                 f"{tag}: {p.relative_to(PROJECT_ROOT)} | shape={X.shape}")
            has_target = "target" in X.columns
            print("      contiene 'target':", has_target)
            if has_target:
                print("      target.describe():")
                try:
                    print(X["target"].describe().to_string())
                except Exception:
                    print("      (no se pudo describir la target)")
        else:
            step(f"2.{ {'train':1,'test':2,'backtest':3}[tag] }",
                 f"{tag}: NO ENCONTRADO → {p.relative_to(PROJECT_ROOT)}")

    banner("INSPECTOR • FIN ✅")

if __name__ == "__main__":
    main()
