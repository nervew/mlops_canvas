# src/app/d_database/genera_base.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Importa la generación desde el módulo de base de datos
from .datos import generate_synthetic_patient_data, DEFAULT_DATA_SEED


def main(
    periods: int = 336,
    seed: int = DEFAULT_DATA_SEED,
    out_path: Path | None = None,
) -> None:
    # Ruta por defecto: .../mlops_canvas/data/raw/complete/df.parquet
    if out_path is None:
        project_root = Path(__file__).resolve().parents[3]
        out_path = project_root / "data" / "raw" / "complete" / "df.parquet"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== d_database • Generación de base sintética ===")
    print(f"- periods: {periods}")
    print(f"- seed   : {seed}")
    print(f"- salida : {out_path}")

    df = generate_synthetic_patient_data(periods=periods, seed=seed)

    # Asegurar tipos consistentes
    df["semana"] = pd.to_datetime(df["semana"])
    float_cols = [c for c in df.columns if c != "semana"]
    df[float_cols] = df[float_cols].astype(float)

    df.to_parquet(out_path, index=False)
    print(f"✔ Parquet guardado en: {out_path}")
    print(f"✔ Shape: {df.shape[0]} filas × {df.shape[1]} columnas")


if __name__ == "__main__":
    main()
