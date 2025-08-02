# m08_feature_reduction/step07_export/adapters/exporter.py
"""
Exporta:
  • transformers/transformador_final.joblib
  • output/report_reduction/{shap_summary.json, metrics_reduction.json, shap_summary.png}
  • data/processed/pipeline_reduction/{train,test,backtest}.parquet
… todos ubicados directamente bajo la raíz del proyecto (mlops_canvas).
"""
from __future__ import annotations
import json
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd

from ..ports.exporter import IExporter


# ───────────────────────── Localizar raíz mlops_canvas ─────────────────────
def get_project_root() -> Path:
    """
    Sube directorios hasta encontrar la carpeta que contiene 'src' y devuelve
    su padre (= raíz del proyecto mlops_canvas).
    """
    d = Path(__file__).resolve()
    while d != d.parent:
        if (d / "src").is_dir():
            return d
        d = d.parent
    raise RuntimeError("No se pudo localizar la raíz del proyecto")

PROJECT_ROOT = get_project_root()        # …/mlops_canvas
# ────────────────────────────────────────────────────────────────────────────


class Exporter(IExporter):
    def export_all(
        self,
        transformer: Any,
        shap_values: np.ndarray,
        shap_summary: np.ndarray,
        dfs: Dict[str, pd.DataFrame],
        metrics_json: Dict[str, Any],
    ) -> None:
        # 1) Serializar pipeline final
        (PROJECT_ROOT / "transformers").mkdir(parents=True, exist_ok=True)
        joblib.dump(
            transformer,
            PROJECT_ROOT / "transformers" / "transformador_final.joblib"
        )

        # 2) Normalizar shap_values a 2-D
        shap_arr = shap_values[0] if isinstance(shap_values, (list, tuple)) else shap_values
        shap_mat = np.asarray(shap_arr)
        if shap_mat.ndim > 2:
            shap_mat = shap_mat.reshape(shap_mat.shape[0], -1)
        elif shap_mat.ndim == 1:
            shap_mat = shap_mat.reshape(-1, 1)

        # 3) Directorio de reportes
        report_dir = PROJECT_ROOT / "output" / "report_reduction"
        report_dir.mkdir(parents=True, exist_ok=True)

        # Guardar summary y métricas
        (report_dir / "shap_summary.json").write_text(
            json.dumps(np.asarray(shap_summary).ravel().tolist(), indent=4),
            encoding="utf-8",
        )
        (report_dir / "metrics_reduction.json").write_text(
            json.dumps(metrics_json, indent=4), encoding="utf-8"
        )

        # 4) Gráfico SHAP (rng explícito para evitar FutureWarning)
        n_plot = min(100, shap_mat.shape[0])
        comp_names = [f"feature_{i+1}" for i in range(shap_mat.shape[1])]
        shap_df = pd.DataFrame(shap_mat[:n_plot], columns=comp_names)

        rng = np.random.default_rng(42)          # ← generador explícito
        plt.figure()
        shap.summary_plot(
            shap_mat[:n_plot],
            shap_df,
            show=False,
            rng=rng
        )
        plt.savefig(report_dir / "shap_summary.png", bbox_inches="tight")
        plt.close()

        # 5) Exportar DataFrames procesados
        out_dir = PROJECT_ROOT / "data" / "processed" / "pipeline_reduction"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, df in dfs.items():
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
