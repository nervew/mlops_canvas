# m08_feature_reduction/step07_export/adapters/exporter.py
import joblib
import json
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import numpy as np

from ..ports.exporter import IExporter

# carpeta src real (mlops_canvas/src)
PROJECT_SRC = Path(__file__).resolve().parents[5]


class Exporter(IExporter):
    def export_all(
        self,
        transformer: Any,
        shap_values: np.ndarray,
        shap_summary: np.ndarray,
        dfs: Dict[str, pd.DataFrame],
    ) -> None:
        # 1) Serializar pipeline final
        (PROJECT_SRC / "transformers").mkdir(parents=True, exist_ok=True)
        joblib.dump(
            transformer,
            PROJECT_SRC / "transformers" / "transformador_final.joblib",
        )

        # 2) Normalizar shap_values → matriz 2-D (n_samples × n_features)
        shap_arr = np.asarray(shap_values[0] if isinstance(shap_values, list) else shap_values)
        if shap_arr.ndim == 1:                       # (n,)
            shap_arr = shap_arr.reshape(-1, 1)
        elif shap_arr.ndim > 2:                      # (n, p, q, …) → (n, p*q*…)
            n_samples = shap_arr.shape[0]
            shap_arr = shap_arr.reshape(n_samples, -1)
        # (ndim==2) queda tal cual

        # 3) Guardar shap_summary
        report_dir = PROJECT_SRC / "output" / "report_reduction"
        report_dir.mkdir(parents=True, exist_ok=True)
        with open(report_dir / "shap_summary.json", "w", encoding="utf-8") as f:
            json.dump(np.asarray(shap_summary).tolist(), f, indent=4)

        # 4) Gráfico SHAP
        n_plot = min(100, shap_arr.shape[0])
        col_names = [f"feature_{i+1}" for i in range(shap_arr.shape[1])]
        shap_df = pd.DataFrame(shap_arr[:n_plot], columns=col_names)
        rng = np.random.default_rng(42)
        plt.figure()
        shap.summary_plot(
        shap_arr[:n_plot],
        shap_df,
        show=False,
        rng=rng                
        )
        plt.savefig(report_dir / "shap_summary.png", bbox_inches="tight")
        plt.close()

        # 5) Exportar DataFrames procesados finales
        out_dir = PROJECT_SRC / "data" / "processed" / "pipeline_reduction"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, df in dfs.items():
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
