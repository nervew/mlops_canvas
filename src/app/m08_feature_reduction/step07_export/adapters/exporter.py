# m08_feature_reduction/step07_export/adapters/exporter.py
import joblib, json, shap, matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import numpy as np
from ..ports.exporter import IExporter

# ===========  NUEVO: localizamos la carpeta src real  ===========
PROJECT_SRC = Path(__file__).resolve().parents[4]   # mlops_canvas/src
# ==============================================================

class Exporter(IExporter):
    def export_all(
        self,
        transformer: Any,
        shap_values: np.ndarray,
        shap_summary: np.ndarray,
        dfs: Dict[str, pd.DataFrame],
    ) -> None:

        # 1) Serializar pipeline final
        transformers_dir = PROJECT_SRC / "transformers"
        transformers_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(transformer, transformers_dir / "transformador_final.joblib")

        # 2) Normalizar shap_values a 2-D
        shap_mat = np.asarray(shap_values[0] if isinstance(shap_values, list) else shap_values)
        shap_mat = shap_mat.squeeze() if shap_mat.ndim == 3 else shap_mat
        if shap_mat.ndim == 1:
            shap_mat = shap_mat.reshape(-1, 1)

        # 3) Guardar shap_summary JSON
        report_dir = PROJECT_SRC / "output" / "report_reduction"
        report_dir.mkdir(parents=True, exist_ok=True)
        with open(report_dir / "shap_summary.json", "w", encoding="utf-8") as f:
            json.dump(shap_summary.tolist(), f, indent=4)

        # 4) Gráfico SHAP
        n_plot = min(100, shap_mat.shape[0])
        comp_names = [f"component_{i+1}" for i in range(shap_mat.shape[1])]
        shap_df = pd.DataFrame(shap_mat[:n_plot], columns=comp_names)
        plt.figure()
        shap.summary_plot(shap_mat[:n_plot], shap_df, show=False)
        plt.savefig(report_dir / "shap_summary.png", bbox_inches="tight")
        plt.close()

        # 5) Exportar DataFrames procesados
        out_dir = PROJECT_SRC / "data" / "processed" / "pipeline_reduction"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, df in dfs.items():
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
