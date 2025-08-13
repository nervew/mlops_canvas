# src/app/m06__feature_selection/run.py

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .step01_import     import ParquetPartitionLoader2
from .step02_filtering  import filter_partitions
from .step03_frame      import frame_partitions
from .step04_abess      import abess_partitions
from .step05_shap_select import shap_partitions
from .step06_permutation import permutation_partitions
from .step07_export     import export_partitions

def run_pipeline(
    techniques: List[str] = ["filter", "frame", "abess", "shap", "permutation"],
    forward_k: int = 3,
    final_k:   int = 2,
    top_n:     int = 3,
    tol:       float = 0.01,
    save_logs: bool  = False,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    List[Dict[str, object]]  # logs
]:
    """
    Ejecuta la pipeline de selección de features:

      * filter → frame → abess → shap → permutation

    Si `techniques` contiene un subconjunto, sólo se aplican esas etapas,
    en el orden dado.

    Parámetros:
    -----------
    techniques : lista de nombres de técnicas a aplicar.
                 Puede incluir: "filter", "frame", "abess", "shap", "permutation".
    forward_k, final_k : para frame_partitions
    top_n               : para shap_partitions
    tol                 : para permutation_partitions
    save_logs           : si True, genera un JSON con tiempos y n_features
                          en src/output/log_selection/feature_selection_logs.json

    Devuelve:
    ---------
    X_train_sel, X_test_sel, X_back_sel, logs
    """
    # 1) carga particiones originales
    loader = ParquetPartitionLoader2()
    X_train, X_test, X_back, y_train, _, _ = loader.load()

    logs: List[Dict[str, object]] = []
    # 2) Run each técnica
    for step in techniques:
        t0 = perf_counter()
        if step == "filter":
            X_train, X_test, X_back, _sel = filter_partitions(
                X_train, X_test, X_back, y_train
            )
        elif step == "frame":
            X_train, X_test, X_back, _sel = frame_partitions(
                X_train, X_test, X_back, y_train,
                estimator=LinearRegression(),
                forward_k=forward_k,
                final_k=final_k,
            )
        elif step == "abess":
            X_train, X_test, X_back, _sel = abess_partitions(
                X_train, X_test, X_back, y_train, mode="regression"
            )
        elif step == "shap":
            X_train, X_test, X_back, _sel = shap_partitions(
                X_train, X_test, X_back, y_train,
                top_n=top_n, task="regression"
            )
        elif step == "permutation":
            X_train, X_test, X_back, _sel = permutation_partitions(
                X_train, X_test, X_back, y_train,
                tol=tol, task="regression", scoring="r2"
            )
        else:
            raise ValueError(f"Técnica desconocida: {step!r}")
        dt = perf_counter() - t0
        logs.append({
            "step": step,
            "time_sec": round(dt, 4),
            "n_features": X_train.shape[1],
        })

    # 3) export final
    export_partitions(X_train, X_test, X_back)

    # 4) (opcional) guardado de logs en disco
    if save_logs:
        # carpeta: <proyecto>/logs/log_selection
        log_dir = Path(__file__).resolve().parents[3] / "logs" / "log_selection"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "feature_selection_logs.json"
        log_file.write_text(json.dumps(logs, indent=2), encoding="utf-8")

        print(f"✔ Logs guardados en {log_file}")

    print("Pipeline finalizado. Variables númericas seleccionadas finales:")
    print(X_train.columns.tolist())

    return X_train, X_test, X_back, logs
