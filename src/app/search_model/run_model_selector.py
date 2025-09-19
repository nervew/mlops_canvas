# -*- coding: utf-8 -*-
# search_model/run_model_selector.py
from __future__ import annotations

from typing import Optional

import pandas as pd

from .flaml_wrapper import FLAMLWrapper


def run_model_selector(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    framework: str = "flaml",
    task: str = "regression",
    time_budget: int = 300,
    metric: str = "mae",
    log_file: Optional[str] = None,
):
    """
    Orquesta la búsqueda de modelo usando FLAML (sin mlflow).
    Devuelve el wrapper con el modelo ya entrenado.
    """
    if framework.lower() != "flaml":
        raise ValueError("Por ahora solo se soporta framework='flaml'.")

    automl = FLAMLWrapper(
        task=task,
        metric=metric,
        time_budget=time_budget,
        log_file=log_file,
        verbose=1,
    )

    if X_test is not None and y_test is not None:
        automl.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    else:
        automl.fit(X_train, y_train)

    return automl
