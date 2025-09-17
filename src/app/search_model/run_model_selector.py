# search_model/run_model_selector.py
from __future__ import annotations
from typing import Any

def run_model_selector(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    framework: str = "flaml",
    **kwargs
) -> Any:
    """
    Ejecuta el selector de modelos según el framework elegido.

    Parámetros
    ----------
    framework : {"flaml", "mljar"}
        - "flaml": usa FLAMLWrapper (requiere X_test e y_test para validación).
        - "mljar": usa MLJARWrapper (no requiere X_test/y_test aquí).

    kwargs : dict
        Se pasan directamente al wrapper correspondiente.
    """
    fw = (framework or "flaml").lower()

    if fw == "flaml":
        # Import perezoso para evitar dependencias innecesarias en tiempo de carga
        from .flaml_wrapper import FLAMLWrapper
        if X_test is None or y_test is None:
            raise ValueError("FLAML requiere X_test e y_test para la validación holdout.")
        automl = FLAMLWrapper(**kwargs)
        automl.fit(X_train, y_train, X_test, y_test)
        return automl

    if fw == "mljar":
        # Solo importamos MLJAR si realmente se solicita (evita errores de scipy/statsmodels)
        from .mljar_wrapper import MLJARWrapper
        automl = MLJARWrapper()
        automl.fit(X_train, y_train)
        return automl

    raise ValueError(f"Framework '{framework}' no soportado. Usa 'flaml' o 'mljar'.")
