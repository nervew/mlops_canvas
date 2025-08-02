# search_model/run_model_selector.py

from .flaml_wrapper import FLAMLWrapper
from .mljar_wrapper import MLJARWrapper

def run_model_selector(
    X_train, y_train, X_test, y_test,
    framework="flaml",
    **kwargs
):
    if framework == "flaml":
        automl = FLAMLWrapper(**kwargs)
        automl.fit(X_train, y_train, X_test, y_test)
    elif framework == "mljar":
        automl = MLJARWrapper()
        automl.fit(X_train, y_train)
    else:
        raise ValueError(f"Framework '{framework}' no soportado.")
    return automl
