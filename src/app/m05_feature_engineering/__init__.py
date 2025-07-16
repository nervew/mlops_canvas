"""
m05_feature_engineering
───────────────────────
Paquete raíz de la fase de Feature Engineering.

Para evitar ciclos de importación **no** cargamos los sub-módulos aquí.
Quien necesite una clase la debe importar así, por ejemplo:

    from m05_feature_engineering.step03_outliers import IQRHandler
    from m05_feature_engineering.step02_imputation import SimpleImputerAdapter
"""

from importlib import import_module as _imp

def run_pipeline(*args, **kwargs):
    """Proxy a m05_feature_engineering.pipeline_engineering.run_pipeline()"""
    return _imp(__name__ + ".pipeline_engineering").run_pipeline(*args, **kwargs)

__all__ = ["run_pipeline"]
