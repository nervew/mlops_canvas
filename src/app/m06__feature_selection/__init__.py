# /Workspace/Users/jorgee.lopez@adres.gov.co/mlops_canvas/src/app/m06__feature_selection/__init__.py

"""
m06__feature_selection
───────────────────────
Paquete raíz de la fase de Feature Selection.

Para evitar ciclos de importación **no** cargamos los sub-módulos aquí.
Quien necesite una clase o el pipeline de ejecución debe importarlos así, por ejemplo:

    from m06__feature_selection.step01_import import ParquetPartitionLoader2
    from m06__feature_selection.step02_filtering.filter_roughfs import FilterRoughFS
    # …etc.

Y para ejecutar todo el pipeline:

    from m06__feature_selection import run_pipeline
"""

from importlib import import_module as _imp

def run_pipeline(*args, **kwargs):
    """Proxy a m06__feature_selection.run.run_pipeline()"""
    return _imp(__name__ + ".run").run_pipeline(*args, **kwargs)

__all__ = ["run_pipeline"]
