"""
Paquete m05_feature_engineering
———————————————
Para evitar dependencias circulares NO importamos los sub-módulos aquí.
Quien los necesite debe hacer:

    from m05_feature_engineering.step03_outliers import IQRHandler
    from m05_feature_engineering.step02_imputation import SimpleImputerAdapter
    …
"""

# (opcional) helper de importación perezosa
import importlib

def __getattr__(name: str):
    """Permite acceder perezosamente a sub-módulos por atributo."""
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as e:
        raise AttributeError(f"{__name__} no contiene {name}") from e
