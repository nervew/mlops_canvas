"""
Módulo instalador.  Solo expone ``install_requirements`` para que
``pipeline.py`` no tenga que conocer los detalles de instalación.
"""
from .service import install_requirements

__all__ = ["install_requirements"]
