# /.../m06__feature_selection/step02_filtering/__init__.py

"""
step02_filtering
─────────────────
Filtro estadístico inicial de variables.
"""

from .core.filter_roughfs import FilterRoughFS, filter_partitions

__all__ = ["FilterRoughFS", "filter_partitions"]
