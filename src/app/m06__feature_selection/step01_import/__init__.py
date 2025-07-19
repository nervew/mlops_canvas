
"""
Submódulo step01_import del módulo m06__feature_selection.

Expone ParquetPartitionLoader2 para cargar las tres particiones procesadas.
"""

from .adapters.parquet_loader import ParquetPartitionLoader2

__all__ = ["ParquetPartitionLoader2"]
