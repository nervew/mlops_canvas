# m08_feature_reduction/step05_pipeline/ports/builder.py

from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from pandas import DataFrame
from typing import List, Any

class IPipelineBuilder(ABC):
    @abstractmethod
    def build(
        self,
        transformer: Any,
        features: List[str],
        reducer: Any
    ) -> Pipeline:
        """
        Construye y devuelve un sklearn.pipeline.Pipeline que encadena:
        1) transformer (preprocesamiento)
        2) selector de columnas (features)
        3) reductor (opcional)
        """
        pass
