# m08_feature_reduction/step05_pipeline/adapters/pipeline_builder.py

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from ..ports.builder import IPipelineBuilder

class PipelineBuilder(IPipelineBuilder):
    """
    Construye un pipeline final que:
      1) Aplica el transformer inicial completo.
      2) Selecciona solo las columnas finales.
      3) Aplica el reductor (si está activado).
    """

    def build(self, transformer, features, reducer):
        # 1) Selector de columnas: passthrough solo para las columnas en 'features'
        selector = ColumnTransformer(
            transformers=[("selector", "passthrough", features)],
            remainder="drop"
        )

        steps = [
            ("preprocessor", transformer),
            ("selector", selector)
        ]

        # 2) Agregar reductor si existe
        if getattr(reducer, "reducer", None) is not None:
            steps.append(("reducer", reducer.reducer))

        return Pipeline(steps)
