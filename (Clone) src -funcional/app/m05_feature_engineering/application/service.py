import pandas as pd

from ..infrastructure.robust_fe import build_transformer
from ..domain.transformer import FeatureTransformer


def run(df: pd.DataFrame, *, target: str = "target") -> FeatureTransformer:
    """Orquesta la construcción del pipeline de FE."""
    transformer_pipeline = build_transformer(df, target=target)
    return FeatureTransformer(transformer=transformer_pipeline)
