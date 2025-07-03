import pandas as pd
from ..infrastructure.simple_fe import build_transformer
from ..domain.transformer import FeatureTransformer


def run(df: pd.DataFrame) -> FeatureTransformer:
    transformer = build_transformer(df)
    return FeatureTransformer(transformer=transformer)
