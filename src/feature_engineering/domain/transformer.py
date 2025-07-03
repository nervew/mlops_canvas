from dataclasses import dataclass
from sklearn.base import TransformerMixin

@dataclass
class FeatureTransformer:
    transformer: TransformerMixin
