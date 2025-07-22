from dataclasses import dataclass
from sklearn.decomposition import PCA

@dataclass
class FeatureReducer:
    reducer: PCA
