import pandas as pd
from sklearn.decomposition import PCA


def build_reducer(n_components: int) -> PCA:
    return PCA(n_components=n_components)
