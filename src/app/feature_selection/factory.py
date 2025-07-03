# factory.py
from fs_variance import VarianceThresholdSelector
from fs_mutual_info import MutualInfoSelector
from fs_rfe import RFESelector

class SelectorFactory:
    """Fábrica para crear instancias de FeatureSelector por nombre."""
    _registry = {
        "variance": VarianceThresholdSelector,
        "mutual_info": MutualInfoSelector,
        "rfe": RFESelector,
    }

    @classmethod
    def create(cls, name: str, **params) -> FeatureSelector:
        if name not in cls._registry:
            raise ValueError(f"Selector '{name}' no registrado.")
        return cls._registry[name](**params)