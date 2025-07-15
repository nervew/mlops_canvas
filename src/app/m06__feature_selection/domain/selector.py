from dataclasses import dataclass
from sklearn.feature_selection import SelectKBest

@dataclass
class FeatureSelector:
    selector: SelectKBest
