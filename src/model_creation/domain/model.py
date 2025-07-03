from dataclasses import dataclass
from sklearn.base import BaseEstimator

@dataclass
class TrainedModel:
    model: BaseEstimator
