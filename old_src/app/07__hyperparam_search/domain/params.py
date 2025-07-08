from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV

@dataclass
class HyperParamResult:
    grid: GridSearchCV
