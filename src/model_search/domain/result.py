from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV

@dataclass
class SearchResult:
    grid: GridSearchCV
