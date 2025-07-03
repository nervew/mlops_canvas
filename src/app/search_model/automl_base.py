from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Definir una clase base para AutoML
class AutoMLBase(ABC):
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.best_params = None
        self.metrics = {}
        self.model_ranking = None

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        pass

    @abstractmethod
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def get_best_model(self) -> Any:
        pass

    @abstractmethod
    def get_model_ranking(self) -> Any:
        pass

    @abstractmethod
    def get_best_params(self) -> Dict:
        pass

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        y_pred_proba = self.predict_proba(X_test)
        y_pred = self.predict(X_test)

        # Calcular ROC AUC solo si es binario o multiclase
        if len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        self.metrics = {
            'ROC AUC': roc_auc,
            'Accuracy': acc,
            'F1 Score': f1
        }