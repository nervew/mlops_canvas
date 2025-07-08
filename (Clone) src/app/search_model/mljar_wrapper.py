import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
try:
    from supervised.automl import AutoML
except ImportError:
    print("MLJAR no está instalado. Instálalo usando `poetry add mljar-supervised`.")
    sys.exit(1)

from .automl_base import AutoMLBase

# Wrapper para MLJAR
class MLJARWrapper(AutoMLBase):
    def __init__(self):
        super().__init__("MLJAR")
        self.automl = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        # Calcular ROC AUC solo si es binario o multiclase
        if len(np.unique(y_train)) == 2:
            model_type = 'binary_classification'
            eval_metric = 'auc'
        else:
            model_type = 'multiclass_classification'
            eval_metric = 'f1'

        # Configurar el AutoML
        self.automl = AutoML(
            mode="Compete",  # 'Compete' para competencia de modelos
            algorithms=["CatBoost", "Xgboost", "LightGBM", "Random Forest", "Extra Trees", "Neural Network"],  # Puedes añadir más algoritmos
            total_time_limit=10,  # Tiempo total en segundos
            ml_task=model_type,
            eval_metric=eval_metric,
            random_state=42
        )
        # Entrenar el modelo
        self.automl.fit(X_train, y_train)
        # Obtener el leaderboard
        leaderboard = self.automl.get_leaderboard()
        self.model = leaderboard.iloc[0]['model_type']
        self.best_params = self.automl._best_model.learner_params
        leaderboard_model = pd.DataFrame(self.automl.report()["leaderboard"])
        self.model_ranking = leaderboard_model[["model_name", "metric_value"]] #TODO: Cambiar el nombre de las columas por estandres

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.automl.predict_proba(X_test)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.automl.predict(X_test)

    def get_best_model(self) -> Any:
        return self.automl
    
    def get_model_ranking(self):
        return self.model_ranking

    def get_best_params(self) -> Dict:
        return self.best_params
