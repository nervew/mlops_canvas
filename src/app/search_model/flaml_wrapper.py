import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from melitk import logging
logger = logging.getLogger("Model_Selector")

from .automl_base import AutoMLBase

try:
    from flaml import AutoML
except ImportError:
    print("FLAML no está instalado. Instálalo usando `poetry add flaml`.")
    sys.exit(1)


class FLAMLWrapper(AutoMLBase):
    """
    Wrapper para FLAML que facilita:
      - Configuración personalizada de búsqueda AutoML
      - Conversión de series pandas a arrays NumPy
      - Mapeo de nombres de estimadores a variantes legibles
    """

    _NAME_MAP: Dict[str, str] = {
        "lgbm": "LightGBM",
        "xgb_limitdepth": "XGBoost (Limit Depth)",
        "xgboost": "XGBoost",
        "rf": "Random Forest",
        "extra_tree": "Extra Trees",
        "lrl1": "Logistic Regression (L1)",
        "sgd": "Stochastic Gradient Descent",
    }

    def __init__(
        self,
        time_budget: int = 360,
        metric: Optional[str] = None,
        verbose: int = -1,
        log_file: str = "flaml.log",
        preprocess: bool = False
    ):
        """
        Args:
            time_budget: Máximo tiempo (segundos) para la búsqueda AutoML.
            metric: Métrica a usar; si None, se determina automáticamente.
            verbose: Nivel de verbosidad de FLAML.
            log_file: Nombre del archivo de log de FLAML.
        """
        super().__init__("FLAML")
        self.time_budget = time_budget
        self.metric_override = metric
        self.verbose = verbose
        self.log_file = log_file
        self.preprocess = preprocess
        self.automl = AutoML()
        self.model: Any = None
        self.best_params: Dict[str, Any] = {}
        self.model_ranking: pd.DataFrame = pd.DataFrame()

    def fit(self, 
            X_train: Union[pd.DataFrame, np.ndarray],
            y_train: Union[pd.Series, np.ndarray, List[Any]],
            X_test: Union[pd.DataFrame, np.ndarray],
            y_test: Union[pd.Series, np.ndarray, List[Any]]
    ) -> None:
        """
        Entrena el buscador AutoML con datos de entrenamiento.

        Args:
            X_train: DataFrame o array de características.
            y_train: Series, array o lista de etiquetas.

        Raises:
            ValueError: Si X_train o y_train están vacíos.
        """
        # Validar X_train
        if isinstance(X_train, np.ndarray):
            X = pd.DataFrame(X_train)
        else:
            X = X_train.copy()
        if X.empty:
            raise ValueError("❌ X_train no puede estar vacío.")

        # Convertir y_train a numpy array y validar
        y_arr = (
            y_train.values
            if isinstance(y_train, pd.Series)
            else np.asarray(y_train)
        )
        if y_arr.size == 0:
            raise ValueError("❌ y_train no puede estar vacío.")

        # Determinar métrica
        if self.metric_override:
            metric = self.metric_override
        else:
            n_classes = np.unique(y_arr).size
            metric = "roc_auc" if n_classes == 2 else "roc_auc_ovr"

        settings: Dict[str, Any] = {
            "task": "classification",
            "time_budget": self.time_budget,
            "log_file_name": self.log_file,
            "skip_transform": self.preprocess,
            "model_history": True,
            "log_training_metric": True,
            "verbose": self.verbose,
            "metric": metric,
        }

        # Iniciar entrenamiento
        self.automl.fit(X_train=X, y_train=y_arr, 
                        X_val = X_test, y_val = y_test, **settings)
        model_name = self.automl.best_estimator #self.automl.model
        self.model = [self._NAME_MAP.get(model_name, model_name), self.automl.model]
        self.best_params = dict(self.automl.best_config)

        # Construir ranking como DataFrame
        records: List[Dict[str, Any]] = [] 
        for est in self.automl.estimator_list:
            best_md = self.automl.best_model_for_estimator(est)
            if best_md is None:
                continue
            hiperparameters_est = self.automl.best_config_per_estimator[est]
            train_score = 1 - self.automl.best_loss_per_estimator[est]
            name = self._NAME_MAP.get(est, est)
            try:
                test_score = best_md.score(X_val=X_test, y_val=y_test)
            except Exception as e:
                logger.warning("⚠️ Error while testing score for model '%s': %s", name, e)
                test_score = 0
            records.append({
                "estimator_name": name, 
                "metric_train": train_score, 
                "metric_test": test_score,
                "hiperparameters": hiperparameters_est
                })

        self.model_ranking = (
            pd.DataFrame(records)
            .sort_values(by="metric_test", ascending=False)
            .reset_index(drop=True)
        )

    def predict_proba(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predice probabilidades para nuevos datos."""
        return self.automl.predict_proba(
            X_test if isinstance(X_test, np.ndarray) else X_test.values
        )

    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predice etiquetas para nuevos datos."""
        return self.automl.predict(
            X_test if isinstance(X_test, np.ndarray) else X_test.values
        )

    def get_best_model(self) -> Any:
        """Devuelve el modelo con mejor desempeño."""
        return self.model

    def get_model_ranking(self) -> pd.DataFrame:
        """Devuelve un DataFrame ordenado de estimadores y sus métricas."""
        return self.model_ranking.copy()

    def get_best_params(self) -> Dict[str, Any]:
        """Devuelve los hiperparámetros del mejor modelo."""
        return dict(self.best_params)
