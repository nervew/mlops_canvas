import os
import json
import pandas as pd
import great_expectations as ge
from great_expectations.profile.user_configurable_profiler import UserConfigurableProfiler
import mlflow

class InitialFeatureValidator:
    """
    Entrena un conjunto de expectativas Great Expectations sobre un DataFrame inicial,
    genera un reporte univariado y guarda las reglas en un archivo (JSON) para su uso
    posterior en blob storage y, opcionalmente, en MLflow.
    """
    def __init__(
        self,
        expectations_filepath: str,
        mlflow_experiment_name: str = None
    ):
        """
        :param expectations_filepath: Ruta local o montada a Blob Storage para guardar el suite JSON.
        :param mlflow_experiment_name: (Opcional) nombre del experimento en MLflow para trazar artefactos.
        """
        self.expectations_filepath = expectations_filepath
        if mlflow_experiment_name:
            mlflow.set_experiment(mlflow_experiment_name)

    def fit(
        self,
        df: pd.DataFrame,
        target: str = None,
        suite_name: str = "initial_feature_suite"
    ) -> dict:
        """
        Ajusta (fit) un suite de expectativas a df, genera el reporte y guarda el suite.
        
        :param df: DataFrame con features originales.
        :param target: Nombre de la columna target (si aplica).
        :param suite_name: Nombre interno del expectation suite.
        :return: Reporte de validación como dict.
        """
        # 1. Crear dataset GE
        dataset = ge.from_pandas(df)

        # 2. Crear o sobreescribir expectation suite
        suite = dataset.create_expectation_suite(
            expectation_suite_name=suite_name,
            overwrite_existing=True
        )

        # 3. Construir un suite con profiler automático
        profiler = UserConfigurableProfiler(
            dataset=dataset,
            excluded_expectations=None,
            ignored_columns=[target] if target else None,
            not_null_only=False
        )
        suite = profiler.build_suite()

        # 4. Guardar suite en JSON local / blob
        os.makedirs(os.path.dirname(self.expectations_filepath), exist_ok=True)
        with open(self.expectations_filepath, "w") as f:
            f.write(json.dumps(suite.to_json_dict(), indent=2))

        # 5. Log del artefacto a MLflow (si hay run activa)
        if mlflow.active_run():
            mlflow.log_artifact(self.expectations_filepath, artifact_path="expectations")

        # 6. Validar de inmediato para generar reporte univariado inicial
        result = dataset.validate(
            expectation_suite=suite,
            result_format="SUMMARY"
        )
        return result.to_json_dict()


class PipelineFeatureValidator:
    """
    Carga un suite de expectativas previamente entrenado y valida un DataFrame nuevo,
    devolviendo un reporte de éxito/fallo.
    """
    def __init__(self, expectations_filepath: str):
        """
        :param expectations_filepath: Ruta al JSON con el expectation suite entrenado.
        """
        self.expectations_filepath = expectations_filepath
        # Cargar suite desde JSON
        with open(self.expectations_filepath, "r") as f:
            suite_dict = json.load(f)
        self.suite = ge.core.ExpectationSuite(**suite_dict)

    def validate(
        self,
        df: pd.DataFrame
    ) -> dict:
        """
        Valida df contra el expectation suite cargado.
        
        :param df: DataFrame con features crudos.
        :return: Reporte de validación como dict, incluyendo un campo "success" True/False.
        """
        dataset = ge.from_pandas(df)
        result = dataset.validate(
            expectation_suite=self.suite,
            result_format="SUMMARY"
        )
        return result.to_json_dict()


# -------------------------
# Ejemplo de uso en Databricks:
# -------------------------
# from pyspark.sql import SparkSession
# spark = SparkSession.builder.getOrCreate()
#
# # Leer tabla temporal de features originales
# df_original = spark.table("temp_features_originales").toPandas()
# df_crudos   = spark.table("temp_features_crudos").toPandas()
#
# # Inicialización
# trainer = InitialFeatureValidator(
#     expectations_filepath="/dbfs/mnt/blobstore/expectations/feature_suite.json",
#     mlflow_experiment_name="FeatureValidation"
# )
# reporte_inicial = trainer.fit(df_original, target="mi_target")
#
# # Pipeline de validación
# validator = PipelineFeatureValidator(
#     expectations_filepath="/dbfs/mnt/blobstore/expectations/feature_suite.json"
# )
# reporte_pipeline = validator.validate(df_crudos)
#
# print("Inicial success:", reporte_inicial["success"])
# print("Pipeline success:", reporte_pipeline["success"])

