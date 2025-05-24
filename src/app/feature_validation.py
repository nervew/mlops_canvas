import os
import json
import pandas as pd
import great_expectations as ge
from great_expectations.profile.basic_dataset_profiler import BasicDatasetProfiler
import mlflow

class InitialFeatureValidator:
    """
    Entrena un conjunto de expectativas Great Expectations sobre un DataFrame inicial,
    genera un reporte univariado y guarda el suite en JSON para uso posterior.
    """
    def __init__(self, expectations_filepath: str, mlflow_experiment_name: str = None):
        self.expectations_filepath = expectations_filepath
        if mlflow_experiment_name:
            mlflow.set_experiment(mlflow_experiment_name)

    def fit(self, df: pd.DataFrame, target: str = None, suite_name: str = "initial_feature_suite") -> dict:
        # 1. Convertir a GE Dataset
        dataset = ge.from_pandas(df)

        # 2. Crear o sobrescribir expectation suite
        dataset.create_expectation_suite(suite_name, overwrite_existing=True)

        # 3. Perfilado automático con BasicDatasetProfiler
        profiler = BasicDatasetProfiler()
        suite = profiler.profile(dataset)

        # 4. Guardar suite como JSON en Blob (o DBFS)
        os.makedirs(os.path.dirname(self.expectations_filepath), exist_ok=True)
        dataset.save_expectation_suite(
            expectation_suite=suite,
            expectation_suite_name=suite_name,
            filepath=self.expectations_filepath
        )

        # 5. Log en MLflow si hay run activa
        if mlflow.active_run():
            mlflow.log_artifact(self.expectations_filepath, artifact_path="expectations")

        # 6. Validación inmediata para reporte univariado inicial
        result = dataset.validate(expectation_suite=suite, result_format="SUMMARY")
        return result.to_json_dict()


class PipelineFeatureValidator:
    """
    Carga el expectation suite entrenado y valida un DataFrame nuevo,
    devolviendo un reporte de éxito/fallo.
    """
    def __init__(self, expectations_filepath: str):
        self.expectations_filepath = expectations_filepath
        # Cargar suite desde JSON
        with open(self.expectations_filepath, "r") as f:
            suite_dict = json.load(f)
        # Reconstruir ExpectationSuite
        self.suite = ge.core.ExpectationSuite(**suite_dict)

    def validate(self, df: pd.DataFrame) -> dict:
        dataset = ge.from_pandas(df)
        result = dataset.validate(expectation_suite=self.suite, result_format="SUMMARY")
        return result.to_json_dict()


# -------------------------
# Ejemplo de uso en Databricks
# -------------------------
# from pyspark.sql import SparkSession
# spark = SparkSession.builder.getOrCreate()
#
# # Leer tablas temporales
# df_original = spark.table("temp_features_originales").toPandas()
# df_crudos   = spark.table("temp_features_crudos").toPandas()
#
# # Paso inicial (fit)
# trainer = InitialFeatureValidator(
#     expectations_filepath="/dbfs/mnt/blobstore/expectations/feature_suite.json",
#     mlflow_experiment_name="FeatureValidation"
# )
# reporte_inicial = trainer.fit(df_original, target="mi_target")
#
# # Paso de pipeline (validate)
# validator = PipelineFeatureValidator(
#     expectations_filepath="/dbfs/mnt/blobstore/expectations/feature_suite.json"
# )
# reporte_pipeline = validator.validate(df_crudos)
#
# print("Inicial success:", reporte_inicial["success"])
# print("Pipeline success:", reporte_pipeline["success"])
