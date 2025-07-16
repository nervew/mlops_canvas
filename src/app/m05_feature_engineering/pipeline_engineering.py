from __future__ import annotations
from pathlib import Path

from .step01_import      import ParquetPartitionLoader
from .step02_imputation  import SimpleImputerAdapter
from .step03_outliers    import IQRHandler
from .step04_transformation import StandardScaleTransformer
from .step05_encoding    import OneHotEncoderAdapter
from .step06_feature_gen import PolynomialFeatureGenerator
from .step07_metrics     import JsonMetricsExporter
from .step08_drift       import PSIDriftDetector
from .step09_export      import ParquetExportAdapter


class FeatureEngineeringPipeline:
    def __init__(self) -> None:
        self.loader      = ParquetPartitionLoader()
        self.imputer     = SimpleImputerAdapter()
        self.outliers    = IQRHandler()
        self.transformer = StandardScaleTransformer()
        self.encoder     = OneHotEncoderAdapter()
        self.generator   = PolynomialFeatureGenerator()
        self.metrics     = JsonMetricsExporter()
        self.drift       = PSIDriftDetector()
        # Ruta relativa: src/data/processed
        self.exporter    = ParquetExportAdapter()

    def run(self) -> None:
        X_train, X_test, X_back, y_train, y_test, y_back = self.loader.load()

        # --- Imputación ---
        X_train = self.imputer.fit_transform(X_train)
        X_test  = self.imputer.transform(X_test)
        X_back  = self.imputer.transform(X_back)

        # --- Outliers ---
        X_train = self.outliers.fit_transform(X_train)
        X_test  = self.outliers.transform(X_test)
        X_back  = self.outliers.transform(X_back)

        # --- Escalado ---
        X_train = self.transformer.fit_transform(X_train)
        X_test  = self.transformer.transform(X_test)
        X_back  = self.transformer.transform(X_back)

        # --- One-Hot Encoding ---
        X_train = self.encoder.fit_transform(X_train)
        X_test  = self.encoder.transform(X_test)
        X_back  = self.encoder.transform(X_back)

        # --- Features polinomiales ---
        X_train = self.generator.fit_transform(X_train)
        X_test  = self.generator.transform(X_test)
        X_back  = self.generator.transform(X_back)

        # Añadimos target
        X_train_final = X_train.copy(); X_train_final["target"] = y_train
        X_test_final  = X_test.copy();  X_test_final["target"]  = y_test
        X_back_final  = X_back.copy();  X_back_final["target"]  = y_back

        # --- Guardar DataFrames procesados ---
        self.exporter.export(X_train_final, X_test_final, X_back_final)

        # --- Métricas de train ---
        Path("metrics").mkdir(exist_ok=True)
        self.metrics.export(X_train_final, "metrics/feature_metrics.json")

        # --- Drift (train vs test) ---
        drift = self.drift.compute(X_train, X_test)
        with open("metrics/drift.json", "w", encoding="utf-8") as fh:
            import json; json.dump(drift, fh, indent=4)


def run_pipeline() -> None:
    FeatureEngineeringPipeline().run()


if __name__ == "__main__":
    run_pipeline()
