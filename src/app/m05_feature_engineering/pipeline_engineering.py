from __future__ import annotations

from pathlib import Path
import json
import joblib
from sklearn.pipeline import Pipeline

# Pasos del módulo
from .step01_import        import ParquetPartitionLoader
from .step02_imputation    import SimpleImputerAdapter
from .step03_outliers      import IQRHandler
from .step04_transformation import StandardScaleTransformer
from .step05_encoding      import OneHotEncoderAdapter
from .step06_feature_gen   import PolynomialFeatureGenerator
from .step07_metrics       import JsonMetricsExporter
from .step08_drift         import PSIDriftDetector
from .step09_export        import ParquetExportAdapter


class FeatureEngineeringPipeline:
    def __init__(self) -> None:
        # Creamos un único pipeline con todos los transformers
        self.preprocess = Pipeline(steps=[
            ("imputer",   SimpleImputerAdapter()),
            ("outliers",  IQRHandler()),
            ("scaler",    StandardScaleTransformer()),
            ("encoder",   OneHotEncoderAdapter()),
            ("generator", PolynomialFeatureGenerator()),
        ])

        self.loader   = ParquetPartitionLoader()
        self.exporter = ParquetExportAdapter()
        self.metrics  = JsonMetricsExporter()
        self.drift    = PSIDriftDetector()

    def run(self) -> None:
        # 1) Cargo train/test/back
        X_train, X_test, X_back, y_train, y_test, y_back = self.loader.load()

        # 2) FIT del pipeline (marca la instancia como fitted)
        self.preprocess.fit(X_train, y_train)

        # 3) TRANSFORM en los tres splits (sin warnings)
        X_train = self.preprocess.transform(X_train)
        X_test  = self.preprocess.transform(X_test)
        X_back  = self.preprocess.transform(X_back)

        # 4) Serializo el pipeline ajustado
        transformers_dir = Path(__file__).resolve().parents[2] / "transformers"
        transformers_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocess, transformers_dir / "transformador_inicial.joblib")

        # 5) Añado target y exporto Parquet
        X_train_final = X_train.copy(); X_train_final["target"] = y_train
        X_test_final  = X_test.copy();  X_test_final["target"]  = y_test
        X_back_final  = X_back.copy();  X_back_final["target"]  = y_back
        self.exporter.export(X_train_final, X_test_final, X_back_final)

        # 6) Métricas descriptivas
        Path("metrics").mkdir(exist_ok=True)
        self.metrics.export(X_train_final, "metrics/feature_metrics.json")

        # 7) Drift train vs test
        drift = self.drift.compute(X_train, X_test)
        with open("metrics/drift.json", "w", encoding="utf-8") as fh:
            json.dump(drift, fh, indent=4)


def run_pipeline() -> None:
    FeatureEngineeringPipeline().run()


if __name__ == "__main__":
    run_pipeline()
