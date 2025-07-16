from __future__ import annotations

from pathlib import Path

from . import (
    # Reemplazamos IrisLoader por ParquetPartitionLoader
    ParquetPartitionLoader,
    SimpleImputerAdapter,
    IQRHandler,
    StandardScaleTransformer,
    OneHotEncoderAdapter,
    PolynomialFeatureGenerator,
    JsonMetricsExporter,
    PSIDriftDetector,
)


class FeatureEngineeringPipeline:
    def __init__(self) -> None:
        # Creamos nuestro loader de Parquet
        self.loader = ParquetPartitionLoader()
        self.imputer = SimpleImputerAdapter()
        self.outliers = IQRHandler()
        self.transformer = StandardScaleTransformer()
        self.encoder = OneHotEncoderAdapter()
        self.generator = PolynomialFeatureGenerator()
        self.metrics = JsonMetricsExporter()
        self.drift = PSIDriftDetector()

    def run(self) -> None:
        # Ahora loader.load() devuelve: X_train, X_test, X_backtest, y_train, y_test, y_backtest
        X_train, X_test, X_backtest, y_train, y_test, y_backtest = self.loader.load()

        # --- Imputación ---
        X_train = self.imputer.fit_transform(X_train)
        X_test = self.imputer.transform(X_test)
        X_backtest = self.imputer.transform(X_backtest)

        # --- Detección y tratamiento de outliers ---
        X_train = self.outliers.fit_transform(X_train)
        X_test = self.outliers.transform(X_test)
        X_backtest = self.outliers.transform(X_backtest)

        # --- Escalado/Estandarización ---
        X_train = self.transformer.fit_transform(X_train)
        X_test = self.transformer.transform(X_test)
        X_backtest = self.transformer.transform(X_backtest)

        # --- Codificación de categóricas ---
        X_train = self.encoder.fit_transform(X_train)
        X_test = self.encoder.transform(X_test)
        X_backtest = self.encoder.transform(X_backtest)

        # --- Generación de features adicionales ---
        X_train = self.generator.fit_transform(X_train)
        X_test = self.generator.transform(X_test)
        X_backtest = self.generator.transform(X_backtest)

        # --- Exportar métricas de train ---
        final_train = X_train.copy()
        final_train["target"] = y_train
        Path("metrics").mkdir(exist_ok=True)
        self.metrics.export(final_train, "metrics/feature_metrics.json")

        # --- Cálculo de drift (train vs. test) ---
        drift = self.drift.compute(X_train, X_test)
        with open("metrics/drift.json", "w", encoding="utf-8") as fh:
            import json

            json.dump(drift, fh, indent=4)


def run_pipeline() -> None:
    pipeline = FeatureEngineeringPipeline()
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
