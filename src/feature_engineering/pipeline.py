from __future__ import annotations

from pathlib import Path

from . import (
    IrisLoader,
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
        self.loader = IrisLoader()
        self.imputer = SimpleImputerAdapter()
        self.outliers = IQRHandler()
        self.transformer = StandardScaleTransformer()
        self.encoder = OneHotEncoderAdapter()
        self.generator = PolynomialFeatureGenerator()
        self.metrics = JsonMetricsExporter()
        self.drift = PSIDriftDetector()

    def run(self) -> None:
        X_train, X_test, X_val, y_train, y_test, y_val = self.loader.load()
        X_train = self.imputer.fit_transform(X_train)
        X_test = self.imputer.transform(X_test)
        X_val = self.imputer.transform(X_val)

        X_train = self.outliers.fit_transform(X_train)
        X_test = self.outliers.transform(X_test)
        X_val = self.outliers.transform(X_val)

        X_train = self.transformer.fit_transform(X_train)
        X_test = self.transformer.transform(X_test)
        X_val = self.transformer.transform(X_val)

        X_train = self.encoder.fit_transform(X_train)
        X_test = self.encoder.transform(X_test)
        X_val = self.encoder.transform(X_val)

        X_train = self.generator.fit_transform(X_train)
        X_test = self.generator.transform(X_test)
        X_val = self.generator.transform(X_val)

        final_train = X_train.copy()
        final_train["target"] = y_train
        self.metrics.export(final_train, "metrics/feature_metrics.json")

        drift = self.drift.compute(X_train, X_test)
        Path("metrics").mkdir(exist_ok=True)
        with open("metrics/drift.json", "w", encoding="utf-8") as fh:
            import json

            json.dump(drift, fh, indent=4)


def run_pipeline() -> None:
    pipeline = FeatureEngineeringPipeline()
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
