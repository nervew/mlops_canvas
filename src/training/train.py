"""Model training entrypoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MLFLOW_EXPERIMENT = "mlops-canvas"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_params(path: Path) -> dict[str, float]:
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    params = load_params(args.parameters)
    df = pd.read_parquet(args.data / "prepared.parquet")

    X = df.drop(columns=[params["target"]])
    y = df[params["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=params.get("n_estimators", 200),
                    max_depth=params.get("max_depth"),
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run():
        mlflow.log_params({k: v for k, v in params.items() if k != "target"})
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        mlflow.log_metric("auc", auc)
        args.output.mkdir(parents=True, exist_ok=True)
        model_path = args.output / "model.pkl"
        mlflow_model_path = args.output / "mlflow-model"
        mlflow.sklearn.save_model(pipeline, path=str(mlflow_model_path))
        mlflow.log_artifact(str(mlflow_model_path), artifact_path="model")
        mlflow.log_artifact(
            args.data / "metadata.json",
            artifact_path="data",
        )
        mlflow.log_metric("roc_auc", auc)
        model_path.write_bytes(b"")


if __name__ == "__main__":
    main()
